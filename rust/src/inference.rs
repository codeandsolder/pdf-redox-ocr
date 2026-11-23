//! Native Rust PP-OCRv6 Small adapter built on OAR task adapters and ONNX Runtime.
//!
//! Detection and both primary/body recognition share one loaded model pair.
//! Original raster geometry is retained; there is no document rectification.

use std::{
    collections::BTreeSet,
    num::NonZeroUsize,
    sync::{Arc, OnceLock},
};

use image::RgbImage;
use num_traits::ToPrimitive;
use oar_ocr::{
    core::{
        OCRError,
        config::{OrtExecutionProvider, OrtSessionConfig},
        traits::{
            OrtConfigurable,
            adapter::{AdapterBuilder, ModelAdapter},
            task::ImageTaskInput,
        },
    },
    domain::{
        adapters::{
            TextDetectionAdapter, TextDetectionAdapterBuilder, TextLineOrientationAdapter,
            TextLineOrientationAdapterBuilder,
        },
        tasks::TextDetectionConfig,
    },
    download::resolve_path,
    models::recognition::crnn::{CRNNModel, CRNNModelBuilder, CRNNModelOutput},
    oarocr::{EdgeProcessor, TextCroppingProcessor},
    processors::{BoundingBox, CTCLabelDecode, LimitType, sort_quad_boxes},
    utils::get_rotate_crop_image,
};

use crate::{
    Error,
    character_alignment::{
        BatchGeometry, CharacterAuditReport, CharacterConfirmation, ConfirmationOutcome,
        INPUT_SHAPE, LineOrientationObservation, MAX_INPUT_WIDTH,
    },
    crop_geometry::CropGeometry,
    crop_repair::{BodyPrediction, RepairCrop, RepairPolicy, RepairReport, repair_crops},
    ctc_evidence::{CtcEvidence, CtcOutcome},
    geometry::{Point, Quad},
    glyph_confirmation::ConfirmationPolicy,
    recognition::{PixelLine, RasterRecognition, RecognitionBackend},
    recognition_model::RecognitionModel,
};

const DETECTION_MODEL: &str = "pp-ocrv6_small_det.onnx";
impl RecognitionModel {
    const fn filename(self) -> &'static str {
        match self {
            Self::Small => "pp-ocrv6_small_rec.onnx",
            Self::Medium => "pp-ocrv6_medium_rec.onnx",
        }
    }
    const fn profile(self) -> &'static str {
        match self {
            Self::Small => "oar-ocr PP-OCRv6 Small detector-max-2048",
            Self::Medium => "oar-ocr PP-OCRv6 Small detector-max-2048 + Medium recognizer",
        }
    }
}
/// Optional classification of the rectified text-line reading direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum LineOrientationMode {
    #[default]
    Disabled,
    PpLcnet,
}

const DICTIONARY: &str = "ppocrv6_dict.txt";
const REGION_BATCH_SIZE: usize = 4;
const ORIENTATION_MODEL: &str = "pp-lcnet_x0_25_textline_ori.onnx";
const ORIENTATION_BATCH_SIZE: usize = 32;
const HALF_TURN_THRESHOLD: f32 = 0.9;

/// Explicit evidence depth; ordinary recognition keeps the compact OAR path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterEvidence {
    Off,
    Geometry,
    Ctc,
}

impl CharacterEvidence {
    const fn positions(self) -> bool {
        !matches!(self, Self::Off)
    }

    const fn confirmation(self) -> Self {
        if matches!(self, Self::Ctc) {
            Self::Ctc
        } else {
            Self::Geometry
        }
    }
}

/// One lazily initialized CPU OCR pipeline reused across multiple rasters.
pub struct OcrInference {
    backend: OnceLock<Result<OcrModels, String>>,
    threads: Option<NonZeroUsize>,
    repair: RepairPolicy,
    character_audit: CharacterEvidence,
    model: RecognitionModel,
    alternative: Option<AlternativeRecognizer>,
    line_orientation: LineOrientationMode,
    alternative_backend: OnceLock<Result<RecognitionSessionModel, String>>,
    profile: String,
}

#[derive(Clone, Copy, Debug)]
pub struct AlternativeRecognizer {
    pub model: RecognitionModel,
    pub policy: ConfirmationPolicy,
}

struct OcrModels {
    detection: TextDetectionAdapter,
    recognition: RecognitionSessionModel,
    orientation: Option<TextLineOrientationAdapter>,
}

/// Retain actual page/native-source detail up to a bounded detector size.
///
/// The OAR preset limits the longest side to 960 pixels, which erases
/// small supplier-PDF labels. Unlike minimum-side resizing, a 2048-pixel
/// maximum never needlessly upscales tiny embedded image sources.
/// Postprocessing thresholds remain the official PP-OCRv6 Small defaults.
const fn small_detector_config() -> TextDetectionConfig {
    TextDetectionConfig {
        score_threshold: 0.2,
        box_threshold: 0.45,
        unclip_ratio: 1.4,
        max_candidates: 3000,
        limit_side_len: Some(2048),
        limit_type: Some(LimitType::Max),
        max_side_len: Some(2048),
    }
}

fn inference_error(error: &OCRError) -> Error {
    Error::Inference(error.to_string())
}

struct PrimaryCrop {
    index: usize,
    bbox: BoundingBox,
    pixels: Arc<RgbImage>,
    ratio: f32,
    rotate180: bool,
}

impl PrimaryCrop {
    fn new(index: usize, bbox: BoundingBox, pixels: Arc<RgbImage>) -> Self {
        // Match OAR's width-sorted four-region batches exactly.
        let ratio = pixels.width().to_f32().unwrap_or_default()
            / pixels.height().max(1).to_f32().unwrap_or(1.0);
        Self {
            index,
            bbox,
            pixels,
            ratio,
            rotate180: false,
        }
    }
}

fn ort_config(threads: Option<NonZeroUsize>) -> OrtSessionConfig {
    // The implicit ONNX CPU provider retains a separate high-water arena
    // per session. OAR's explicit CPU provider disables that arena.
    // Variable page and line dimensions otherwise compound retained memory.
    let config = OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]);
    if let Some(threads) = threads {
        config
            .with_intra_threads(threads.get())
            .with_inter_threads(1)
    } else {
        config
    }
}

// The same CRNN model supplies compact greedy output or full diagnostic
// distributions. No duplicate session, preprocessing, or model invocation.
pub(crate) struct RecognitionSessionModel {
    model: CRNNModel,
    vocabulary: Vec<char>,
}

pub(crate) struct CropPredictions {
    pub(crate) output: CRNNModelOutput,
    pub(crate) ctc: Vec<Option<CtcOutcome>>,
}

fn score_observation_sequences(
    vocabulary: &[char],
    rows: &[Vec<&[f32]>],
    evidence: &mut [Option<CtcOutcome>],
    primary: &str,
) {
    match (rows, evidence) {
        ([rows], [Some(CtcOutcome::Measured { evidence })]) => {
            evidence.query_primary(vocabulary, rows, primary);
        }
        (
            [before_rows, after_rows],
            [
                Some(CtcOutcome::Measured { evidence: before }),
                Some(CtcOutcome::Measured { evidence: after }),
            ],
        ) => {
            CtcEvidence::pair_queries(vocabulary, before, before_rows, after, after_rows, primary);
        }
        _ => {}
    }
}

impl RecognitionSessionModel {
    pub(crate) fn load(
        model: RecognitionModel,
        threads: Option<NonZeroUsize>,
    ) -> Result<Self, Error> {
        let dictionary = std::fs::read_to_string(
            resolve_path(DICTIONARY).map_err(|error| inference_error(&error))?,
        )?;
        let dictionary = dictionary.lines().map(str::to_owned).collect::<Vec<_>>();
        // Obtain the exact class mapping from OAR, including blank and space.
        let vocabulary = CTCLabelDecode::from_string_list(Some(&dictionary), true, false)
            .get_character_list()
            .to_vec();
        let rec_path = resolve_path(model.filename()).map_err(|error| inference_error(&error))?;
        let model = CRNNModelBuilder::new()
            .model_input_shape(INPUT_SHAPE)
            .max_img_w(MAX_INPUT_WIDTH)
            .character_dict(dictionary)
            .with_ort_config(ort_config(threads))
            .build(rec_path)
            .map_err(|error| inference_error(&error))?;
        Ok(Self { model, vocabulary })
    }

    pub(crate) fn predict(
        &self,
        images: &[&RgbImage],
        mode: CharacterEvidence,
    ) -> Result<CropPredictions, Error> {
        self.predict_impl(images, mode, None)
    }

    pub(crate) fn predict_with_primary(
        &self,
        images: &[&RgbImage],
        primary: &str,
    ) -> Result<CropPredictions, Error> {
        if !(1..=2).contains(&images.len())
            || images
                .windows(2)
                .any(|p| p[0].dimensions() != p[1].dimensions())
        {
            return Err(Error::Invalid("paired CTC crop dimensions mismatch".into()));
        }
        self.predict_impl(images, CharacterEvidence::Ctc, Some(primary))
    }

    fn predict_impl(
        &self,
        images: &[&RgbImage],
        mode: CharacterEvidence,
        primary: Option<&str>,
    ) -> Result<CropPredictions, Error> {
        let (output, ctc) = if mode == CharacterEvidence::Ctc {
            let input = self
                .model
                .preprocess_refs(images)
                .map_err(|error| inference_error(&error))?;
            let probabilities = self
                .model
                .infer(&input)
                .map_err(|error| inference_error(&error))?;
            let output = self.model.postprocess(&probabilities, true);
            let mut ctc = probabilities
                .outer_iter()
                .enumerate()
                .map(|(index, sequence)| {
                    let result = (|| {
                        let text = output
                            .texts
                            .get(index)
                            .ok_or_else(|| Error::Invalid("missing CTC text".into()))?;
                        let columns = output
                            .char_col_indices
                            .get(index)
                            .ok_or_else(|| Error::Invalid("missing CTC columns".into()))?;
                        let data = sequence
                            .as_slice()
                            .ok_or_else(|| Error::Invalid("noncontiguous CTC sequence".into()))?;
                        let width = sequence.shape()[1];
                        if width == 0 {
                            return Err(Error::Invalid("empty CTC vocabulary".into()));
                        }
                        CtcEvidence::measure(
                            &self.vocabulary,
                            data.chunks_exact(width),
                            text,
                            columns,
                        )
                    })();
                    Some(match result {
                        Ok(evidence) => CtcOutcome::Measured { evidence },
                        Err(error) => CtcOutcome::Unavailable {
                            reason: error.to_string(),
                        },
                    })
                })
                .collect::<Vec<_>>();
            if let Some(primary) = primary {
                let matrices = probabilities.outer_iter().collect::<Vec<_>>();
                let rows = matrices
                    .iter()
                    .map(|matrix| {
                        let width = matrix.shape()[1];
                        if width == 0 {
                            return Err(Error::Invalid("empty CTC vocabulary".into()));
                        }
                        Ok(matrix
                            .as_slice()
                            .ok_or_else(|| Error::Invalid("noncontiguous CTC tensor".into()))?
                            .chunks_exact(width)
                            .collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                score_observation_sequences(&self.vocabulary, &rows, &mut ctc, primary);
            }
            (output, ctc)
        } else {
            (
                self.model
                    .forward_refs(images, mode.positions())
                    .map_err(|error| inference_error(&error))?,
                Vec::new(),
            )
        };
        if output.texts.len() != images.len() || output.scores.len() != images.len() {
            return Err(Error::Inference(
                "recognizer returned inconsistent batch dimensions".into(),
            ));
        }
        Ok(CropPredictions { output, ctc })
    }
}

struct ConfirmationCrop {
    detection_index: usize,
    line_index: usize,
    pixels: Arc<RgbImage>,
}

/// The detector polygon remains in its original corner order until every
/// source-crop repair and confirmation has finished.
struct PrimaryObservation {
    line: PixelLine,
    rotate180: bool,
    reading_order: [usize; 4],
}

struct PrimaryRecognition {
    lines: Vec<PixelLine>,
    reading_orders: Vec<[usize; 4]>,
    half_turns: BTreeSet<usize>,
    audit: CharacterAuditReport,
    candidates: Vec<ConfirmationCrop>,
}

impl PrimaryRecognition {
    fn scatter(
        ordered: Vec<Option<PrimaryObservation>>,
        mut audit: CharacterAuditReport,
        mut candidate_crops: std::collections::BTreeMap<usize, Arc<RgbImage>>,
    ) -> Self {
        audit
            .lines
            .sort_unstable_by_key(|entry| entry.detection_index);
        let mut lines = Vec::new();
        let mut candidates = Vec::new();
        let mut reading_orders = Vec::new();
        let mut half_turns = BTreeSet::new();
        for (index, line) in ordered.into_iter().enumerate() {
            if let Some(PrimaryObservation {
                line,
                rotate180,
                reading_order,
            }) = line
            {
                if rotate180 {
                    half_turns.insert(lines.len());
                }
                if let Some(pixels) = candidate_crops.remove(&index) {
                    candidates.push(ConfirmationCrop {
                        detection_index: index,
                        line_index: lines.len(),
                        pixels,
                    });
                }
                lines.push(line);
                reading_orders.push(reading_order);
            }
        }
        Self {
            lines,
            reading_orders,
            half_turns,
            audit,
            candidates,
        }
    }
}

impl OcrModels {
    fn load(
        threads: Option<NonZeroUsize>,
        model: RecognitionModel,
        line_orientation: LineOrientationMode,
    ) -> Result<Self, Error> {
        let det_path = resolve_path(DETECTION_MODEL).map_err(|error| inference_error(&error))?;
        let detection = TextDetectionAdapterBuilder::new()
            .with_config(small_detector_config())
            .with_ort_config(ort_config(threads));
        let orientation = if line_orientation == LineOrientationMode::PpLcnet {
            let path = resolve_path(ORIENTATION_MODEL).map_err(|error| inference_error(&error))?;
            let builder =
                TextLineOrientationAdapterBuilder::new().with_ort_config(ort_config(threads));
            Some(
                builder
                    .build(path)
                    .map_err(|error| inference_error(&error))?,
            )
        } else {
            None
        };
        Ok(Self {
            orientation,
            detection: detection
                .build(det_path)
                .map_err(|error| inference_error(&error))?,
            recognition: RecognitionSessionModel::load(model, threads)?,
        })
    }

    fn primary_crops(&self, image: &Arc<RgbImage>) -> Result<(usize, Vec<PrimaryCrop>), Error> {
        let detection = self
            .detection
            .execute(
                ImageTaskInput::from_arc_images(vec![Arc::clone(image)]),
                None,
            )
            .map_err(|error| inference_error(&error))?;
        let Some(detected) = detection.detections.into_iter().next() else {
            return Err(Error::Inference("detector returned no image result".into()));
        };
        let boxes = sort_quad_boxes(&detected.into_iter().map(|d| d.bbox).collect::<Vec<_>>());
        let count = boxes.len();
        if count == 0 {
            return Ok((0, Vec::new()));
        }
        let pixels = TextCroppingProcessor::new(true)
            .process((Arc::clone(image), boxes.clone()))
            .map_err(|error| inference_error(&error))?;
        if pixels.len() != count {
            return Err(Error::Inference(
                "crop processor returned inconsistent batch dimensions".into(),
            ));
        }
        let mut crops = boxes
            .into_iter()
            .zip(pixels)
            .enumerate()
            .filter_map(|(index, (bbox, pixels))| {
                pixels.map(|pixels| PrimaryCrop::new(index, bbox, pixels))
            })
            .collect::<Vec<_>>();
        // Stable order for equal ratios, then scatter into detection order.
        crops.sort_by(|a, b| a.ratio.total_cmp(&b.ratio));
        Ok((count, crops))
    }

    fn orient_crops(
        &self,
        crops: &mut [PrimaryCrop],
    ) -> Result<Vec<LineOrientationObservation>, Error> {
        let Some(classifier) = &self.orientation else {
            return Ok(Vec::new());
        };
        let mut evidence = Vec::with_capacity(crops.len());
        for batch in crops.chunks_mut(ORIENTATION_BATCH_SIZE) {
            let images = batch.iter().map(|crop| Arc::clone(&crop.pixels)).collect();
            let result = classifier
                .execute(ImageTaskInput::from_arc_images(images), None)
                .map_err(|error| inference_error(&error))?;
            if result.classifications.len() != batch.len() {
                return Err(Error::Inference(
                    "orientation classifier returned wrong batch size".into(),
                ));
            }
            for (crop, classes) in batch.iter_mut().zip(result.classifications) {
                let score = classes
                    .iter()
                    .find(|class| class.class_id == 1)
                    .map(|class| class.score)
                    .filter(|score| (0.0..=1.0).contains(score))
                    .ok_or_else(|| {
                        Error::Inference("missing valid half-turn classification".into())
                    })?;
                crop.rotate180 = score >= HALF_TURN_THRESHOLD;
                if crop.rotate180 {
                    crop.pixels = Arc::new(image::imageops::rotate180(&*crop.pixels));
                }
                evidence.push(LineOrientationObservation {
                    detection_index: crop.index,
                    half_turn_score: score,
                    applied: crop.rotate180,
                });
            }
        }
        evidence.sort_unstable_by_key(|entry| entry.detection_index);
        Ok(evidence)
    }

    fn recognize_primary(
        &self,
        image: &Arc<RgbImage>,
        character_audit: CharacterEvidence,
        confirmation_threshold: Option<f64>,
    ) -> Result<PrimaryRecognition, Error> {
        let (width, height) = (f64::from(image.width()), f64::from(image.height()));
        let (count, mut crops) = self.primary_crops(image)?;
        let mut ordered = (0..count).map(|_| None).collect::<Vec<_>>();
        let mut audit = CharacterAuditReport {
            orientation: self.orient_crops(&mut crops)?,
            ..Default::default()
        };
        let mut candidate_crops = std::collections::BTreeMap::new();
        let mut crops = crops.into_iter();
        let mode = if confirmation_threshold.is_some() {
            character_audit.confirmation()
        } else {
            character_audit
        };
        let positions = mode.positions();
        loop {
            let batch = crops.by_ref().take(REGION_BATCH_SIZE).collect::<Vec<_>>();
            if batch.is_empty() {
                break;
            }
            let images = batch
                .iter()
                .map(|crop| crop.pixels.as_ref())
                .collect::<Vec<_>>();
            let geometry = positions
                .then(|| BatchGeometry::new(images.iter().map(|image| image.dimensions())))
                .transpose()?;
            let CropPredictions {
                output: predictions,
                ctc,
            } = self.recognition.predict(&images, mode)?;
            let mut ctc = ctc.into_iter();
            // Keep original detection identities until the scatter.
            for (batch_index, (crop, (text, confidence))) in batch
                .into_iter()
                .zip(predictions.texts.into_iter().zip(predictions.scores))
                .enumerate()
            {
                let PrimaryCrop {
                    index,
                    bbox,
                    pixels,
                    rotate180,
                    ..
                } = crop;
                let dimensions = pixels.dimensions();
                let evidence = ctc.next().flatten();
                if !text.trim().is_empty() && confidence >= 0.0 {
                    let quad = image_polygon_to_quad(&bbox.points, width, height)?;
                    let line = PixelLine {
                        detection_index: index,
                        text,
                        confidence: f64::from(confidence),
                        quad,
                    };
                    let crop_geometry = if bbox.points.len() == 4 {
                        CropGeometry::from_quad(quad, image.dimensions(), dimensions)
                            .map(|geometry| geometry.with_half_turn(rotate180))
                    } else {
                        None
                    };
                    let reading_order = crop_geometry
                        .and_then(|geometry| geometry.reading_order(quad))
                        .unwrap_or([0, 1, 2, 3]);
                    let unqualified =
                        confirmation_threshold.is_some_and(|threshold| line.confidence < threshold);
                    if let Some(batch) = geometry
                        && (character_audit.positions() || unqualified)
                    {
                        let columns = predictions
                            .char_col_indices
                            .get(batch_index)
                            .map_or(&[][..], Vec::as_slice);
                        let length = predictions
                            .sequence_lengths
                            .get(batch_index)
                            .copied()
                            .unwrap_or_default();
                        if let Some(entry) = audit.observe(&line, &pixels, batch, columns, length) {
                            entry.geometry = crop_geometry;
                            entry.ctc = evidence;
                            if unqualified {
                                candidate_crops.insert(index, pixels);
                            }
                        }
                    }
                    ordered[index] = Some(PrimaryObservation {
                        line,
                        rotate180,
                        reading_order,
                    });
                }
            }
        }
        Ok(PrimaryRecognition::scatter(ordered, audit, candidate_crops))
    }

    fn recognize_bodies(
        &self,
        image: &RgbImage,
        quads: &[RepairCrop],
    ) -> Result<Vec<BodyPrediction>, Error> {
        let mut crops = Vec::with_capacity(quads.len());
        for quad in quads {
            let points = quad
                .quad
                .iter()
                .map(|&Point(x, y)| {
                    Ok(oar_ocr::processors::Point::new(
                        x.to_f32()
                            .ok_or_else(|| Error::Invalid("unrepresentable body crop X".into()))?,
                        y.to_f32()
                            .ok_or_else(|| Error::Invalid("unrepresentable body crop Y".into()))?,
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let crop =
                get_rotate_crop_image(image, &points).map_err(|error| inference_error(&error))?;
            crops.push(if quad.rotate180 {
                image::imageops::rotate180(&crop)
            } else {
                crop
            });
        }
        // Keep all real hypotheses in the audit, including low scores.
        let output = self
            .recognition
            .predict(&crops.iter().collect::<Vec<_>>(), CharacterEvidence::Off)?
            .output;
        Ok(output
            .texts
            .into_iter()
            .zip(output.scores)
            .map(|(text, confidence)| BodyPrediction {
                text,
                confidence: f64::from(confidence),
            })
            .collect())
    }
}

fn failed_confirmation(
    audit: &crate::character_alignment::CharacterLineAudit,
    error: String,
) -> CharacterConfirmation {
    CharacterConfirmation {
        detection_index: audit.detection_index,
        original: audit.original.clone(),
        crop_sha256: audit.crop_sha256.clone(),
        outcome: ConfirmationOutcome::InferenceFailed { error },
    }
}

fn confirm_batch(
    recognizer: Result<&RecognitionSessionModel, &str>,
    batch: &[ConfirmationCrop],
    lines: &mut [PixelLine],
    audit: &mut CharacterAuditReport,
    policy: ConfirmationPolicy,
    mode: CharacterEvidence,
) {
    let infer = || -> Result<(BatchGeometry, CropPredictions), Error> {
        let recognizer = recognizer.map_err(|error| Error::Inference(error.into()))?;
        let images = batch
            .iter()
            .map(|candidate| candidate.pixels.as_ref())
            .collect::<Vec<_>>();
        let geometry = BatchGeometry::new(images.iter().map(|image| image.dimensions()))?;
        let output = recognizer.predict(&images, mode)?;
        Ok((geometry, output))
    };
    if recognizer.is_ok() {
        audit.confirmation_batches += 1;
    }
    let mut predictions = infer();
    for (index, candidate) in batch.iter().enumerate() {
        let Ok(position) = audit
            .lines
            .binary_search_by_key(&candidate.detection_index, |line| line.detection_index)
        else {
            continue;
        };
        let source = &audit.lines[position];
        let confirmation = match &mut predictions {
            Err(error) => failed_confirmation(source, error.to_string()),
            Ok((geometry, predictions)) => {
                let evidence = predictions.ctc.get_mut(index).and_then(Option::take);
                let predictions = &predictions.output;
                let prediction = BodyPrediction {
                    text: predictions.texts[index].clone(),
                    confidence: f64::from(predictions.scores[index]),
                };
                let columns = predictions
                    .char_col_indices
                    .get(index)
                    .map_or(&[][..], Vec::as_slice);
                let length = predictions
                    .sequence_lengths
                    .get(index)
                    .copied()
                    .unwrap_or_default();
                match source.confirmation(prediction, *geometry, columns, length, policy) {
                    Err(error) => failed_confirmation(source, error.to_string()),
                    Ok(mut result) => {
                        result.ctc = evidence;
                        if result.decision.accepted() {
                            lines[candidate.line_index]
                                .text
                                .clone_from(&result.prediction.text);
                            lines[candidate.line_index].confidence = result.prediction.confidence;
                        }
                        CharacterConfirmation {
                            detection_index: source.detection_index,
                            original: source.original.clone(),
                            crop_sha256: source.crop_sha256.clone(),
                            outcome: ConfirmationOutcome::Predicted { result },
                        }
                    }
                }
            }
        };
        audit.confirmations.push(confirmation);
    }
}

impl OcrInference {
    /// Configure the official PP-OCRv6 Small detector/recognizer pair.
    /// Resolve and load models only on the first recognition. A native-only
    /// document needs no model or ONNX session.
    #[must_use = "retain the configured OCR backend for recognition"]
    pub fn pp_ocrv6(
        threads: Option<NonZeroUsize>,
        repair: RepairPolicy,
        character_audit: CharacterEvidence,
        model: RecognitionModel,
        alternative: Option<AlternativeRecognizer>,
        line_orientation: LineOrientationMode,
    ) -> Self {
        let profile = threads.map_or_else(
            || model.profile().to_owned(),
            |count| format!("{} CPU threads={count}", model.profile()),
        );
        let profile = format!(
            "{profile}; crop confirmation >= {} (leader={}, boundary={})",
            repair.minimum_confidence, repair.leaders, repair.boundaries
        );
        let profile = alternative.map_or_else(
            || profile.clone(),
            |alt| format!("{profile}; alternative={}", alt.model.filename()),
        );
        let profile = if line_orientation == LineOrientationMode::PpLcnet {
            format!(
                "{profile}; line orientation={ORIENTATION_MODEL} threshold={HALF_TURN_THRESHOLD}"
            )
        } else {
            profile
        };
        Self {
            line_orientation,
            backend: OnceLock::new(),
            threads,
            repair,
            character_audit,
            model,
            alternative,
            alternative_backend: OnceLock::new(),
            profile,
        }
    }

    fn confirm(&self, primary: &mut PrimaryRecognition) {
        let Some(alternative) = self.alternative else {
            return;
        };
        // Layout-repaired and already eligible observations retain precedence.
        primary.candidates.retain(|candidate| {
            primary
                .audit
                .lines
                .binary_search_by_key(&candidate.detection_index, |line| line.detection_index)
                .is_ok_and(|index| {
                    primary.lines[candidate.line_index] == primary.audit.lines[index].original
                        && primary.lines[candidate.line_index].confidence
                            < alternative.policy.protected_score
                })
        });
        if primary.candidates.is_empty() {
            return;
        }
        let backend = self
            .alternative_backend
            .get_or_init(|| {
                RecognitionSessionModel::load(alternative.model, self.threads)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(String::as_str);
        // Deterministic width ordering to minimize padding-induced drift.
        let ratio = |crop: &ConfirmationCrop| {
            f64::from(crop.pixels.width()) / f64::from(crop.pixels.height().max(1))
        };
        primary.candidates.sort_by(|a, b| {
            ratio(a)
                .total_cmp(&ratio(b))
                .then(a.detection_index.cmp(&b.detection_index))
        });
        for batch in primary.candidates.chunks(REGION_BATCH_SIZE) {
            confirm_batch(
                backend,
                batch,
                &mut primary.lines,
                &mut primary.audit,
                alternative.policy,
                self.character_audit.confirmation(),
            );
        }
    }

    fn backend(&self) -> Result<&OcrModels, Error> {
        self.backend
            .get_or_init(|| {
                OcrModels::load(self.threads, self.model, self.line_orientation)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| Error::Inference(error.clone()))
    }
}

impl RecognitionBackend for OcrInference {
    fn model_profile(&self) -> &str {
        &self.profile
    }

    /// Recognize the original input raster without applying PDF placement geometry.
    ///
    /// # Errors
    /// Returns an error for invalid image geometry or ONNX inference failure.
    fn recognize_pixels(&self, rgb: RgbImage) -> Result<RasterRecognition, Error> {
        let rgb = Arc::new(rgb);
        let backend = self.backend()?;
        let mut primary = backend.recognize_primary(
            &rgb,
            self.character_audit,
            self.alternative.map(|alt| alt.policy.protected_score),
        )?;
        let crop_repair = if self.repair.leaders || self.repair.boundaries {
            repair_crops(
                &rgb,
                &mut primary.lines,
                &primary.half_turns,
                self.repair,
                |quads| backend.recognize_bodies(&rgb, quads),
            )
        } else {
            RepairReport::default()
        };
        self.confirm(&mut primary);
        // Only the final observations acquire reading-frame corner order.
        // Raw audits and crop-repair source identities remain detector-ordered.
        for (line, order) in primary.lines.iter_mut().zip(primary.reading_orders) {
            line.quad = order.map(|index| line.quad[index]);
        }
        Ok(RasterRecognition {
            lines: primary.lines,
            crop_repair,
            character_audit: primary.audit,
        })
    }
}

fn image_polygon_to_quad(
    points: &[oar_ocr::processors::Point],
    width: f64,
    height: f64,
) -> Result<Quad, Error> {
    if points.len() < 3
        || !width.is_finite()
        || !height.is_finite()
        || width <= 0.0
        || height <= 0.0
        || points
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err(Error::Inference("invalid OCR image-region geometry".into()));
    }
    let normalized =
        |point: oar_ocr::processors::Point| Point(f64::from(point.x), f64::from(point.y));
    if let [top_left, top_right, bottom_right, bottom_left] = points {
        return Ok([
            normalized(*bottom_left),
            normalized(*bottom_right),
            normalized(*top_right),
            normalized(*top_left),
        ]);
    }
    // Detectors may emit polygons with >4 vertices for curved text.
    // Choose a conservative envelope rather than dropping the observation.
    let min_x = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let max_x = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
    let min_y = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let max_y = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
    let from_pixel = |x: f32, y: f32| Point(f64::from(x), f64::from(y));
    Ok([
        from_pixel(min_x, max_y),
        from_pixel(max_x, max_y),
        from_pixel(max_x, min_y),
        from_pixel(min_x, min_y),
    ])
}

#[cfg(test)]
mod tests {
    use super::{image_polygon_to_quad, ort_config, small_detector_config};
    use crate::geometry::{DisplayMapper, PageBox, Point, Rotation};
    use oar_ocr::core::config::OrtExecutionProvider;
    use oar_ocr::processors::{LimitType, Point as OarPoint};
    use std::num::NonZeroUsize;

    #[test]
    fn explicit_cpu_allocation_is_independent_of_thread_override() {
        for threads in [None, NonZeroUsize::new(6)] {
            let config = ort_config(threads);
            assert_eq!(
                config.execution_providers,
                Some(vec![OrtExecutionProvider::CPU])
            );
            assert_eq!(config.intra_threads, threads.map(NonZeroUsize::get));
            assert_eq!(config.inter_threads, threads.map(|_| 1));
        }
    }

    #[test]
    fn detector_retains_small_source_pixels_without_upscaling() {
        let config = small_detector_config();
        assert_eq!(config.limit_side_len, Some(2048));
        assert!(matches!(config.limit_type, Some(LimitType::Max)));
        assert_eq!(config.max_side_len, Some(2048));
        assert_eq!(config.score_threshold, 0.2);
        assert_eq!(config.box_threshold, 0.45);
        assert_eq!(config.unclip_ratio, 1.4);
        assert_eq!(config.max_candidates, 3000);
    }

    #[test]
    fn oar_bounding_boxes_translate_to_ccw_pdf_space() -> Result<(), crate::Error> {
        let mapper = DisplayMapper::new(PageBox::new([0.0, 0.0, 800.0, 600.0])?, Rotation::Deg0);
        let box_ = [
            OarPoint::new(20.0, 10.0),
            OarPoint::new(120.0, 10.0),
            OarPoint::new(120.0, 30.0),
            OarPoint::new(20.0, 30.0),
        ];
        let quad = image_polygon_to_quad(&box_, 800.0, 600.0)?
            .map(|point| mapper.to_pdf(point.0 / 800.0, point.1 / 600.0));
        assert_eq!(
            quad,
            [
                Point(20.0, 570.0),
                Point(120.0, 570.0),
                Point(120.0, 590.0),
                Point(20.0, 590.0),
            ]
        );
        Ok(())
    }
}
