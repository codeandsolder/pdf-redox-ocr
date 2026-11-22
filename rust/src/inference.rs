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
            TextLineOrientationAdapterBuilder, TextRecognitionAdapter,
            TextRecognitionAdapterBuilder,
        },
        tasks::{TextDetectionConfig, TextRecognitionConfig, TextRecognitionOutput},
    },
    download::resolve_path,
    oarocr::{EdgeProcessor, TextCroppingProcessor},
    processors::{BoundingBox, LimitType, sort_quad_boxes},
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
    geometry::{Point, Quad},
    glyph_confirmation::ConfirmationPolicy,
    recognition::{PixelLine, RasterRecognition, RecognitionBackend},
};

const DETECTION_MODEL: &str = "pp-ocrv6_small_det.onnx";
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum RecognitionModel {
    #[default]
    Small,
    Medium,
}

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

/// One lazily initialized CPU OCR pipeline reused across multiple rasters.
pub struct OcrInference {
    backend: OnceLock<Result<OcrModels, String>>,
    threads: Option<NonZeroUsize>,
    repair: RepairPolicy,
    character_audit: bool,
    model: RecognitionModel,
    alternative: Option<AlternativeRecognizer>,
    line_orientation: LineOrientationMode,
    alternative_backend: OnceLock<Result<TextRecognitionAdapter, String>>,
    profile: String,
}

#[derive(Clone, Copy, Debug)]
pub struct AlternativeRecognizer {
    pub model: RecognitionModel,
    pub policy: ConfirmationPolicy,
}

struct OcrModels {
    detection: TextDetectionAdapter,
    recognition: TextRecognitionAdapter,
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

fn load_recognition(
    model: RecognitionModel,
    threads: Option<NonZeroUsize>,
    positions: bool,
) -> Result<TextRecognitionAdapter, Error> {
    let dictionary = std::fs::read_to_string(
        resolve_path(DICTIONARY).map_err(|error| inference_error(&error))?,
    )?;
    let rec_path = resolve_path(model.filename()).map_err(|error| inference_error(&error))?;
    let builder = TextRecognitionAdapterBuilder::new()
        .model_input_shape(INPUT_SHAPE)
        .max_img_w(MAX_INPUT_WIDTH)
        .return_word_box(positions)
        .character_dict(dictionary.lines().map(str::to_owned).collect())
        .with_ort_config(ort_config(threads));
    builder
        .build(rec_path)
        .map_err(|error| inference_error(&error))
}

fn predict_crops(
    recognizer: &TextRecognitionAdapter,
    images: Vec<Arc<RgbImage>>,
    config: Option<&TextRecognitionConfig>,
) -> Result<TextRecognitionOutput, Error> {
    let count = images.len();
    let prediction = recognizer
        .execute(ImageTaskInput::from_arc_images(images), config)
        .map_err(|error| inference_error(&error))?;
    if prediction.texts.len() != count || prediction.scores.len() != count {
        return Err(Error::Inference(
            "recognizer returned inconsistent batch dimensions".into(),
        ));
    }
    Ok(prediction)
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
        character_audit: bool,
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
            recognition: load_recognition(model, threads, character_audit)?,
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
        character_audit: bool,
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
        let positions = character_audit || confirmation_threshold.is_some();
        loop {
            let (descriptors, images): (Vec<_>, Vec<_>) = crops
                .by_ref()
                .take(REGION_BATCH_SIZE)
                .map(|crop| {
                    (
                        (
                            crop.index,
                            crop.bbox,
                            crop.rotate180,
                            crop.pixels.dimensions(),
                        ),
                        crop.pixels,
                    )
                })
                .unzip();
            if descriptors.is_empty() {
                break;
            }
            let geometry = if positions {
                Some(BatchGeometry::new(
                    images
                        .iter()
                        .map(|image: &Arc<RgbImage>| image.dimensions()),
                )?)
            } else {
                None
            };
            let retained = if positions {
                images.clone()
            } else {
                Vec::new()
            };
            let predictions = predict_crops(&self.recognition, images, None)?;
            // The adapter preserves batch positions including empty/filtered
            // observations. Keep original detection indices until the scatter.
            for (batch_index, ((index, bbox, rotate180, dimensions), (text, confidence))) in
                descriptors
                    .into_iter()
                    .zip(predictions.texts.into_iter().zip(predictions.scores))
                    .enumerate()
            {
                if !text.trim().is_empty() {
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
                        && (character_audit || unqualified)
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
                        let crop = &retained[batch_index];
                        if let Some(entry) = audit.observe(&line, crop, batch, columns, length) {
                            entry.geometry = crop_geometry;
                            if unqualified {
                                candidate_crops.insert(index, Arc::clone(crop));
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
            crops.push(Arc::new(if quad.rotate180 {
                image::imageops::rotate180(&crop)
            } else {
                crop
            }));
        }
        // Keep all real hypotheses in the audit, including low scores.
        let output = predict_crops(
            &self.recognition,
            crops,
            Some(&TextRecognitionConfig {
                score_threshold: 0.0,
            }),
        )?;
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
    recognizer: Result<&TextRecognitionAdapter, &str>,
    batch: &[ConfirmationCrop],
    lines: &mut [PixelLine],
    audit: &mut CharacterAuditReport,
    policy: ConfirmationPolicy,
) {
    let infer = || -> Result<(BatchGeometry, TextRecognitionOutput), Error> {
        let recognizer = recognizer.map_err(|error| Error::Inference(error.into()))?;
        let images = batch
            .iter()
            .map(|candidate| Arc::clone(&candidate.pixels))
            .collect::<Vec<_>>();
        let geometry = BatchGeometry::new(images.iter().map(|image| image.dimensions()))?;
        let output = predict_crops(recognizer, images, None)?;
        Ok((geometry, output))
    };
    if recognizer.is_ok() {
        audit.confirmation_batches += 1;
    }
    let predictions = infer();
    for (index, candidate) in batch.iter().enumerate() {
        let Ok(position) = audit
            .lines
            .binary_search_by_key(&candidate.detection_index, |line| line.detection_index)
        else {
            continue;
        };
        let source = &audit.lines[position];
        let confirmation = match &predictions {
            Err(error) => failed_confirmation(source, error.to_string()),
            Ok((geometry, predictions)) => {
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
                    Ok(result) => {
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
    #[must_use]
    pub fn pp_ocrv6(
        threads: Option<NonZeroUsize>,
        repair: RepairPolicy,
        character_audit: bool,
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
                load_recognition(alternative.model, self.threads, true)
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
            );
        }
    }

    fn backend(&self) -> Result<&OcrModels, Error> {
        self.backend
            .get_or_init(|| {
                OcrModels::load(
                    self.threads,
                    self.character_audit || self.alternative.is_some(),
                    self.model,
                    self.line_orientation,
                )
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
