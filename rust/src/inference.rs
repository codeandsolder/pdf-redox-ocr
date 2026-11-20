//! Native Rust PP-OCRv6 Small adapter built on OAR task adapters and ONNX Runtime.
//!
//! Detection and both primary/body recognition share one loaded model pair.
//! Original raster geometry is retained; there is no document rectification.

use std::{
    num::NonZeroUsize,
    sync::{Arc, OnceLock},
};

use image::RgbImage;
use num_traits::ToPrimitive;
use oar_ocr::{
    core::{
        OCRError,
        config::OrtSessionConfig,
        traits::{
            OrtConfigurable,
            adapter::{AdapterBuilder, ModelAdapter},
            task::ImageTaskInput,
        },
    },
    domain::{
        adapters::{
            TextDetectionAdapter, TextDetectionAdapterBuilder, TextRecognitionAdapter,
            TextRecognitionAdapterBuilder,
        },
        tasks::{TextDetectionConfig, TextRecognitionConfig},
    },
    download::resolve_path,
    oarocr::{EdgeProcessor, TextCroppingProcessor},
    processors::{BoundingBox, LimitType, sort_quad_boxes},
    utils::get_rotate_crop_image,
};

use crate::{
    Error,
    geometry::{Point, Quad},
    leader_repair::{BodyPrediction, repair_leaders},
    recognition::{PixelLine, RasterRecognition, RecognitionBackend},
};

const DETECTION_MODEL: &str = "pp-ocrv6_small_det.onnx";
const RECOGNITION_MODEL: &str = "pp-ocrv6_small_rec.onnx";
const DICTIONARY: &str = "ppocrv6_dict.txt";
const REGION_BATCH_SIZE: usize = 4;

/// One lazily initialized CPU OCR pipeline reused across multiple rasters.
pub struct OcrInference {
    backend: OnceLock<Result<OcrModels, String>>,
    threads: Option<NonZeroUsize>,
    leader_threshold: Option<f64>,
    profile: String,
}

struct OcrModels {
    detection: TextDetectionAdapter,
    recognition: TextRecognitionAdapter,
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

fn inference_error(error: OCRError) -> Error {
    Error::Inference(error.to_string())
}

struct PrimaryCrop {
    index: usize,
    bbox: BoundingBox,
    pixels: Arc<RgbImage>,
    ratio: f32,
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
        }
    }
}

impl OcrModels {
    fn load(threads: Option<NonZeroUsize>) -> Result<Self, Error> {
        let dictionary =
            std::fs::read_to_string(resolve_path(DICTIONARY).map_err(inference_error)?)?;
        let det_path = resolve_path(DETECTION_MODEL).map_err(inference_error)?;
        let rec_path = resolve_path(RECOGNITION_MODEL).map_err(inference_error)?;
        let mut detection = TextDetectionAdapterBuilder::new().with_config(small_detector_config());
        let mut recognition = TextRecognitionAdapterBuilder::new()
            .character_dict(dictionary.lines().map(str::to_owned).collect());
        if let Some(threads) = threads {
            let config = OrtSessionConfig::new()
                .with_intra_threads(threads.get())
                .with_inter_threads(1)
                .resolve_auto();
            detection = detection.with_ort_config(config.clone());
            recognition = recognition.with_ort_config(config);
        }
        Ok(Self {
            detection: detection.build(det_path).map_err(inference_error)?,
            recognition: recognition.build(rec_path).map_err(inference_error)?,
        })
    }

    fn predict_crops(
        &self,
        images: Vec<Arc<RgbImage>>,
        config: Option<&TextRecognitionConfig>,
    ) -> Result<Vec<BodyPrediction>, Error> {
        let count = images.len();
        let prediction = self
            .recognition
            .execute(ImageTaskInput::from_arc_images(images), config)
            .map_err(inference_error)?;
        if prediction.texts.len() != count || prediction.scores.len() != count {
            return Err(Error::Inference(
                "recognizer returned inconsistent batch dimensions".into(),
            ));
        }
        Ok(prediction
            .texts
            .into_iter()
            .zip(prediction.scores)
            .map(|(text, confidence)| BodyPrediction {
                text,
                confidence: f64::from(confidence),
            })
            .collect())
    }

    fn primary_crops(&self, image: &Arc<RgbImage>) -> Result<(usize, Vec<PrimaryCrop>), Error> {
        let detection = self
            .detection
            .execute(
                ImageTaskInput::from_arc_images(vec![Arc::clone(image)]),
                None,
            )
            .map_err(inference_error)?;
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
            .map_err(inference_error)?;
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

    fn recognize_primary(&self, image: &Arc<RgbImage>) -> Result<Vec<PixelLine>, Error> {
        let (width, height) = (f64::from(image.width()), f64::from(image.height()));
        let (count, crops) = self.primary_crops(image)?;
        let mut ordered = vec![None; count];
        let mut crops = crops.into_iter();
        loop {
            let (descriptors, images): (Vec<_>, Vec<_>) = crops
                .by_ref()
                .take(REGION_BATCH_SIZE)
                .map(|crop| ((crop.index, crop.bbox), crop.pixels))
                .unzip();
            if descriptors.is_empty() {
                break;
            }
            let predictions = self.predict_crops(images, None)?;
            for ((index, bbox), prediction) in descriptors.into_iter().zip(predictions) {
                if !prediction.text.trim().is_empty() {
                    let quad = image_polygon_to_quad(&bbox.points, width, height)?;
                    ordered[index] = Some(PixelLine {
                        text: prediction.text,
                        confidence: prediction.confidence,
                        quad,
                    });
                }
            }
        }
        Ok(ordered.into_iter().flatten().collect())
    }

    fn recognize_bodies(
        &self,
        image: &RgbImage,
        quads: &[Quad],
    ) -> Result<Vec<BodyPrediction>, Error> {
        let mut crops = Vec::with_capacity(quads.len());
        for quad in quads {
            let points = quad
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
            crops.push(Arc::new(
                get_rotate_crop_image(image, &points).map_err(inference_error)?,
            ));
        }
        // Keep all real hypotheses in the audit, including low scores.
        self.predict_crops(
            crops,
            Some(&TextRecognitionConfig {
                score_threshold: 0.0,
            }),
        )
    }
}

impl OcrInference {
    /// Stable draft provenance including the detector's resolution policy.
    pub const MODEL_PROFILE: &str = "oar-ocr PP-OCRv6 Small detector-max-2048";

    /// Configure the official PP-OCRv6 Small detector/recognizer pair.
    /// Resolve and load models only on the first recognition. A native-only
    /// document needs no model or ONNX session.
    #[must_use]
    pub fn pp_ocrv6_small(threads: Option<NonZeroUsize>, leader_threshold: Option<f64>) -> Self {
        let profile = threads.map_or_else(
            || Self::MODEL_PROFILE.to_owned(),
            |count| format!("{} CPU threads={count}", Self::MODEL_PROFILE),
        );
        let profile = if let Some(threshold) = leader_threshold {
            format!("{profile}; periodic-leader body confirmation >= {threshold}")
        } else {
            profile
        };
        Self {
            backend: OnceLock::new(),
            threads,
            leader_threshold,
            profile,
        }
    }

    fn backend(&self) -> Result<&OcrModels, Error> {
        self.backend
            .get_or_init(|| OcrModels::load(self.threads).map_err(|error| error.to_string()))
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
        let mut lines = backend.recognize_primary(&rgb)?;
        let leader_repair = self
            .leader_threshold
            .map_or_else(Default::default, |threshold| {
                repair_leaders(&rgb, &mut lines, threshold, |quads| {
                    backend.recognize_bodies(&rgb, quads)
                })
            });
        Ok(RasterRecognition {
            lines,
            leader_repair,
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
    use super::{image_polygon_to_quad, small_detector_config};
    use crate::geometry::{DisplayMapper, PageBox, Point, Rotation};
    use oar_ocr::processors::{LimitType, Point as OarPoint};

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
