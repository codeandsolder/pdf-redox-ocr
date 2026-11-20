//! Native Rust PP-OCRv6 Small adapter built on oar-ocr and ONNX Runtime.
//!
//! No Python interpreter, bindings or subprocess is involved in inference.
//! All recognitions use the original page-image frame, with no `UVDoc` warping.

use std::sync::OnceLock;

use image::RgbImage;
use oar_ocr::{
    domain::tasks::TextDetectionConfig,
    prelude::{OAROCR, OAROCRBuilder, PpOcrV6Size},
    processors::LimitType,
};

use crate::{
    Error,
    geometry::{Point, Quad},
    recognition::{PixelLine, RecognitionBackend},
};

/// One lazily initialized CPU OCR pipeline reused across multiple raster and PDF pages.
pub struct OcrInference {
    backend: OnceLock<Result<OAROCR, String>>,
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

impl OcrInference {
    /// Stable draft provenance including the detector's resolution policy.
    pub const MODEL_PROFILE: &str = "oar-ocr PP-OCRv6 Small detector-max-2048";

    /// Configure the official PP-OCRv6 Small detector/recognizer pair.
    /// Resolve and load models only on the first recognition; a native-only
    /// document needs no model files or ONNX session at all.
    #[must_use]
    pub const fn pp_ocrv6_small() -> Self {
        Self {
            backend: OnceLock::new(),
        }
    }

    fn backend(&self) -> Result<&OAROCR, Error> {
        self.backend
            .get_or_init(|| {
                OAROCRBuilder::pp_ocrv6(PpOcrV6Size::Small)
                    .text_detection_config(small_detector_config())
                    .image_batch_size(1)
                    .region_batch_size(4)
                    .build()
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| Error::Inference(error.clone()))
    }
}

impl RecognitionBackend for OcrInference {
    fn model_profile(&self) -> &str {
        Self::MODEL_PROFILE
    }

    /// Recognize the original input raster without applying PDF placement geometry.
    ///
    /// # Errors
    /// Returns an error for invalid image geometry or ONNX inference failure.
    fn recognize_pixels(&self, rgb: RgbImage) -> Result<Vec<PixelLine>, Error> {
        let width = f64::from(rgb.width());
        let height = f64::from(rgb.height());
        let results = self
            .backend()?
            .predict(vec![rgb])
            .map_err(|error| Error::Inference(error.to_string()))?;
        let Some(result) = results.into_iter().next() else {
            return Err(Error::Inference(
                "OCR backend returned no page result".into(),
            ));
        };
        let mut lines = Vec::new();
        for region in result.text_regions {
            let Some((text, confidence)) = region.text_with_confidence() else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            let quad = image_polygon_to_quad(&region.bounding_box.points, width, height)?;
            lines.push(PixelLine {
                text: text.to_owned(),
                confidence: f64::from(confidence),
                quad,
            });
        }
        Ok(lines)
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
