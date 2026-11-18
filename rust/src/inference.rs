//! Native Rust PP-OCRv6 Small adapter built on oar-ocr and ONNX Runtime.
//!
//! No Python interpreter, bindings or subprocess is involved in inference.
//! All recognitions use the original page-image frame, with no `UVDoc` warping.

use image::RgbImage;
use oar_ocr::prelude::{OAROCR, OAROCRBuilder, PpOcrV6Size};

use crate::{
    Error,
    draft::OcrLine,
    geometry::{DisplayMapper, Point, Quad},
};

/// One warmed CPU OCR pipeline reused across multiple raster and PDF pages.
pub struct OcrInference {
    backend: OAROCR,
}

impl OcrInference {
    /// Initialize the official PP-OCRv6 Small detector/recognizer pair.
    ///
    /// Model files are resolved using the configured oar-ocr model registry,
    /// and downloaded into the model cache only when absent.
    ///
    /// # Errors
    /// Returns an error when models or ONNX Runtime cannot initialize.
    pub fn pp_ocrv6_small() -> Result<Self, Error> {
        let backend = OAROCRBuilder::pp_ocrv6(PpOcrV6Size::Small)
            .image_batch_size(1)
            .region_batch_size(4)
            .build()
            .map_err(|error| Error::Inference(error.to_string()))?;
        Ok(Self { backend })
    }

    /// Detect and recognize text, returning importer-ready PDF-space quads.
    ///
    /// # Errors
    /// Returns an error when the ONNX runtime fails to process an image.
    pub fn recognize_page(
        &self,
        rgb: RgbImage,
        mapper: DisplayMapper,
    ) -> Result<Vec<OcrLine>, Error> {
        let width = f64::from(rgb.width());
        let height = f64::from(rgb.height());
        self.recognize_mapped(rgb, |point| {
            mapper.to_pdf(point.0 / width, point.1 / height)
        })
    }

    /// Recognize a cropped or reoriented raster while preserving the exact
    /// pixel-to-PDF map instead of guessing a full-page display rectangle.
    ///
    /// # Errors
    /// Returns an error for invalid image geometry or ONNX inference failure.
    pub fn recognize_mapped(
        &self,
        rgb: RgbImage,
        pixel_to_pdf: impl Fn(Point) -> Point,
    ) -> Result<Vec<OcrLine>, Error> {
        let width = f64::from(rgb.width());
        let height = f64::from(rgb.height());
        let results = self
            .backend
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
            let quad =
                image_polygon_to_quad(&region.bounding_box.points, width, height, &pixel_to_pdf)?;
            lines.push(OcrLine {
                text: text.to_owned(),
                confidence: f64::from(confidence),
                quad,
                reviewed: false,
            });
        }
        Ok(lines)
    }
}

fn image_polygon_to_quad(
    points: &[oar_ocr::processors::Point],
    width: f64,
    height: f64,
    pixel_to_pdf: &impl Fn(Point) -> Point,
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
    let normalized = |point: oar_ocr::processors::Point| {
        pixel_to_pdf(Point(f64::from(point.x), f64::from(point.y)))
    };
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
    let from_pixel =
        |pixel_x: f32, pixel_y: f32| pixel_to_pdf(Point(f64::from(pixel_x), f64::from(pixel_y)));
    Ok([
        from_pixel(min_x, max_y),
        from_pixel(max_x, max_y),
        from_pixel(max_x, min_y),
        from_pixel(min_x, min_y),
    ])
}

#[cfg(test)]
mod tests {
    use super::image_polygon_to_quad;
    use crate::geometry::{DisplayMapper, PageBox, Point, Rotation};
    use oar_ocr::processors::Point as OarPoint;

    #[test]
    fn oar_bounding_boxes_translate_to_ccw_pdf_space() -> Result<(), crate::Error> {
        let mapper = DisplayMapper::new(PageBox::new([0.0, 0.0, 800.0, 600.0])?, Rotation::Deg0);
        let box_ = [
            OarPoint::new(20.0, 10.0),
            OarPoint::new(120.0, 10.0),
            OarPoint::new(120.0, 30.0),
            OarPoint::new(20.0, 30.0),
        ];
        let quad = image_polygon_to_quad(&box_, 800.0, 600.0, &|point| {
            mapper.to_pdf(point.0 / 800.0, point.1 / 600.0)
        })?;
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
