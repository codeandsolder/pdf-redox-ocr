//! Backend-independent OCR over an explicitly mapped raster.

use image::RgbImage;

use crate::{
    Error,
    draft::OcrLine,
    geometry::{DisplayMapper, Point},
};

/// A reusable OCR recognizer; mapping is supplied by the caller for every raster.
pub trait RecognitionBackend {
    /// Stable model and preprocessing provenance for the generated draft.
    fn model_profile(&self) -> &str;

    /// Recognize text and return observations in the supplied coordinate frame.
    ///
    /// # Errors
    /// Returns an error when recognition or geometry conversion fails.
    fn recognize_mapped(
        &self,
        rgb: RgbImage,
        pixel_to_pdf: &dyn Fn(Point) -> Point,
    ) -> Result<Vec<OcrLine>, Error>;

    /// Recognize a complete raster using its normalized display mapping.
    ///
    /// # Errors
    /// Returns an error for empty images or unsuccessful recognition.
    fn recognize_page(&self, rgb: RgbImage, mapper: DisplayMapper) -> Result<Vec<OcrLine>, Error> {
        let width = f64::from(rgb.width());
        let height = f64::from(rgb.height());
        if width == 0.0 || height == 0.0 {
            return Err(Error::Invalid("empty OCR raster".into()));
        }
        self.recognize_mapped(rgb, &|point| {
            mapper.to_pdf(point.0 / width, point.1 / height)
        })
    }
}
