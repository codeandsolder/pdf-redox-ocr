//! Typed, structure-aware PDF OCR pipeline.
//!
//! The deterministic geometry, native text extraction and source sampling
//! components are independent of the inference backend.

pub mod artifacts;
pub mod boundary_ink;
pub mod candidate_ink;
pub mod candidate_ranking;
#[cfg(feature = "pdf-integration")]
pub mod catalog_evidence;
pub mod character_alignment;
pub mod crop_geometry;
pub mod crop_repair;
pub mod ctc_alignment;
pub mod ctc_evidence;
#[cfg(feature = "pdf-integration")]
pub mod direct_images;
mod disjoint_set;
pub mod draft;
pub mod draft_compare;
#[cfg(any(feature = "pdf-integration", test))]
mod fragment_ink;
#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
mod fragment_reconciliation;
#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
pub mod fragment_refinement;
pub mod geometry;
pub mod glyph_appearance;
pub mod glyph_confirmation;
pub mod glyph_isolation;
pub mod glyph_matching;
pub mod glyph_topology;
pub mod hybrid;
#[cfg(feature = "native-inference")]
pub mod inference;
pub mod ink_polarity;
#[cfg(feature = "pdf-integration")]
pub mod input_preparation;
pub mod layout_leaders;
#[cfg(feature = "pdf-integration")]
pub mod line_audit;
mod line_frame;
#[cfg(feature = "pdf-integration")]
pub mod mixed_visibility;
#[cfg(feature = "pdf-integration")]
pub mod mosaic;
#[cfg(feature = "pdf-integration")]
pub mod mosaic_ocr;
#[cfg(feature = "pdf-integration")]
pub mod native_font_mask;
pub mod native_images;
pub mod native_words;
pub mod page_selection;
#[cfg(feature = "pdf-integration")]
pub mod paint_isolation;
#[cfg(feature = "pdf-integration")]
pub mod paint_scan;
#[cfg(feature = "pdf-integration")]
pub mod pdf_metadata;
#[cfg(feature = "pdf-integration")]
pub mod pipeline;
pub mod pixel_regions;
pub mod polygon;
pub mod quality_eval;
pub mod rasterize;
pub mod recognition;
pub mod recognition_model;
#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
pub mod refinement;
pub mod refinement_evidence;
#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
pub mod refinement_recognition;
#[cfg(feature = "pdf-integration")]
pub mod region_ocr;
pub mod reviewed_glyphs;
#[cfg(feature = "pdf-integration")]
pub mod sample_layout;
#[cfg(feature = "pdf-integration")]
pub mod source_appearances;
#[cfg(feature = "pdf-integration")]
pub mod source_glyphs;
#[cfg(feature = "pdf-integration")]
pub mod source_mask;
#[cfg(feature = "pdf-integration")]
pub mod source_ocr;
#[cfg(feature = "pdf-integration")]
mod source_samples;
pub mod source_selection;

use thiserror::Error;

/// Invalid input, unavailable text extractor, or I/O failure.
#[derive(Debug, Error)]
pub enum Error {
    /// An invalid PDF structure, numeric value or page selection.
    #[error("{0}")]
    Invalid(String),
    /// A filesystem or subprocess operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// XML parser error.
    #[error(transparent)]
    Xml(#[from] quick_xml::Error),
    /// Bad number in external PDF text data.
    #[error(transparent)]
    Number(#[from] std::num::ParseFloatError),
    /// An error in the original Rust pdf-redox parser or writer.
    #[cfg(feature = "pdf-integration")]
    #[error(transparent)]
    Pdf(#[from] pdf_redox::Error),
    /// Raster decoding failed.
    #[error(transparent)]
    Image(#[from] image::ImageError),
    /// Native model inference could not complete.
    #[error("native OCR inference failed: {0}")]
    Inference(String),
    /// Poppler command failed.
    #[error("pdftotext failed: {0}")]
    Poppler(String),
}

/// A one-based PDF page number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(transparent)]
pub struct PageNumber(u32);

impl PageNumber {
    /// Reject zero before the page index reaches any subprocess or PDF parser.
    ///
    /// # Errors
    /// Returns an error for page zero.
    pub fn new(number: u32) -> Result<Self, Error> {
        if number == 0 {
            return Err(Error::Invalid("page numbers start at one".into()));
        }
        Ok(Self(number))
    }

    /// Underlying one-based page number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}
