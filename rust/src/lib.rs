//! Typed, structure-aware PDF OCR pipeline.
//!
//! The deterministic geometry, native text extraction and source sampling
//! components are independent of the inference backend.

pub mod draft;
pub mod geometry;
pub mod hybrid;
pub mod native_words;
pub mod polygon;
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
    /// Poppler command failed.
    #[error("pdftotext failed: {0}")]
    Poppler(String),
}

/// A one-based PDF page number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
