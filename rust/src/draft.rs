//! Version 3 OCR draft consumed by the pdf-redox Rust overlay importer.
//!
//! OCR observations are tied to one exact source PDF, not only a filename.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, PageNumber, geometry::Quad};

/// A PDF-bound OCR draft with original, unmodified Unicode observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrDraft {
    /// Importer schema version.
    pub schema_version: u32,
    /// Lowercase SHA-256 digest of input PDF bytes.
    pub pdf_sha256: String,
    /// Engine and model provenance.
    pub model: String,
    /// Selected pages, in one-based document order.
    pub pages: Vec<DraftPage>,
}

/// Per-page observations and native text exclusion geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftPage {
    /// One-based page index.
    pub page: u32,
    /// OCR-detected text.
    pub lines: Vec<OcrLine>,
    /// Existing native text quads that the overlay must not duplicate.
    pub native_text_regions: Vec<Quad>,
}

/// A single OCR observation; source-only identifiers are not part of the import schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrLine {
    /// Recognized Unicode string.
    pub text: String,
    /// Model confidence, not a guarantee of textual correctness.
    pub confidence: f64,
    /// PDF user-space quadrilateral in reading-frame BL/BR/TR/TL order.
    pub quad: Quad,
    /// Only an explicit review process may set this true.
    pub reviewed: bool,
}

impl OcrLine {
    /// Match the importer's automatic text/confidence eligibility.
    /// Geometric native-text and duplicate checks are separate.
    #[must_use = "check overlay eligibility before replacing source text"]
    pub fn overlay_qualified(&self, threshold: f64) -> bool {
        !self.text.trim().is_empty() && (self.reviewed || self.confidence >= threshold)
    }
}

impl OcrDraft {
    /// Construct a draft with the supported importer schema.
    #[must_use = "retain the initialized draft for further OCR refinement"]
    pub const fn new(digest: String, model: String, pages: Vec<DraftPage>) -> Self {
        Self {
            schema_version: 3,
            pdf_sha256: digest,
            model,
            pages,
        }
    }

    /// Validate the importer schema and shared structural constraints.
    ///
    /// The returned index borrows the draft, preventing mutation while it
    /// is in use. Empty drafts, blank text, and finite degenerate quads are
    /// legitimate observations; sampling and import eligibility impose
    /// their own stronger constraints.
    ///
    /// # Errors
    /// Rejects unsupported schema, malformed source identity, duplicate or
    /// zero page numbers, invalid confidence or nonfinite coordinates.
    pub fn validated_pages(&self) -> Result<BTreeMap<PageNumber, &DraftPage>, Error> {
        if self.schema_version != 3 {
            return Err(Error::Invalid("unsupported OCR importer schema".into()));
        }
        if self.pdf_sha256.len() != 64
            || !self
                .pdf_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(Error::Invalid(
                "invalid lowercase OCR source SHA-256".into(),
            ));
        }
        let mut pages = BTreeMap::new();
        let finite = |quad: &Quad| {
            quad.iter()
                .all(|point| point.0.is_finite() && point.1.is_finite())
        };
        for page in &self.pages {
            let number = PageNumber::new(page.page)?;
            if pages.insert(number, page).is_some() {
                return Err(Error::Invalid(format!("duplicate OCR page {}", page.page)));
            }
            for (index, line) in page.lines.iter().enumerate() {
                if !(0.0..=1.0).contains(&line.confidence) || !finite(&line.quad) {
                    return Err(Error::Invalid(format!(
                        "invalid OCR confidence or coordinates on page {}, line {index}",
                        page.page
                    )));
                }
            }
            if !page.native_text_regions.iter().all(finite) {
                return Err(Error::Invalid(format!(
                    "invalid native text coordinates on page {}",
                    page.page
                )));
            }
        }
        Ok(pages)
    }

    /// Verify exact input provenance before any merge or overlay.
    ///
    /// # Errors
    /// Returns an error for an unreadable PDF, invalid draft or differing digest.
    pub fn verify_pdf(&self, pdf: &Path) -> Result<(), Error> {
        self.validated_pages()?;
        if self.pdf_sha256 != sha256_file(pdf)? {
            return Err(Error::Invalid("OCR source PDF changed".into()));
        }
        Ok(())
    }

    /// Serialize directly to a provided stream, preserving UTF-8.
    ///
    /// # Errors
    /// Returns an error when serialization or writing fails.
    pub fn write_json(&self, output: impl Write) -> Result<(), serde_json::Error> {
        serde_json::to_writer(output, self)
    }
}

/// Stream SHA-256 to avoid loading large PDFs into RAM.
///
/// # Errors
/// Returns an error when the file cannot be read.
pub fn sha256_file(path: &Path) -> Result<String, Error> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::{DraftPage, OcrDraft, OcrLine, sha256_file};
    use crate::geometry::Point;

    #[test]
    fn draft_coordinates_roundtrip_to_identical_f64_bits() -> Result<(), serde_json::Error> {
        // A real source quad coordinate that the default fast JSON parser
        // previously rounded one ULP down when loading a saved draft.
        let point = Point(368.986_683_687_943_24, 668.412_680_729_166_6);
        let encoded = serde_json::to_string(&point)?;
        let decoded: Point = serde_json::from_str(&encoded)?;
        assert_eq!(point.0.to_bits(), decoded.0.to_bits());
        assert_eq!(point.1.to_bits(), decoded.1.to_bits());
        Ok(())
    }

    #[test]
    fn admission_rejects_malformed_structure_before_any_consumer_uses_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let pdf = tempfile::NamedTempFile::new()?;
        let original = OcrDraft::new(
            sha256_file(pdf.path())?,
            "fixture".into(),
            vec![DraftPage {
                page: 1,
                lines: vec![OcrLine {
                    text: String::new(),
                    confidence: 0.0,
                    quad: [Point(1.0, 2.0); 4],
                    reviewed: false,
                }],
                native_text_regions: vec![],
            }],
        );
        assert_eq!(original.validated_pages()?.len(), 1);
        original.verify_pdf(pdf.path())?;
        let mut empty = original.clone();
        empty.pages.clear();
        empty.verify_pdf(pdf.path())?;

        let truth = crate::quality_eval::GroundTruth {
            schema_version: 1,
            source_pdf_sha256: original.pdf_sha256.clone(),
            provenance: "fixture".into(),
            coordinate_system: "PDF user space".into(),
            samples: vec![crate::quality_eval::ReferenceSample {
                page: 1,
                center: Point(1.0, 2.0),
                text: "reference".into(),
                kind: "fixture".into(),
            }],
        };
        assert_eq!(
            crate::quality_eval::evaluate(&truth, &original)?
                .overall
                .missing,
            1
        );
        let mutations: [fn(&mut OcrDraft); 8] = [
            |draft| draft.schema_version = 0,
            |draft| draft.pdf_sha256.make_ascii_uppercase(),
            |draft| draft.pages[0].page = 0,
            |draft| draft.pages.push(draft.pages[0].clone()),
            |draft| draft.pages[0].lines[0].confidence = f64::NAN,
            |draft| draft.pages[0].lines[0].confidence = 1.1,
            |draft| draft.pages[0].lines[0].quad[2].0 = f64::INFINITY,
            |draft| {
                draft.pages[0]
                    .native_text_regions
                    .push([Point(f64::NAN, 1.0); 4])
            },
        ];
        for mutate in mutations {
            let mut invalid = original.clone();
            mutate(&mut invalid);
            assert!(invalid.validated_pages().is_err());
            assert!(invalid.verify_pdf(pdf.path()).is_err());
            assert!(crate::draft_compare::compare(&original, &invalid, 0.5, 0.9).is_err());
            assert!(crate::quality_eval::evaluate(&truth, &invalid).is_err());
            #[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
            assert!(
                crate::refinement::refinement_plan(
                    &invalid,
                    crate::refinement::RefinementScope::ByConfidence { pages: None },
                    1,
                    crate::refinement::RefinementPolicy::default(),
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn produces_exact_v3_importer_shape() -> Result<(), serde_json::Error> {
        let draft = OcrDraft::new(
            "01".repeat(32),
            "PP-OCRv6".into(),
            vec![DraftPage {
                page: 8,
                native_text_regions: vec![],
                lines: vec![OcrLine {
                    text: "Resistance 47 Ω".into(),
                    quad: [
                        Point(1.0, 2.0),
                        Point(3.0, 4.0),
                        Point(5.0, 6.0),
                        Point(7.0, 8.0),
                    ],
                    confidence: 0.98,
                    reviewed: false,
                }],
            }],
        );
        let value = serde_json::to_value(&draft)?;
        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["pages"][0]["page"], 8);
        assert_eq!(
            value["pages"][0]["lines"][0]["quad"][0],
            serde_json::json!([1.0, 2.0])
        );
        assert_eq!(value["pages"][0]["lines"][0]["reviewed"], false);
        assert_eq!(value["pages"][0]["lines"][0]["text"], "Resistance 47 Ω");
        Ok(())
    }
}
