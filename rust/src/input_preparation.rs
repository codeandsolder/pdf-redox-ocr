//! A complete, SHA-bound input preparation transaction.
//!
//! Optimization runs before OCR, with the standard pdf-redox processing
//! preset. Only the exact resulting bytes can authorize semantic overlays.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use pdf_redox::{
    Config, GlyphCatalog, OptimizationReport, optimize_pdf, optimize_pdf_with_glyph_catalog,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::{
    Error,
    draft::{OcrDraft, sha256_file},
};

/// Actual optimizer policy and report, not inferred from the OCR output.
#[derive(Debug, Serialize)]
pub struct ProcessingEvidence {
    pub config: Config,
    pub report: OptimizationReport,
}

/// Both identities are retained even when the intermediate PDF is temporary.
#[derive(Debug, Serialize)]
pub struct PreparationAudit {
    pub original_sha256: String,
    pub ocr_input_sha256: String,
    pub processing: Option<ProcessingEvidence>,
}

/// Scoped prepared input, including any temporary optimized PDF needed by
/// Poppler. The original source is never written.
pub struct PreparedInput {
    source: PathBuf,
    optimized: Option<NamedTempFile>,
    audit: PreparationAudit,
    catalog: Option<GlyphCatalog>,
}

/// Source data and processing options for one exact prepared input.
#[derive(Clone, Copy)]
pub enum PreparationMode<'a> {
    Original {
        catalog: Option<&'a Path>,
    },
    Processing {
        work_dir: Option<&'a Path>,
        collect_catalog: bool,
    },
}

impl PreparedInput {
    /// Snapshot the source identity and optionally run processing optimization.
    ///
    /// # Errors
    /// Returns an error for unreadable/mutated source, optimizer failure or
    /// temporary-file I/O failure.
    pub fn open(source: &Path, mode: PreparationMode<'_>) -> Result<Self, Error> {
        let source = fs::canonicalize(source)?;
        let (work_dir, collect_catalog) = match mode {
            PreparationMode::Original { catalog } => {
                let digest = sha256_file(&source)?;
                let catalog = catalog
                    .map(|path| -> Result<GlyphCatalog, Error> {
                        let data: GlyphCatalog =
                            serde_json::from_reader(std::io::BufReader::new(fs::File::open(path)?))
                                .map_err(|error| {
                                    Error::Invalid(format!("invalid glyph catalog: {error}"))
                                })?;
                        if data.schema_version != 2 || data.optimized_pdf_sha256 != digest {
                            return Err(Error::Invalid(
                                "glyph catalog does not match original PDF".into(),
                            ));
                        }
                        Ok(data)
                    })
                    .transpose()?;
                return Ok(Self {
                    source,
                    optimized: None,
                    audit: PreparationAudit {
                        original_sha256: digest.clone(),
                        ocr_input_sha256: digest,
                        processing: None,
                    },
                    catalog,
                });
            }
            PreparationMode::Processing {
                work_dir,
                collect_catalog,
            } => (work_dir, collect_catalog),
        };
        let original = fs::read(&source)?;
        let original_sha256 = hex::encode(Sha256::digest(&original));
        let config = Config::optimize_only().for_processing();
        let (bytes, report, catalog) = if collect_catalog {
            let (bytes, report, catalog) = optimize_pdf_with_glyph_catalog(&original, &config)?;
            (bytes, report, Some(catalog))
        } else {
            let (bytes, report) = optimize_pdf(&original, &config)?;
            (bytes, report, None)
        };
        drop(original);
        let ocr_input_sha256 = hex::encode(Sha256::digest(&bytes));
        let mut file = match work_dir {
            Some(dir) => NamedTempFile::new_in(dir)?,
            None => NamedTempFile::new()?,
        };
        file.write_all(&bytes)?;
        file.flush()?;
        let prepared = Self {
            source,
            optimized: Some(file),
            audit: PreparationAudit {
                original_sha256,
                ocr_input_sha256,
                processing: Some(ProcessingEvidence { config, report }),
            },
            catalog,
        };
        prepared.verify_source()?;
        Ok(prepared)
    }

    /// Exact PDF against which the recognition geometry must be interpreted.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.optimized
            .as_ref()
            .map_or(self.source.as_path(), NamedTempFile::path)
    }

    /// Provenance of the source preparation, independently of OCR.
    #[must_use]
    pub const fn audit(&self) -> &PreparationAudit {
        &self.audit
    }

    /// Exact optimizer-produced outline catalog, if requested.
    #[must_use]
    pub const fn catalog(&self) -> Option<&GlyphCatalog> {
        self.catalog.as_ref()
    }

    /// Reject a changed original PDF before publishing results.
    ///
    /// # Errors
    /// Returns an error for a different source hash or inaccessible input.
    pub fn verify_source(&self) -> Result<(), Error> {
        if sha256_file(&self.source)? != self.audit.original_sha256 {
            return Err(Error::Invalid("original PDF changed during OCR job".into()));
        }
        Ok(())
    }

    /// Reject any OCR draft/bytes not derived from this exact preparation.
    ///
    /// # Errors
    /// Returns an error for a mismatched OCR input, or a changed original PDF.
    pub fn verify_recognized(&self, draft: &OcrDraft) -> Result<(), Error> {
        if draft.pdf_sha256 != self.audit.ocr_input_sha256 {
            return Err(Error::Invalid(
                "OCR draft does not match prepared PDF".into(),
            ));
        }
        draft.verify_pdf(self.path())?;
        if self.optimized.is_some() {
            self.verify_source()?;
        }
        Ok(())
    }
}
