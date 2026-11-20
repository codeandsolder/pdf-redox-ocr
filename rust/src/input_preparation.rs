//! A complete, SHA-bound input preparation transaction.
//!
//! Optimization runs before OCR, with the standard pdf-redox processing
//! preset. Only the exact resulting bytes can authorize semantic overlays.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use pdf_redox::{Config, OptimizationReport, optimize_pdf};
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
}

impl PreparedInput {
    /// Snapshot the source identity and optionally run processing optimization.
    ///
    /// # Errors
    /// Returns an error for unreadable/mutated source, optimizer failure or
    /// temporary-file I/O failure.
    pub fn open(source: &Path, optimize: bool, work_dir: Option<&Path>) -> Result<Self, Error> {
        let source = fs::canonicalize(source)?;
        if !optimize {
            let digest = sha256_file(&source)?;
            return Ok(Self {
                source,
                optimized: None,
                audit: PreparationAudit {
                    original_sha256: digest.clone(),
                    ocr_input_sha256: digest,
                    processing: None,
                },
            });
        }
        let original = fs::read(&source)?;
        let original_sha256 = hex::encode(Sha256::digest(&original));
        let config = Config::optimize_only().for_processing();
        let (bytes, report) = optimize_pdf(&original, &config)?;
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
