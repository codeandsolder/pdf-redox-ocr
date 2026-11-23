//! Typed, model-independent artifacts for source refinement and offline ranking.

use crate::{
    Error,
    candidate_ink::{SourceInk, ViewInkMapping},
    ctc_evidence::CtcOutcome,
    draft::OcrDraft,
    geometry::Quad,
    recognition_model::RecognitionModel,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectionMode {
    Confidence,
    ExplicitLines,
}

/// Measured evidence for exactly one model and horizontal resampling.
#[derive(Debug, Serialize, Deserialize)]
pub struct ViewEvidence {
    pub model: RecognitionModel,
    pub stretch_percent: u32,
    pub dimensions: (u32, u32),
    pub rgb_sha256: String,
    pub greedy_text: String,
    pub greedy_score: f64,
    pub ctc: CtcOutcome,
    pub ink_mapping: Option<ViewInkMapping>,
}

/// Why a research hypothesis is (or is not) proposed.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RefinementDecision {
    Greedy {
        text: String,
    },
    Consensus {
        text: String,
        view_indices: Vec<usize>,
    },
    Abstained,
}

impl RefinementDecision {
    #[must_use = "use observed text when deciding whether refinement is meaningful"]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Greedy { text } | Self::Consensus { text, .. } => Some(text),
            Self::Abstained => None,
        }
    }
}

/// Evidence and an optional research-only proposal, with no invented score.
#[derive(Debug, Serialize, Deserialize)]
pub struct Recognition {
    pub views: Vec<ViewEvidence>,
    pub source_ink: Box<SourceInk>,
    pub decision: RefinementDecision,
}

pub(crate) fn canonical(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Exact source sampling provenance.
#[derive(Debug, Serialize, Deserialize)]
pub struct SourceCropEvidence {
    pub rgb_sha256: String,
    pub dimensions: (u32, u32),
    pub rendered_dimensions: (u32, u32),
    pub rendered_quad: Quad,
    pub charged_pixels: u64,
}

/// Per-line diagnostic result, including explicit geometry/budget failures.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RefinementOutcome {
    Recognized {
        crop: SourceCropEvidence,
        recognition: Recognition,
    },
    Skipped {
        reason: String,
    },
}

/// Report for a single original OCR observation.
#[derive(Debug, Serialize, Deserialize)]
pub struct RefinementProposal {
    pub page: u32,
    pub line_index: usize,
    pub quad: Quad,
    pub primary_text: String,
    pub primary_score: f64,
    pub outcome: RefinementOutcome,
}

impl RefinementProposal {
    pub(crate) fn revised_text(&self) -> Option<&str> {
        let RefinementOutcome::Recognized { recognition, .. } = &self.outcome else {
            return None;
        };
        recognition
            .decision
            .text()
            .filter(|text| meaningful_change(&self.primary_text, text))
    }
}

/// Summary of attempted recognition-only revisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementSummary {
    pub source_lines: usize,
    pub selected_lines: usize,
    pub revised_predictions: usize,
    pub recognized_lines: usize,
    pub skipped_lines: usize,
    pub charged_pixels: u64,
    pub seconds: f64,
    pub source_pages: usize,
    pub pages_refined: usize,
}

/// Research-only provenance-bound comparison report.
#[derive(Debug, Serialize, Deserialize)]
pub struct RefinementReport {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub source_model: String,
    pub secondary_model: String,
    pub selection_mode: SelectionMode,
    pub max_primary_score: Option<f64>,
    pub min_chars: Option<usize>,
    pub dpi: u32,
    pub max_line_pixels: u64,
    pub max_page_pixels: u64,
    pub summary: RefinementSummary,
    pub proposals: Vec<RefinementProposal>,
}

/// Ignore only whitespace; retain literal engineering symbols, punctuation,
/// superscripts and full-width forms.
#[must_use = "check whether the refinement changes recognized content"]
pub fn meaningful_change(left: &str, right: &str) -> bool {
    canonical(left) != canonical(right)
}

/// Construct a research-only draft. No inherited confidence or review survives.
pub(crate) fn experimental_draft<'a>(
    source: &OcrDraft,
    model: &str,
    revisions: impl IntoIterator<Item = (u32, usize, &'a str)>,
) -> Result<OcrDraft, Error> {
    let mut experimental = source.clone();
    experimental.model = format!("{} + research-only {model} hypotheses", source.model);
    for page in &mut experimental.pages {
        for line in &mut page.lines {
            line.confidence = 0.0;
            line.reviewed = false;
        }
    }
    for (page, index, text) in revisions {
        let line = experimental
            .pages
            .iter_mut()
            .find(|p| p.page == page)
            .and_then(|p| p.lines.get_mut(index))
            .ok_or_else(|| Error::Invalid("refinement proposal line missing".into()))?;
        text.clone_into(&mut line.text);
    }
    Ok(experimental)
}
