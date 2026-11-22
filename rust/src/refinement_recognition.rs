//! Recognition-only views of a single bounded, reading-order RGB source crop.
//! No detector, orientation classifier or automatic admission is involved.

use std::{collections::BTreeMap, num::NonZeroUsize};

use image::{RgbImage, imageops::FilterType};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Error,
    ctc_evidence::{CtcEvidence, CtcOutcome},
    inference::{CharacterEvidence, RecognitionModel, RecognitionSessionModel},
};

const STRETCHES: [u32; 4] = [85, 100, 120, 150];
const MIN_VOTES: usize = 5;
const MAX_RUNNER_RATIO: f64 = 0.25;

/// Independent, fixed research recognizer configurations.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum RefinementModel {
    Small,
    #[default]
    Medium,
    Consensus,
}

impl RefinementModel {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Small => "PP-OCRv6 Small",
            Self::Medium => "PP-OCRv6 Medium",
            Self::Consensus => "PP-OCRv6 Small+Medium four-stretch CTC consensus",
        }
    }

    const fn models(self) -> &'static [RecognitionModel] {
        match self {
            Self::Small => &[RecognitionModel::Small],
            Self::Medium => &[RecognitionModel::Medium],
            Self::Consensus => &[RecognitionModel::Small, RecognitionModel::Medium],
        }
    }

    const fn stretches(self) -> &'static [u32] {
        match self {
            Self::Consensus => &STRETCHES,
            Self::Small | Self::Medium => &[100],
        }
    }
}

/// Measured evidence for exactly one model and horizontal resampling.
#[derive(Debug, Serialize)]
pub struct ViewEvidence {
    pub model: RecognitionModel,
    pub stretch_percent: u32,
    pub dimensions: (u32, u32),
    pub rgb_sha256: String,
    pub greedy_text: String,
    pub greedy_score: f64,
    pub ctc: CtcOutcome,
}

/// Why a research hypothesis is (or is not) proposed.
#[derive(Debug, Serialize)]
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
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Greedy { text } | Self::Consensus { text, .. } => Some(text),
            Self::Abstained => None,
        }
    }
}

/// Evidence and an optional research-only proposal, with no invented score.
#[derive(Debug, Serialize)]
pub struct Recognition {
    pub views: Vec<ViewEvidence>,
    pub decision: RefinementDecision,
}

pub(crate) fn canonical(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

fn separated(evidence: &CtcEvidence) -> Option<&str> {
    let best = evidence.hypotheses.first()?;
    let key = canonical(&best.text);
    if key.is_empty() {
        return None;
    }
    // The rank and margin compare retained path mass, not calibrated
    // probabilities. Formatting variants are not distinct candidates.
    let runner = evidence
        .hypotheses
        .iter()
        .skip(1)
        .find(|h| canonical(&h.text) != key)?;
    let ratio = (runner.log_mass_lower_bound - best.log_mass_lower_bound).exp();
    (ratio <= MAX_RUNNER_RATIO).then_some(best.text.as_str())
}

fn consensus(views: &[ViewEvidence]) -> RefinementDecision {
    // The fixed experiment requires one vote per model/stretch pair.
    // Duplicate observations cannot create an additional vote.
    let mut seen = std::collections::BTreeSet::new();
    let mut seen_samples = std::collections::BTreeSet::new();
    let mut votes = BTreeMap::<String, Vec<(usize, &str)>>::new();
    for (index, view) in views.iter().enumerate() {
        if !STRETCHES.contains(&view.stretch_percent)
            || !seen.insert((view.model, view.stretch_percent))
            || !seen_samples.insert((view.model, view.dimensions, view.rgb_sha256.as_str()))
        {
            continue;
        }
        if let CtcOutcome::Measured { evidence } = &view.ctc
            && let Some(text) = separated(evidence)
        {
            votes
                .entry(canonical(text))
                .or_default()
                .push((index, text));
        }
    }
    let mut ranked = votes.into_values().collect::<Vec<_>>();
    ranked.sort_by_key(|v| std::cmp::Reverse(v.len()));
    let Some(winner) = ranked.first() else {
        return RefinementDecision::Abstained;
    };
    if winner.len() < MIN_VOTES
        || ranked.get(1).is_some_and(|next| next.len() >= winner.len())
        || !winner
            .iter()
            .any(|&(i, _)| views[i].model == RecognitionModel::Small)
        || !winner
            .iter()
            .any(|&(i, _)| views[i].model == RecognitionModel::Medium)
    {
        return RefinementDecision::Abstained;
    }
    // Retain literal text from one actual hypothesis; prefer unstretched,
    // then deterministic view order. Do not synthesize a spelling.
    let Some(&(_, text)) = winner
        .iter()
        .min_by_key(|&&(i, _)| (views[i].stretch_percent.abs_diff(100), i))
    else {
        return RefinementDecision::Abstained;
    };
    RefinementDecision::Consensus {
        text: text.to_owned(),
        view_indices: winner.iter().map(|&(i, _)| i).collect(),
    }
}

/// Lazily loaded recognition-only sessions. Both use the same preprocessing,
/// vocabulary and CPU provider configuration as primary recognition.
pub struct RefinementRecognizer {
    profile: RefinementModel,
    threads: Option<NonZeroUsize>,
    sessions: BTreeMap<RecognitionModel, RecognitionSessionModel>,
}

impl RefinementRecognizer {
    #[must_use]
    pub const fn new(profile: RefinementModel, threads: Option<NonZeroUsize>) -> Self {
        Self {
            profile,
            threads,
            sessions: BTreeMap::new(),
        }
    }

    /// Recognize a single original reading-order RGB crop, retaining each
    /// complete view separately. Crop width variation is bounded at 150%.
    ///
    /// # Errors
    /// Returns an error for model failures, unrepresentable view dimensions
    /// or unexpected inference batch sizes.
    pub fn recognize(&mut self, image: &RgbImage) -> Result<Recognition, Error> {
        let mut views = Vec::new();
        for &model in self.profile.models() {
            let session = match self.sessions.entry(model) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(RecognitionSessionModel::load(model, self.threads)?)
                }
            };
            for &stretch in self.profile.stretches() {
                let width =
                    u32::try_from((u64::from(image.width()) * u64::from(stretch)).div_ceil(100))
                        .map_err(|_| Error::Invalid("refinement view width overflow".into()))?;
                if width == 0 || image.height() == 0 {
                    return Err(Error::Invalid("empty refinement view".into()));
                }
                let resampled;
                let pixels = if stretch == 100 {
                    image
                } else {
                    resampled =
                        image::imageops::resize(image, width, image.height(), FilterType::Triangle);
                    &resampled
                };
                let mut prediction = session.predict(&[pixels], CharacterEvidence::Ctc)?;
                let text = prediction
                    .output
                    .texts
                    .pop()
                    .ok_or_else(|| Error::Inference("missing view text".into()))?;
                let score = prediction
                    .output
                    .scores
                    .pop()
                    .ok_or_else(|| Error::Inference("missing view score".into()))?;
                let ctc = prediction
                    .ctc
                    .pop()
                    .flatten()
                    .ok_or_else(|| Error::Inference("missing view CTC evidence".into()))?;
                if !(0.0..=1.0).contains(&score) {
                    return Err(Error::Inference("invalid view score".into()));
                }
                views.push(ViewEvidence {
                    model,
                    stretch_percent: stretch,
                    dimensions: pixels.dimensions(),
                    rgb_sha256: hex::encode(Sha256::digest(pixels.as_raw())),
                    greedy_text: text,
                    greedy_score: f64::from(score),
                    ctc,
                });
            }
        }
        let decision = match self.profile {
            RefinementModel::Consensus => consensus(&views),
            RefinementModel::Small | RefinementModel::Medium => views
                .first()
                .filter(|v| !v.greedy_text.trim().is_empty())
                .map_or(RefinementDecision::Abstained, |v| {
                    RefinementDecision::Greedy {
                        text: v.greedy_text.clone(),
                    }
                }),
        };
        Ok(Recognition { views, decision })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc_evidence::SequenceHypothesis;

    fn view(
        model: RecognitionModel,
        stretch_percent: u32,
        best: &str,
        runner: &str,
        ratio: f64,
    ) -> ViewEvidence {
        ViewEvidence {
            model,
            stretch_percent,
            dimensions: (100, 20),
            rgb_sha256: stretch_percent.to_string(),
            greedy_text: best.into(),
            greedy_score: 0.99,
            ctc: CtcOutcome::Measured {
                evidence: CtcEvidence {
                    sequence_length: 2,
                    vocabulary_size: 3,
                    row_class_limit: 8,
                    beam_width: 12,
                    greedy_text: best.into(),
                    emissions: Vec::new(),
                    hypotheses: vec![
                        SequenceHypothesis {
                            text: best.into(),
                            classes: Vec::new(),
                            log_mass_lower_bound: -0.2,
                        },
                        SequenceHypothesis {
                            text: runner.into(),
                            classes: Vec::new(),
                            log_mass_lower_bound: -0.2 + ratio.ln(),
                        },
                    ],
                },
            },
        }
    }

    #[test]
    fn consensus_needs_distinct_views_and_both_models() {
        let mut views = STRETCHES
            .iter()
            .map(|&s| view(RecognitionModel::Small, s, "3.5", "35", 0.1))
            .collect::<Vec<_>>();
        assert!(consensus(&views).text().is_none());
        views.push(view(RecognitionModel::Small, 100, "3.5", "35", 0.1));
        assert!(consensus(&views).text().is_none());
        views.push(view(RecognitionModel::Medium, 100, "3.5", "35", 0.5));
        assert!(consensus(&views).text().is_none());
        views.push(view(RecognitionModel::Medium, 120, "3.5", "35", 0.1));
        assert_eq!(consensus(&views).text(), Some("3.5"));
        views[3].rgb_sha256 = views[0].rgb_sha256.clone();
        assert!(consensus(&views).text().is_none());
    }

    #[test]
    fn consensus_abstains_instead_of_combining_distinct_strings() {
        let views = [RecognitionModel::Small, RecognitionModel::Medium]
            .into_iter()
            .flat_map(|m| {
                STRETCHES
                    .into_iter()
                    .map(move |s| view(m, s, if s < 110 { "Ø20" } else { "020" }, "20", 0.1))
            })
            .collect::<Vec<_>>();
        assert!(consensus(&views).text().is_none());
    }
}
