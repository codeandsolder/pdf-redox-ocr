//! Candidate-specific localization of CTC evidence on measured source ink.
//!
//! All horizontal views use one segmentation of the unmodified
//! reading-order crop, so alternative hypotheses cannot create new pixels.

#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
use crate::ctc_evidence::SequenceReference;
use image::RgbImage;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    Error,
    character_alignment::{
        BatchGeometry, CharacterAnchor, EmissionCell, RunAlignment, align_runs, anchor_candidates,
    },
    ctc_alignment::{SequenceAlignment, TokenLocalization},
    ctc_evidence::{CtcEvidence, SequenceData},
    glyph_appearance::GlyphShape,
    glyph_isolation::{InkGroup, MeasuredInk},
    ink_polarity::PolarityEvidence,
    pixel_regions::InkBounds,
};

/// A single independent measurement of source ink, shared by all hypotheses.
#[derive(Debug, Serialize, Deserialize)]
pub struct SourceInk {
    pub dimensions: (u32, u32),
    pub rgb_sha256: String,
    pub normalized_sha256: String,
    pub polarity: PolarityEvidence,
    pub groups: Vec<InkGroup>,
    /// No appearance is assigned to a clipped group.
    pub appearances: Vec<Option<GlyphShape>>,
}

/// Positions in the exact, unstretched source crop. The interval reflects
/// conditional CTC entry localization, not a measured glyph bounding box.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct CandidateAnchor {
    #[serde(flatten)]
    pub anchor: CharacterAnchor,
    /// Outer envelope of the entry interval, mode, and monotone token order.
    pub entry_interval: [f64; 2],
}

/// Per-hypothesis evidence; unlocalized hypotheses remain actual model
/// hypotheses and are never silently replaced with greedy geometry.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CandidateMapping {
    Mapped { anchors: Vec<CandidateAnchor> },
    Unavailable { reason: String },
}

impl CandidateMapping {
    /// Localized non-whitespace tokens for the specified canonical text.
    #[must_use]
    pub fn visible_anchors(&self, text: &str) -> Option<Vec<&CandidateAnchor>> {
        let Self::Mapped { anchors } = self else {
            return None;
        };
        let visible = anchors
            .iter()
            .filter(|a| !a.anchor.character.is_whitespace())
            .collect::<Vec<_>>();
        visible
            .iter()
            .map(|a| a.anchor.character)
            .eq(text.chars().filter(|c| !c.is_whitespace()))
            .then_some(visible)
    }

    /// A unique, strictly ordered source-run identity for each token.
    #[must_use]
    pub fn unique_runs(&self, text: &str) -> Option<Vec<usize>> {
        let runs = self
            .visible_anchors(text)?
            .iter()
            .map(|a| match a.anchor.alignment {
                RunAlignment::Unique { run } => Some(run),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        runs.windows(2).all(|p| p[0] < p[1]).then_some(runs)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ViewGeometry {
    pub tensor_width: usize,
    pub resized_width: usize,
    pub sequence_length: usize,
    /// Source pixels per CTC column, accounting for actual horizontal
    /// stretch dimensions and right padding in the model input.
    pub source_column_pitch: f64,
}

impl ViewGeometry {
    fn new(source: (u32, u32), view: (u32, u32), sequence_length: usize) -> Option<Self> {
        if source.0 == 0 || source.1 != view.1 || sequence_length == 0 {
            return None;
        }
        let batch = BatchGeometry::new([view]).ok()?;
        let resized_width = batch.resized_width(view).ok()?;
        let source_column_pitch = batch.tensor_width.to_f64()? / sequence_length.to_f64()?
            * f64::from(source.0)
            / resized_width.to_f64()?;
        if !source_column_pitch.is_finite() || source_column_pitch <= 0.0 {
            return None;
        }
        Some(Self {
            tensor_width: batch.tensor_width,
            resized_width,
            sequence_length,
            source_column_pitch,
        })
    }

    fn position(&self, column: usize) -> Option<f64> {
        if column >= self.sequence_length {
            return None;
        }
        Some((column.to_f64()? + 0.5) * self.source_column_pitch)
    }

    fn cell(&self, character: char, token: &TokenLocalization) -> Option<EmissionCell> {
        if token.interval[0] > token.interval[1] {
            return None;
        }
        Some(EmissionCell {
            character,
            column: token.mode,
            position: self.position(token.mode)?,
            interval: [
                self.position(token.interval[0].min(token.mode))?,
                self.position(token.interval[1].max(token.mode))?,
            ],
        })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ViewInkMapping {
    pub geometry: ViewGeometry,
    /// Same order as the measured CTC hypotheses, even on partial failure.
    pub hypotheses: Vec<CandidateMapping>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub queries: Vec<CandidateMapping>,
}

#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
impl ViewInkMapping {
    pub(crate) fn sequence(&self, reference: SequenceReference) -> Option<&CandidateMapping> {
        match reference {
            SequenceReference::Beam { index } => self.hypotheses.get(index),
            SequenceReference::Query { index } => self.queries.get(index),
        }
    }
}

impl SourceInk {
    /// Segment the source RGB once, without any OCR character count or text.
    ///
    /// # Errors
    /// Rejects empty or invalid source crops.
    pub fn measure(image: &RgbImage) -> Result<Self, Error> {
        let MeasuredInk {
            normalized,
            polarity,
            isolation,
        } = MeasuredInk::from_rgb(image)?;
        Self::from_groups(image, &normalized, polarity, isolation.groups)
    }

    #[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
    pub(crate) fn from_shared_raster(
        image: &RgbImage,
        normalized: &image::GrayImage,
        polarity: PolarityEvidence,
        threshold: u8,
        topology_thresholds: &[u8],
    ) -> Result<Self, Error> {
        let (width, height) = image.dimensions();
        if normalized.dimensions() != (width, height) {
            return Err(Error::Invalid(
                "inconsistent recognition raster dimensions".into(),
            ));
        }
        let isolation = crate::glyph_isolation::isolate_ink(
            normalized,
            crate::pixel_regions::InkBounds {
                left: 0,
                top: 0,
                right: width,
                bottom: height,
            },
            threshold,
            topology_thresholds,
        )?;
        Self::from_groups(image, normalized, polarity, isolation.groups)
    }

    fn from_groups(
        image: &RgbImage,
        normalized: &image::GrayImage,
        polarity: PolarityEvidence,
        groups: Vec<InkGroup>,
    ) -> Result<Self, Error> {
        let dimensions = image.dimensions();
        let appearances = groups
            .iter()
            .map(|group| {
                if group.touches_line_edge {
                    Ok(None)
                } else {
                    GlyphShape::measure(normalized, group.run.ink).map(Some)
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            dimensions,
            rgb_sha256: hex::encode(Sha256::digest(image.as_raw())),
            normalized_sha256: hex::encode(Sha256::digest(normalized.as_raw())),
            polarity,
            groups,
            appearances,
        })
    }

    fn map_sequence(
        &self,
        geometry: &ViewGeometry,
        hypothesis: SequenceData<'_>,
        runs: &[InkBounds],
    ) -> CandidateMapping {
        let SequenceAlignment::Measured { tokens, .. } = hypothesis.alignment else {
            return CandidateMapping::Unavailable {
                reason: "no conditional token localization".into(),
            };
        };
        if hypothesis.text.chars().count() != tokens.len()
            || tokens
                .iter()
                .map(|t| t.class_index)
                .ne(hypothesis.classes.iter().copied())
        {
            return CandidateMapping::Unavailable {
                reason: "inconsistent CTC token sequence".into(),
            };
        }
        let Some(mut cells) = hypothesis
            .text
            .chars()
            .zip(tokens)
            .map(|(character, token)| geometry.cell(character, token))
            .collect::<Option<Vec<_>>>()
        else {
            return CandidateMapping::Unavailable {
                reason: "invalid CTC token interval".into(),
            };
        };
        // First-entry quantiles are monotone on complete paths, but the
        // marginal modes need not be. Widen to a monotone outer envelope
        // rather than invent spatial order from those modes.
        for index in (1..cells.len()).rev() {
            cells[index - 1].interval[0] =
                cells[index - 1].interval[0].min(cells[index].interval[0]);
        }
        for index in 1..cells.len() {
            cells[index].interval[1] = cells[index].interval[1].max(cells[index - 1].interval[1]);
        }
        let mut anchors = anchor_candidates(&cells, runs);
        let width = f64::from(self.dimensions.0);
        let mut possible_padding_claims = std::collections::BTreeSet::new();
        let mut excluded = Vec::new();
        for (anchor, cell) in anchors.iter_mut().zip(&cells) {
            let inside = cell.interval.iter().all(|x| (0.0..width).contains(x));
            if !cell.character.is_whitespace() && !inside {
                // Do not force a partly padded token to consume a visible
                // glyph. Its possible claim still prevents other tokens
                // from presenting that same ink as uniquely identified.
                let runs = std::mem::take(&mut anchor.candidate_runs);
                if cell.interval[0] < width && cell.interval[1] >= 0.0 {
                    possible_padding_claims.extend(runs.iter().copied());
                }
                excluded.push(Some(runs));
            } else {
                excluded.push(None);
            }
        }
        align_runs(&mut anchors, runs.len(), self.dimensions.0);
        let anchors = anchors
            .into_iter()
            .zip(cells)
            .zip(excluded)
            .map(|((mut anchor, cell), excluded)| {
                if let Some(runs) = excluded {
                    anchor.candidate_runs = runs;
                    anchor.alignment = RunAlignment::Padding;
                } else if let RunAlignment::Unique { run } = anchor.alignment
                    && possible_padding_claims.contains(&run)
                {
                    anchor.alignment = RunAlignment::UncertainNeighbor;
                }
                CandidateAnchor {
                    anchor,
                    entry_interval: cell.interval,
                }
            })
            .collect();
        CandidateMapping::Mapped { anchors }
    }

    /// Apply the exact input-width contract of one actual single-crop view.
    /// Each candidate is localized independently against the same source
    /// groups. Uncertain intervals only broaden geometric compatibility.
    #[must_use]
    pub fn map_view(
        &self,
        view_dimensions: (u32, u32),
        evidence: &CtcEvidence,
    ) -> Option<ViewInkMapping> {
        let bounds = self.groups.iter().map(|g| g.run.ink).collect::<Vec<_>>();
        self.map_view_with_bounds(view_dimensions, evidence, &bounds)
    }

    /// Re-localize the same measured CTC paths against an independently
    /// established ordered partition of the exact original source raster.
    pub(crate) fn map_view_with_bounds(
        &self,
        view_dimensions: (u32, u32),
        evidence: &CtcEvidence,
        runs: &[InkBounds],
    ) -> Option<ViewInkMapping> {
        let geometry =
            ViewGeometry::new(self.dimensions, view_dimensions, evidence.sequence_length)?;
        let hypotheses = evidence
            .hypotheses
            .iter()
            .map(|candidate| self.map_sequence(&geometry, candidate.data(), runs))
            .collect();
        let queries = evidence
            .queries
            .iter()
            .map(|query| self.map_sequence(&geometry, query.data(), runs))
            .collect();
        Some(ViewInkMapping {
            geometry,
            hypotheses,
            queries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc_evidence::SequenceHypothesis;
    use image::Rgb;

    fn source() -> Result<SourceInk, Error> {
        SourceInk::measure(&RgbImage::from_fn(100, 20, |x, y| {
            if ((10..25).contains(&x) || (70..85).contains(&x)) && (4..17).contains(&y) {
                Rgb([0; 3])
            } else {
                Rgb([255; 3])
            }
        }))
    }

    fn hypothesis(text: &str, tokens: Vec<TokenLocalization>) -> SequenceHypothesis {
        SequenceHypothesis {
            text: text.into(),
            classes: tokens.iter().map(|t| t.class_index).collect(),
            log_mass_lower_bound: -1.0,
            alignment: SequenceAlignment::Measured {
                log_mass: -0.5,
                tokens,
            },
        }
    }

    fn token(class_index: usize, mode: usize, interval: [usize; 2]) -> TokenLocalization {
        TokenLocalization {
            class_index,
            mode,
            interval,
            mode_probability: 0.9,
        }
    }

    fn mapped_fixture(
        view: (u32, u32),
        hypothesis: &SequenceHypothesis,
    ) -> Result<Vec<CandidateAnchor>, Error> {
        let source = source()?;
        assert_eq!(source.groups.len(), 2);
        let geometry = ViewGeometry::new(source.dimensions, view, 80)
            .ok_or_else(|| Error::Invalid("fixture geometry".into()))?;
        let bounds = source.groups.iter().map(|g| g.run.ink).collect::<Vec<_>>();
        match source.map_sequence(&geometry, hypothesis.data(), &bounds) {
            CandidateMapping::Mapped { anchors } => Ok(anchors),
            CandidateMapping::Unavailable { reason } => Err(Error::Invalid(reason)),
        }
    }

    #[test]
    fn alternate_labels_share_exact_source_ink_without_resampling() -> Result<(), Error> {
        let a = hypothesis("AB", vec![token(1, 13, [12, 14]), token(2, 57, [56, 58])]);
        let b = hypothesis("XB", vec![token(3, 14, [12, 15]), token(2, 57, [56, 58])]);
        for h in [&a, &b] {
            let anchors = mapped_fixture((150, 20), h)?;
            assert_eq!(anchors[0].anchor.alignment, RunAlignment::Unique { run: 0 });
            assert_eq!(anchors[1].anchor.alignment, RunAlignment::Unique { run: 1 });
        }
        Ok(())
    }

    #[test]
    fn intervals_that_extend_into_padding_do_not_bind_a_glyph() -> Result<(), Error> {
        let h = hypothesis("A", vec![token(1, 40, [39, 60])]);
        let anchors = mapped_fixture((85, 20), &h)?;
        assert_eq!(anchors[0].anchor.alignment, RunAlignment::Padding);
        Ok(())
    }

    #[test]
    fn partially_padded_token_does_not_prove_or_steal_neighbor_ink() -> Result<(), Error> {
        let h = hypothesis(
            "ABC",
            vec![
                token(1, 10, [9, 11]),
                token(2, 45, [44, 46]),
                token(3, 65, [45, 70]),
            ],
        );
        let anchors = mapped_fixture((100, 20), &h)?;
        assert_eq!(anchors[0].anchor.alignment, RunAlignment::Unique { run: 0 });
        assert_eq!(anchors[1].anchor.alignment, RunAlignment::UncertainNeighbor);
        assert_eq!(anchors[2].anchor.alignment, RunAlignment::Padding);
        assert!(anchors[2].anchor.candidate_runs.contains(&1));
        Ok(())
    }

    #[test]
    fn uncertain_entries_cannot_split_one_visible_run() -> Result<(), Error> {
        let h = hypothesis(
            "AAB",
            vec![
                token(1, 4, [3, 6]),
                token(1, 5, [4, 7]),
                token(2, 46, [45, 47]),
            ],
        );
        let anchors = mapped_fixture((100, 20), &h)?;
        assert_ne!(anchors[0].anchor.alignment, RunAlignment::Unique { run: 0 });
        assert_eq!(anchors[2].anchor.alignment, RunAlignment::Unique { run: 1 });
        Ok(())
    }
}
