//! Candidate-specific localization of CTC evidence on measured source ink.
//!
//! All horizontal views use one segmentation of the unmodified
//! reading-order crop, so alternative hypotheses cannot create new pixels.

use image::{Pixel, RgbImage};
use num_traits::ToPrimitive;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Error,
    character_alignment::{
        BatchGeometry, CharacterAnchor, EmissionCell, RunAlignment, align_runs, anchor_candidates,
    },
    ctc_alignment::{SequenceAlignment, TokenLocalization},
    ctc_evidence::{CtcEvidence, SequenceHypothesis},
    glyph_appearance::GlyphShape,
    glyph_isolation::{InkGroup, isolate_ink},
    ink_polarity::{Polarity, PolarityEvidence, estimate},
    pixel_regions::InkBounds,
};

/// A single independent measurement of source ink, shared by all hypotheses.
#[derive(Debug, Serialize)]
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
#[derive(Debug, Serialize)]
pub struct CandidateAnchor {
    #[serde(flatten)]
    pub anchor: CharacterAnchor,
    pub entry_interval: [f64; 2],
}

/// Per-hypothesis evidence; unlocalized hypotheses remain actual model
/// hypotheses and are never silently replaced with greedy geometry.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CandidateMapping {
    Mapped { anchors: Vec<CandidateAnchor> },
    Unavailable { reason: &'static str },
}

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
pub struct ViewInkMapping {
    pub geometry: ViewGeometry,
    /// Same order as the measured CTC hypotheses, even on partial failure.
    pub hypotheses: Vec<CandidateMapping>,
}

impl SourceInk {
    /// Segment the source RGB once, without any OCR character count or text.
    ///
    /// # Errors
    /// Rejects empty or invalid source crops.
    pub fn measure(image: &RgbImage) -> Result<Self, Error> {
        let dimensions = image.dimensions();
        let gray = image::GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
            image.get_pixel(x, y).to_luma()
        });
        let polarity = estimate(&gray, Polarity::Auto)?;
        let normalized = polarity.normalize(&gray);
        let groups = isolate_ink(
            &normalized,
            InkBounds {
                left: 0,
                top: 0,
                right: dimensions.0,
                bottom: dimensions.1,
            },
            192,
            &[128, 192, 224],
        )?
        .groups;
        let appearances = groups
            .iter()
            .map(|group| {
                if group.touches_line_edge {
                    Ok(None)
                } else {
                    GlyphShape::measure(&normalized, group.run.ink).map(Some)
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

    fn map_hypothesis(
        &self,
        geometry: &ViewGeometry,
        hypothesis: &SequenceHypothesis,
    ) -> CandidateMapping {
        let SequenceAlignment::Measured { tokens, .. } = &hypothesis.alignment else {
            return CandidateMapping::Unavailable {
                reason: "no conditional token localization",
            };
        };
        if hypothesis.text.chars().count() != tokens.len()
            || tokens
                .iter()
                .map(|t| t.class_index)
                .ne(hypothesis.classes.iter().copied())
        {
            return CandidateMapping::Unavailable {
                reason: "inconsistent CTC token sequence",
            };
        }
        let Some(cells) = hypothesis
            .text
            .chars()
            .zip(tokens)
            .map(|(character, token)| geometry.cell(character, token))
            .collect::<Option<Vec<_>>>()
        else {
            return CandidateMapping::Unavailable {
                reason: "invalid CTC token interval",
            };
        };
        let mut anchors = anchor_candidates(&cells, &self.groups);
        align_runs(&mut anchors, self.groups.len(), self.dimensions.0);
        let anchors = anchors
            .into_iter()
            .zip(cells)
            .map(|(mut anchor, cell)| {
                let inside = cell
                    .interval
                    .iter()
                    .all(|x| (0.0..f64::from(self.dimensions.0)).contains(x));
                if !cell.character.is_whitespace() && !inside {
                    // Even when the modal entry falls inside the source, an
                    // interval extending into padding cannot prove a visible run.
                    anchor.alignment = RunAlignment::Padding;
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
        let geometry =
            ViewGeometry::new(self.dimensions, view_dimensions, evidence.sequence_length)?;
        let hypotheses = evidence
            .hypotheses
            .iter()
            .map(|candidate| self.map_hypothesis(&geometry, candidate))
            .collect();
        Some(ViewInkMapping {
            geometry,
            hypotheses,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn alternate_labels_share_exact_source_ink_without_resampling() -> Result<(), Error> {
        let source = source()?;
        assert_eq!(source.groups.len(), 2);
        let geometry = ViewGeometry::new((100, 20), (150, 20), 80)
            .ok_or_else(|| Error::Invalid("fixture geometry".into()))?;
        let a = hypothesis("AB", vec![token(1, 13, [12, 14]), token(2, 57, [56, 58])]);
        let b = hypothesis("XB", vec![token(3, 14, [12, 15]), token(2, 57, [56, 58])]);
        for h in [&a, &b] {
            let CandidateMapping::Mapped { anchors } = source.map_hypothesis(&geometry, h) else {
                return Err(Error::Invalid("fixture mapping".into()));
            };
            assert_eq!(anchors[0].anchor.alignment, RunAlignment::Unique { run: 0 });
            assert_eq!(anchors[1].anchor.alignment, RunAlignment::Unique { run: 1 });
        }
        Ok(())
    }

    #[test]
    fn intervals_that_extend_into_padding_do_not_bind_a_glyph() -> Result<(), Error> {
        let source = source()?;
        let geometry = ViewGeometry::new((100, 20), (85, 20), 80)
            .ok_or_else(|| Error::Invalid("fixture geometry".into()))?;
        let h = hypothesis("A", vec![token(1, 40, [39, 60])]);
        let CandidateMapping::Mapped { anchors } = source.map_hypothesis(&geometry, &h) else {
            return Err(Error::Invalid("fixture mapping".into()));
        };
        assert_eq!(anchors[0].anchor.alignment, RunAlignment::Padding);
        Ok(())
    }

    #[test]
    fn uncertain_entries_cannot_split_one_visible_run() -> Result<(), Error> {
        let source = source()?;
        let geometry = ViewGeometry::new((100, 20), (100, 20), 80)
            .ok_or_else(|| Error::Invalid("fixture geometry".into()))?;
        let h = hypothesis(
            "AAB",
            vec![
                token(1, 4, [3, 6]),
                token(1, 5, [4, 7]),
                token(2, 46, [45, 47]),
            ],
        );
        let CandidateMapping::Mapped { anchors } = source.map_hypothesis(&geometry, &h) else {
            return Err(Error::Invalid("fixture mapping".into()));
        };
        assert_ne!(anchors[0].anchor.alignment, RunAlignment::Unique { run: 0 });
        assert_eq!(anchors[2].anchor.alignment, RunAlignment::Unique { run: 1 });
        Ok(())
    }
}
