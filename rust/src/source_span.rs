//! Ordered ownership bands on an unmodified shared OCR raster.
//!
//! These bands are *hypotheses* about pixel-to-character ownership, not
//! semantic ground truth. Unlike disconnected vertical projection runs they
//! can partition a connected or touching glyph group without inventing or
//! removing pixels. Recognition is allowed to proceed only when the shared
//! source raster independently supports every claimed band.

use std::collections::BTreeSet;

use image::GrayImage;
use num_traits::ToPrimitive;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Error,
    candidate_ink::CandidateAnchor,
    ctc_evidence::{CtcOutcome, SequenceReference},
    fragment_ink::{RunAccounting, Sample},
    geometry::Quad,
    pixel_regions::InkBounds,
    recognition_model::RecognitionModel,
    refinement_evidence::{Recognition, RefinementDecision, ViewEvidence, canonical},
    source_reconciliation::{Embedding, SourceEdits, embedding_compatible},
};

/// Exact, disjoint source columns attributed to a visible CTC token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ColumnBand {
    pub token: usize,
    /// Source crop x coordinates, half-open, with full source crop height.
    pub bounds: InkBounds,
    /// Exact foreground count, never a bounding-box estimate.
    pub foreground: u64,
    /// Whether measured ink touches any original source raster boundary.
    pub touches_source_edge: bool,
}

/// All accepted CTC views' tokens in the same canonical reading order.
/// Their entry intervals must be strictly orderable across all views;
/// overlapping interval families cannot be made exact by sorting modes.
pub fn ordered_partition(
    views: &[Vec<&CandidateAnchor>],
    dimensions: (u32, u32),
) -> Option<Vec<InkBounds>> {
    let (width, height) = dimensions;
    let tokens = views.first()?.len();
    if width == 0 || height == 0 || tokens == 0 || views.iter().any(|v| v.len() != tokens) {
        return None;
    }
    let w = f64::from(width);
    let mut lows = vec![f64::INFINITY; tokens];
    let mut highs = vec![f64::NEG_INFINITY; tokens];
    for view in views {
        for (i, token) in view.iter().enumerate() {
            let [lo, hi] = token.entry_interval;
            if !lo.is_finite()
                || !hi.is_finite()
                || lo > hi
                || lo < 0.0
                || hi >= w
                || !(0.0..w).contains(&token.anchor.crop_x)
            {
                return None;
            }
            lows[i] = lows[i].min(lo);
            highs[i] = highs[i].max(hi);
        }
    }
    if lows.iter().zip(&highs).any(|(lo, hi)| lo > hi) {
        return None;
    }
    let mut cuts = Vec::with_capacity(tokens + 1);
    cuts.push(0);
    for i in 1..tokens {
        // The cut lies in the *unclaimed* gap between every earlier entry
        // interval and every later entry interval. The intervening real
        // pixels are assigned once, never counted twice or dropped.
        let a = highs[i - 1];
        let b = lows[i];
        if a >= b {
            return None;
        }
        let cut = ((a + b) / 2.0).round().to_u32()?;
        if cut <= *cuts.last()? || cut >= width {
            return None;
        }
        cuts.push(cut);
    }
    cuts.push(width);
    let mut result = Vec::with_capacity(tokens);
    for (i, edge) in cuts.windows(2).enumerate() {
        let bounds = InkBounds {
            left: edge[0],
            top: 0,
            right: edge[1],
            bottom: height,
        };
        if lows[i] < f64::from(bounds.left) || highs[i] >= f64::from(bounds.right) {
            return None;
        }
        result.push(bounds);
    }
    Some(result)
}

/// Count exact foreground, including ink crossing a band boundary.
/// A touching component is counted in multiple adjacent bands by its
/// *disjoint pixels*, not duplicated as a whole component.
pub fn measured_bands(
    raster: &GrayImage,
    threshold: u8,
    bounds: &[InkBounds],
    remaining_checks: &mut u64,
) -> Result<Option<Vec<ColumnBand>>, Error> {
    let (width, height) = raster.dimensions();
    if threshold == 0
        || bounds.is_empty()
        || bounds[0].left != 0
        || bounds.last().is_none_or(|b| b.right != width)
        || bounds
            .iter()
            .any(|b| b.top != 0 || b.bottom != height || b.left >= b.right)
        || bounds.windows(2).any(|p| p[0].right != p[1].left)
    {
        return Err(Error::Invalid(
            "invalid ordered source ink partition".into(),
        ));
    }
    let pixels = u64::from(width) * u64::from(height);
    let Some(remaining) = remaining_checks.checked_sub(pixels) else {
        return Ok(None);
    };
    *remaining_checks = remaining;
    let mut result = Vec::with_capacity(bounds.len());
    for (token, &b) in bounds.iter().enumerate() {
        let mut foreground = 0_u64;
        let mut touches_source_edge = false;
        for y in 0..height {
            for x in b.left..b.right {
                if raster.get_pixel(x, y).0[0] < threshold {
                    foreground += 1;
                    touches_source_edge |= x == 0 || x + 1 == width || y == 0 || y + 1 == height;
                }
            }
        }
        result.push(ColumnBand {
            token,
            bounds: b,
            foreground,
            touches_source_edge,
        });
    }
    Ok(Some(result))
}

/// A geometrically proposed text revision that has not been committed.
/// These are measured source pixels and exact class-sequence likelihoods,
/// not a truth label or an automatically admitted document change.
#[derive(Debug, Serialize)]
pub struct SpanVote {
    pub view: usize,
    pub primary: SequenceReference,
    pub log_odds: f64,
    pub embedding: Embedding,
}

#[derive(Debug, Serialize)]
pub struct SourceSpanEvidence {
    pub source_rgb_sha256: String,
    pub normalized_sha256: String,
    pub revised_text: String,
    pub bands: Vec<ColumnBand>,
    pub edits: SourceEdits,
    pub votes: Vec<SpanVote>,
    pub source_accounting: RunAccounting,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SourceSpanOutcome {
    Skipped { reason: &'static str },
    Measured { evidence: SourceSpanEvidence },
}

/// Build all-view token bands without relying on injective projection runs.
fn candidate_anchors<'a>(
    recognition: &'a Recognition,
    text: &str,
    view_indices: &[usize],
) -> Option<Vec<(usize, &'a ViewEvidence, Vec<&'a CandidateAnchor>)>> {
    view_indices
        .iter()
        .copied()
        .map(|i| {
            let view = recognition.views.get(i)?;
            let map = view.ink_mapping.as_ref()?.hypotheses.first()?;
            let visible = map.visible_anchors(text)?;
            Some((i, view, visible))
        })
        .collect()
}

fn view_vote(
    i: usize,
    view: &ViewEvidence,
    old: &str,
    revised: &str,
    bands: &[ColumnBand],
) -> Option<SpanVote> {
    let CtcOutcome::Measured { evidence } = &view.ctc else {
        return None;
    };
    let new = evidence.hypotheses.first()?;
    if canonical(&new.text) != revised {
        return None;
    }
    let primary = evidence.strongest_matching(|text| canonical(text) == old)?;
    let odds = new.data().log_mass()? - evidence.sequence(primary)?.log_mass()?;
    if odds <= 0.0 {
        return None;
    }
    let mapping = view.ink_mapping.as_ref()?;
    let old_anchors = mapping.sequence(primary)?.visible_anchors(old)?;
    let revised_characters = revised.chars().collect::<Vec<_>>();
    let old_characters = old.chars().collect::<Vec<_>>();
    let editable = bands
        .iter()
        .map(|band| band.foreground > 0 && !band.touches_source_edge)
        .collect::<Vec<_>>();
    let embedding =
        embedding_compatible(&old_characters, &revised_characters, &editable, |j, i| {
            let [lo, hi] = old_anchors[j].entry_interval;
            let span = bands[i].bounds;
            lo >= f64::from(span.left)
                && hi < f64::from(span.right)
                && (f64::from(span.left)..f64::from(span.right))
                    .contains(&old_anchors[j].anchor.crop_x)
        })?;
    Some(SpanVote {
        view: i,
        primary,
        log_odds: odds,
        embedding,
    })
}

/// Requires two independent model families, not two stretches of one model.
/// All accepted views share the same exact unmodified RGB/source raster.
pub struct SpanInput<'a> {
    pub recognition: &'a Recognition,
    pub original: &'a str,
    pub rgb: &'a image::RgbImage,
    pub normalized: &'a GrayImage,
    pub sample: Sample<'a>,
    pub observation: InkBounds,
    pub peers: &'a [Quad],
}

pub fn probe(input: SpanInput<'_>, remaining_checks: &mut u64) -> Result<SourceSpanOutcome, Error> {
    let SpanInput {
        recognition,
        original,
        rgb,
        normalized,
        sample,
        observation,
        peers,
    } = input;
    let skip = |reason| SourceSpanOutcome::Skipped { reason };
    let RefinementDecision::Consensus { text, view_indices } = &recognition.decision else {
        return Ok(skip("no_consensus"));
    };
    let primary = canonical(original);
    let revised = canonical(text);
    if primary == revised {
        return Ok(skip("unchanged"));
    }
    if primary.chars().count() > revised.chars().count() {
        return Ok(skip("would_delete_text"));
    }
    if rgb.dimensions() != normalized.dimensions()
        || normalized.dimensions() != recognition.source_ink.dimensions
        || hex::encode(Sha256::digest(rgb.as_raw())) != recognition.source_ink.rgb_sha256
        || hex::encode(Sha256::digest(normalized.as_raw()))
            != recognition.source_ink.normalized_sha256
    {
        return Err(Error::Invalid(
            "source-span recognition raster provenance mismatch".into(),
        ));
    }
    let Some(views) = candidate_anchors(recognition, &revised, view_indices) else {
        return Ok(skip("missing_ctc_entry_evidence"));
    };
    let anchor_sets = views.iter().map(|(_, _, v)| v.clone()).collect::<Vec<_>>();
    let Some(bounds) = ordered_partition(&anchor_sets, normalized.dimensions()) else {
        return Ok(skip("unresolved_ordered_partition"));
    };
    let Some(bands) = measured_bands(normalized, sample.threshold, &bounds, remaining_checks)?
    else {
        return Ok(skip("source_pixel_budget"));
    };
    let mut votes = Vec::new();
    let mut seen = BTreeSet::new();
    let mut models = BTreeSet::new();
    for (i, view, _) in views {
        if !seen.insert((view.model, view.stretch_percent)) {
            continue;
        }
        if let Some(vote) = view_vote(i, view, &primary, &revised, &bands) {
            models.insert(view.model);
            votes.push(vote);
        }
    }
    if !models.contains(&RecognitionModel::Small) || !models.contains(&RecognitionModel::Medium) {
        return Ok(skip("unlocalized_original_text"));
    }
    let edits = SourceEdits::combine(votes.iter().map(|v| &v.embedding.edits))
        .ok_or_else(|| Error::Invalid("missing source-span embedding votes".into()))?;
    let accounting = sample.account(observation, peers, remaining_checks)?;
    let Some(source_accounting) =
        sample.account_spans(observation, &bands, peers, &accounting, remaining_checks)?
    else {
        return Ok(skip("source_membership_budget"));
    };
    for position in edits.affected() {
        if source_accounting.runs[position]
            .peers
            .values()
            .any(|count| count.original != 0)
        {
            return Ok(skip("competing_peer_foreground"));
        }
    }
    Ok(SourceSpanOutcome::Measured {
        evidence: SourceSpanEvidence {
            source_rgb_sha256: recognition.source_ink.rgb_sha256.clone(),
            normalized_sha256: recognition.source_ink.normalized_sha256.clone(),
            revised_text: text.clone(),
            bands,
            edits,
            votes,
            source_accounting,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::character_alignment::{CharacterAnchor, RunAlignment};
    use image::Luma;

    fn anchor(c: char, x: f64, interval: [f64; 2]) -> CandidateAnchor {
        CandidateAnchor {
            anchor: CharacterAnchor {
                character: c,
                column: 0,
                crop_x: x,
                candidate_runs: Vec::new(),
                alignment: RunAlignment::NoInjectiveAssignment,
            },
            entry_interval: interval,
        }
    }

    #[test]
    fn two_touching_glyphs_still_have_disjoint_exact_source_pixel_bands() -> Result<(), Error> {
        // One continuous connected horizontal stroke. Strict projection
        // segmentation sees one run, yet both token bands own real ink.
        let image = GrayImage::from_fn(20, 8, |x, y| {
            if y == 4 && (2..18).contains(&x) {
                Luma([0])
            } else {
                Luma([255])
            }
        });
        let first = [
            anchor('A', 5.0, [4.0, 6.0]),
            anchor('B', 14.0, [13.0, 15.0]),
        ];
        let second = [
            anchor('A', 5.5, [4.5, 6.5]),
            anchor('B', 14.5, [13.5, 15.5]),
        ];
        let views = vec![first.iter().collect(), second.iter().collect()];
        let bounds = ordered_partition(&views, image.dimensions())
            .ok_or_else(|| Error::Invalid("expected ordered partition".into()))?;
        let mut budget = 20 * 8;
        let measured = measured_bands(&image, 128, &bounds, &mut budget)?
            .ok_or_else(|| Error::Invalid("expected measured bands".into()))?;
        assert_eq!(budget, 0);
        assert_eq!(measured.len(), 2);
        assert_eq!(measured.iter().map(|b| b.foreground).sum::<u64>(), 16);
        assert!(
            measured
                .iter()
                .all(|b| b.foreground > 0 && !b.touches_source_edge)
        );
        assert_eq!(measured[0].bounds.right, measured[1].bounds.left);
        Ok(())
    }

    #[test]
    fn refuses_overlapping_or_padded_ctc_intervals() {
        let a = [anchor('A', 7.0, [4.0, 9.0]), anchor('B', 10.0, [8.0, 13.0])];
        assert!(ordered_partition(&[a.iter().collect()], (20, 8)).is_none());
        let a = [
            anchor('A', 7.0, [-1.0, 9.0]),
            anchor('B', 14.0, [12.0, 16.0]),
        ];
        assert!(ordered_partition(&[a.iter().collect()], (20, 8)).is_none());
        let a = [
            anchor('A', 7.0, [4.0, 9.0]),
            anchor('B', 17.0, [12.0, 20.0]),
        ];
        assert!(ordered_partition(&[a.iter().collect()], (20, 8)).is_none());
    }

    #[test]
    fn source_budget_and_partition_integrity() -> Result<(), Error> {
        let raster = GrayImage::from_pixel(10, 10, Luma([0]));
        let bounds = [
            InkBounds {
                left: 0,
                right: 5,
                top: 0,
                bottom: 10,
            },
            InkBounds {
                left: 5,
                right: 10,
                top: 0,
                bottom: 10,
            },
        ];
        let mut budget = 99;
        assert!(measured_bands(&raster, 128, &bounds, &mut budget)?.is_none());
        assert_eq!(budget, 99);
        let mut budget = 100;
        assert_eq!(
            measured_bands(&raster, 128, &bounds, &mut budget)?
                .ok_or_else(|| Error::Invalid("budget".into()))?
                .iter()
                .map(|b| b.foreground)
                .sum::<u64>(),
            100
        );
        assert_eq!(budget, 0);
        let invalid = [
            InkBounds {
                left: 0,
                right: 6,
                top: 0,
                bottom: 10,
            },
            InkBounds {
                left: 5,
                right: 10,
                top: 0,
                bottom: 10,
            },
        ];
        assert!(measured_bands(&raster, 128, &invalid, &mut budget).is_err());
        Ok(())
    }
}
