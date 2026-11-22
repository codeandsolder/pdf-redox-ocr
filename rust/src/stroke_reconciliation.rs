//! Source-supported deletion of external stroke transcriptions. All retained
//! and removed ink is partitioned in a single native sampling frame, and the
//! original and contracted CTC hypotheses must agree on surviving token runs.

use std::{collections::BTreeSet, ops::Range};

use serde::Serialize;

use crate::{
    candidate_ink::{CandidateMapping, SourceInk},
    character_alignment::RunAlignment,
    ctc_evidence::CtcOutcome,
    fragment_reconciliation::{Operation, Rejection, Transaction},
    layout_leaders::LeaderEdge,
    line_audit::{AuditedLine, MappedEdgeContraction},
    pixel_regions::InkBounds,
    recognition_model::RecognitionModel,
    refinement_evidence::{Recognition, RefinementDecision, canonical},
};

/// Independent projection partitions, in the common padded raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartitionRun {
    pub bounds: InkBounds,
    pub foreground_pixels: u64,
    /// Index of the retained-crop run; absent for a removed stripe.
    pub retained_run: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ContractionVote {
    pub original_view: usize,
    pub original_hypotheses: Vec<usize>,
    pub contracted_view: usize,
}

/// Bind all original non-space tokens to one ordered source partition.
/// The absent retained-run identity identifies exactly the deleted tokens.
#[derive(Debug, Serialize)]
pub struct Contraction {
    pub contraction_index: usize,
    pub removed_pixels: u64,
    pub retained_pixels: u64,
    pub partition: Vec<PartitionRun>,
    pub original_token_runs: Vec<usize>,
    pub votes: Vec<ContractionVote>,
}

fn dimensions(b: InkBounds) -> (u32, u32) {
    (b.right - b.left, b.bottom - b.top)
}

fn translated(b: InkBounds, x: u32, y: u32) -> InkBounds {
    InkBounds {
        left: b.left + x,
        right: b.right + x,
        top: b.top + y,
        bottom: b.bottom + y,
    }
}

fn within(b: InkBounds, outer: InkBounds) -> bool {
    b.left >= outer.left
        && b.right <= outer.right
        && b.top >= outer.top
        && b.bottom <= outer.bottom
        && b.left < b.right
        && b.top < b.bottom
}

fn pixels(ink: &SourceInk) -> u64 {
    ink.groups
        .iter()
        .map(|g| g.run.observation_support.inside_pixels)
        .sum()
}

/// The rejected components can share a projection column with retained
/// components at a half-open boundary. Keep the two independently measured
/// partitions separate, instead of pretending an empty column exists.
fn partition(
    original: &SourceInk,
    retained: &SourceInk,
    observation: InkBounds,
    contraction: &MappedEdgeContraction,
) -> Option<Vec<PartitionRun>> {
    let e = &contraction.evidence;
    let b = e.retained;
    if !within(b, observation)
        || b.top != observation.top
        || b.bottom != observation.bottom
        || original.dimensions != dimensions(observation)
        || retained.dimensions != dimensions(b)
        || pixels(original) != e.removed_pixels.checked_add(e.retained_pixels)?
        || pixels(retained) != e.retained_pixels
        || e.removed_pixels == 0
        || e.retained_pixels == 0
    {
        return None;
    }
    let removed = match e.edge {
        LeaderEdge::Left if b.right == observation.right => InkBounds {
            right: b.left,
            ..observation
        },
        LeaderEdge::Right if b.left == observation.left => InkBounds {
            left: b.right,
            ..observation
        },
        _ => return None,
    };
    let mut strokes = e.strokes.iter().collect::<Vec<_>>();
    strokes.sort_unstable_by_key(|s| (s.inside_bounds.left, s.inside_bounds.right));
    let mut runs: Vec<PartitionRun> = Vec::new();
    for stroke in strokes {
        let bounds = stroke.inside_bounds;
        if !within(bounds, removed) || stroke.inside.pixels == 0 {
            return None;
        }
        if let Some(last) = runs.last_mut().filter(|p| bounds.left <= p.bounds.right) {
            last.bounds = last.bounds.union(bounds);
            last.foreground_pixels = last.foreground_pixels.checked_add(stroke.inside.pixels)?;
        } else {
            runs.push(PartitionRun {
                bounds,
                foreground_pixels: stroke.inside.pixels,
                retained_run: None,
            });
        }
    }
    if runs.iter().map(|r| r.foreground_pixels).sum::<u64>() != e.removed_pixels {
        return None;
    }
    for (i, g) in retained.groups.iter().enumerate() {
        let bounds = translated(g.run.ink, b.left, b.top);
        if !within(bounds, b) {
            return None;
        }
        runs.push(PartitionRun {
            bounds,
            foreground_pixels: g.run.observation_support.inside_pixels,
            retained_run: Some(i),
        });
    }
    runs.sort_unstable_by_key(|r| (r.bounds.left, r.bounds.right));
    if runs
        .windows(2)
        .any(|r| r[0].bounds.right > r[1].bounds.left)
    {
        return None;
    }
    Some(runs)
}

fn removed_range(original: &str, revised: &str, edge: LeaderEdge) -> Option<Range<usize>> {
    let old = canonical(original);
    let new = canonical(revised);
    let (a, b) = (old.chars().count(), new.chars().count());
    if b == 0 || a <= b {
        return None;
    }
    match edge {
        LeaderEdge::Left if old.ends_with(&new) => Some(0..a - b),
        LeaderEdge::Right if old.starts_with(&new) => Some(b..a),
        _ => None,
    }
}

fn assignment(mapping: &CandidateMapping, text: &str) -> Option<Vec<usize>> {
    let CandidateMapping::Mapped { anchors } = mapping else {
        return None;
    };
    let visible = anchors
        .iter()
        .filter(|a| !a.anchor.character.is_whitespace())
        .collect::<Vec<_>>();
    if !visible.iter().map(|a| a.anchor.character).eq(text.chars()) {
        return None;
    }
    let runs = visible
        .iter()
        .map(|a| match a.anchor.alignment {
            RunAlignment::Unique { run } => Some(run),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    runs.windows(2).all(|p| p[0] < p[1]).then_some(runs)
}

/// Every removed run must be bound and no retained token may change its
/// measured source identity. Unclaimed retained ink is allowed.
fn conserved(
    old: &[usize],
    new: &[usize],
    removed: &Range<usize>,
    partition: &[PartitionRun],
) -> bool {
    if old.len().checked_sub(removed.len()) != Some(new.len()) {
        return false;
    }
    let mut removed_runs = BTreeSet::new();
    let mut survivors = Vec::new();
    for (position, &run) in old.iter().enumerate() {
        let Some(p) = partition.get(run) else {
            return false;
        };
        if removed.contains(&position) {
            if p.retained_run.is_some() || !removed_runs.insert(run) {
                return false;
            }
        } else {
            let Some(retained) = p.retained_run else {
                return false;
            };
            survivors.push(retained);
        }
    }
    survivors == new
        && removed_runs.len()
            == partition
                .iter()
                .filter(|p| p.retained_run.is_none())
                .count()
}

fn original_assignment(
    recognition: &Recognition,
    view_index: usize,
    text: &str,
    bounds: &[InkBounds],
) -> Option<(Vec<usize>, Vec<usize>)> {
    let view = recognition.views.get(view_index)?;
    let CtcOutcome::Measured { evidence } = &view.ctc else {
        return None;
    };
    let map = recognition
        .source_ink
        .map_view_with_bounds(view.dimensions, evidence, bounds)?;
    let mut result = None;
    let mut indices = Vec::new();
    for (index, (h, m)) in evidence.hypotheses.iter().zip(&map.hypotheses).enumerate() {
        if canonical(&h.text) != text {
            continue;
        }
        let runs = assignment(m, text)?;
        if result.as_ref().is_some_and(|old| old != &runs) {
            return None;
        }
        result = Some(runs);
        indices.push(index);
    }
    Some((result?, indices))
}

struct VoteContext<'a> {
    original: &'a Recognition,
    revised: &'a Recognition,
    old: &'a str,
    new: &'a str,
    bounds: &'a [InkBounds],
    removed: &'a Range<usize>,
    partition: &'a [PartitionRun],
}

impl VoteContext<'_> {
    fn vote(&self, view_index: usize) -> Option<(Vec<usize>, ContractionVote)> {
        let view = self.revised.views.get(view_index)?;
        let CtcOutcome::Measured { evidence } = &view.ctc else {
            return None;
        };
        if canonical(&evidence.hypotheses.first()?.text) != self.new {
            return None;
        }
        let revised_runs = assignment(view.ink_mapping.as_ref()?.hypotheses.first()?, self.new)?;
        let mut candidates =
            self.original.views.iter().enumerate().filter(|(_, v)| {
                v.model == view.model && v.stretch_percent == view.stretch_percent
            });
        let (index, _) = candidates.next()?;
        if candidates.next().is_some() {
            return None;
        }
        let (runs, indices) = original_assignment(self.original, index, self.old, self.bounds)?;
        if !conserved(&runs, &revised_runs, self.removed, self.partition) {
            return None;
        }
        Some((
            runs,
            ContractionVote {
                original_view: index,
                original_hypotheses: indices,
                contracted_view: view_index,
            },
        ))
    }
}

/// Require a literal consensus edge deletion and matching source-token
/// identities in both recognizer models.
///
/// # Errors
/// Returns the measured reason that a contraction cannot be supported.
pub fn propose(
    line: &AuditedLine,
    contraction_index: usize,
    original: &Recognition,
    revised: &Recognition,
) -> Result<Transaction, Rejection> {
    let contraction = line
        .foreign_edge_contractions
        .as_ref()
        .and_then(|c| c.get(contraction_index))
        .ok_or(Rejection::NoMeasuredInk)?;
    let RefinementDecision::Consensus { text, view_indices } = &revised.decision else {
        return Err(Rejection::NoConsensus);
    };
    let old = canonical(&line.original_text);
    let new = canonical(text);
    if old == new {
        return Err(Rejection::Unchanged);
    }
    let removed = removed_range(&old, &new, contraction.evidence.edge)
        .ok_or(Rejection::NotLiteralContraction)?;
    let partition = partition(
        &original.source_ink,
        &revised.source_ink,
        line.observation_bounds,
        contraction,
    )
    .ok_or(Rejection::ForegroundPartitionMismatch)?;
    let bounds = partition
        .iter()
        .map(|p| InkBounds {
            left: p.bounds.left - line.observation_bounds.left,
            right: p.bounds.right - line.observation_bounds.left,
            top: p.bounds.top - line.observation_bounds.top,
            bottom: p.bounds.bottom - line.observation_bounds.top,
        })
        .collect::<Vec<_>>();
    let mut votes = Vec::new();
    let mut models = BTreeSet::new();
    let mut assigned = None;
    let context = VoteContext {
        original,
        revised,
        old: &old,
        new: &new,
        bounds: &bounds,
        removed: &removed,
        partition: &partition,
    };
    for &i in view_indices {
        let Some((runs, evidence)) = context.vote(i) else {
            continue;
        };
        if assigned.as_ref().is_some_and(|a| a != &runs) {
            return Err(Rejection::UnlocalizedRemovedTokens);
        }
        assigned = Some(runs);
        models.insert(revised.views[i].model);
        votes.push(evidence);
    }
    if !models.contains(&RecognitionModel::Small) || !models.contains(&RecognitionModel::Medium) {
        return Err(Rejection::UnlocalizedRemovedTokens);
    }
    Ok(Transaction {
        page: line.page,
        line_index: line.line_index,
        quad: contraction.proposed_quad,
        text: text.clone(),
        operation: Operation::Contraction(Contraction {
            contraction_index,
            removed_pixels: contraction.evidence.removed_pixels,
            retained_pixels: contraction.evidence.retained_pixels,
            partition,
            original_token_runs: assigned.ok_or(Rejection::UnlocalizedRemovedTokens)?,
            votes,
        }),
        consumed: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(x: u32, retained_run: Option<usize>) -> PartitionRun {
        PartitionRun {
            bounds: InkBounds {
                left: x,
                right: x + 4,
                top: 0,
                bottom: 5,
            },
            foreground_pixels: 8,
            retained_run,
        }
    }

    #[test]
    fn detailed_partition_supports_adjacent_runs_and_rejects_lost_glyphs() {
        let p = [
            run(0, None),
            run(4, Some(0)),
            run(9, Some(1)),
            run(15, Some(2)),
        ];
        assert!(conserved(&[0, 1, 2, 3], &[0, 1, 2], &(0..1), &p));
        assert!(!conserved(&[0, 1, 2], &[1, 2], &(0..1), &p));
        assert!(!conserved(&[0, 1, 2], &[0], &(0..2), &p));
        assert!(!conserved(&[1, 2, 3], &[1, 2], &(0..1), &p));
        assert!(!conserved(&[0, 1, 2], &[0, 1], &(1..2), &p));
        let right = [
            run(0, Some(0)),
            run(4, Some(1)),
            run(9, None),
            run(15, None),
        ];
        assert!(conserved(&[0, 1, 2, 3], &[0, 1], &(2..4), &right));
        assert!(!conserved(&[0, 1, 2], &[0, 1], &(2..3), &right));
    }

    #[test]
    fn only_literal_edge_deletions_are_permitted() {
        assert_eq!(removed_range("−1.0", "1.0", LeaderEdge::Left), Some(0..1));
        assert_eq!(
            removed_range("a 中 x", "a 中", LeaderEdge::Right),
            Some(2..3)
        );
        assert_eq!(removed_range("a1x", "ax", LeaderEdge::Left), None);
        assert_eq!(removed_range("abc", "", LeaderEdge::Right), None);
        assert_eq!(removed_range("A B", "AB", LeaderEdge::Left), None);
    }
}
