//! Paired CTC evidence for corrections confined to foreground runs affected by
//! independently proven drawing-stroke masks. A mask never creates ink.

use serde::Serialize;

use crate::{
    candidate_ink::{CandidateAnchor, CandidateMapping, SourceInk},
    character_alignment::RunAlignment,
    ctc_alignment::SequenceAlignment,
    ctc_evidence::{CtcOutcome, SequenceHypothesis},
    fragment_reconciliation::{Operation, Rejection, Transaction, consistent_votes},
    line_audit::AuditedLine,
    orthogonal_mask::MaskEvidence,
    pixel_regions::InkBounds,
    refinement_evidence::{Recognition, RefinementDecision, ViewEvidence, canonical},
};

/// Every masked run descends from exactly one original run. Removal can
/// split an original projection group but cannot merge or add any ink.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct RunPartition {
    pub bounds: InkBounds,
    pub foreground_pixels: u64,
    pub removed_pixels: u64,
    pub retained_runs: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct MaskVote {
    pub original_view: usize,
    pub masked_view: usize,
    /// Primary and revised class sequence indices, respectively.
    pub original_hypotheses: [usize; 2],
    pub masked_hypotheses: [usize; 2],
    /// Full conditional path log mass difference, revised minus primary.
    pub log_odds_before: f64,
    pub log_odds_after: f64,
    /// One feasible ordered binding preserving every unaffected run.
    pub primary_token_groups: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct MaskCorrection {
    pub mask_sha256: String,
    pub partition: Vec<RunPartition>,
    pub revised_token_runs: Vec<usize>,
    pub votes: Vec<MaskVote>,
}

struct Partition {
    groups: Vec<RunPartition>,
    parents: Vec<usize>,
}

const fn contains(a: InkBounds, b: InkBounds) -> bool {
    a.left <= b.left && b.right <= a.right && a.top <= b.top && b.bottom <= a.bottom
}

impl Partition {
    fn measure(original: &SourceInk, masked: &SourceInk, mask: &MaskEvidence) -> Option<Self> {
        if original.dimensions != masked.dimensions {
            return None;
        }
        let mut groups = original
            .groups
            .iter()
            .map(|g| RunPartition {
                bounds: g.run.ink,
                foreground_pixels: g.run.observation_support.inside_pixels,
                removed_pixels: g.run.observation_support.inside_pixels,
                retained_runs: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut parents = Vec::new();
        for (i, g) in masked.groups.iter().enumerate() {
            let bounds = g.run.ink;
            let parent = groups.partition_point(|p| p.bounds.right <= bounds.left);
            let p = groups.get_mut(parent)?;
            if !contains(p.bounds, bounds) {
                return None;
            }
            p.removed_pixels = p
                .removed_pixels
                .checked_sub(g.run.observation_support.inside_pixels)?;
            p.retained_runs.push(i);
            parents.push(parent);
        }
        for p in &groups {
            if p.removed_pixels == 0 {
                let &[run] = p.retained_runs.as_slice() else {
                    return None;
                };
                if masked.groups[run].run.ink != p.bounds {
                    return None;
                }
            }
        }
        let before = groups.iter().map(|p| p.foreground_pixels).sum::<u64>();
        let removed = groups.iter().map(|p| p.removed_pixels).sum::<u64>();
        if before != mask.foreground_before
            || removed != mask.removed_pixels
            || before.checked_sub(removed)? != mask.foreground_after
        {
            return None;
        }
        Some(Self { groups, parents })
    }

    fn labels(&self, text: &str, runs: &[usize]) -> Option<Vec<String>> {
        let mut labels = vec![String::new(); self.groups.len()];
        let mut characters = text.chars();
        for &run in runs {
            labels[*self.parents.get(run)?].push(characters.next()?);
        }
        characters.next().is_none().then_some(labels)
    }
}

fn supports(anchor: &CandidateAnchor, group: usize, affected: bool) -> bool {
    let a = &anchor.anchor;
    match a.alignment {
        RunAlignment::Unique { run } => group == run,
        RunAlignment::Ambiguous { first, last } => {
            (first..=last).contains(&group) && a.candidate_runs.contains(&group)
        }
        RunAlignment::NoInjectiveAssignment => affected && a.candidate_runs.contains(&group),
        _ => false,
    }
}

/// Ordered dynamic program. In unaffected groups the old transcription
/// must preserve exactly the revised label sequence. In contaminated groups
/// it may use any compatible consecutive sequence, including no token.
/// Return an explicit witness, not a guessed assignment for ambiguous CTC.
fn localized_assignment(
    anchors: &[&CandidateAnchor],
    groups: &[RunPartition],
    revised_labels: &[String],
) -> Option<Vec<usize>> {
    // Match the existing bounded conditional CTC alignment lattice budget.
    // No quadratic allocation for pathological very-wide diagnostic crops.
    let width = anchors.len().checked_add(1)?;
    if groups.len().checked_mul(width)? > crate::ctc_alignment::MAX_CELLS
        || groups.len() != revised_labels.len()
    {
        return None;
    }
    let mut previous = vec![None; width];
    previous[0] = Some(0);
    let mut rows = Vec::with_capacity(groups.len());
    for (index, (group, labels)) in groups.iter().zip(revised_labels).enumerate() {
        let mut next = vec![None; width];
        if group.removed_pixels > 0 {
            for end in 0..width {
                next[end] = if previous[end].is_some() {
                    Some(end)
                } else if end > 0 && supports(anchors[end - 1], index, true) {
                    next[end - 1]
                } else {
                    None
                };
            }
        } else {
            let count = labels.chars().count();
            for (start, reachable) in previous.iter().enumerate() {
                if reachable.is_none() || start + count >= width {
                    continue;
                }
                let end = start + count;
                if anchors[start..end]
                    .iter()
                    .zip(labels.chars())
                    .all(|(a, c)| a.anchor.character == c && supports(a, index, false))
                {
                    next[end] = Some(start);
                }
            }
        }
        previous.clone_from(&next);
        rows.push(next);
    }
    let mut end = anchors.len();
    let mut assignment = vec![0; end];
    for (i, row) in rows.iter().enumerate().rev() {
        let start = row[end]?;
        assignment[start..end].fill(i);
        end = start;
    }
    (end == 0).then_some(assignment)
}

struct Sequences<'a> {
    hypotheses: &'a [SequenceHypothesis],
    mappings: &'a [CandidateMapping],
}

impl<'a> Sequences<'a> {
    fn from_view(view: &'a ViewEvidence) -> Option<Self> {
        let CtcOutcome::Measured { evidence } = &view.ctc else {
            return None;
        };
        let mappings = &view.ink_mapping.as_ref()?.hypotheses;
        (mappings.len() == evidence.hypotheses.len()).then_some(Self {
            hypotheses: &evidence.hypotheses,
            mappings,
        })
    }

    fn index_of(&self, classes: &[usize]) -> Option<usize> {
        self.hypotheses.iter().position(|h| h.classes == classes)
    }

    fn strongest(&self, text: &str) -> Option<usize> {
        self.hypotheses
            .iter()
            .enumerate()
            .filter_map(|(i, h)| {
                (canonical(&h.text) == text)
                    .then(|| log_mass(h).map(|mass| (i, mass)))
                    .flatten()
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }
}

fn log_mass(h: &SequenceHypothesis) -> Option<f64> {
    let SequenceAlignment::Measured { log_mass, .. } = h.alignment else {
        return None;
    };
    log_mass.is_finite().then_some(log_mass)
}

struct Comparison<'a> {
    original: &'a Recognition,
    masked: &'a Recognition,
    primary: &'a str,
    revised: &'a str,
    partition: &'a Partition,
}

impl Comparison<'_> {
    fn vote(&self, index: usize) -> Option<(Vec<usize>, MaskVote)> {
        let view = self.masked.views.get(index)?;
        let mut originals =
            self.original.views.iter().enumerate().filter(|(_, v)| {
                v.model == view.model && v.stretch_percent == view.stretch_percent
            });
        let (original_index, old_view) = originals.next()?;
        if originals.next().is_some() || old_view.dimensions != view.dimensions {
            return None;
        }
        let before = Sequences::from_view(old_view)?;
        let after = Sequences::from_view(view)?;
        let winner = after.hypotheses.first()?;
        if canonical(&winner.text) != self.revised {
            return None;
        }
        let original_revised = before.index_of(&winner.classes)?;
        let original_primary = before.strongest(self.primary)?;
        let masked_primary = after.index_of(&before.hypotheses[original_primary].classes)?;
        let before_odds = log_mass(&before.hypotheses[original_revised])?
            - log_mass(&before.hypotheses[original_primary])?;
        let after_odds = log_mass(winner)? - log_mass(&after.hypotheses[masked_primary])?;
        // Compare the very same class sequences, not separately pruned beam
        // mass, nor probability across unrelated model configurations.
        if after_odds <= 0.0 || after_odds <= before_odds {
            return None;
        }
        let runs = after.mappings.first()?.unique_runs(self.revised)?;
        let labels = self.partition.labels(self.revised, &runs)?;
        let original_anchors = before.mappings[original_revised].visible_anchors(self.revised)?;
        if original_anchors.len() != runs.len()
            || !original_anchors.iter().zip(&runs).all(|(a, &r)| {
                let group = self.partition.parents[r];
                supports(a, group, self.partition.groups[group].removed_pixels > 0)
            })
        {
            return None;
        }
        let old_anchors = before.mappings[original_primary].visible_anchors(self.primary)?;
        let primary_token_groups =
            localized_assignment(&old_anchors, &self.partition.groups, &labels)?;
        Some((
            runs,
            MaskVote {
                original_view: original_index,
                masked_view: index,
                original_hypotheses: [original_primary, original_revised],
                masked_hypotheses: [masked_primary, 0],
                log_odds_before: before_odds,
                log_odds_after: after_odds,
                primary_token_groups,
            },
        ))
    }
}

/// Require paired CTC odds improvement and a source-localized edit in both
/// recognizer models. All unaffected projection groups preserve their labels.
///
/// # Errors
/// Returns the measured reason that a masked correction cannot be supported.
pub fn propose(
    line: &AuditedLine,
    original: &Recognition,
    masked: &Recognition,
    mask: &MaskEvidence,
) -> Result<Transaction, Rejection> {
    let RefinementDecision::Consensus { text, view_indices } = &masked.decision else {
        return Err(Rejection::NoConsensus);
    };
    let primary = canonical(&line.original_text);
    let revised = canonical(text);
    if primary == revised {
        return Err(Rejection::Unchanged);
    }
    let partition = Partition::measure(&original.source_ink, &masked.source_ink, mask)
        .ok_or(Rejection::ForegroundPartitionMismatch)?;
    let comparison = Comparison {
        original,
        masked,
        primary: &primary,
        revised: &revised,
        partition: &partition,
    };
    let (assigned, votes) = consistent_votes(masked, view_indices, |i, _| comparison.vote(i))
        .ok_or(Rejection::UnlocalizedMaskChange)?;
    Ok(Transaction {
        page: line.page,
        line_index: line.line_index,
        quad: line.original_quad,
        text: text.clone(),
        operation: Operation::MaskCorrection(MaskCorrection {
            mask_sha256: mask.mask_sha256.clone(),
            partition: partition.groups,
            revised_token_runs: assigned,
            votes,
        }),
        consumed: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::character_alignment::CharacterAnchor;

    fn anchor(character: char, candidates: Vec<usize>, alignment: RunAlignment) -> CandidateAnchor {
        CandidateAnchor {
            anchor: CharacterAnchor {
                character,
                column: 0,
                crop_x: 0.0,
                candidate_runs: candidates,
                alignment,
            },
            entry_interval: [0.0, 0.0],
        }
    }

    fn group(removed_pixels: u64) -> RunPartition {
        RunPartition {
            bounds: InkBounds {
                left: 0,
                right: 5,
                top: 0,
                bottom: 10,
            },
            foreground_pixels: 20,
            removed_pixels,
            retained_runs: Vec::new(),
        }
    }

    #[test]
    fn affected_groups_allow_edits_but_unchanged_ink_keeps_literal_identity() {
        let a = anchor(
            'A',
            vec![0, 1],
            RunAlignment::Ambiguous { first: 0, last: 1 },
        );
        let b = anchor('B', vec![1], RunAlignment::Unique { run: 1 });
        let c = anchor('C', vec![2], RunAlignment::Unique { run: 2 });
        let groups = [group(0), group(5), group(0)];
        let primary = [&a, &b, &c];
        let text = |labels: [&str; 3]| labels.map(str::to_owned);
        assert_eq!(
            localized_assignment(&primary, &groups, &text(["A", "xy", "C"])),
            Some(vec![0, 1, 2])
        );
        assert_eq!(
            localized_assignment(&primary, &groups, &text(["A", "", "C"])),
            Some(vec![0, 1, 2])
        );
        assert_eq!(
            localized_assignment(&primary, &groups, &text(["Z", "xy", "C"])),
            None
        );
        assert_eq!(
            localized_assignment(&primary, &groups, &text(["A", "xy", "Z"])),
            None
        );
        assert_eq!(
            localized_assignment(&[&a, &c], &groups, &text(["A", "xy", "C"])),
            Some(vec![0, 2])
        );
        assert_eq!(
            localized_assignment(
                &primary,
                &[group(0), group(0), group(0)],
                &text(["A", "xy", "C"])
            ),
            None
        );
    }

    #[test]
    fn merged_contaminated_group_supports_multiple_tokens_without_reordering() {
        let a = anchor('A', vec![0], RunAlignment::NoInjectiveAssignment);
        let b = anchor('B', vec![0], RunAlignment::NoInjectiveAssignment);
        let c = anchor('C', vec![1], RunAlignment::Unique { run: 1 });
        let groups = [group(5), group(0)];
        let labels = ["XY".to_owned(), "C".to_owned()];
        assert_eq!(
            localized_assignment(&[&a, &b, &c], &groups, &labels),
            Some(vec![0, 0, 1])
        );
        assert_eq!(localized_assignment(&[&a, &c, &b], &groups, &labels), None);
        assert_eq!(
            localized_assignment(&[&a, &b, &c], &[group(0), group(0)], &labels),
            None
        );
    }
}
