//! Source-localized nondeleting text revisions without changing the
//! observation geometry or pixels.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::{
    candidate_ink::{CandidateAnchor, SourceInk},
    ctc_evidence::{CtcOutcome, SequenceReference},
    fragment_ink::RunAccounting,
    fragment_reconciliation::{Rejection, consistent_votes},
    line_audit::{AuditedLine, BoundaryPeer},
    refinement_evidence::{Recognition, RefinementDecision, ViewEvidence, canonical},
    source_completeness::CompleteRun,
};

/// Token identities over every minimum-substitution complete embedding.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct ChangePositions {
    /// Union: every slot which can have this change in some embedding.
    pub possible: Vec<usize>,
    /// Intersection: slots which must have this change in every embedding.
    pub required: Vec<usize>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct SourceEdits {
    pub inserted: ChangePositions,
    pub substituted: ChangePositions,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Embedding {
    pub substitutions: usize,
    #[serde(flatten)]
    pub edits: SourceEdits,
}

#[derive(Debug, Serialize)]
pub struct SourceVote {
    pub view: usize,
    pub primary: SequenceReference,
    pub log_odds: f64,
    pub embedding: Embedding,
}

#[derive(Debug, Serialize)]
pub struct SourceRecognition {
    pub source_rgb_sha256: String,
    /// One uniquely assigned run per revised non-whitespace character.
    pub revised_runs: Vec<usize>,
    pub edits: SourceEdits,
    pub votes: Vec<SourceVote>,
}

#[derive(Debug, Serialize)]
pub struct SourceCorrection {
    #[serde(flatten)]
    pub recognition: SourceRecognition,
    pub peers: Vec<BoundaryPeer>,
    pub run_accounting: RunAccounting,
    pub complete_runs: Vec<CompleteRun>,
}

#[derive(Clone, Copy)]
struct Slot {
    character: char,
    run: usize,
    editable: bool,
}

impl Slot {
    /// Literal retention has zero cost; a substitution needs independently
    /// editable ink. Every consumed old token must support this exact run.
    fn match_cost(self, old: &[&CandidateAnchor], j: usize) -> Option<usize> {
        let anchor = old.get(j)?;
        if !anchor.supports_run(self.run) {
            return None;
        }
        let changed = anchor.anchor.character != self.character;
        (!changed || self.editable).then_some(usize::from(changed))
    }
}

const UNREACHABLE: usize = usize::MAX;

struct Prefix {
    width: usize,
    costs: Vec<usize>,
    optimum: usize,
}

impl Prefix {
    fn measure(old: &[&CandidateAnchor], slots: &[Slot]) -> Option<Self> {
        if old.len() > slots.len() {
            return None;
        }
        let width = old.len().checked_add(1)?;
        let cells = slots.len().checked_add(1)?.checked_mul(width)?;
        if cells > crate::ctc_alignment::MAX_CELLS {
            return None;
        }
        let mut costs = vec![UNREACHABLE; cells];
        costs[0] = 0;
        for (i, &slot) in slots.iter().enumerate() {
            for j in 0..width {
                let cost = costs[i * width + j];
                if cost == UNREACHABLE {
                    continue;
                }
                let row = (i + 1) * width;
                if slot.editable {
                    costs[row + j] = costs[row + j].min(cost);
                }
                if let Some(change) = slot.match_cost(old, j) {
                    costs[row + j + 1] = costs[row + j + 1].min(cost + change);
                }
            }
        }
        let optimum = costs[cells - 1];
        (optimum != UNREACHABLE).then_some(Self {
            width,
            costs,
            optimum,
        })
    }

    fn optimal_edge(&self, i: usize, j: usize, cost: usize, remaining: usize) -> bool {
        self.costs[i * self.width + j]
            .saturating_add(cost)
            .saturating_add(remaining)
            == self.optimum
    }
}

#[derive(Default)]
struct SlotPossibilities {
    inserted: bool,
    retained: bool,
    substituted: bool,
}

impl SlotPossibilities {
    fn record(self, position: usize, edits: &mut SourceEdits) {
        if self.inserted {
            edits.inserted.possible.push(position);
            if !self.retained && !self.substituted {
                edits.inserted.required.push(position);
            }
        }
        if self.substituted {
            edits.substituted.possible.push(position);
            if !self.retained && !self.inserted {
                edits.substituted.required.push(position);
            }
        }
    }
}

impl ChangePositions {
    const fn empty() -> Self {
        Self {
            possible: Vec::new(),
            required: Vec::new(),
        }
    }

    fn reverse(&mut self) {
        self.possible.reverse();
        self.required.reverse();
    }

    fn combine<'a>(mut views: impl Iterator<Item = &'a Self>) -> Option<Self> {
        let first = views.next()?;
        let mut possible = BTreeSet::from_iter(first.possible.iter().copied());
        let mut required = BTreeSet::from_iter(first.required.iter().copied());
        for view in views {
            possible.extend(view.possible.iter().copied());
            required.retain(|i| view.required.binary_search(i).is_ok());
        }
        Some(Self {
            possible: possible.into_iter().collect(),
            required: required.into_iter().collect(),
        })
    }
}

impl SourceEdits {
    const fn empty() -> Self {
        Self {
            inserted: ChangePositions::empty(),
            substituted: ChangePositions::empty(),
        }
    }

    pub fn affected(&self) -> impl Iterator<Item = usize> + '_ {
        self.inserted
            .possible
            .iter()
            .chain(&self.substituted.possible)
            .copied()
    }

    fn combine(votes: &[SourceVote]) -> Option<Self> {
        Some(Self {
            inserted: ChangePositions::combine(votes.iter().map(|v| &v.embedding.edits.inserted))?,
            substituted: ChangePositions::combine(
                votes.iter().map(|v| &v.embedding.edits.substituted),
            )?,
        })
    }
}

/// Every path consumes all old tokens in reading order and all uniquely
/// localized new runs. The number of insertions is fixed by the length
/// difference. Among geometrically compatible paths, minimize substitutions
/// so literal preservation needs no gratuitous relabeling. Compute the
/// complete set of minimum-cost edges without enumerating embeddings.
fn embedding(old: &[&CandidateAnchor], slots: &[Slot]) -> Option<Embedding> {
    let prefix = Prefix::measure(old, slots)?;
    let mut suffix = vec![UNREACHABLE; prefix.width];
    suffix[old.len()] = 0;
    let mut edits = SourceEdits::empty();
    for (i, &slot) in slots.iter().enumerate().rev() {
        let mut possible = SlotPossibilities::default();
        // Ascending j preserves the unread suffix[j+1] from the next row.
        for j in 0..prefix.width {
            let inserted = if slot.editable {
                suffix[j]
            } else {
                UNREACHABLE
            };
            possible.inserted |= prefix.optimal_edge(i, j, 0, inserted);
            let consumed = slot.match_cost(old, j).map_or(UNREACHABLE, |cost| {
                let tail = suffix[j + 1];
                if prefix.optimal_edge(i, j, cost, tail) {
                    possible.retained |= cost == 0;
                    possible.substituted |= cost != 0;
                }
                tail.saturating_add(cost)
            });
            suffix[j] = inserted.min(consumed);
        }
        possible.record(i, &mut edits);
    }
    edits.inserted.reverse();
    edits.substituted.reverse();
    Some(Embedding {
        substitutions: prefix.optimum,
        edits,
    })
}

fn vote(
    view_index: usize,
    view: &ViewEvidence,
    old: &str,
    revised: &str,
    ink: &SourceInk,
    complete: &BTreeSet<usize>,
) -> Option<(Vec<usize>, SourceVote)> {
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
    let runs = mapping.hypotheses.first()?.unique_runs(revised)?;
    let old = mapping.sequence(primary)?.visible_anchors(old)?;
    let slots = revised
        .chars()
        .zip(&runs)
        .map(|(character, &run)| {
            let group = ink.groups.get(run)?;
            Some(Slot {
                character,
                run,
                editable: !group.touches_line_edge
                    && group.run.observation_support.inside_pixels > 0,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let embedding = embedding(&old, &slots)?;
    Some((
        runs,
        SourceVote {
            view: view_index,
            primary,
            log_odds: odds,
            embedding,
        },
    ))
}

pub fn propose(
    line: &AuditedLine,
    recognition: &Recognition,
    complete_runs: &[CompleteRun],
) -> Result<SourceRecognition, Rejection> {
    let RefinementDecision::Consensus { text, view_indices } = &recognition.decision else {
        return Err(Rejection::NoConsensus);
    };
    let old = canonical(&line.original_text);
    let revised = canonical(text);
    if old == revised {
        return Err(Rejection::Unchanged);
    }
    if old.chars().count() > revised.chars().count() {
        return Err(Rejection::WouldDeleteSourceToken);
    }
    let ink = &recognition.source_ink;
    let complete: BTreeSet<_> = complete_runs.iter().map(|p| p.run).collect();
    let (revised_runs, votes) = consistent_votes(recognition, view_indices, |i, v| {
        vote(i, v, &old, &revised, ink, &complete)
    })
    .ok_or(Rejection::UnlocalizedSourceEdit)?;
    // Text and geometry are identical across all feasible explanations;
    // keep the whole uncertainty set, not a fabricated unique old identity.
    let edits = SourceEdits::combine(&votes).ok_or(Rejection::UnlocalizedSourceEdit)?;
    Ok(SourceRecognition {
        source_rgb_sha256: ink.rgb_sha256.clone(),
        revised_runs,
        edits,
        votes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::character_alignment::{CharacterAnchor, RunAlignment};

    fn anchor(c: char, runs: Vec<usize>) -> CandidateAnchor {
        CandidateAnchor {
            anchor: CharacterAnchor {
                character: c,
                column: 0,
                crop_x: 0.0,
                alignment: RunAlignment::Ambiguous {
                    first: runs.first().copied().unwrap_or_default(),
                    last: runs.last().copied().unwrap_or_default(),
                },
                candidate_runs: runs,
            },
            entry_interval: [0.0, 0.0],
        }
    }

    fn slots(new: &[char], clean: &[bool]) -> Vec<Slot> {
        new.iter()
            .zip(clean)
            .enumerate()
            .map(|(run, (&character, &editable))| Slot {
                character,
                run,
                editable,
            })
            .collect()
    }

    fn positions(possible: &[usize], required: &[usize]) -> ChangePositions {
        ChangePositions {
            possible: possible.to_vec(),
            required: required.to_vec(),
        }
    }

    #[test]
    fn minimum_substitutions_preserve_literals_and_all_repeated_identities() {
        let a = anchor('A', vec![0, 1]);
        let b = anchor('B', vec![1, 2]);
        assert_eq!(
            embedding(&[&a, &b], &slots(&['A', '字', 'B'], &[true, true, true])),
            Some(Embedding {
                substitutions: 0,
                edits: SourceEdits {
                    inserted: positions(&[1], &[1]),
                    substituted: ChangePositions::empty(),
                },
            })
        );
        assert_eq!(
            embedding(&[&a, &b], &slots(&['A', '字', 'B'], &[true, false, true])),
            None
        );
        assert_eq!(
            embedding(&[&a], &slots(&['A', 'A'], &[true, true])),
            Some(Embedding {
                substitutions: 0,
                edits: SourceEdits {
                    inserted: positions(&[0, 1], &[]),
                    substituted: ChangePositions::empty(),
                },
            })
        );
    }

    #[test]
    fn mixed_and_same_length_changes_require_compatible_editable_ink() {
        let a = anchor('A', vec![0]);
        let b = anchor('B', vec![2]);
        let mixed = slots(&['A', '字', 'C'], &[true, true, true]);
        assert_eq!(
            embedding(&[&a, &b], &mixed),
            Some(Embedding {
                substitutions: 1,
                edits: SourceEdits {
                    inserted: positions(&[1], &[1]),
                    substituted: positions(&[2], &[2]),
                },
            })
        );
        assert_eq!(
            embedding(&[&a, &b], &slots(&['A', '字', 'C'], &[true, true, false])),
            None
        );
        assert_eq!(embedding(&[&b, &a], &mixed), None);
        assert_eq!(
            embedding(&[&a], &slots(&['B'], &[true])),
            Some(Embedding {
                substitutions: 1,
                edits: SourceEdits {
                    inserted: ChangePositions::empty(),
                    substituted: positions(&[0], &[0]),
                },
            })
        );
        assert_eq!(embedding(&[&a], &slots(&['B'], &[false])), None);
        assert_eq!(embedding(&[&a, &b], &slots(&['A'], &[true])), None);
    }

    /// Independent brute-force oracle enumerates omitted position sets,
    /// rather than computing the same forward/reverse path recurrence.
    fn enumerate(old: &[&CandidateAnchor], slots: &[Slot]) -> Option<Embedding> {
        let mut best = UNREACHABLE;
        let mut paths = Vec::new();
        let size = slots.len();
        for skip in 0_usize..1 << size {
            if usize::try_from(skip.count_ones()).ok()? + old.len() != size {
                continue;
            }
            let mut j = 0;
            let mut changes = 0;
            let mut valid = true;
            for (i, slot) in slots.iter().enumerate() {
                let bit = 1 << i;
                if skip & bit != 0 {
                    valid &= slot.editable;
                    continue;
                }
                let previous = old[j];
                j += 1;
                valid &= previous.supports_run(slot.run);
                if previous.anchor.character != slot.character {
                    changes |= bit;
                    valid &= slot.editable;
                }
            }
            let count = usize::try_from(changes.count_ones()).ok()?;
            if valid && count <= best {
                if count < best {
                    paths.clear();
                    best = count;
                }
                paths.push((skip, changes));
            }
        }
        let summarize = |index: usize| {
            let bits = paths.iter().map(|p| <[usize; 2]>::from(*p)[index]);
            let (possible, required) = bits.fold((0, usize::MAX), |(p, r), b| (p | b, r & b));
            ChangePositions {
                possible: (0..size).filter(|i| possible & (1 << i) != 0).collect(),
                required: (0..size).filter(|i| required & (1 << i) != 0).collect(),
            }
        };
        (!paths.is_empty()).then(|| Embedding {
            substitutions: best,
            edits: SourceEdits {
                inserted: summarize(0),
                substituted: summarize(1),
            },
        })
    }

    #[test]
    fn minimum_cost_edges_match_exhaustive_subsets() {
        let runs = [0, 1, 2, 3];
        for word in 0_u8..16 {
            let new = runs.map(|i| if word & (1 << i) == 0 { 'A' } else { 'B' });
            for old_word in 0_u8..4 {
                for support in 0_u16..256 {
                    let a = anchor(
                        if old_word & 1 == 0 { 'A' } else { 'B' },
                        runs.iter()
                            .copied()
                            .filter(|&r| support & (1 << r) != 0)
                            .collect(),
                    );
                    let b = anchor(
                        if old_word & 2 == 0 { 'A' } else { 'B' },
                        runs.iter()
                            .copied()
                            .filter(|&r| support & (1 << (r + 4)) != 0)
                            .collect(),
                    );
                    let old = [&a, &b];
                    for clean in 0_u8..16 {
                        let current = slots(&new, &runs.map(|r| clean & (1 << r) != 0));
                        assert_eq!(embedding(&old, &current), enumerate(&old, &current));
                    }
                }
            }
        }
    }
}
