//! Source-localized insertion of omitted tokens without changing the
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
};

/// Positions of omitted tokens over every feasible complete embedding.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct InsertionPositions {
    /// Union: every slot which can be an insertion in some embedding.
    pub possible: Vec<usize>,
    /// Intersection: slots which must be an insertion in every embedding.
    pub required: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct InsertionVote {
    pub view: usize,
    pub primary: SequenceReference,
    pub log_odds: f64,
    pub positions: InsertionPositions,
}

#[derive(Debug, Serialize)]
pub struct InsertionRecognition {
    pub source_rgb_sha256: String,
    /// One uniquely assigned run per revised non-whitespace character.
    pub revised_runs: Vec<usize>,
    pub positions: InsertionPositions,
    pub votes: Vec<InsertionVote>,
}

#[derive(Debug, Serialize)]
pub struct SourceInsertion {
    #[serde(flatten)]
    pub recognition: InsertionRecognition,
    pub peers: Vec<BoundaryPeer>,
    pub run_accounting: RunAccounting,
}

fn retained_match(old: &[&CandidateAnchor], j: usize, c: char, run: usize) -> bool {
    old.get(j)
        .is_some_and(|a| a.anchor.character == c && a.supports_run(run))
}

/// An ordered path consumes a revised token by retaining the next old
/// token (compatible literal and ink run) or inserting at a clean run.
///
/// Forward and reverse reachability characterize *all* complete paths,
/// without enumerating exponentially many equivalent embeddings.
fn embedding(
    old: &[&CandidateAnchor],
    new: &[char],
    runs: &[usize],
    insertable: &[bool],
) -> Option<InsertionPositions> {
    if new.len() != runs.len() || new.len() != insertable.len() || old.len() >= new.len() {
        return None;
    }
    let width = old.len().checked_add(1)?;
    let height = new.len().checked_add(1)?;
    let cells = height.checked_mul(width)?;
    if cells > crate::ctc_alignment::MAX_CELLS {
        return None;
    }
    let mut prefix = vec![false; cells];
    prefix[0] = true;
    for (i, ((&c, &run), &insert)) in new.iter().zip(runs).zip(insertable).enumerate() {
        for j in 0..width {
            if !prefix[i * width + j] {
                continue;
            }
            let row = (i + 1) * width;
            prefix[row + j] |= insert;
            if retained_match(old, j, c, run) {
                prefix[row + j + 1] = true;
            }
        }
    }
    if !prefix[cells - 1] {
        return None;
    }
    let mut suffix = vec![false; width];
    suffix[old.len()] = true;
    let mut positions = InsertionPositions {
        possible: Vec::new(),
        required: Vec::new(),
    };
    for i in (0..new.len()).rev() {
        // Read the next row in place: ascending j leaves suffix[j+1]
        // unchanged until after the current retain transition is tested.
        let mut can_insert = false;
        let mut can_retain = false;
        for j in 0..width {
            let insert = insertable[i] && suffix[j];
            let retain = retained_match(old, j, new[i], runs[i]) && suffix[j + 1];
            if prefix[i * width + j] {
                can_insert |= insert;
                can_retain |= retain;
            }
            suffix[j] = insert || retain;
        }
        if can_insert {
            positions.possible.push(i);
            if !can_retain {
                positions.required.push(i);
            }
        }
    }
    positions.possible.reverse();
    positions.required.reverse();
    Some(positions)
}

fn combine_positions(votes: &[InsertionVote]) -> Option<InsertionPositions> {
    let mut possible = BTreeSet::new();
    let mut required: BTreeSet<_> = votes.first()?.positions.required.iter().copied().collect();
    for vote in votes {
        possible.extend(vote.positions.possible.iter().copied());
        required.retain(|i| vote.positions.required.binary_search(i).is_ok());
    }
    Some(InsertionPositions {
        possible: possible.into_iter().collect(),
        required: required.into_iter().collect(),
    })
}

fn vote(
    view_index: usize,
    view: &ViewEvidence,
    old: &str,
    revised: &str,
    ink: &SourceInk,
) -> Option<(Vec<usize>, InsertionVote)> {
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
    let insertable = runs
        .iter()
        .map(|&r| {
            ink.groups
                .get(r)
                .map(|g| !g.touches_line_edge && g.run.observation_support.inside_pixels > 0)
        })
        .collect::<Option<Vec<_>>>()?;
    let positions = embedding(
        &old,
        &revised.chars().collect::<Vec<_>>(),
        &runs,
        &insertable,
    )?;
    Some((
        runs,
        InsertionVote {
            view: view_index,
            primary,
            log_odds: odds,
            positions,
        },
    ))
}

pub fn propose(
    line: &AuditedLine,
    recognition: &Recognition,
) -> Result<InsertionRecognition, Rejection> {
    let RefinementDecision::Consensus { text, view_indices } = &recognition.decision else {
        return Err(Rejection::NoConsensus);
    };
    let old = canonical(&line.original_text);
    let revised = canonical(text);
    if old == revised {
        return Err(Rejection::Unchanged);
    }
    if old.chars().count() >= revised.chars().count() {
        return Err(Rejection::NotLiteralInsertion);
    }
    let ink = &recognition.source_ink;
    let (revised_runs, votes) = consistent_votes(recognition, view_indices, |i, v| {
        vote(i, v, &old, &revised, ink)
    })
    .ok_or(Rejection::UnlocalizedInsertedTokens)?;
    // Text and geometry are identical across all feasible explanations;
    // keep the whole uncertainty set, not a fabricated unique old identity.
    let positions = combine_positions(&votes).ok_or(Rejection::UnlocalizedInsertedTokens)?;
    Ok(InsertionRecognition {
        source_rgb_sha256: ink.rgb_sha256.clone(),
        revised_runs,
        positions,
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

    fn positions(possible: &[usize], required: &[usize]) -> InsertionPositions {
        InsertionPositions {
            possible: possible.to_vec(),
            required: required.to_vec(),
        }
    }

    #[test]
    fn embedding_preserves_original_ink_and_rejects_clipped_additions() {
        let a = anchor('A', vec![0, 1]);
        let b = anchor('B', vec![1, 2]);
        assert_eq!(
            embedding(
                &[&a, &b],
                &['A', '字', 'B'],
                &[0, 1, 2],
                &[true, true, true]
            ),
            Some(positions(&[1], &[1]))
        );
        assert_eq!(
            embedding(
                &[&a, &b],
                &['A', '字', 'B'],
                &[0, 1, 2],
                &[true, false, true]
            ),
            None
        );
        assert_eq!(
            embedding(
                &[&a, &b],
                &['A', '字', 'C'],
                &[0, 1, 2],
                &[true, true, true]
            ),
            None
        );
        assert_eq!(
            embedding(
                &[&b, &a],
                &['A', '字', 'B'],
                &[0, 1, 2],
                &[true, true, true]
            ),
            None
        );
    }

    #[test]
    fn repeated_literals_preserve_the_complete_uncertainty_set() {
        let a = anchor('A', vec![0, 1]);
        assert_eq!(
            embedding(&[&a], &['A', 'A'], &[0, 1], &[true, true]),
            Some(positions(&[0, 1], &[]))
        );
        assert_eq!(
            embedding(&[&a], &['A', 'A'], &[0, 1], &[false, true]),
            Some(positions(&[1], &[1]))
        );
        assert_eq!(
            embedding(&[&a], &['A', 'A', 'B'], &[0, 1, 2], &[true, true, true]),
            Some(positions(&[0, 1, 2], &[2]))
        );
        let a = anchor('A', vec![0]);
        assert_eq!(
            embedding(&[&a], &['A', 'A'], &[0, 1], &[true, true]),
            Some(positions(&[1], &[1]))
        );
    }

    #[test]
    fn reachability_matches_exhaustive_subsets() {
        // All length-four binary strings, length-two primary strings,
        // support masks for each primary token and clean-run masks.
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
                        let allowed = runs.map(|r| clean & (1 << r) != 0);
                        let mut possible = 0;
                        let mut required = 15;
                        let mut found = false;
                        for skip in 0_u8..16 {
                            if skip.count_ones() != 2 || skip & !clean != 0 {
                                continue;
                            }
                            if runs
                                .iter()
                                .copied()
                                .filter(|&r| skip & (1 << r) == 0)
                                .enumerate()
                                .all(|(j, r)| retained_match(&old, j, new[r], r))
                            {
                                possible |= skip;
                                required &= skip;
                                found = true;
                            }
                        }
                        let expected = found.then(|| InsertionPositions {
                            possible: runs
                                .iter()
                                .copied()
                                .filter(|&r| possible & (1 << r) != 0)
                                .collect(),
                            required: runs
                                .iter()
                                .copied()
                                .filter(|&r| required & (1 << r) != 0)
                                .collect(),
                        });
                        assert_eq!(embedding(&old, &new, &runs, &allowed), expected);
                    }
                }
            }
        }
    }
}
