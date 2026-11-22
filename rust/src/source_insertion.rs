//! Source-localized insertion of omitted tokens without changing the
//! observation geometry or pixels.

use serde::Serialize;

use crate::{
    candidate_ink::{CandidateAnchor, SourceInk},
    ctc_evidence::{CtcOutcome, SequenceReference},
    fragment_ink::RunAccounting,
    fragment_reconciliation::{Rejection, consistent_votes},
    line_audit::{AuditedLine, BoundaryPeer},
    refinement_evidence::{Recognition, RefinementDecision, ViewEvidence, canonical},
};

#[derive(Debug, Serialize)]
pub struct InsertionVote {
    pub view: usize,
    pub primary: SequenceReference,
    pub log_odds: f64,
    /// Each original character's position in the revised transcription.
    pub retained_positions: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct InsertionRecognition {
    pub source_rgb_sha256: String,
    /// One uniquely assigned run per revised non-whitespace character.
    pub revised_runs: Vec<usize>,
    pub inserted_positions: Vec<usize>,
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

/// Count feasible embeddings of the primary sequence in the revised one,
/// capping counts at two. Insertion is allowed only at clean, independent
/// source runs. Every retained character preserves its measured run support.
fn embedding(
    old: &[&CandidateAnchor],
    new: &[char],
    runs: &[usize],
    insertable: &[bool],
) -> Option<Vec<usize>> {
    if new.len() != runs.len() || new.len() != insertable.len() || old.len() >= new.len() {
        return None;
    }
    let width = old.len().checked_add(1)?;
    let height = new.len().checked_add(1)?;
    if height.checked_mul(width)? > crate::ctc_alignment::MAX_CELLS {
        return None;
    }
    let mut counts = vec![0_u8; height * width];
    counts[0] = 1;
    for (i, ((&c, &run), &insert)) in new.iter().zip(runs).zip(insertable).enumerate() {
        for j in 0..width {
            let n = counts[i * width + j];
            if n == 0 {
                continue;
            }
            let row = (i + 1) * width;
            if insert {
                counts[row + j] = (counts[row + j] + n).min(2);
            }
            if retained_match(old, j, c, run) {
                counts[row + j + 1] = (counts[row + j + 1] + n).min(2);
            }
        }
    }
    let mut j = old.len();
    if counts[new.len() * width + j] != 1 {
        return None;
    }
    let mut retained = Vec::with_capacity(j);
    for i in (0..new.len()).rev() {
        // A unique complete path has precisely one reachable predecessor.
        if insertable[i] && counts[i * width + j] > 0 {
            continue;
        }
        j = j.checked_sub(1)?;
        if !retained_match(old, j, new[i], runs[i]) {
            return None;
        }
        retained.push(i);
    }
    retained.reverse();
    (j == 0).then_some(retained)
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
    let retained_positions = embedding(
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
            retained_positions,
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
    let kept = &votes
        .first()
        .ok_or(Rejection::UnlocalizedInsertedTokens)?
        .retained_positions;
    if votes.iter().any(|v| &v.retained_positions != kept) {
        return Err(Rejection::UnlocalizedInsertedTokens);
    }
    let inserted_positions = (0..revised_runs.len())
        .filter(|i| kept.binary_search(i).is_err())
        .collect();
    Ok(InsertionRecognition {
        source_rgb_sha256: ink.rgb_sha256.clone(),
        revised_runs,
        inserted_positions,
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

    #[test]
    fn unique_embedding_preserves_original_ink_and_rejects_clipped_additions() {
        let a = anchor('A', vec![0, 1]);
        let b = anchor('B', vec![1, 2]);
        assert_eq!(
            embedding(
                &[&a, &b],
                &['A', '字', 'B'],
                &[0, 1, 2],
                &[true, true, true]
            ),
            Some(vec![0, 2])
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
    fn repeated_literal_requires_one_geometrically_supported_embedding() {
        let a = anchor('A', vec![0, 1]);
        assert_eq!(embedding(&[&a], &['A', 'A'], &[0, 1], &[true, true]), None);
        assert_eq!(
            embedding(&[&a], &['A', 'A'], &[0, 1], &[false, true]),
            Some(vec![0])
        );
        let a = anchor('A', vec![0]);
        assert_eq!(
            embedding(&[&a], &['A', 'A'], &[0, 1], &[true, true]),
            Some(vec![0])
        );
    }
}
