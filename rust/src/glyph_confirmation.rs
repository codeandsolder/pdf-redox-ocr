//! Alternative-model admission on an unchanged, independently segmented crop.
//!
//! Geometry establishes which source ink is shared between the predictions.
//! It does not itself determine Unicode. Never replace an already eligible
//! primary observation, or introduce unanchored edits.

use crate::{
    character_alignment::{CharacterAnchor, RunAlignment},
    disjoint_set::DisjointSet,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfirmationDecision {
    Agreement,
    InkSupportedSubstitution,
    ScoreTooLow,
    AlreadyEligible,
    Unanchored,
    ChangedSegmentation,
    NoSharedText,
    UnstableAlignment,
}

impl ConfirmationDecision {
    #[must_use = "check the glyph confirmation decision before applying replacements"]
    pub const fn accepted(self) -> bool {
        matches!(self, Self::Agreement | Self::InkSupportedSubstitution)
    }
}

fn supported_runs(anchor: &CharacterAnchor) -> Vec<usize> {
    match anchor.alignment {
        RunAlignment::Unique { run } => vec![run],
        RunAlignment::Ambiguous { .. }
        | RunAlignment::NoInjectiveAssignment
        | RunAlignment::UncertainNeighbor => anchor.candidate_runs.clone(),
        RunAlignment::Whitespace | RunAlignment::Padding | RunAlignment::NoInk => vec![],
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConfirmationPolicy {
    pub protected_score: f64,
    pub minimum_score: f64,
}

impl ConfirmationPolicy {
    #[must_use = "retain the overlay confirmation policy when evaluating glyphs"]
    pub const fn for_overlay(threshold: f64) -> Self {
        Self {
            protected_score: threshold,
            minimum_score: threshold.max(0.97),
        }
    }
}

#[derive(Default)]
struct Cluster {
    original: String,
    alternative: String,
    original_tokens: Vec<Vec<usize>>,
    alternative_tokens: Vec<Vec<usize>>,
    original_runs: BTreeSet<usize>,
    alternative_runs: BTreeSet<usize>,
}

/// Compare observable token partitions on the same source ink.
///
/// The union of
/// both candidate graphs allows a genuinely joined glyph cluster; all edits
/// must preserve its ink coverage and scalar cardinality. Exact canonical
/// Unicode equivalence and whitespace changes do not require substitution.
#[must_use = "use the glyph confirmation decision to filter tentative matches"]
pub fn decide(
    primary_score: f64,
    alternative_score: f64,
    policy: ConfirmationPolicy,
    primary: &[CharacterAnchor],
    alternative: &[CharacterAnchor],
    run_count: usize,
) -> ConfirmationDecision {
    if primary_score >= policy.protected_score {
        return ConfirmationDecision::AlreadyEligible;
    }
    if !(policy.minimum_score..=1.0).contains(&alternative_score) {
        return ConfirmationDecision::ScoreTooLow;
    }
    let primary = primary
        .iter()
        .filter(|a| !a.character.is_whitespace())
        .collect::<Vec<_>>();
    let alternative = alternative
        .iter()
        .filter(|a| !a.character.is_whitespace())
        .collect::<Vec<_>>();
    if primary.is_empty() || alternative.is_empty() {
        return ConfirmationDecision::NoSharedText;
    }
    let all = primary
        .iter()
        .chain(&alternative)
        .copied()
        .collect::<Vec<_>>();
    let mut sets = DisjointSet::new(all.len());
    let mut owner = vec![None; run_count];
    let runs = all.iter().map(|a| supported_runs(a)).collect::<Vec<_>>();
    if runs
        .iter()
        .any(|r| r.is_empty() || r.iter().any(|&i| i >= run_count))
    {
        return ConfirmationDecision::Unanchored;
    }
    for (i, group) in runs.iter().enumerate() {
        for &run in group {
            if let Some(first) = owner[run] {
                sets.connect(i, first);
            } else {
                owner[run] = Some(i);
            }
        }
    }
    let mut clusters = BTreeMap::<usize, Cluster>::new();
    let mut original_order = Vec::new();
    let mut alternative_order = Vec::new();
    for (i, (token, runs)) in all.into_iter().zip(runs).enumerate() {
        let root = sets.root(i);
        let cluster = clusters.entry(root).or_default();
        let (text, ink, tokens, order) = if i < primary.len() {
            (
                &mut cluster.original,
                &mut cluster.original_runs,
                &mut cluster.original_tokens,
                &mut original_order,
            )
        } else {
            (
                &mut cluster.alternative,
                &mut cluster.alternative_runs,
                &mut cluster.alternative_tokens,
                &mut alternative_order,
            )
        };
        text.push(token.character);
        ink.extend(runs.iter().copied());
        tokens.push(runs);
        if order.last() != Some(&root) {
            order.push(root);
        }
    }
    // Do not accept moved/reordered groups or a different number of glyphs
    // in the same connected source cluster.
    if original_order != alternative_order {
        return ConfirmationDecision::ChangedSegmentation;
    }
    compare_clusters(clusters.values())
}

fn compare_clusters<'a>(clusters: impl Iterator<Item = &'a Cluster>) -> ConfirmationDecision {
    let mut changes = false;
    let mut shared = false;
    for cluster in clusters {
        if cluster.original_runs != cluster.alternative_runs {
            return ConfirmationDecision::ChangedSegmentation;
        }
        if cluster.original.nfc().eq(cluster.alternative.nfc()) {
            shared = true;
            continue;
        }
        if cluster.original.chars().count() != cluster.alternative.chars().count() {
            return ConfirmationDecision::ChangedSegmentation;
        }
        // Equal aggregate coverage alone can hide a changed token that has
        // drifted into its neighbour. Preserve its individual support even
        // when the two models agree that several tokens share joined ink.
        for ((a, b), (left, right)) in cluster
            .original
            .chars()
            .zip(cluster.alternative.chars())
            .zip(
                cluster
                    .original_tokens
                    .iter()
                    .zip(&cluster.alternative_tokens),
            )
        {
            if a != b && left != right {
                return ConfirmationDecision::UnstableAlignment;
            }
        }
        changes = true;
        // A joined cluster may contain unchanged characters mixed with
        // substitutions. They still count as agreement, not new labels.
        shared |= cluster
            .original
            .chars()
            .zip(cluster.alternative.chars())
            .any(|(a, b)| a == b);
    }
    if !shared {
        return ConfirmationDecision::NoSharedText;
    }
    if changes {
        ConfirmationDecision::InkSupportedSubstitution
    } else {
        ConfirmationDecision::Agreement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn token(ch: char, run: usize) -> CharacterAnchor {
        CharacterAnchor {
            character: ch,
            column: 0,
            crop_x: 1.,
            candidate_runs: vec![run],
            alignment: RunAlignment::Unique { run },
        }
    }
    #[test]
    fn actual_score_and_ink_identity_gate_local_substitution() {
        let primary = [token('A', 0), token('0', 1), token('1', 2)];
        let corrected = [token('A', 0), token('Ø', 1), token('1', 2)];
        let deleted = [token('A', 0), token('1', 2)];
        let unrelated = [token('B', 0), token('C', 1), token('D', 2)];
        for (score, alternative_score, candidate, expected) in [
            (
                0.93,
                0.98,
                corrected.as_slice(),
                ConfirmationDecision::InkSupportedSubstitution,
            ),
            (
                0.98,
                0.99,
                corrected.as_slice(),
                ConfirmationDecision::AlreadyEligible,
            ),
            (
                0.93,
                0.96,
                corrected.as_slice(),
                ConfirmationDecision::ScoreTooLow,
            ),
            (
                0.93,
                0.98,
                primary.as_slice(),
                ConfirmationDecision::Agreement,
            ),
            (
                0.93,
                0.98,
                deleted.as_slice(),
                ConfirmationDecision::ChangedSegmentation,
            ),
            (
                0.93,
                0.98,
                unrelated.as_slice(),
                ConfirmationDecision::NoSharedText,
            ),
        ] {
            assert_eq!(
                decide(
                    score,
                    alternative_score,
                    ConfirmationPolicy::for_overlay(0.97),
                    &primary,
                    candidate,
                    3
                ),
                expected
            );
        }
    }
    #[test]
    fn substitution_cannot_expand_individual_ink_support() {
        let ambiguous = |ch, runs: Vec<usize>| CharacterAnchor {
            character: ch,
            column: 0,
            crop_x: 1.,
            alignment: RunAlignment::Ambiguous {
                first: runs[0],
                last: runs[runs.len() - 1],
            },
            candidate_runs: runs,
        };
        let primary = [
            ambiguous('1', vec![0, 1]),
            ambiguous('0', vec![1, 2]),
            token('3', 3),
        ];
        let alternative = [token('1', 0), token('0', 1), ambiguous('^', vec![2, 3])];
        assert_eq!(
            decide(
                0.75,
                0.98,
                ConfirmationPolicy::for_overlay(0.97),
                &primary,
                &alternative,
                4
            ),
            ConfirmationDecision::UnstableAlignment
        );
        assert_eq!(
            decide(
                0.96,
                0.98,
                ConfirmationPolicy::for_overlay(0.95),
                &primary,
                &alternative,
                4
            ),
            ConfirmationDecision::AlreadyEligible
        );
        assert_eq!(
            decide(
                0.93,
                0.96,
                ConfirmationPolicy::for_overlay(0.95),
                &primary,
                &alternative,
                4
            ),
            ConfirmationDecision::ScoreTooLow
        );
    }

    #[test]
    fn joined_ink_clusters_are_compared_without_forcing_char_boxes() {
        let joined = |ch| CharacterAnchor {
            character: ch,
            column: 0,
            crop_x: 1.,
            candidate_runs: vec![1],
            alignment: RunAlignment::NoInjectiveAssignment,
        };
        let primary = [token('A', 0), joined('c'), joined('n'), token('t', 2)];
        let secondary = [token('A', 0), joined('e'), joined('n'), token('t', 2)];
        assert_eq!(
            decide(
                0.93,
                0.98,
                ConfirmationPolicy::for_overlay(0.97),
                &primary,
                &secondary,
                3
            ),
            ConfirmationDecision::InkSupportedSubstitution
        );
        let partial = [token('A', 0), joined('e'), token('t', 2)];
        assert_eq!(
            decide(
                0.93,
                0.98,
                ConfirmationPolicy::for_overlay(0.97),
                &primary,
                &partial,
                3
            ),
            ConfirmationDecision::ChangedSegmentation
        );
        let padded = CharacterAnchor {
            character: 'X',
            column: 0,
            crop_x: 100.,
            candidate_runs: vec![],
            alignment: RunAlignment::Padding,
        };
        assert_eq!(
            decide(
                0.93,
                0.98,
                ConfirmationPolicy::for_overlay(0.97),
                &primary,
                &[token('A', 0), padded, joined('n'), token('t', 2)],
                3
            ),
            ConfirmationDecision::Unanchored
        );
    }
}
