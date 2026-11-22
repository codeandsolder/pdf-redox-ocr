//! Research-only, atomic replacement of fragmented OCR observations.
//! Recognition support is constrained by the actual newly included ink
//! in one shared raster, not by text length or overlapping boxes.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use serde::Serialize;

use crate::{
    candidate_ink::{CandidateAnchor, CandidateMapping},
    character_alignment::{CharacterAnchor, RunAlignment, align_runs},
    ctc_evidence::CtcOutcome,
    draft::{OcrDraft, OcrLine},
    fragment_ink::{InkAccounting, InkCounts, RunAccounting},
    geometry::{Point, Quad, QuadBounds},
    layout_leaders::LeaderEdge,
    line_audit::{BoundaryPeer, MappedBoundaryExtension},
    line_frame::LineFrame,
    polygon::{area, clip_convex},
    recognition_model::RecognitionModel,
    refinement_evidence::{Recognition, RefinementDecision, RefinementOutcome, canonical},
};

#[derive(Debug, Clone, Serialize)]
pub struct TokenBinding {
    /// Non-whitespace Unicode scalar index in the proposed transcription.
    pub position: usize,
    pub character: char,
    pub run: usize,
    pub views: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Consumption {
    AllForegroundCaptured,
    LocalizedPeerTextPreserved,
}

#[derive(Debug, Clone, Serialize)]
pub struct PeerClaim {
    pub line_index: usize,
    pub ink: InkCounts,
    pub reason: Consumption,
    pub token_positions: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct Transaction {
    pub page: u32,
    pub line_index: usize,
    pub extension_index: usize,
    pub quad: Quad,
    pub text: String,
    pub bindings: Vec<TokenBinding>,
    pub consumed: Vec<PeerClaim>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    NoMeasuredInk,
    NoConsensus,
    Unchanged,
    NotLiteralExtension,
    UnlocalizedAddedTokens,
    NativeTextConflict,
    IncompletePeer,
    MissingPeerEvidence,
    UnboundPeerInk,
    SplitPeerComponent,
    ResidualPeerText,
    PixelBudgetExceeded,
    InkBudgetExceeded,
}

fn added_range(primary: &str, candidate: &str, edge: LeaderEdge) -> Option<Range<usize>> {
    let old = canonical(primary);
    let new = canonical(candidate);
    let (a, b) = (old.chars().count(), new.chars().count());
    if a == 0 || b <= a {
        return None;
    }
    match edge {
        LeaderEdge::Left if new.ends_with(&old) => Some(0..b - a),
        LeaderEdge::Right if new.starts_with(&old) => Some(a..b),
        _ => None,
    }
}

fn top_anchors<'a>(
    view: &'a crate::refinement_evidence::ViewEvidence,
    text: &str,
) -> Option<Vec<&'a CandidateAnchor>> {
    let CtcOutcome::Measured { evidence } = &view.ctc else {
        return None;
    };
    if canonical(&evidence.hypotheses.first()?.text) != canonical(text) {
        return None;
    }
    let CandidateMapping::Mapped { anchors } = view.ink_mapping.as_ref()?.hypotheses.first()?
    else {
        return None;
    };
    let visible = anchors
        .iter()
        .filter(|a| !a.anchor.character.is_whitespace())
        .collect::<Vec<_>>();
    visible
        .iter()
        .map(|a| a.anchor.character)
        .eq(canonical(text).chars())
        .then_some(visible)
}

fn constrained_view(
    anchors: &[&CandidateAnchor],
    added: &Range<usize>,
    ink: &RunAccounting,
    width: u32,
) -> Option<Vec<usize>> {
    let mut padding_claims = BTreeSet::new();
    let mut constrained = Vec::with_capacity(anchors.len());
    for (position, &candidate) in anchors.iter().enumerate() {
        let anchor = &candidate.anchor;
        let runs = if matches!(anchor.alignment, RunAlignment::Padding) {
            padding_claims.extend(anchor.candidate_runs.iter().copied());
            Vec::new()
        } else {
            anchor
                .candidate_runs
                .iter()
                .copied()
                .filter(|&run| {
                    ink.runs.get(run).is_some_and(|r| {
                        if added.contains(&position) {
                            r.sample.added > 0
                        } else {
                            r.sample.original > 0
                        }
                    })
                })
                .collect()
        };
        // An already localizable original character may not lose its
        // support to manufacture a new character assignment.
        if runs.is_empty()
            && matches!(
                anchor.alignment,
                RunAlignment::Unique { .. } | RunAlignment::Ambiguous { .. }
            )
        {
            return None;
        }
        constrained.push(CharacterAnchor {
            character: anchor.character,
            column: anchor.column,
            crop_x: anchor.crop_x,
            candidate_runs: runs,
            alignment: anchor.alignment,
        });
    }
    align_runs(&mut constrained, ink.runs.len(), width);
    let assigned = added
        .clone()
        .map(|position| {
            let RunAlignment::Unique { run } = constrained.get(position)?.alignment else {
                return None;
            };
            (!padding_claims.contains(&run)
                && !matches!(
                    anchors.get(position)?.anchor.alignment,
                    RunAlignment::Padding
                ))
            .then_some(run)
        })
        .collect::<Option<Vec<_>>>()?;
    if assigned.windows(2).any(|r| r[0] >= r[1]) {
        return None;
    }
    let first = *assigned.first()?;
    let last = *assigned.last()?;
    for (position, anchor) in constrained.iter().enumerate() {
        if added.contains(&position) || anchor.candidate_runs.is_empty() {
            continue;
        }
        if !anchor.candidate_runs.iter().any(|&run| {
            if position < added.start {
                run < first
            } else {
                run > last
            }
        }) {
            return None;
        }
    }
    Some(assigned)
}

fn bind_added(
    recognition: &Recognition,
    text: &str,
    view_indices: &[usize],
    added: Range<usize>,
    ink: &RunAccounting,
) -> Option<Vec<TokenBinding>> {
    let mut votes = Vec::new();
    let mut models = BTreeSet::new();
    let mut assignment = None;
    for &index in view_indices {
        let view = recognition.views.get(index)?;
        let Some(anchors) = top_anchors(view, text) else {
            continue;
        };
        let Some(runs) =
            constrained_view(&anchors, &added, ink, recognition.source_ink.dimensions.0)
        else {
            continue;
        };
        if assignment
            .as_ref()
            .is_some_and(|previous| previous != &runs)
        {
            return None;
        }
        assignment = Some(runs);
        models.insert(view.model);
        votes.push(index);
    }
    if !both_models(&models) {
        return None;
    }
    let characters = canonical(text).chars().collect::<Vec<_>>();
    Some(
        added
            .zip(assignment?)
            .map(|(position, run)| TokenBinding {
                position,
                character: characters[position],
                run,
                views: votes.clone(),
            })
            .collect(),
    )
}

fn both_models(models: &BTreeSet<RecognitionModel>) -> bool {
    models.contains(&RecognitionModel::Small) && models.contains(&RecognitionModel::Medium)
}

fn run_quad(quad: Quad, recognition: &Recognition, run: usize) -> Option<Quad> {
    let (width, height) = recognition.source_ink.dimensions;
    let frame = LineFrame {
        corners: quad,
        width,
        height,
        axis_swapped: false,
    };
    let b = recognition.source_ink.groups.get(run)?.run.ink;
    let (l, t, r, b) = (
        f64::from(b.left),
        f64::from(b.top),
        f64::from(b.right),
        f64::from(b.bottom),
    );
    Some([Point(l, b), Point(r, b), Point(r, t), Point(l, t)].map(|p| frame.to_source(p)))
}

fn intersection(a: &Quad, b: &Quad) -> f64 {
    if QuadBounds::new(a).intersects(QuadBounds::new(b)) {
        area(&clip_convex(a, b))
    } else {
        0.0
    }
}

fn contained(a: &Quad, b: &Quad) -> bool {
    let size = area(a);
    size > 0.0 && (size - intersection(a, b)).abs() <= size.max(1.0) * 1e-8
}

/// Stable first-entry intervals are stronger localization evidence than a
/// single token's unconstrained midpoint cell, which spans the whole line.
fn peer_runs(recognition: &Recognition, text: &str) -> Option<Vec<usize>> {
    let mut assignment = None;
    let mut models = BTreeSet::new();
    for view in &recognition.views {
        let anchors = top_anchors(view, text)?;
        let runs = anchors
            .iter()
            .map(|a| {
                let [lo, hi] = a.entry_interval;
                let mut matching =
                    recognition
                        .source_ink
                        .groups
                        .iter()
                        .enumerate()
                        .filter(|(_, g)| {
                            let b = g.run.ink;
                            !g.touches_line_edge
                                && f64::from(b.left) <= lo
                                && lo <= hi
                                && hi < f64::from(b.right)
                        });
                let (index, _) = matching.next()?;
                matching.next().is_none().then_some(index)
            })
            .collect::<Option<Vec<_>>>()?;
        if runs.windows(2).any(|p| p[0] >= p[1])
            || assignment.as_ref().is_some_and(|old| old != &runs)
        {
            return None;
        }
        assignment = Some(runs);
        models.insert(view.model);
    }
    both_models(&models).then_some(assignment).flatten()
}

fn preserve_partial_peer(
    line: &OcrLine,
    peer: &Recognition,
    variant: &Recognition,
    quad: Quad,
    bindings: &[TokenBinding],
    ink: &RunAccounting,
    peer_index: usize,
) -> Option<Vec<usize>> {
    let runs = peer_runs(peer, &line.text)?;
    let mut positions = Vec::new();
    for (character, run) in canonical(&line.text).chars().zip(runs) {
        let source = run_quad(line.quad, peer, run)?;
        if !contained(&source, &quad) {
            return None;
        }
        let mut matched = bindings.iter().filter(|b| {
            b.character == character
                && ink.runs[b.run]
                    .peers
                    .get(&peer_index)
                    .is_some_and(|c| c.added > 0)
                && run_quad(quad, variant, b.run).is_some_and(|q| intersection(&source, &q) > 0.0)
        });
        let b = matched.next()?;
        if matched.next().is_some() {
            return None;
        }
        positions.push(b.position);
    }
    positions
        .windows(2)
        .all(|p| p[0] < p[1])
        .then_some(positions)
}

struct ProposalContext<'a> {
    extension: &'a MappedBoundaryExtension,
    variant: &'a Recognition,
    ink: &'a RunAccounting,
    bindings: &'a [TokenBinding],
}

impl ProposalContext<'_> {
    fn claim(
        &self,
        peer: usize,
        index: usize,
        line: &OcrLine,
        recognition: &Recognition,
    ) -> Result<PeerClaim, Rejection> {
        let InkAccounting::Measured {
            peers,
            peers_fully_sampled,
            components,
            ..
        } = &self.extension.ink_accounting
        else {
            return Err(Rejection::NoMeasuredInk);
        };
        let count = peers.get(&peer).copied().unwrap_or_default();
        if !peers_fully_sampled.get(peer).copied().unwrap_or(false) {
            return Err(Rejection::IncompletePeer);
        }
        // Every added pixel belonging to this peer must reside in a run
        // associated with a newly recognized token, not nearby diagram ink.
        let mut accounted = 0;
        let mut positions = Vec::new();
        for binding in self.bindings {
            let pixels = self.ink.runs[binding.run]
                .peers
                .get(&peer)
                .map_or(0, |c| c.added);
            if pixels > 0 {
                accounted += pixels;
                positions.push(binding.position);
            }
        }
        if accounted != count.added {
            return Err(Rejection::UnboundPeerInk);
        }
        let reason = if count.outside_expansion == 0 {
            Consumption::AllForegroundCaptured
        } else {
            if components.iter().any(|c| {
                c.peers
                    .get(&peer)
                    .is_some_and(|p| p.added > 0 && p.outside_expansion > 0)
            }) {
                return Err(Rejection::SplitPeerComponent);
            }
            let preserved = preserve_partial_peer(
                line,
                recognition,
                self.variant,
                self.extension.proposed_quad,
                self.bindings,
                self.ink,
                peer,
            )
            .ok_or(Rejection::ResidualPeerText)?;
            if preserved != positions {
                return Err(Rejection::ResidualPeerText);
            }
            Consumption::LocalizedPeerTextPreserved
        };
        Ok(PeerClaim {
            line_index: index,
            ink: count,
            reason,
            token_positions: positions,
        })
    }
}

pub fn propose(
    page: &crate::draft::DraftPage,
    line_index: usize,
    extension_index: usize,
    extension: &MappedBoundaryExtension,
    variant: &Recognition,
    ink: &RunAccounting,
    peers: &BTreeMap<usize, RefinementOutcome>,
) -> Result<Transaction, Rejection> {
    let InkAccounting::Measured {
        sample,
        peers: counts,
        ..
    } = &extension.ink_accounting
    else {
        return Err(Rejection::NoMeasuredInk);
    };
    if sample.added == 0 {
        return Err(Rejection::NoMeasuredInk);
    }
    let RefinementDecision::Consensus { text, view_indices } = &variant.decision else {
        return Err(Rejection::NoConsensus);
    };
    let original = &page.lines[line_index].text;
    if canonical(original) == canonical(text) {
        return Err(Rejection::Unchanged);
    }
    let added = added_range(original, text, extension.evidence.edge)
        .ok_or(Rejection::NotLiteralExtension)?;
    let bindings = bind_added(variant, text, view_indices, added, ink)
        .ok_or(Rejection::UnlocalizedAddedTokens)?;
    let context = ProposalContext {
        extension,
        variant,
        ink,
        bindings: &bindings,
    };
    let mut consumed = Vec::new();
    for (i, overlap) in extension.overlaps.iter().enumerate() {
        if counts.get(&i).is_none_or(|c| c.added == 0) {
            continue;
        }
        let BoundaryPeer::OcrLine { line_index } = overlap.peer else {
            return Err(Rejection::NativeTextConflict);
        };
        let Some(RefinementOutcome::Recognized { recognition, .. }) = peers.get(&line_index) else {
            return Err(Rejection::MissingPeerEvidence);
        };
        consumed.push(context.claim(i, line_index, &page.lines[line_index], recognition)?);
    }
    Ok(Transaction {
        page: page.page,
        line_index,
        extension_index,
        quad: extension.proposed_quad,
        text: text.clone(),
        bindings,
        consumed,
    })
}

/// Conflicts are symmetric, including multiple proposals for one observation
/// and overlapping replacement geometry. No score or input order breaks ties.
pub fn conflicts(operations: &[Transaction]) -> BTreeSet<usize> {
    let mut pages = BTreeMap::<u32, Vec<usize>>::new();
    let mut identities = BTreeMap::<(u32, usize), usize>::new();
    let mut blocked = BTreeSet::new();
    for (i, op) in operations.iter().enumerate() {
        for index in std::iter::once(op.line_index).chain(op.consumed.iter().map(|p| p.line_index))
        {
            if let Some(previous) = identities.insert((op.page, index), i) {
                if previous != i {
                    blocked.extend([previous, i]);
                }
            }
        }
        let earlier = pages.entry(op.page).or_default();
        for &j in earlier.iter() {
            if intersection(&op.quad, &operations[j].quad) > 1e-8 {
                blocked.extend([i, j]);
            }
        }
        earlier.push(i);
    }
    blocked
}

pub fn experimental_draft(
    source: &OcrDraft,
    operations: &[Transaction],
    blocked: &BTreeSet<usize>,
) -> OcrDraft {
    let mut draft = source.clone();
    // Unlike ordinary diagnostic refinements, this removes observations;
    // it must not be accepted even by an importer with a zero score threshold.
    draft.schema_version = 0;
    draft.model = "NON-IMPORTABLE fragment reconciliation research".into();
    let mut replacements = BTreeMap::new();
    let mut consumed = BTreeSet::new();
    for (i, op) in operations.iter().enumerate() {
        if blocked.contains(&i) {
            continue;
        }
        replacements.insert((op.page, op.line_index), op);
        consumed.extend(op.consumed.iter().map(|p| (op.page, p.line_index)));
    }
    for page in &mut draft.pages {
        let mut index = 0;
        page.lines.retain_mut(|line| {
            let key = (page.page, index);
            index += 1;
            if consumed.contains(&key) {
                return false;
            }
            if let Some(op) = replacements.get(&key) {
                line.text.clone_from(&op.text);
                line.quad = op.quad;
            }
            line.confidence = 0.;
            line.reviewed = false;
            true
        });
    }
    draft
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{draft::DraftPage, fragment_ink::RunInk};

    fn ink(counts: &[(u64, u64)]) -> RunAccounting {
        RunAccounting {
            runs: counts
                .iter()
                .enumerate()
                .map(|(run, &(original, added))| RunInk {
                    run,
                    sample: InkCounts {
                        original,
                        added,
                        outside_expansion: 0,
                    },
                    peers: BTreeMap::new(),
                })
                .collect(),
            charged_checks: 0,
        }
    }

    fn anchor(c: char, x: f64, runs: Vec<usize>, alignment: RunAlignment) -> CandidateAnchor {
        CandidateAnchor {
            anchor: CharacterAnchor {
                character: c,
                column: 0,
                crop_x: x,
                candidate_runs: runs,
                alignment,
            },
            entry_interval: [x, x],
        }
    }

    #[test]
    fn actual_added_pixels_resolve_ambiguous_punctuation_without_stealing_original_ink() {
        // A vertical drawing stroke lies between the two original digits
        // and the newly captured dot. It is a legal contextual candidate
        // before source accounting but has no new foreground membership.
        let a = [
            anchor('1', 10., vec![0], RunAlignment::Unique { run: 0 }),
            anchor('2', 20., vec![1], RunAlignment::Unique { run: 1 }),
            anchor(
                '.',
                30.,
                vec![1, 2, 3],
                RunAlignment::Ambiguous { first: 2, last: 3 },
            ),
            anchor('4', 40., vec![4], RunAlignment::Unique { run: 4 }),
        ];
        let refs = a.iter().collect::<Vec<_>>();
        let accounting = ink(&[(20, 0), (30, 0), (90, 0), (4, 8), (0, 35)]);
        assert_eq!(
            constrained_view(&refs, &(2..4), &accounting, 50),
            Some(vec![3, 4])
        );
        let ambiguous = ink(&[(20, 0), (30, 0), (90, 1), (4, 8), (0, 35)]);
        assert_eq!(constrained_view(&refs, &(2..4), &ambiguous, 50), None);
        let missing = ink(&[(20, 0), (30, 0), (90, 0), (4, 0), (0, 35)]);
        assert_eq!(constrained_view(&refs, &(2..4), &missing, 50), None);
        let stolen = ink(&[(20, 0), (0, 30), (90, 0), (4, 8), (0, 35)]);
        assert_eq!(constrained_view(&refs, &(2..4), &stolen, 50), None);
    }

    #[test]
    fn left_extensions_and_repeated_symbols_use_literal_scalar_positions() {
        assert_eq!(added_range(" Ω² ", "µΩ²", LeaderEdge::Left), Some(0..1));
        assert_eq!(added_range("Ω", "ΩΩ", LeaderEdge::Right), Some(1..2));
        assert_eq!(added_range("Ω", "OΩ", LeaderEdge::Right), None);
        assert_eq!(added_range("Ω", "Ω ", LeaderEdge::Right), None);
        let a = [
            anchor(
                'A',
                10.,
                vec![0, 1],
                RunAlignment::Ambiguous { first: 0, last: 1 },
            ),
            anchor('A', 20., vec![0, 1], RunAlignment::Unique { run: 1 }),
        ];
        let accounting = ink(&[(0, 20), (30, 0)]);
        assert_eq!(
            constrained_view(&a.iter().collect::<Vec<_>>(), &(0..1), &accounting, 30),
            Some(vec![0])
        );
    }

    #[test]
    fn disjoint_but_crossed_extension_assignments_are_rejected() {
        let a = [
            anchor('A', 10., vec![1], RunAlignment::Unique { run: 1 }),
            anchor('B', 20., vec![0], RunAlignment::Unique { run: 0 }),
        ];
        let accounting = ink(&[(0, 20), (30, 0)]);
        assert_eq!(
            constrained_view(&a.iter().collect::<Vec<_>>(), &(1..2), &accounting, 30),
            None
        );
    }

    fn rect(x: f64) -> Quad {
        [
            Point(x, 0.),
            Point(x + 1., 0.),
            Point(x + 1., 1.),
            Point(x, 1.),
        ]
    }

    fn operation(page: u32, line_index: usize, consumed: Vec<usize>, x: f64) -> Transaction {
        Transaction {
            page,
            line_index,
            extension_index: 0,
            quad: rect(x),
            text: "AB".into(),
            bindings: vec![],
            consumed: consumed
                .into_iter()
                .map(|line_index| PeerClaim {
                    line_index,
                    ink: InkCounts::default(),
                    reason: Consumption::AllForegroundCaptured,
                    token_positions: vec![],
                })
                .collect(),
        }
    }

    #[test]
    fn group_conflicts_are_symmetric_and_replacements_preserve_original_indices() {
        let operations = [
            operation(1, 0, vec![1], 0.),
            operation(1, 1, vec![2], 2.),
            operation(1, 4, vec![], 2.5),
            operation(2, 0, vec![1], 0.),
            operation(2, 2, vec![], 2.),
        ];
        let blocked = conflicts(&operations);
        assert_eq!(blocked, BTreeSet::from([0, 1, 2]));
        let source = OcrDraft::new(
            "digest".into(),
            "source".into(),
            (1..=2)
                .map(|page| DraftPage {
                    page,
                    native_text_regions: vec![rect(20.)],
                    lines: (0..5)
                        .map(|i| OcrLine {
                            text: i.to_string(),
                            confidence: 0.99,
                            quad: rect(f64::from(i) * 2.),
                            reviewed: true,
                        })
                        .collect(),
                })
                .collect(),
        );
        let draft = experimental_draft(&source, &operations, &blocked);
        assert_eq!(draft.schema_version, 0);
        assert_eq!(source.pages[1].lines.len(), 5);
        assert_eq!(
            draft.pages[0]
                .lines
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>(),
            ["0", "1", "2", "3", "4"]
        );
        assert_eq!(
            draft.pages[1]
                .lines
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>(),
            ["AB", "AB", "3", "4"]
        );
        assert_eq!(
            draft.pages[1].native_text_regions,
            source.pages[1].native_text_regions
        );
        assert!(
            draft
                .pages
                .iter()
                .flat_map(|p| &p.lines)
                .all(|l| l.confidence.to_bits() == 0.0_f64.to_bits() && !l.reviewed)
        );
        assert!(source.pages[1].lines[1].reviewed);
    }
}
