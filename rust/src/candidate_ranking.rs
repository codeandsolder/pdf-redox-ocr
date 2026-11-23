//! Offline comparison of competing recognition hypotheses against reviewed ink.
//!
//! All proposed substitutions must already exist in the retained CTC candidates.
//! Group geometry is recomputed from the recorded conditional token intervals.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    Error,
    candidate_ink::{CandidateMapping, SourceInk},
    character_alignment::RunAlignment,
    ctc_alignment::SequenceAlignment,
    ctc_evidence::CtcOutcome,
    draft::{OcrDraft, OcrLine},
    glyph_matching::validate_bank,
    recognition_model::RecognitionModel,
    refinement_evidence::{
        Recognition, RefinementOutcome, RefinementProposal, RefinementReport, canonical,
        experimental_draft,
    },
    reviewed_glyphs::ReviewedGlyphBank,
};

const MIN_DONORS: usize = 2;

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Debug, Clone, Serialize)]
struct Donor<'a> {
    bank: usize,
    pdf_sha256: &'a str,
    page: u32,
    detection_index: usize,
    character_index: usize,
    normalized_line_sha256: &'a str,
    holes: usize,
}

type Donors<'a> = BTreeMap<char, Vec<Donor<'a>>>;

fn collect_donors<'a>(banks: &[&'a ReviewedGlyphBank], dpi: u32) -> Result<Donors<'a>, Error> {
    let mut donors = Donors::new();
    for (bank_index, bank) in banks.iter().enumerate() {
        validate_bank(bank)?;
        if bank.source_sampling.as_ref().map(|s| s.dpi) != Some(dpi) {
            return Err(invalid("donors must use the same source DPI"));
        }
        for glyph in &bank.glyphs {
            let Some(source) = &glyph.source_appearance else {
                continue;
            };
            if let Some(holes) = source.shape.stable_hole_count()? {
                donors.entry(glyph.label).or_default().push(Donor {
                    bank: bank_index,
                    pdf_sha256: &bank.pdf_sha256,
                    page: glyph.page,
                    detection_index: glyph.detection_index,
                    character_index: glyph.character_index,
                    normalized_line_sha256: &source.normalized_line_sha256,
                    holes,
                });
            }
        }
    }
    Ok(donors)
}

#[derive(Debug, Serialize)]
struct Binding {
    run: usize,
    views: BTreeSet<usize>,
    models: BTreeSet<RecognitionModel>,
}
impl Binding {
    fn supported(&self) -> bool {
        self.models.contains(&RecognitionModel::Small)
            && self.models.contains(&RecognitionModel::Medium)
    }
}

/// Each view contributes at most once; aliases with different localization
/// within that view cannot supply a unique binding.
#[derive(Default)]
struct CandidateBuilder<'a> {
    literal: Option<&'a str>,
    per_view: BTreeMap<usize, Vec<Option<usize>>>,
}

struct Candidate<'a> {
    literal: &'a str,
    bindings: Vec<Option<Binding>>,
}

impl<'a> CandidateBuilder<'a> {
    fn add(&mut self, view: usize, literal: &'a str, runs: Vec<Option<usize>>) {
        if self
            .literal
            .is_none_or(|old| (literal.chars().count(), literal) < (old.chars().count(), old))
        {
            self.literal = Some(literal);
        }
        self.per_view
            .entry(view)
            .and_modify(|prior| {
                for (a, b) in prior.iter_mut().zip(&runs) {
                    if a != b {
                        *a = None;
                    }
                }
            })
            .or_insert(runs);
    }

    fn finish(self, recognition: &Recognition) -> Option<Candidate<'a>> {
        let literal = self.literal?;
        let bindings = canonical(literal)
            .chars()
            .enumerate()
            .map(|(index, _)| {
                let mut binding: Option<Binding> = None;
                for (&view, runs) in &self.per_view {
                    let Some(run) = runs[index] else { continue };
                    let entry = binding.get_or_insert_with(|| Binding {
                        run,
                        views: BTreeSet::new(),
                        models: BTreeSet::new(),
                    });
                    if entry.run != run {
                        return None;
                    }
                    entry.views.insert(view);
                    entry.models.insert(recognition.views[view].model);
                }
                binding
            })
            .collect();
        Some(Candidate { literal, bindings })
    }
}

fn validate_ink(ink: &SourceInk) -> Result<(), Error> {
    let (width, height) = ink.dimensions;
    if width == 0
        || height == 0
        || !digest(&ink.rgb_sha256)
        || !digest(&ink.normalized_sha256)
        || ink.groups.len() != ink.appearances.len()
    {
        return Err(invalid("invalid shared source ink"));
    }
    let mut previous_right = 0;
    for (group, shape) in ink.groups.iter().zip(&ink.appearances) {
        let b = group.run.ink;
        let clipped = b.left == 0 || b.top == 0 || b.right == width || b.bottom == height;
        if b.left >= b.right
            || b.top >= b.bottom
            || b.right > width
            || b.bottom > height
            || b.left < previous_right
            || group.touches_line_edge != clipped
            || shape.is_some() == clipped
            || shape
                .as_ref()
                .is_some_and(|s| s.dimensions() != (b.right - b.left, b.bottom - b.top))
        {
            return Err(invalid("invalid source group or appearance"));
        }
        previous_right = b.right;
    }
    Ok(())
}

fn candidates(recognition: &Recognition) -> Result<BTreeMap<String, Candidate<'_>>, Error> {
    let ink = &recognition.source_ink;
    validate_ink(ink)?;
    let mut result: BTreeMap<String, CandidateBuilder<'_>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for (index, view) in recognition.views.iter().enumerate() {
        let expected_width =
            (u64::from(ink.dimensions.0) * u64::from(view.stretch_percent)).div_ceil(100);
        if ![85, 100, 120, 150].contains(&view.stretch_percent)
            || expected_width != u64::from(view.dimensions.0)
            || view.dimensions.1 != ink.dimensions.1
            || !digest(&view.rgb_sha256)
            || (view.stretch_percent == 100 && view.rgb_sha256 != ink.rgb_sha256)
            || !(0.0..=1.0).contains(&view.greedy_score)
        {
            return Err(invalid("invalid recorded recognition view"));
        }
        let CtcOutcome::Measured { evidence } = &view.ctc else {
            if view.ink_mapping.is_some() {
                return Err(invalid("mapping without measured CTC"));
            }
            continue;
        };
        if evidence.greedy_text != view.greedy_text || evidence.hypotheses.len() > 5 {
            return Err(invalid("inconsistent CTC evidence"));
        }
        for hypothesis in &evidence.hypotheses {
            if hypothesis
                .classes
                .iter()
                .any(|&class| class == 0 || class >= evidence.vocabulary_size)
                || hypothesis.classes.len() != hypothesis.text.chars().count()
                || !hypothesis.log_mass_lower_bound.is_finite()
            {
                return Err(invalid("invalid retained class sequence"));
            }
            if let SequenceAlignment::Measured { log_mass, tokens } = &hypothesis.alignment
                && (!log_mass.is_finite()
                    || *log_mass + 1e-8 < hypothesis.log_mass_lower_bound
                    || tokens
                        .iter()
                        .any(|t| !(0.0..=1.0).contains(&t.mode_probability)))
            {
                return Err(invalid("invalid conditional alignment"));
            }
        }
        let mapping = ink.map_view(view.dimensions, evidence);
        if mapping != view.ink_mapping {
            return Err(invalid("recorded mapping differs from recomputed geometry"));
        }
        let Some(mapping) = mapping else { continue };
        if !seen.insert((view.model, view.dimensions, &view.rgb_sha256)) {
            continue;
        }
        for (hypothesis, mapped) in evidence.hypotheses.iter().zip(mapping.hypotheses) {
            let CandidateMapping::Mapped { anchors } = mapped else {
                continue;
            };
            let key = canonical(&hypothesis.text);
            if key.is_empty() {
                continue;
            }
            let runs = anchors
                .iter()
                .filter(|a| !a.anchor.character.is_whitespace())
                .map(|a| match a.anchor.alignment {
                    RunAlignment::Unique { run } => Some(run),
                    _ => None,
                })
                .collect();
            result
                .entry(key)
                .or_default()
                .add(index, &hypothesis.text, runs);
        }
    }
    Ok(result
        .into_iter()
        .filter_map(|(key, builder)| builder.finish(recognition).map(|c| (key, c)))
        .collect())
}

#[derive(Debug, Serialize)]
struct Dispute<'a> {
    position: usize,
    run: usize,
    incumbent: char,
    alternative: char,
    incumbent_binding: &'a Binding,
    alternative_binding: &'a Binding,
}

/// Compare substitutions only when both candidates bind the same observed
/// source run. Distinct runs and unchanged context must not disagree.
fn disputes<'a>(
    old: &str,
    new: &str,
    a: &'a Candidate<'_>,
    b: &'a Candidate<'_>,
) -> Option<Vec<Dispute<'a>>> {
    if a.bindings.len() != b.bindings.len() {
        return None;
    }
    let mut differences = Vec::new();
    let mut used = BTreeSet::new();
    for (position, ((incumbent, alternative), (left, right))) in old
        .chars()
        .zip(new.chars())
        .zip(a.bindings.iter().zip(&b.bindings))
        .enumerate()
    {
        if let (Some(l), Some(r)) = (left, right)
            && l.run != r.run
        {
            return None;
        }
        if incumbent == alternative {
            continue;
        }
        let (l, r) = (left.as_ref()?, right.as_ref()?);
        if !l.supported() || !r.supported() || !used.insert(l.run) {
            return None;
        }
        differences.push(Dispute {
            position,
            run: l.run,
            incumbent,
            alternative,
            incumbent_binding: l,
            alternative_binding: r,
        });
    }
    (!differences.is_empty()).then_some(differences)
}

#[derive(Debug, Serialize)]
struct TopologyEvidence<'a> {
    query_holes: Option<usize>,
    incumbent_donors: Vec<Donor<'a>>,
    alternative_donors: Vec<Donor<'a>>,
    supported: bool,
}

fn topology_evidence<'a>(
    donors: &Donors<'a>,
    pdf: &str,
    page: u32,
    ink: &SourceInk,
    dispute: &Dispute<'_>,
) -> Result<TopologyEvidence<'a>, Error> {
    let query_holes = ink
        .appearances
        .get(dispute.run)
        .and_then(Option::as_ref)
        .map(crate::glyph_appearance::GlyphShape::stable_hole_count)
        .transpose()?
        .flatten();
    let eligible = |label| {
        donors
            .get(&label)
            .into_iter()
            .flatten()
            .filter(|d| {
                (d.pdf_sha256 != pdf || d.page != page)
                    && d.normalized_line_sha256 != ink.normalized_sha256
            })
            .cloned()
            .collect::<Vec<_>>()
    };
    let incumbent_donors = eligible(dispute.incumbent);
    let alternative_donors = eligible(dispute.alternative);
    let supported = query_holes.is_some_and(|holes| {
        let old = incumbent_donors
            .iter()
            .map(|d| d.normalized_line_sha256)
            .collect::<BTreeSet<_>>();
        let new = alternative_donors
            .iter()
            .filter(|d| d.holes == holes)
            .map(|d| d.normalized_line_sha256)
            .collect::<BTreeSet<_>>();
        old.len() >= MIN_DONORS
            && new.len() >= MIN_DONORS
            && incumbent_donors.iter().all(|d| d.holes != holes)
    });
    Ok(TopologyEvidence {
        query_holes,
        incumbent_donors,
        alternative_donors,
        supported,
    })
}

#[derive(Debug, Serialize)]
struct SubstitutionEvidence<'a> {
    position: usize,
    run: usize,
    incumbent: char,
    alternative: char,
    incumbent_views: BTreeSet<usize>,
    alternative_views: BTreeSet<usize>,
    topology: TopologyEvidence<'a>,
}

#[derive(Debug, Serialize)]
struct Alternative<'a> {
    candidate: String,
    substitutions: Vec<SubstitutionEvidence<'a>>,
    supported: bool,
}

#[derive(Debug, Serialize)]
struct RankedLine<'a> {
    page: u32,
    line_index: usize,
    incumbent: String,
    alternatives: Vec<Alternative<'a>>,
    selected: Option<String>,
}

fn rank_line<'a>(
    proposal: &RefinementProposal,
    donors: &Donors<'a>,
    pdf: &str,
) -> Result<RankedLine<'a>, Error> {
    let incumbent = proposal.revised_text().unwrap_or(&proposal.primary_text);
    let mut line = RankedLine {
        page: proposal.page,
        line_index: proposal.line_index,
        incumbent: incumbent.into(),
        alternatives: Vec::new(),
        selected: None,
    };
    let RefinementOutcome::Recognized { crop, recognition } = &proposal.outcome else {
        return Ok(line);
    };
    let ink = &recognition.source_ink;
    if crop.dimensions != ink.dimensions || crop.rgb_sha256 != ink.rgb_sha256 {
        return Err(invalid("source crop and shared ink disagree"));
    }
    let competition = candidates(recognition)?;
    let key = canonical(incumbent);
    let Some(primary) = competition.get(&key) else {
        return Ok(line);
    };
    for (candidate, alternative) in &competition {
        if candidate == &key {
            continue;
        }
        let Some(differences) = disputes(&key, candidate, primary, alternative) else {
            continue;
        };
        let substitutions = differences
            .iter()
            .map(|d| {
                Ok(SubstitutionEvidence {
                    position: d.position,
                    run: d.run,
                    incumbent: d.incumbent,
                    alternative: d.alternative,
                    incumbent_views: d.incumbent_binding.views.clone(),
                    alternative_views: d.alternative_binding.views.clone(),
                    topology: topology_evidence(donors, pdf, proposal.page, ink, d)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let supported = substitutions.iter().all(|s| s.topology.supported);
        line.alternatives.push(Alternative {
            candidate: alternative.literal.into(),
            substitutions,
            supported,
        });
    }
    let mut supported = line.alternatives.iter().filter(|a| a.supported);
    if let Some(first) = supported.next()
        && supported.next().is_none()
    {
        line.selected = Some(first.candidate.clone());
    }
    Ok(line)
}

#[derive(Debug, Serialize)]
pub struct RankingReport<'a> {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub source_model: String,
    pub donor_pdf_sha256: Vec<String>,
    pub source_dpi: u32,
    pub selected_changes: usize,
    lines: Vec<RankedLine<'a>>,
}

fn validate_proposal(proposal: &RefinementProposal, source: &OcrLine) -> Result<(), Error> {
    if proposal.primary_text != source.text
        || proposal.primary_score.to_bits() != source.confidence.to_bits()
        || proposal
            .quad
            .iter()
            .zip(source.quad)
            .any(|(a, b)| a.0.to_bits() != b.0.to_bits() || a.1.to_bits() != b.1.to_bits())
    {
        return Err(invalid(
            "refinement observation differs from original draft",
        ));
    }
    Ok(())
}

/// Rank saved CTC evidence against independently transcribed donor banks.
///
/// The returned draft is research-only, with confidence and review cleared.
///
/// # Errors
/// Rejects inconsistent artifact provenance, mappings, or invalid donor banks.
pub fn rank<'a>(
    source: &OcrDraft,
    report: &RefinementReport,
    banks: &[&'a ReviewedGlyphBank],
) -> Result<(RankingReport<'a>, OcrDraft), Error> {
    if source.schema_version != 3
        || report.schema_version != 4
        || source.pdf_sha256 != report.pdf_sha256
        || source.model != report.source_model
        || banks.is_empty()
        || !(36..=2400).contains(&report.dpi)
    {
        return Err(invalid("incompatible source, report, or donor inputs"));
    }
    let donors = collect_donors(banks, report.dpi)?;
    let mut pages = BTreeMap::new();
    for page in &source.pages {
        if page.page == 0 || pages.insert(page.page, page).is_some() {
            return Err(invalid("duplicate or invalid source page"));
        }
    }
    let mut seen = BTreeSet::new();
    let mut lines = Vec::new();
    for proposal in &report.proposals {
        if !seen.insert((proposal.page, proposal.line_index)) {
            return Err(invalid("duplicate refinement observation"));
        }
        let line = pages
            .get(&proposal.page)
            .and_then(|p| p.lines.get(proposal.line_index))
            .ok_or_else(|| invalid("unknown refinement observation"))?;
        validate_proposal(proposal, line)?;
        lines.push(rank_line(proposal, &donors, &report.pdf_sha256)?);
    }
    let draft = experimental_draft(
        source,
        &format!("{} + candidate topology", report.secondary_model),
        lines.iter().map(|l| {
            (
                l.page,
                l.line_index,
                l.selected.as_deref().unwrap_or(&l.incumbent),
            )
        }),
    )?;
    let evidence = RankingReport {
        schema_version: 1,
        pdf_sha256: source.pdf_sha256.clone(),
        source_model: source.model.clone(),
        donor_pdf_sha256: banks.iter().map(|b| b.pdf_sha256.clone()).collect(),
        source_dpi: report.dpi,
        selected_changes: lines.iter().filter(|l| l.selected.is_some()).count(),
        lines,
    };
    Ok((evidence, draft))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        candidate_ink::SourceInk,
        ctc_alignment::{SequenceAlignment, TokenLocalization},
        ctc_evidence::{CtcEvidence, SequenceHypothesis},
        refinement_evidence::{RefinementDecision, ViewEvidence},
    };
    use image::{Rgb, RgbImage};

    fn source_ink() -> Result<SourceInk, Error> {
        // Two enclosed regions, separated by a horizontal stroke. The
        // topology is measured independently of either semantic label.
        let image = RgbImage::from_fn(40, 24, |x, y| {
            if (4..20).contains(&x)
                && (3..21).contains(&y)
                && (!(7..17).contains(&x) || !(6..18).contains(&y) || (11..14).contains(&y))
            {
                Rgb([0; 3])
            } else {
                Rgb([255; 3])
            }
        });
        SourceInk::measure(&image)
    }
    fn view(ink: &SourceInk, model: RecognitionModel) -> ViewEvidence {
        let hypotheses = [("X", 1), ("Y", 2)]
            .into_iter()
            .map(|(text, class)| SequenceHypothesis {
                text: text.into(),
                classes: vec![class],
                log_mass_lower_bound: -2.,
                alignment: SequenceAlignment::Measured {
                    log_mass: -1.,
                    tokens: vec![TokenLocalization {
                        class_index: class,
                        mode: 10,
                        mode_probability: 1.,
                        interval: [10, 10],
                    }],
                },
            })
            .collect();
        let evidence = CtcEvidence {
            queries: Vec::new(),
            sequence_length: 80,
            vocabulary_size: 3,
            row_class_limit: 3,
            beam_width: 2,
            greedy_text: "X".into(),
            emissions: Vec::new(),
            hypotheses,
        };
        ViewEvidence {
            model,
            stretch_percent: 100,
            dimensions: ink.dimensions,
            rgb_sha256: ink.rgb_sha256.clone(),
            greedy_text: "X".into(),
            greedy_score: 0.99,
            ink_mapping: ink.map_view(ink.dimensions, &evidence),
            ctc: CtcOutcome::Measured { evidence },
        }
    }
    fn donor(page: u32, label_index: usize, digest: &str, holes: usize) -> Donor<'_> {
        Donor {
            bank: 0,
            pdf_sha256: "test-pdf",
            page,
            detection_index: 0,
            character_index: label_index,
            normalized_line_sha256: digest,
            holes,
        }
    }
    #[test]
    fn candidate_support_requires_distinct_models_and_recomputed_mapping() -> Result<(), Error> {
        let ink = source_ink()?;
        let small = view(&ink, RecognitionModel::Small);
        let medium = view(&ink, RecognitionModel::Medium);
        let mut recognition = Recognition {
            source_ink: Box::new(ink),
            views: vec![small],
            decision: RefinementDecision::Abstained,
        };
        let one = candidates(&recognition)?;
        assert!(disputes("X", "Y", &one["X"], &one["Y"]).is_none());
        recognition.views.push(medium);
        let both = candidates(&recognition)?;
        let difference = disputes("X", "Y", &both["X"], &both["Y"])
            .ok_or_else(|| invalid("expected same-run disagreement"))?;
        assert_eq!(difference[0].run, 0);
        recognition.views[1].ink_mapping = None;
        assert!(candidates(&recognition).is_err());
        Ok(())
    }

    #[test]
    fn topology_requires_independent_donors_for_both_classes_and_holds_out_source()
    -> Result<(), Error> {
        let ink = source_ink()?;
        assert_eq!(
            ink.appearances[0]
                .as_ref()
                .ok_or_else(|| invalid("missing fixture shape"))?
                .stable_hole_count()?,
            Some(2)
        );
        let binding = Binding {
            run: 0,
            views: BTreeSet::from([0, 1]),
            models: BTreeSet::from([RecognitionModel::Small, RecognitionModel::Medium]),
        };
        let dispute = Dispute {
            position: 0,
            run: 0,
            incumbent: 'X',
            alternative: 'Y',
            incumbent_binding: &binding,
            alternative_binding: &binding,
        };
        let mut donors = BTreeMap::from([
            ('X', vec![donor(2, 0, "old1", 1), donor(3, 0, "old2", 1)]),
            ('Y', vec![donor(2, 0, "new1", 2), donor(3, 0, "new1", 2)]),
        ]);
        assert!(!topology_evidence(&donors, "test-pdf", 1, &ink, &dispute)?.supported);
        donors
            .entry('Y')
            .or_default()
            .push(donor(1, 0, "held-out-page", 2));
        donors
            .entry('Y')
            .or_default()
            .push(donor(4, 0, &ink.normalized_sha256, 2));
        assert!(!topology_evidence(&donors, "test-pdf", 1, &ink, &dispute)?.supported);
        donors.entry('Y').or_default().push(donor(4, 0, "new2", 2));
        assert!(topology_evidence(&donors, "test-pdf", 1, &ink, &dispute)?.supported);
        donors
            .entry('X')
            .or_default()
            .push(donor(4, 0, "compatible-incumbent", 2));
        assert!(!topology_evidence(&donors, "test-pdf", 1, &ink, &dispute)?.supported);
        Ok(())
    }
}
