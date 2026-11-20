//! Hold-out glyph recognition using independently reviewed source shapes.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    Error,
    reviewed_glyphs::{ReviewedGlyph, ReviewedGlyphBank},
};

/// Distance and ambiguity policy, separate from any model confidence.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct MatchPolicy {
    pub max_distance: f64,
    pub min_margin: f64,
    pub min_independent_crops: usize,
}

impl MatchPolicy {
    /// # Errors
    /// Rejects degenerate distance, margin or support policies.
    pub fn validate(self) -> Result<(), Error> {
        if !(0.0..=1.2).contains(&self.max_distance)
            || !(0.0..=1.2).contains(&self.min_margin)
            || self.min_independent_crops == 0
        {
            return Err(Error::Invalid("invalid glyph matching policy".into()));
        }
        Ok(())
    }
}

/// Actual nearest donor occurrence, not a class confidence.
#[derive(Clone, Debug, Serialize)]
pub struct DonorMatch<'a> {
    pub label: char,
    pub distance: f64,
    pub pdf_sha256: &'a str,
    pub page: u32,
    pub crop_sha256: &'a str,
    pub character_index: usize,
}

#[derive(Debug, Serialize)]
pub struct ClassMatch<'a> {
    #[serde(flatten)]
    pub nearest: DonorMatch<'a>,
    pub supporting_crops: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchDecision {
    Candidate,
    NoExemplars,
    NoCompetitor,
    TooDistant,
    Ambiguous,
    InsufficientSupport,
}

#[derive(Debug, Serialize)]
pub struct GlyphMatch<'a> {
    pub page: u32,
    pub crop_sha256: &'a str,
    pub character_index: usize,
    pub reference: char,
    pub primary_prediction: char,
    pub primary_line_score: f64,
    pub nearest: Option<ClassMatch<'a>>,
    pub runner_up: Option<ClassMatch<'a>>,
    pub reference_class_available: bool,
    pub decision: MatchDecision,
}

#[derive(Debug, Default, Serialize)]
pub struct MatchSummary {
    pub queries: usize,
    pub reference_class_available: usize,
    pub primary_correct: usize,
    pub candidates: usize,
    pub correct_candidates: usize,
    pub corrected_primary_errors: usize,
    pub changed_correct_primary: usize,
    pub changed_wrong_to_wrong: usize,
}

#[derive(Debug, Serialize)]
pub struct MatchReport<'a> {
    pub schema_version: u32,
    pub query_pdf_sha256: &'a str,
    pub donor_pdf_sha256: Vec<&'a str>,
    pub policy: MatchPolicy,
    pub summary: MatchSummary,
    pub matches: Vec<GlyphMatch<'a>>,
}

#[derive(Default)]
struct ClassEvidence<'a> {
    nearest: Option<DonorMatch<'a>>,
    supporting: BTreeSet<&'a str>,
}

impl<'a> ClassEvidence<'a> {
    fn observe(&mut self, donor: DonorMatch<'a>, limit: f64) {
        if donor.distance <= limit {
            self.supporting.insert(donor.crop_sha256);
        }
        if self
            .nearest
            .as_ref()
            .is_none_or(|prior| donor.distance < prior.distance)
        {
            self.nearest = Some(donor);
        }
    }

    fn finish(self) -> Option<ClassMatch<'a>> {
        Some(ClassMatch {
            nearest: self.nearest?,
            supporting_crops: self.supporting.len(),
        })
    }
}

fn decide(
    nearest: Option<&ClassMatch<'_>>,
    runner: Option<&ClassMatch<'_>>,
    policy: MatchPolicy,
) -> MatchDecision {
    let Some(best) = nearest else {
        return MatchDecision::NoExemplars;
    };
    let Some(other) = runner else {
        return MatchDecision::NoCompetitor;
    };
    if best.nearest.distance > policy.max_distance {
        return MatchDecision::TooDistant;
    }
    if other.nearest.distance - best.nearest.distance <= policy.min_margin {
        return MatchDecision::Ambiguous;
    }
    if best.supporting_crops < policy.min_independent_crops {
        return MatchDecision::InsufficientSupport;
    }
    MatchDecision::Candidate
}

fn classify<'a>(
    query_pdf: &str,
    query: &'a ReviewedGlyph,
    banks: &[&'a ReviewedGlyphBank],
    policy: MatchPolicy,
) -> GlyphMatch<'a> {
    let mut labels = BTreeMap::<char, ClassEvidence<'a>>::new();
    for bank in banks {
        for donor in &bank.glyphs {
            // A whole source page is held out. Identical recognition crops
            // on different pages or documents cannot leak into training.
            if (bank.pdf_sha256 == query_pdf && donor.page == query.page)
                || donor.crop_sha256 == query.crop_sha256
            {
                continue;
            }
            let evidence = DonorMatch {
                label: donor.label,
                distance: query.shape.distance(&donor.shape),
                pdf_sha256: &bank.pdf_sha256,
                page: donor.page,
                crop_sha256: &donor.crop_sha256,
                character_index: donor.character_index,
            };
            labels
                .entry(donor.label)
                .or_default()
                .observe(evidence, policy.max_distance);
        }
    }
    let reference_class_available = labels.contains_key(&query.label);
    let mut classes = labels
        .into_values()
        .filter_map(ClassEvidence::finish)
        .collect::<Vec<_>>();
    classes.sort_by(|a, b| {
        a.nearest
            .distance
            .total_cmp(&b.nearest.distance)
            .then(a.nearest.label.cmp(&b.nearest.label))
    });
    let mut classes = classes.into_iter();
    let nearest = classes.next();
    let runner_up = classes.next();
    let decision = decide(nearest.as_ref(), runner_up.as_ref(), policy);
    GlyphMatch {
        page: query.page,
        crop_sha256: &query.crop_sha256,
        character_index: query.character_index,
        reference: query.label,
        primary_prediction: query.prediction,
        primary_line_score: query.line_score,
        nearest,
        runner_up,
        reference_class_available,
        decision,
    }
}

impl MatchSummary {
    fn observe(&mut self, result: &GlyphMatch<'_>) {
        self.queries += 1;
        let primary_correct = result.primary_prediction == result.reference;
        self.primary_correct += usize::from(primary_correct);
        self.reference_class_available += usize::from(result.reference_class_available);
        if result.decision != MatchDecision::Candidate {
            return;
        }
        let Some(best) = &result.nearest else {
            return;
        };
        self.candidates += 1;
        let correct = best.nearest.label == result.reference;
        self.correct_candidates += usize::from(correct);
        self.corrected_primary_errors += usize::from(correct && !primary_correct);
        self.changed_correct_primary += usize::from(!correct && primary_correct);
        self.changed_wrong_to_wrong += usize::from(
            !correct && !primary_correct && best.nearest.label != result.primary_prediction,
        );
    }
}

fn validate_bank(bank: &ReviewedGlyphBank) -> Result<(), Error> {
    let digest = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
    if bank.schema_version != 1
        || !digest(&bank.pdf_sha256)
        || bank.reference_provenance.trim().is_empty()
    {
        return Err(Error::Invalid("invalid glyph bank provenance".into()));
    }
    let mut occurrences = BTreeSet::new();
    for glyph in &bank.glyphs {
        if glyph.page == 0
            || !digest(&glyph.crop_sha256)
            || glyph.label.is_whitespace()
            || glyph.prediction.is_whitespace()
            || !(0.0..=1.0).contains(&glyph.line_score)
            || !occurrences.insert((glyph.page, glyph.detection_index, glyph.character_index))
        {
            return Err(Error::Invalid(
                "invalid or duplicated reviewed glyph occurrence".into(),
            ));
        }
    }
    Ok(())
}

/// Compare withheld source pages to labeled donors, retaining competing
/// labels and abstention reasons. No importer draft is modified.
///
/// # Errors
/// Rejects malformed reference banks and invalid policy.
pub fn evaluate<'a>(
    query: &'a ReviewedGlyphBank,
    banks: &[&'a ReviewedGlyphBank],
    pages: &BTreeSet<u32>,
    policy: MatchPolicy,
) -> Result<MatchReport<'a>, Error> {
    policy.validate()?;
    validate_bank(query)?;
    for bank in banks {
        validate_bank(bank)?;
    }
    if pages.contains(&0) {
        return Err(Error::Invalid("query pages start at one".into()));
    }
    let mut summary = MatchSummary::default();
    let matches = query
        .glyphs
        .iter()
        .filter(|g| pages.is_empty() || pages.contains(&g.page))
        .map(|g| {
            let result = classify(&query.pdf_sha256, g, banks, policy);
            summary.observe(&result);
            result
        })
        .collect();
    Ok(MatchReport {
        schema_version: 1,
        query_pdf_sha256: &query.pdf_sha256,
        donor_pdf_sha256: banks.iter().map(|b| b.pdf_sha256.as_str()).collect(),
        policy,
        summary,
        matches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geometry::Point, glyph_appearance::GlyphShape, pixel_regions::InkBounds,
        reviewed_glyphs::ReviewStats,
    };
    use image::{GrayImage, Luma};
    fn bank(pdf: u8, exemplars: &[(u32, u8, char, bool)]) -> Result<ReviewedGlyphBank, Error> {
        let mut glyphs = Vec::new();
        for (index, &(page, crop, label, vertical)) in exemplars.iter().enumerate() {
            let image = GrayImage::from_fn(12, 12, |x, y| {
                Luma([if (vertical && x < 4) || (!vertical && y < 4) {
                    0
                } else {
                    255
                }])
            });
            glyphs.push(ReviewedGlyph {
                page,
                crop_sha256: format!("{crop:064x}"),
                detection_index: index,
                character_index: 0,
                label,
                prediction: 'B',
                line_score: 0.999,
                ink_quad: [Point(0., 0.); 4],
                shape: GlyphShape::measure(
                    &image,
                    InkBounds {
                        left: 0,
                        top: 0,
                        right: 12,
                        bottom: 12,
                    },
                )?,
            });
        }
        Ok(ReviewedGlyphBank {
            schema_version: 1,
            pdf_sha256: format!("{pdf:064x}"),
            model: "fixture".into(),
            reference_provenance: "fixture review".into(),
            stats: ReviewStats::default(),
            glyphs,
        })
    }
    #[test]
    fn excludes_held_out_page_and_identical_crop_and_needs_competitor() -> Result<(), Error> {
        let q = bank(1, &[(1, 1, 'A', true)])?;
        let same_doc = bank(1, &[(1, 2, 'B', true), (2, 3, 'A', true)])?;
        let other = bank(
            2,
            &[(1, 1, 'B', true), (2, 4, 'A', true), (2, 5, 'B', false)],
        )?;
        let policy = MatchPolicy {
            max_distance: 0.1,
            min_margin: 0.02,
            min_independent_crops: 2,
        };
        let report = evaluate(&q, &[&same_doc, &other], &BTreeSet::new(), policy)?;
        assert_eq!(report.summary.corrected_primary_errors, 1);
        assert_eq!(
            report.matches[0]
                .nearest
                .as_ref()
                .map(|m| m.supporting_crops),
            Some(2)
        );
        let report = evaluate(&q, &[&same_doc], &BTreeSet::new(), policy)?;
        assert_eq!(report.matches[0].decision, MatchDecision::NoCompetitor);
        Ok(())
    }
    #[test]
    fn indistinguishable_labels_abstain_even_from_many_donors() -> Result<(), Error> {
        let q = bank(1, &[(1, 1, 'A', true)])?;
        let other = bank(
            2,
            &[(1, 2, 'A', true), (1, 3, 'B', true), (1, 4, 'A', true)],
        )?;
        let report = evaluate(
            &q,
            &[&other],
            &BTreeSet::new(),
            MatchPolicy {
                max_distance: 0.1,
                min_margin: 0.0,
                min_independent_crops: 2,
            },
        )?;
        assert_eq!(report.matches[0].decision, MatchDecision::Ambiguous);
        assert_eq!(report.summary.candidates, 0);
        Ok(())
    }
}
