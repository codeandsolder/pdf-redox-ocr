//! Hold-out glyph recognition using independently reviewed source shapes.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    Error,
    glyph_appearance::GlyphShape,
    recognition::RasterLocation,
    reviewed_glyphs::{ReviewedGlyph, ReviewedGlyphBank},
};

/// Compare only like-for-like actual appearance measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum AppearanceMode {
    #[default]
    Primary,
    Source,
}

impl AppearanceMode {
    fn sample(self, glyph: &ReviewedGlyph) -> Option<(&GlyphShape, &str)> {
        match self {
            Self::Primary => Some((&glyph.shape, &glyph.crop_sha256)),
            Self::Source => glyph.source_appearance.as_ref().map(|appearance| {
                (
                    &appearance.shape,
                    appearance.normalized_line_sha256.as_str(),
                )
            }),
        }
    }
}

/// Distance and ambiguity policy, separate from any model confidence.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct MatchPolicy {
    pub appearance: AppearanceMode,
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
    pub raster_location: RasterLocation,
    pub raster_sha256: &'a str,
    pub detection_index: usize,
    pub crop_sha256: &'a str,
    pub appearance_sha256: &'a str,
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
    NoQueryAppearance,
    NoExemplars,
    NoCompetitor,
    TooDistant,
    Ambiguous,
    InsufficientSupport,
}

#[derive(Debug, Serialize)]
pub struct GlyphMatch<'a> {
    pub page: u32,
    pub raster_location: RasterLocation,
    pub raster_sha256: &'a str,
    pub detection_index: usize,
    pub crop_sha256: &'a str,
    pub appearance_sha256: Option<&'a str>,
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
    pub query_appearances: usize,
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
    pub source_dpi: Option<u32>,
    pub summary: MatchSummary,
    pub matches: Vec<GlyphMatch<'a>>,
}

#[derive(Default)]
struct ClassEvidence<'a> {
    nearest: Option<DonorMatch<'a>>,
    supporting: BTreeSet<&'a str>,
}

impl<'a> ClassEvidence<'a> {
    fn observe(&mut self, donor: DonorMatch<'a>, digest: &'a str, limit: f64) {
        if donor.distance <= limit {
            self.supporting.insert(digest);
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
    let query_sample = policy.appearance.sample(query);
    if let Some((query_shape, query_digest)) = query_sample {
        for bank in banks {
            for donor in &bank.glyphs {
                let Some((donor_shape, donor_digest)) = policy.appearance.sample(donor) else {
                    continue;
                };
                // Entire source page and identical input crops are held out.
                // Source matching additionally excludes identical resampled lines.
                if (bank.pdf_sha256 == query_pdf && donor.page == query.page)
                    || donor.crop_sha256 == query.crop_sha256
                    || donor_digest == query_digest
                {
                    continue;
                }
                let evidence = DonorMatch {
                    label: donor.label,
                    distance: query_shape.distance(donor_shape),
                    pdf_sha256: &bank.pdf_sha256,
                    page: donor.page,
                    raster_location: donor.raster_location,
                    raster_sha256: &donor.raster_sha256,
                    detection_index: donor.detection_index,
                    crop_sha256: &donor.crop_sha256,
                    appearance_sha256: donor_digest,
                    character_index: donor.character_index,
                };
                labels.entry(donor.label).or_default().observe(
                    evidence,
                    donor_digest,
                    policy.max_distance,
                );
            }
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
    let decision = if query_sample.is_some() {
        decide(nearest.as_ref(), runner_up.as_ref(), policy)
    } else {
        MatchDecision::NoQueryAppearance
    };
    GlyphMatch {
        page: query.page,
        raster_location: query.raster_location,
        raster_sha256: &query.raster_sha256,
        detection_index: query.detection_index,
        crop_sha256: &query.crop_sha256,
        appearance_sha256: query_sample.map(|(_, digest)| digest),
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
        self.query_appearances += usize::from(result.appearance_sha256.is_some());
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

fn validate_source_sampling(bank: &ReviewedGlyphBank) -> Result<(), Error> {
    let mut source_lines = BTreeMap::new();
    let mut source_groups = BTreeSet::new();
    let mut count = 0;
    for glyph in &bank.glyphs {
        let Some(source) = &glyph.source_appearance else {
            continue;
        };
        count += 1;
        let key = (glyph.page, glyph.raster_location, glyph.detection_index);
        let digest = &source.normalized_line_sha256;
        if digest.len() != 64
            || !digest.bytes().all(|b| b.is_ascii_hexdigit())
            || source.line_dimensions.0 == 0
            || source.line_dimensions.1 == 0
            || source_lines
                .insert(key, (digest, source.line_dimensions))
                .is_some_and(|prior| prior != (digest, source.line_dimensions))
            || !source_groups.insert((key, source.source_group_index))
        {
            return Err(Error::Invalid(
                "invalid source appearance provenance".into(),
            ));
        }
    }
    if bank.source_sampling.as_ref().map_or(count != 0, |s| {
        !(36..=2400).contains(&s.dpi)
            || s.matched_glyphs != count
            || s.sampled_lines > s.attempted_lines
            || s.sampled_lines.checked_add(s.failures.len()) != Some(s.attempted_lines)
            || source_lines.len() > s.sampled_lines
    }) {
        return Err(Error::Invalid("inconsistent source glyph sampling".into()));
    }
    Ok(())
}

fn validate_bank(bank: &ReviewedGlyphBank) -> Result<(), Error> {
    let digest = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
    if bank.schema_version != 3
        || !digest(&bank.pdf_sha256)
        || bank.reference_provenance.trim().is_empty()
    {
        return Err(Error::Invalid("invalid glyph bank provenance".into()));
    }
    validate_source_sampling(bank)?;
    let mut occurrences = BTreeSet::new();
    let mut rasters = BTreeMap::new();
    for glyph in &bank.glyphs {
        if glyph.page == 0
            || !digest(&glyph.raster_sha256)
            || !digest(&glyph.crop_sha256)
            || rasters
                .insert((glyph.page, glyph.raster_location), &glyph.raster_sha256)
                .is_some_and(|prior| prior != &glyph.raster_sha256)
            || glyph.label.is_whitespace()
            || glyph.prediction.is_whitespace()
            || !(0.0..=1.0).contains(&glyph.line_score)
            || !occurrences.insert((
                glyph.page,
                glyph.raster_location,
                glyph.detection_index,
                glyph.character_index,
            ))
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
    let source_dpi = if policy.appearance == AppearanceMode::Source {
        let dpi = query
            .source_sampling
            .as_ref()
            .ok_or_else(|| Error::Invalid("query bank has no source sampling".into()))?
            .dpi;
        if banks
            .iter()
            .any(|bank| bank.source_sampling.as_ref().map(|s| s.dpi) != Some(dpi))
        {
            return Err(Error::Invalid(
                "source appearance banks require the same sampling DPI".into(),
            ));
        }
        Some(dpi)
    } else {
        None
    };
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
        schema_version: 3,
        query_pdf_sha256: &query.pdf_sha256,
        donor_pdf_sha256: banks.iter().map(|b| b.pdf_sha256.as_str()).collect(),
        policy,
        source_dpi,
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
                raster_location: RasterLocation::WholePage { residual: false },
                raster_sha256: format!("{page:064x}"),
                crop_sha256: format!("{crop:064x}"),
                detection_index: index,
                character_index: 0,
                label,
                prediction: 'B',
                line_score: 0.999,
                ink_quad: [Point(0., 0.); 4],
                source_appearance: None,
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
            schema_version: 3,
            pdf_sha256: format!("{pdf:064x}"),
            model: "fixture".into(),
            reference_provenance: "fixture review".into(),
            stats: ReviewStats::default(),
            glyphs,
            source_sampling: None,
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
            appearance: AppearanceMode::Primary,
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
                appearance: AppearanceMode::Primary,
                max_distance: 0.1,
                min_margin: 0.0,
                min_independent_crops: 2,
            },
        )?;
        assert_eq!(report.matches[0].decision, MatchDecision::Ambiguous);
        assert_eq!(report.summary.candidates, 0);
        Ok(())
    }

    #[test]
    fn raster_scopes_disambiguate_detector_ids_but_not_duplicate_occurrences() -> Result<(), Error>
    {
        let mut reviewed = bank(
            1,
            &[(1, 1, 'A', true), (1, 2, 'B', false), (1, 3, 'C', true)],
        )?;
        reviewed.glyphs[1].raster_location = RasterLocation::Region {
            seed_paint_index: 4,
        };
        reviewed.glyphs[1].detection_index = 0;
        reviewed.glyphs[2].raster_location = RasterLocation::Region {
            seed_paint_index: 9,
        };
        reviewed.glyphs[2].detection_index = 0;
        validate_bank(&reviewed)?;
        reviewed.glyphs[2].raster_location = reviewed.glyphs[1].raster_location;
        assert!(validate_bank(&reviewed).is_err());
        reviewed.glyphs[2].detection_index = 1;
        validate_bank(&reviewed)?;
        reviewed.glyphs[2].raster_sha256 = "ab".repeat(32);
        assert!(validate_bank(&reviewed).is_err());
        Ok(())
    }

    fn attach_source(bank: &mut ReviewedGlyphBank, dpi: u32, digests: &[Option<u8>]) {
        use crate::reviewed_glyphs::{SourceGlyphAppearance, SourceSampling};
        let matched_glyphs = digests.iter().filter(|d| d.is_some()).count();
        for (glyph, digest) in bank.glyphs.iter_mut().zip(digests) {
            glyph.source_appearance = digest.map(|digest| SourceGlyphAppearance {
                normalized_line_sha256: format!("{digest:064x}"),
                line_dimensions: (100, 20),
                source_group_index: 0,
                ink_quad: glyph.ink_quad,
                shape: glyph.shape.clone(),
            });
        }
        bank.source_sampling = Some(SourceSampling {
            dpi,
            attempted_lines: bank.glyphs.len(),
            sampled_lines: bank.glyphs.len(),
            matched_glyphs,
            failures: vec![],
        });
    }

    #[test]
    fn source_matching_never_falls_back_or_counts_identical_source_crops() -> Result<(), Error> {
        let mut q = bank(1, &[(1, 1, 'A', true), (1, 2, 'A', true)])?;
        attach_source(&mut q, 600, &[Some(10), None]);
        let mut donors = bank(
            2,
            &[
                (1, 3, 'A', true),
                (1, 4, 'A', true),
                (1, 5, 'A', true),
                (1, 6, 'B', false),
            ],
        )?;
        attach_source(&mut donors, 600, &[Some(10), Some(11), Some(12), Some(13)]);
        let policy = MatchPolicy {
            appearance: AppearanceMode::Source,
            max_distance: 0.1,
            min_margin: 0.02,
            min_independent_crops: 2,
        };
        let report = evaluate(&q, &[&donors], &BTreeSet::new(), policy)?;
        assert_eq!(report.summary.query_appearances, 1);
        assert_eq!(report.summary.corrected_primary_errors, 1);
        assert_eq!(
            report.matches[0]
                .nearest
                .as_ref()
                .map(|m| m.supporting_crops),
            Some(2)
        );
        assert_eq!(report.matches[1].decision, MatchDecision::NoQueryAppearance);
        attach_source(&mut donors, 300, &[Some(10), Some(11), Some(12), Some(13)]);
        assert!(evaluate(&q, &[&donors], &BTreeSet::new(), policy).is_err());
        Ok(())
    }
}
