//! Recognition-only second-pass diagnostics for uncertain lines in an OCR draft.
//!
//! Only the source PDF supplies pixels. All proposed text changes stay in a
//! separate zero-confidence, unreviewed draft until independently reviewed.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Instant,
};

use image::RgbImage;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Error, PageNumber,
    draft::{DraftPage, OcrDraft, OcrLine},
    geometry::{DisplayMapper, PageBox, Point, Quad, QuadBounds},
    line_frame::{LineAxis, LineFrame, SampleWindow},
    pdf_metadata::PdfMetadata,
    rasterize::{calculate_crop, render_crop, sample_whitened},
    refinement_recognition::{Recognition, canonical},
};

/// Second-pass selection and resource limits.
#[derive(Clone, Copy, Debug)]
pub struct RefinementPolicy {
    /// Maximum primary confidence considered uncertain.
    pub max_primary_score: f64,
    /// Minimum number of Unicode scalar values to refine.
    pub min_chars: usize,
    /// Native source raster DPI.
    pub dpi: u32,
    /// Maximum total rendered and rectified pixels per line.
    pub max_line_pixels: u64,
    /// Maximum total rendered and rectified pixels per page.
    pub max_page_pixels: u64,
}

impl Default for RefinementPolicy {
    fn default() -> Self {
        Self {
            max_primary_score: 0.97,
            min_chars: 2,
            dpi: 600,
            max_line_pixels: 2_000_000,
            max_page_pixels: 32_000_000,
        }
    }
}

impl RefinementPolicy {
    /// Validate thresholds before running or loading models.
    ///
    /// # Errors
    /// Rejects invalid confidence, selection, or render configuration.
    pub fn validate(self) -> Result<Self, Error> {
        if !self.max_primary_score.is_finite()
            || !(0.0..=1.0).contains(&self.max_primary_score)
            || self.min_chars == 0
            || !(36..=2400).contains(&self.dpi)
            || self.max_line_pixels == 0
            || self.max_line_pixels > self.max_page_pixels
        {
            return Err(Error::Invalid(
                "invalid secondary OCR refinement limits".into(),
            ));
        }
        Ok(self)
    }
}

/// Select either low-confidence observations or explicit source-draft line indices.
///
/// Explicit targets bypass confidence and minimum-text-length thresholds, but
/// must refer to real, nonblank observations in the unchanged original draft.
#[derive(Clone, Copy, Debug)]
pub enum RefinementScope<'a> {
    /// Usual automatic candidate selection on all or a subset of draft pages.
    ByConfidence {
        pages: Option<&'a BTreeSet<PageNumber>>,
    },
    /// Research selection of exact zero-based line indices keyed by PDF page.
    ExplicitLines(&'a BTreeMap<PageNumber, BTreeSet<usize>>),
}

/// Exact source sampling provenance.
#[derive(Debug, Serialize)]
pub struct SourceCropEvidence {
    pub rgb_sha256: String,
    pub dimensions: (u32, u32),
    pub rendered_dimensions: (u32, u32),
    pub rendered_quad: Quad,
    pub charged_pixels: u64,
}

/// Per-line diagnostic result, including explicit geometry/budget failures.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RefinementOutcome {
    Recognized {
        crop: SourceCropEvidence,
        recognition: Recognition,
    },
    Skipped {
        reason: String,
    },
}

/// Report for a single original OCR observation.
#[derive(Debug, Serialize)]
pub struct RefinementProposal {
    pub page: u32,
    pub line_index: usize,
    pub quad: Quad,
    pub primary_text: String,
    pub primary_score: f64,
    pub outcome: RefinementOutcome,
}

impl RefinementProposal {
    fn revised_text(&self) -> Option<&str> {
        let RefinementOutcome::Recognized { recognition, .. } = &self.outcome else {
            return None;
        };
        recognition
            .decision
            .text()
            .filter(|text| meaningful_change(&self.primary_text, text))
    }
}

/// Summary of attempted recognition-only revisions.
#[derive(Debug, Clone, Serialize)]
pub struct RefinementSummary {
    pub source_lines: usize,
    pub selected_lines: usize,
    pub revised_predictions: usize,
    pub recognized_lines: usize,
    pub skipped_lines: usize,
    pub charged_pixels: u64,
    pub seconds: f64,
    pub source_pages: usize,
    pub pages_refined: usize,
}

/// Research-only provenance-bound comparison report.
#[derive(Debug, Serialize)]
pub struct RefinementReport {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub source_model: String,
    pub secondary_model: String,
    pub selection_mode: &'static str,
    pub max_primary_score: Option<f64>,
    pub min_chars: Option<usize>,
    pub dpi: u32,
    pub max_line_pixels: u64,
    pub max_page_pixels: u64,
    pub summary: RefinementSummary,
    pub proposals: Vec<RefinementProposal>,
}

/// Ignore only whitespace; retain literal engineering symbols, punctuation,
/// superscripts and full-width forms.
#[must_use]
pub fn meaningful_change(left: &str, right: &str) -> bool {
    canonical(left) != canonical(right)
}

struct SourcePlan {
    frame: LineFrame,
    render: crate::rasterize::CropRequest,
    cost: u64,
}

impl SourcePlan {
    fn new(line: &OcrLine, mapper: DisplayMapper, policy: RefinementPolicy) -> Result<Self, Error> {
        let frame = LineFrame::new(
            line.quad,
            |points| points * f64::from(policy.dpi) / 72.0,
            policy.max_line_pixels,
            LineAxis::ReadingOrder,
        )?;
        // Reject out-of-page or folded geometry before rendering. Preserve the
        // final draft's actual reading order, even for tall single characters.
        SampleWindow::new(
            frame,
            |p| {
                let Point(u, v) = mapper.to_raster(p);
                (-1e-8..=1.0 + 1e-8).contains(&u) && (-1e-8..=1.0 + 1e-8).contains(&v)
            },
            0,
            policy.max_line_pixels,
        )?;
        let bounds = QuadBounds::new(&frame.corners);
        let region = PageBox::new([bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y])?;
        let render = calculate_crop(mapper, region, policy.dpi, policy.max_line_pixels)?;
        let cost = render
            .pixels()
            .checked_add(u64::from(frame.width) * u64::from(frame.height))
            .ok_or_else(|| Error::Invalid("source crop cost overflow".into()))?;
        if cost > policy.max_line_pixels {
            return Err(Error::Invalid(
                "source refinement line pixel budget exceeded".into(),
            ));
        }
        Ok(Self {
            frame,
            render,
            cost,
        })
    }

    fn sample(
        &self,
        pdf: &Path,
        page: PageNumber,
    ) -> Result<(RgbImage, SourceCropEvidence), Error> {
        let crop = render_crop(pdf, page, &self.render)?;
        let mapper = crop.pixel_mapper();
        let frame = self.frame;
        let image = RgbImage::from_fn(frame.width, frame.height, |x, y| {
            let p = frame.to_source(Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
            sample_whitened(&crop.rgb, mapper.to_raster(p))
        });
        let (w, h) = crop.rgb.dimensions();
        let evidence = SourceCropEvidence {
            rgb_sha256: hex::encode(Sha256::digest(image.as_raw())),
            dimensions: image.dimensions(),
            rendered_dimensions: (w, h),
            rendered_quad: [
                Point(0.0, f64::from(h)),
                Point(f64::from(w), f64::from(h)),
                Point(f64::from(w), 0.0),
                Point(0.0, 0.0),
            ]
            .map(|p| mapper.to_pdf(p)),
            charged_pixels: self.cost,
        };
        Ok((image, evidence))
    }
}

/// Select uncertain lines without loading a model or rendering a page.
///
/// # Errors
/// Rejects invalid input page identity, confidence or page selections.
pub fn refinement_plan(
    draft: &OcrDraft,
    scope: RefinementScope<'_>,
    page_count: usize,
    policy: RefinementPolicy,
) -> Result<BTreeMap<PageNumber, Vec<usize>>, Error> {
    let policy = policy.validate()?;
    if draft.schema_version != 3 {
        return Err(Error::Invalid("unsupported OCR importer schema".into()));
    }
    let mut seen = BTreeSet::new();
    for page in &draft.pages {
        let number = PageNumber::new(page.page)?;
        if !usize::try_from(page.page).is_ok_and(|p| p <= page_count) || !seen.insert(number) {
            return Err(Error::Invalid("invalid or duplicate draft page".into()));
        }
        for line in &page.lines {
            if !line.confidence.is_finite() || !(0.0..=1.0).contains(&line.confidence) {
                return Err(Error::Invalid("invalid primary OCR confidence".into()));
            }
        }
    }

    let mut plan = BTreeMap::new();
    match scope {
        RefinementScope::ByConfidence { pages } => {
            if pages.is_some_and(|pages| pages.is_empty() || !pages.is_subset(&seen)) {
                return Err(Error::Invalid(
                    "requested refinement page missing from draft".into(),
                ));
            }
            for page in &draft.pages {
                let number = PageNumber::new(page.page)?;
                if pages.is_some_and(|pages| !pages.contains(&number)) {
                    continue;
                }
                let indices = page
                    .lines
                    .iter()
                    .enumerate()
                    .filter_map(|(index, line)| {
                        (line.confidence <= policy.max_primary_score
                            && line.text.trim().chars().count() >= policy.min_chars)
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();
                if !indices.is_empty() {
                    plan.insert(number, indices);
                }
            }
        }
        RefinementScope::ExplicitLines(targets) => {
            if targets.is_empty() {
                return Err(Error::Invalid("empty explicit OCR targets".into()));
            }
            for (&number, indices) in targets {
                let page = draft
                    .pages
                    .iter()
                    .find(|page| page.page == number.get())
                    .ok_or_else(|| Error::Invalid("explicit OCR target page missing".into()))?;
                if indices.is_empty() {
                    return Err(Error::Invalid("empty explicit OCR target line set".into()));
                }
                for &index in indices {
                    let line = page.lines.get(index).ok_or_else(|| {
                        Error::Invalid("explicit OCR line index out of range".into())
                    })?;
                    if line.text.trim().is_empty() {
                        return Err(Error::Invalid(
                            "cannot refine a blank OCR observation".into(),
                        ));
                    }
                }
                plan.insert(number, indices.iter().copied().collect());
            }
        }
    }
    Ok(plan)
}

/// Render only explicitly budgeted visible source regions, preserving target
/// identities even when a crop cannot be sampled.
fn proposals_for_page(
    pdf: &Path,
    number: PageNumber,
    page: &DraftPage,
    indices: &[usize],
    mapper: DisplayMapper,
    policy: RefinementPolicy,
    recognize: &mut impl FnMut(&RgbImage) -> Result<Recognition, Error>,
) -> Result<(Vec<RefinementProposal>, u64), Error> {
    let mut remaining = policy.max_page_pixels;
    let mut proposals = Vec::with_capacity(indices.len());
    for &index in indices {
        let line = &page.lines[index];
        let sampled = SourcePlan::new(line, mapper, policy).and_then(|plan| {
            remaining = remaining.checked_sub(plan.cost).ok_or_else(|| {
                Error::Invalid("source refinement page pixel budget exhausted".into())
            })?;
            plan.sample(pdf, number)
        });
        let outcome = match sampled {
            Ok((image, crop)) => RefinementOutcome::Recognized {
                crop,
                recognition: recognize(&image)?,
            },
            Err(error) => RefinementOutcome::Skipped {
                reason: error.to_string(),
            },
        };
        proposals.push(RefinementProposal {
            page: number.get(),
            line_index: index,
            quad: line.quad,
            primary_text: line.text.clone(),
            primary_score: line.confidence,
            outcome,
        });
    }
    Ok((proposals, policy.max_page_pixels - remaining))
}

/// Primary and secondary evidence sources for a single refinement run.
#[derive(Clone, Copy)]
pub struct RefinementRequest<'a> {
    /// Immutable exact source PDF.
    pub pdf: &'a Path,
    /// Parsed original page geometry.
    pub metadata: &'a PdfMetadata,
    /// Exact-PDF-bound OCR draft.
    pub source: &'a OcrDraft,
    /// Confidence-based or explicitly indexed input observations.
    pub scope: RefinementScope<'a>,
    /// Human-readable secondary model identity for traceability.
    pub secondary_model: &'a str,
    /// Selection and source render limits.
    pub policy: RefinementPolicy,
}

/// Apply a research-only recognition-only second pass to the original PDF.
///
/// The callback receives rectified RGB crops in actual draft reading order.
/// Only the original visible PDF is rendered; no temporary masking or
/// orientation guessing is involved.
///
/// # Errors
/// Rejects changed PDFs, invalid selection evidence or model failures.
/// Invalid individual sampling regions are reported as skipped.
pub fn refine_draft(
    request: RefinementRequest<'_>,
    mut recognize: impl FnMut(&RgbImage) -> Result<Recognition, Error>,
) -> Result<(RefinementReport, OcrDraft), Error> {
    let RefinementRequest {
        pdf,
        metadata,
        source,
        scope,
        secondary_model,
        policy,
    } = request;
    let started = Instant::now();
    source.verify_pdf(pdf)?;
    let policy = policy.validate()?;
    let plan = refinement_plan(source, scope, metadata.page_count(), policy)?;
    let mappers = metadata.select_mappers(&plan.keys().copied().collect::<Vec<_>>())?;
    let mut proposals = Vec::new();
    let mut charged_pixels = 0_u64;

    for (&number, indices) in &plan {
        let page = source
            .pages
            .iter()
            .find(|entry| entry.page == number.get())
            .ok_or_else(|| Error::Invalid("refinement page disappeared".into()))?;
        let mapper = *mappers
            .get(&number)
            .ok_or_else(|| Error::Invalid("missing refinement page geometry".into()))?;
        let (page_proposals, pixels) =
            proposals_for_page(pdf, number, page, indices, mapper, policy, &mut recognize)?;
        proposals.extend(page_proposals);
        charged_pixels = charged_pixels.saturating_add(pixels);
    }
    let mut experimental = source.clone();
    experimental.model = format!(
        "{} + research-only {secondary_model} hypotheses",
        source.model
    );
    for page in &mut experimental.pages {
        for line in &mut page.lines {
            line.confidence = 0.0;
            line.reviewed = false;
        }
    }
    for proposal in &proposals {
        if let Some(text) = proposal.revised_text() {
            let page = experimental
                .pages
                .iter_mut()
                .find(|p| p.page == proposal.page)
                .ok_or_else(|| Error::Invalid("refinement proposal page missing".into()))?;
            let line: &mut OcrLine = page
                .lines
                .get_mut(proposal.line_index)
                .ok_or_else(|| Error::Invalid("refinement proposal line missing".into()))?;
            text.clone_into(&mut line.text);
        }
    }
    let by_confidence = matches!(scope, RefinementScope::ByConfidence { .. });
    source.verify_pdf(pdf)?;
    let recognized_lines = proposals
        .iter()
        .filter(|p| matches!(p.outcome, RefinementOutcome::Recognized { .. }))
        .count();
    let report = RefinementReport {
        schema_version: 4,
        pdf_sha256: source.pdf_sha256.clone(),
        source_model: source.model.clone(),
        secondary_model: secondary_model.to_owned(),
        selection_mode: if by_confidence {
            "confidence"
        } else {
            "explicit-lines"
        },
        max_primary_score: by_confidence.then_some(policy.max_primary_score),
        min_chars: by_confidence.then_some(policy.min_chars),
        dpi: policy.dpi,
        max_line_pixels: policy.max_line_pixels,
        max_page_pixels: policy.max_page_pixels,
        summary: RefinementSummary {
            source_lines: source.pages.iter().map(|page| page.lines.len()).sum(),
            selected_lines: proposals.len(),
            revised_predictions: proposals
                .iter()
                .filter(|p| p.revised_text().is_some())
                .count(),
            recognized_lines,
            skipped_lines: proposals.len() - recognized_lines,
            charged_pixels,
            seconds: (started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
            source_pages: source.pages.len(),
            pages_refined: plan.len(),
        },
        proposals,
    };
    Ok((report, experimental))
}

#[cfg(test)]
mod tests {
    use super::{
        Recognition, RefinementPolicy, RefinementRequest, RefinementScope, SourcePlan,
        meaningful_change, refine_draft, refinement_plan,
    };
    use crate::{
        PageNumber,
        draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
        geometry::{DisplayMapper, PageBox, Point, Rotation},
        pdf_metadata::PdfMetadata,
        refinement_recognition::RefinementDecision,
    };
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
    };

    #[test]
    fn format_equivalence_preserves_semantically_distinct_tokens() {
        assert!(meaningful_change("12 ＶＡＣ", "12VAC"));
        assert!(meaningful_change("mm²", "mm2"));
        assert!(!meaningful_change("A \t B", "AB"));
        for (first, second) in [("20A", "20V"), ("AB", "ab"), ("Ω", "ohm"), ("10.2", "102")] {
            assert!(meaningful_change(first, second), "{first:?}, {second:?}");
        }
    }

    #[test]
    fn reading_order_and_budget_survive_all_page_rotations() -> Result<(), crate::Error> {
        let bounds = PageBox::new([30., 40., 830., 640.])?;
        let policy = RefinementPolicy::default();
        for angle in [0, 90, 180, 270] {
            let mapper = DisplayMapper::new(bounds, Rotation::try_from(angle)?);
            let mut line = OcrLine {
                text: "A".into(),
                confidence: 0.99,
                quad: [
                    mapper.to_pdf(0.2, 0.6),
                    mapper.to_pdf(0.22, 0.6),
                    mapper.to_pdf(0.22, 0.4),
                    mapper.to_pdf(0.2, 0.4),
                ],
                reviewed: false,
            };
            let plan = SourcePlan::new(&line, mapper, policy)?;
            assert_eq!(plan.frame.corners, line.quad);
            assert!(!plan.frame.axis_swapped);
            assert!(plan.frame.height > plan.frame.width);
            assert!(plan.cost > plan.render.pixels());
            line.quad.rotate_left(2);
            let reversed = SourcePlan::new(&line, mapper, policy)?;
            assert_eq!(reversed.frame.corners, line.quad);
            assert_eq!(plan.cost, reversed.cost);
            assert!(
                SourcePlan::new(
                    &line,
                    mapper,
                    RefinementPolicy {
                        max_line_pixels: 1,
                        ..policy
                    }
                )
                .is_err()
            );
            line.quad.swap(1, 2);
            assert!(SourcePlan::new(&line, mapper, policy).is_err());
        }
        Ok(())
    }

    #[test]
    fn planning_rejects_invalid_evidence_without_loading_models() -> Result<(), crate::Error> {
        let mut draft = OcrDraft::new(
            "00".repeat(32),
            "test".into(),
            vec![DraftPage {
                page: 1,
                lines: vec![OcrLine {
                    text: "Capacitor".into(),
                    confidence: 0.9,
                    quad: [Point(1.0, 2.0); 4],
                    reviewed: false,
                }],
                native_text_regions: Vec::new(),
            }],
        );
        let policy = RefinementPolicy::default();
        assert_eq!(
            refinement_plan(
                &draft,
                RefinementScope::ByConfidence { pages: None },
                1,
                policy
            )?
            .len(),
            1
        );
        assert!(
            refinement_plan(
                &draft,
                RefinementScope::ByConfidence {
                    pages: Some(&BTreeSet::from([PageNumber::new(2)?]))
                },
                2,
                policy
            )
            .is_err()
        );
        draft.pages.push(draft.pages[0].clone());
        assert!(
            refinement_plan(
                &draft,
                RefinementScope::ByConfidence { pages: None },
                2,
                policy
            )
            .is_err()
        );
        draft.pages.pop();
        draft.pages[0].lines[0].confidence = f64::NAN;
        assert!(
            refinement_plan(
                &draft,
                RefinementScope::ByConfidence { pages: None },
                1,
                policy
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn explicit_line_targets_bypass_score_and_length_but_validate_source()
    -> Result<(), crate::Error> {
        let page = PageNumber::new(2)?;
        let mut draft = OcrDraft::new(
            "00".repeat(32),
            "test".into(),
            vec![DraftPage {
                page: page.get(),
                lines: vec![
                    OcrLine {
                        text: "   ".into(),
                        confidence: 0.99,
                        quad: [Point(1.0, 2.0); 4],
                        reviewed: false,
                    },
                    OcrLine {
                        text: "0".into(),
                        confidence: 0.999,
                        quad: [Point(1.0, 2.0); 4],
                        reviewed: false,
                    },
                ],
                native_text_regions: Vec::new(),
            }],
        );
        let policy = RefinementPolicy {
            max_primary_score: 0.01,
            min_chars: 6,
            ..RefinementPolicy::default()
        };
        let mut targets = BTreeMap::from([(page, BTreeSet::from([1_usize]))]);
        let plan = refinement_plan(&draft, RefinementScope::ExplicitLines(&targets), 2, policy)?;
        assert_eq!(plan[&page], vec![1]);
        assert!(
            refinement_plan(
                &draft,
                RefinementScope::ByConfidence { pages: None },
                2,
                policy,
            )?
            .is_empty()
        );

        targets.insert(page, BTreeSet::from([0]));
        assert!(
            refinement_plan(&draft, RefinementScope::ExplicitLines(&targets), 2, policy,).is_err()
        );
        targets.insert(page, BTreeSet::from([20]));
        assert!(
            refinement_plan(&draft, RefinementScope::ExplicitLines(&targets), 2, policy,).is_err()
        );
        targets.clear();
        assert!(
            refinement_plan(&draft, RefinementScope::ExplicitLines(&targets), 2, policy,).is_err()
        );
        targets.insert(PageNumber::new(1)?, BTreeSet::from([0]));
        assert!(
            refinement_plan(&draft, RefinementScope::ExplicitLines(&targets), 2, policy,).is_err()
        );
        draft.pages[0].lines[1].confidence = f64::NAN;
        assert!(
            refinement_plan(&draft, RefinementScope::ExplicitLines(&targets), 2, policy,).is_err()
        );
        Ok(())
    }

    fn source_fixture() -> Result<
        (tempfile::TempDir, std::path::PathBuf, PdfMetadata, OcrDraft),
        Box<dyn std::error::Error>,
    > {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("mixed.pdf");
        fs::write(
            &path,
            include_bytes!("../tests/fixtures/mixed-paint-visible-occluded.pdf"),
        )?;
        let metadata = PdfMetadata::open(&path)?;
        let mapper = metadata.mapper(PageNumber::new(1)?)?;
        let quad = [
            mapper.to_pdf(0.2, 0.65),
            mapper.to_pdf(0.4, 0.65),
            mapper.to_pdf(0.4, 0.55),
            mapper.to_pdf(0.2, 0.55),
        ];
        let source = OcrDraft::new(
            sha256_file(&path)?,
            "primary".into(),
            vec![DraftPage {
                page: 1,
                lines: vec![OcrLine {
                    text: "CAPACITOR".into(),
                    confidence: 0.9,
                    quad,
                    reviewed: true,
                }],
                native_text_regions: Vec::new(),
            }],
        );
        Ok((temp, path, metadata, source))
    }

    #[test]
    fn sampled_half_turn_and_page_budget_are_exact() -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, path, metadata, source) = source_fixture()?;
        let mapper = metadata.mapper(PageNumber::new(1)?)?;
        let policy = RefinementPolicy {
            dpi: 144,
            ..RefinementPolicy::default()
        };
        let request = RefinementRequest {
            pdf: &path,
            metadata: &metadata,
            source: &source,
            scope: RefinementScope::ByConfidence { pages: None },
            secondary_model: "test",
            policy,
        };
        let plan = SourcePlan::new(&source.pages[0].lines[0], mapper, policy)?;
        let cost = plan.cost;
        let (forward, _) = plan.sample(&path, PageNumber::new(1)?)?;
        let mut inverse = source.pages[0].lines[0].clone();
        inverse.quad.rotate_left(2);
        let (reverse, _) =
            SourcePlan::new(&inverse, mapper, policy)?.sample(&path, PageNumber::new(1)?)?;
        assert_eq!(image::imageops::rotate180(&forward), reverse);
        let mut crowded = source.clone();
        crowded.pages[0]
            .lines
            .push(source.pages[0].lines[0].clone());
        let (budget_report, budget_draft) = refine_draft(
            RefinementRequest {
                source: &crowded,
                policy: RefinementPolicy {
                    max_line_pixels: cost,
                    max_page_pixels: cost,
                    ..policy
                },
                ..request
            },
            |_| {
                Ok(Recognition {
                    source_ink: None,
                    views: Vec::new(),
                    decision: RefinementDecision::Abstained,
                })
            },
        )?;
        assert_eq!(budget_report.summary.recognized_lines, 1);
        assert_eq!(budget_report.summary.skipped_lines, 1);
        assert_eq!(budget_report.summary.charged_pixels, cost);
        assert!(
            budget_draft.pages[0]
                .lines
                .iter()
                .all(|l| l.text == "CAPACITOR" && !l.reviewed)
        );
        assert!(matches!(
            budget_report.proposals[1].outcome,
            super::RefinementOutcome::Skipped { .. }
        ));

        Ok(())
    }

    #[test]
    fn exact_pdf_second_pass_preserves_source_and_zeroes_all_hypothesis_confidences()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, path, metadata, source) = source_fixture()?;
        let quad = source.pages[0].lines[0].quad;
        let policy = RefinementPolicy {
            dpi: 144,
            ..RefinementPolicy::default()
        };
        let mut calls = 0;
        let request = RefinementRequest {
            pdf: &path,
            metadata: &metadata,
            source: &source,
            scope: RefinementScope::ByConfidence { pages: None },
            secondary_model: "medium-recognition",
            policy,
        };
        let (report, experimental) = refine_draft(request, |image| {
            calls += 1;
            assert!(image.width() > 2 && image.height() > 2);
            Ok(Recognition {
                source_ink: None,
                views: Vec::new(),
                decision: RefinementDecision::Greedy {
                    text: "CAPAC1TOR".into(),
                },
            })
        })?;
        assert_eq!(calls, 1);
        assert_eq!(report.summary.selected_lines, 1);
        assert_eq!(report.summary.revised_predictions, 1);
        assert_eq!(report.proposals[0].line_index, 0);
        assert_eq!(report.proposals[0].quad, quad);
        assert_eq!(experimental.pages[0].lines[0].text, "CAPAC1TOR");
        assert_eq!(experimental.pages[0].lines[0].confidence, 0.0);
        assert!(!experimental.pages[0].lines[0].reviewed);
        assert_eq!(source.pages[0].lines[0].text, "CAPACITOR");
        assert!(source.pages[0].lines[0].reviewed);

        let (_, skipped) = refine_draft(
            RefinementRequest {
                policy: RefinementPolicy {
                    max_primary_score: 0.1,
                    ..policy
                },
                ..request
            },
            |_| Err(crate::Error::Inference("must not invoke model".into())),
        )?;
        assert_eq!(skipped.pages[0].lines[0].text, "CAPACITOR");
        assert_eq!(skipped.pages[0].lines[0].confidence, 0.0);
        Ok(())
    }
}
