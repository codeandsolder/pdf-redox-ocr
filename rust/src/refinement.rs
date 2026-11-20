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
use num_traits::ToPrimitive;
use oar_ocr::{processors::Point as OarPoint, utils::get_rotate_crop_image};
use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

use crate::{
    Error, PageNumber,
    draft::{DraftPage, OcrDraft, OcrLine},
    geometry::{DisplayMapper, Point, Quad},
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
};

/// Second-pass selection and resource limits.
#[derive(Clone, Copy, Debug)]
pub struct RefinementPolicy {
    /// Maximum primary confidence considered uncertain.
    pub max_primary_score: f64,
    /// Minimum number of Unicode scalar values to refine.
    pub min_chars: usize,
    /// Maximum crops per inference batch.
    pub batch_size: usize,
    /// Page rasterization resolution.
    pub render: RenderPolicy,
}

impl Default for RefinementPolicy {
    fn default() -> Self {
        Self {
            max_primary_score: 0.97,
            min_chars: 2,
            batch_size: 16,
            render: RenderPolicy::default(),
        }
    }
}

impl RefinementPolicy {
    /// Validate thresholds before running or loading models.
    ///
    /// # Errors
    /// Rejects invalid confidence, batch size, or render configuration.
    pub fn validate(self) -> Result<Self, Error> {
        if !self.max_primary_score.is_finite()
            || !(0.0..=1.0).contains(&self.max_primary_score)
            || self.min_chars == 0
            || !(1..=256).contains(&self.batch_size)
        {
            return Err(Error::Invalid(
                "invalid secondary OCR refinement limits".into(),
            ));
        }
        self.render.validate()?;
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

/// Recognition returned for one crop in the original input order.
#[derive(Debug, Clone)]
pub struct Recognition {
    /// Second model's Unicode hypothesis.
    pub text: String,
    /// Second model's confidence score.
    pub confidence: f64,
}

/// Report for a single original OCR observation.
#[derive(Debug, Clone, Serialize)]
pub struct RefinementProposal {
    /// Original one-based PDF page.
    pub page: u32,
    /// Original zero-based line index within its page.
    pub line_index: usize,
    /// Unchanged source PDF text geometry.
    pub quad: Quad,
    pub primary_text: String,
    pub primary_score: f64,
    pub secondary_text: String,
    pub secondary_score: f64,
    /// NFKC and whitespace are formatting differences only.
    pub different: bool,
    pub format_equivalent: bool,
}

/// Summary of attempted recognition-only revisions.
#[derive(Debug, Clone, Serialize)]
pub struct RefinementSummary {
    pub source_lines: usize,
    pub selected_lines: usize,
    pub revised_predictions: usize,
    pub seconds: f64,
    pub source_pages: usize,
    pub pages_refined: usize,
}

/// Research-only provenance-bound comparison report.
#[derive(Debug, Clone, Serialize)]
pub struct RefinementReport {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub source_model: String,
    pub secondary_model: String,
    pub selection_mode: &'static str,
    pub max_primary_score: Option<f64>,
    pub min_chars: Option<usize>,
    pub max_side: u32,
    pub summary: RefinementSummary,
    pub proposals: Vec<RefinementProposal>,
}

/// NFKC-compatible and whitespace-insensitive equivalence, preserving case,
/// punctuation, units and digits as semantically distinct.
#[must_use]
pub fn meaningful_change(left: &str, right: &str) -> bool {
    let canonical = |value: &str| {
        value
            .nfkc()
            .filter(|character| !character.is_whitespace())
            .collect::<String>()
    };
    canonical(left) != canonical(right)
}

/// Convert a PDF BL/BR/TR/TL quad into four TL/TR/BR/BL raster pixels.
///
/// # Errors
/// Rejects nonfinite, out-of-page, degenerate or unrepresentable geometry.
pub fn raster_quad(
    quad: &Quad,
    mapper: DisplayMapper,
    width: u32,
    height: u32,
) -> Result<[OarPoint; 4], Error> {
    if width == 0 || height == 0 {
        return Err(Error::Invalid("empty refinement image".into()));
    }
    let points = [quad[3], quad[2], quad[1], quad[0]].map(|point| {
        let Point(u, v) = mapper.to_raster(point);
        if !(-0.025..=1.025).contains(&u)
            || !(-0.025..=1.025).contains(&v)
            || !u.is_finite()
            || !v.is_finite()
        {
            return Err(Error::Invalid("OCR crop outside PDF page".into()));
        }
        let (x, y) = (u * f64::from(width), v * f64::from(height));
        if !x.is_finite()
            || !y.is_finite()
            || x.abs() > f64::from(f32::MAX)
            || y.abs() > f64::from(f32::MAX)
        {
            return Err(Error::Invalid("unrepresentable OCR crop geometry".into()));
        }
        Ok(OarPoint::new(
            x.to_f32()
                .ok_or_else(|| Error::Invalid("OCR crop X outside f32".into()))?,
            y.to_f32()
                .ok_or_else(|| Error::Invalid("OCR crop Y outside f32".into()))?,
        ))
    });
    let points = points.into_iter().collect::<Result<Vec<_>, _>>()?;
    let points: [OarPoint; 4] = points
        .try_into()
        .map_err(|_| Error::Invalid("OCR crop must have four corners".into()))?;
    for (i, point) in points.iter().enumerate() {
        let other = points[(i + 1) % points.len()];
        let dx = f64::from(point.x - other.x);
        let dy = f64::from(point.y - other.y);
        if dx.mul_add(dx, dy * dy) < 1.0 {
            return Err(Error::Invalid("degenerate OCR crop".into()));
        }
    }
    Ok(points)
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

/// Render one page once, preserving crop order across bounded recognizer batches.
fn proposals_for_page(
    render_pdf: &Path,
    number: PageNumber,
    page: &DraftPage,
    indices: &[usize],
    mapper: DisplayMapper,
    policy: RefinementPolicy,
    recognize: &mut impl FnMut(Vec<RgbImage>) -> Result<Vec<Recognition>, Error>,
) -> Result<Vec<RefinementProposal>, Error> {
    let image = render_rgb_page(render_pdf, number, policy.render)?;
    let mut proposals = Vec::with_capacity(indices.len());
    for chunk in indices.chunks(policy.batch_size) {
        let mut crops = Vec::with_capacity(chunk.len());
        for &index in chunk {
            let quad = raster_quad(
                &page.lines[index].quad,
                mapper,
                image.width(),
                image.height(),
            )?;
            crops.push(
                get_rotate_crop_image(&image, &quad)
                    .map_err(|error| Error::Inference(format!("refinement crop: {error}")))?,
            );
        }
        let recognized = recognize(crops)?;
        if recognized.len() != chunk.len() {
            return Err(Error::Inference(
                "secondary recognizer returned wrong batch size".into(),
            ));
        }
        for (&index, prediction) in chunk.iter().zip(recognized) {
            if !prediction.confidence.is_finite() || !(0.0..=1.0).contains(&prediction.confidence) {
                return Err(Error::Invalid("invalid secondary OCR confidence".into()));
            }
            let original = &page.lines[index];
            let different = meaningful_change(&original.text, &prediction.text);
            proposals.push(RefinementProposal {
                page: number.get(),
                line_index: index,
                quad: original.quad,
                primary_text: original.text.clone(),
                primary_score: original.confidence,
                secondary_text: prediction.text,
                secondary_score: prediction.confidence,
                different,
                format_equivalent: !different,
            });
        }
    }
    Ok(proposals)
}

/// Primary and secondary evidence sources for a single refinement run.
#[derive(Clone, Copy)]
pub struct RefinementRequest<'a> {
    /// Immutable exact source PDF.
    pub pdf: &'a Path,
    /// PDF used only for rendering, optionally with native font masking.
    pub render_pdf: &'a Path,
    /// Parsed original page geometry.
    pub metadata: &'a PdfMetadata,
    /// Exact-PDF-bound OCR draft.
    pub source: &'a OcrDraft,
    /// Confidence-based or explicitly indexed input observations.
    pub scope: RefinementScope<'a>,
    /// Human-readable secondary model identity for traceability.
    pub secondary_model: &'a str,
    /// Selection, batching and render limits.
    pub policy: RefinementPolicy,
}

/// Apply a research-only recognition-only second pass to the original PDF.
///
/// The callback receives perspective-corrected crops in draft line order;
/// it must return precisely one recognition per crop. The caller may lazily
/// initialize any OCR model. Native searchable font masking is supplied by
/// the caller through `render_pdf`, and may be the original PDF when unnecessary.
///
/// # Errors
/// Rejects changed PDFs, invalid source evidence, model failures or bad geometry.
pub fn refine_draft(
    request: RefinementRequest<'_>,
    mut recognize: impl FnMut(Vec<RgbImage>) -> Result<Vec<Recognition>, Error>,
) -> Result<(RefinementReport, OcrDraft), Error> {
    let RefinementRequest {
        pdf,
        render_pdf,
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

    for (&number, indices) in &plan {
        let page = source
            .pages
            .iter()
            .find(|entry| entry.page == number.get())
            .ok_or_else(|| Error::Invalid("refinement page disappeared".into()))?;
        let mapper = *mappers
            .get(&number)
            .ok_or_else(|| Error::Invalid("missing refinement page geometry".into()))?;
        proposals.extend(proposals_for_page(
            render_pdf,
            number,
            page,
            indices,
            mapper,
            policy,
            &mut recognize,
        )?);
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
        if proposal.different && !proposal.secondary_text.trim().is_empty() {
            let page = experimental
                .pages
                .iter_mut()
                .find(|p| p.page == proposal.page)
                .ok_or_else(|| Error::Invalid("refinement proposal page missing".into()))?;
            let line: &mut OcrLine = page
                .lines
                .get_mut(proposal.line_index)
                .ok_or_else(|| Error::Invalid("refinement proposal line missing".into()))?;
            line.text.clone_from(&proposal.secondary_text);
        }
    }
    let by_confidence = matches!(scope, RefinementScope::ByConfidence { .. });
    let report = RefinementReport {
        schema_version: 2,
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
        max_side: policy.render.max_side,
        summary: RefinementSummary {
            source_lines: source.pages.iter().map(|page| page.lines.len()).sum(),
            selected_lines: proposals.len(),
            revised_predictions: proposals.iter().filter(|p| p.different).count(),
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
        Recognition, RefinementPolicy, RefinementRequest, RefinementScope, meaningful_change,
        raster_quad, refine_draft, refinement_plan,
    };
    use crate::{
        PageNumber,
        draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
        geometry::{DisplayMapper, PageBox, Point, Rotation},
        pdf_metadata::PdfMetadata,
        rasterize::RenderPolicy,
    };
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
    };

    #[test]
    fn format_equivalence_preserves_semantically_distinct_tokens() {
        assert!(!meaningful_change("12 ＶＡＣ", "12VAC"));
        assert!(!meaningful_change("A \t B", "AB"));
        for (first, second) in [("20A", "20V"), ("AB", "ab"), ("Ω", "ohm"), ("10.2", "102")] {
            assert!(meaningful_change(first, second), "{first:?}, {second:?}");
        }
    }

    #[test]
    fn perspective_crop_coords_roundtrip_all_page_rotations() -> Result<(), crate::Error> {
        let bounds = PageBox::new([30.0, 40.0, 830.0, 640.0])?;
        for angle in [0, 90, 180, 270] {
            let mapper = DisplayMapper::new(bounds, Rotation::try_from(angle)?);
            let source = [
                mapper.to_pdf(0.2, 0.8),
                mapper.to_pdf(0.35, 0.8),
                mapper.to_pdf(0.35, 0.65),
                mapper.to_pdf(0.2, 0.65),
            ];
            let raster_points = raster_quad(&source, mapper, 1500, 2000)?;
            for (actual, expected) in raster_points.iter().zip([
                Point(300.0, 1300.0),
                Point(525.0, 1300.0),
                Point(525.0, 1600.0),
                Point(300.0, 1600.0),
            ]) {
                assert!((f64::from(actual.x) - expected.0).abs() < 0.03, "{angle}");
                assert!((f64::from(actual.y) - expected.1).abs() < 0.03, "{angle}");
            }
            assert!(raster_quad(&[Point(0.0, 0.0); 4], mapper, 1500, 2000).is_err());
            assert!(raster_quad(&[Point(f64::NAN, 0.0); 4], mapper, 1500, 2000).is_err());
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

    #[test]
    fn exact_pdf_second_pass_preserves_source_and_zeroes_all_hypothesis_confidences()
    -> Result<(), Box<dyn std::error::Error>> {
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
        let policy = RefinementPolicy {
            render: RenderPolicy {
                max_side: 600,
                dpi: 72,
            },
            ..RefinementPolicy::default()
        };
        let mut calls = 0;
        let request = RefinementRequest {
            pdf: &path,
            render_pdf: &path,
            metadata: &metadata,
            source: &source,
            scope: RefinementScope::ByConfidence { pages: None },
            secondary_model: "medium-recognition",
            policy,
        };
        let (report, experimental) = refine_draft(request, |images| {
            calls += 1;
            assert_eq!(images.len(), 1);
            assert!(images[0].width() > 2 && images[0].height() > 2);
            Ok(vec![Recognition {
                text: "CAPAC1TOR".into(),
                confidence: 0.99,
            }])
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
