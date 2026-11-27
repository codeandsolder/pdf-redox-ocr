//! High-detail recognition of actual composited page regions.
//!
//! A regional observation is not a source-image observation: it sees every
//! visible layer in the crop, including vector content, clipping and masks.
//! Only non-overlapping, qualified observations supplement the full-page draft.

use std::{collections::BTreeMap, path::Path};

use crate::{
    Error, PageNumber,
    draft::{DraftPage, OcrDraft, OcrLine},
    geometry::{PageBox, Point, Quad, QuadBounds},
    hybrid::{PageRoute, SourceAdmission},
    mosaic::compositor_handles_reason,
    paint_scan::{ImagePaint, PaintReport, PdfMatrix},
    pdf_metadata::PdfMetadata,
    pipeline::{PipelinePolicy, RasterMode},
    polygon::area,
    rasterize::render_rgb_region,
    recognition::{
        RasterCharacterMapping, RasterMappingScope, RecognitionBackend, RecognitionSession,
    },
    source_selection::SourcePolicy,
};

/// Actual cropped input and decisions, independent of source visibility proofs.
#[derive(Debug, serde::Serialize)]
pub struct RegionAudit {
    pub page: PageNumber,
    pub seed_paint_index: u32,
    pub requested_region: [f64; 4],
    pub dpi: u32,
    pub outcome: RegionOutcome,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RegionOutcome {
    Recognized {
        dimensions: (u32, u32),
        rgb_sha256: String,
        corners: Quad,
        observations: Vec<RegionObservation>,
    },
    Failed {
        reason: String,
    },
}

#[derive(Debug, serde::Serialize)]
pub struct RegionObservation {
    pub detection_index: usize,
    pub line: OcrLine,
    pub within_crop: bool,
    pub page_overlap: Option<f64>,
    pub native_overlap: Option<f64>,
    pub included_as_extra: bool,
}

impl RegionAudit {
    pub(crate) fn admitted(&self) -> usize {
        match &self.outcome {
            RegionOutcome::Recognized { observations, .. } => observations
                .iter()
                .filter(|observation| observation.included_as_extra)
                .count(),
            RegionOutcome::Failed { .. } => 0,
        }
    }
}

#[derive(Clone, Copy)]
struct RegionRequest {
    paint_index: u32,
    region: PageBox,
    dpi: u32,
    gain: f64,
    estimate: u64,
}

fn request_region(
    paint: &ImagePaint,
    page: PageBox,
    policy: PipelinePolicy,
) -> Option<RegionRequest> {
    // A normal independent native source already has its own recognition
    // and visibility path. Only request regions that require the compositor.
    if !paint.sample_layout.contextual
        && !paint.optional_content
        && !paint
            .unsafe_reasons
            .iter()
            .any(|reason| compositor_handles_reason(reason))
    {
        return None;
    }
    let source = SourcePolicy::default();
    let pixels = u64::from(paint.width) * u64::from(paint.height);
    if pixels < source.minimum_source_pixels || paint.width.min(paint.height) < 24 {
        return None;
    }
    let painted_area = PdfMatrix(paint.ctm).area_scale();
    if !painted_area.is_finite()
        || painted_area <= 1e-9
        || paint
            .page_quad
            .iter()
            .any(|&Point(x, y)| !x.is_finite() || !y.is_finite())
    {
        return None;
    }
    let native_scale = (f64::from(paint.width) * f64::from(paint.height) / painted_area).sqrt();
    let dpi = num_traits::ToPrimitive::to_u32(
        &(native_scale * 72.0)
            .round()
            .min(f64::from(policy.mosaic_budget.dpi)),
    )?;
    let scale = f64::from(dpi) / 72.0;
    let ordinary_scale = f64::from(policy.render.max_side) / page.width().max(page.height());
    let gain = (scale / ordinary_scale).powi(2);
    let min_gain = if policy.raster_mode == RasterMode::All {
        1.0
    } else {
        source.min_gain
    };
    if dpi < 36 || !gain.is_finite() || gain <= min_gain {
        return None;
    }
    let bounds = QuadBounds::new(&paint.page_quad);
    // Keep context around the entire image footprint rather than cutting
    // words at the source edge. The OCR admission gate also excludes lines
    // touching an artificial crop edge.
    let pad = 12.0 / scale;
    let region = PageBox::new([
        (bounds.min_x - pad).max(page.x0()),
        (bounds.min_y - pad).max(page.y0()),
        (bounds.max_x + pad).min(page.x1()),
        (bounds.max_y + pad).min(page.y1()),
    ])
    .ok()?;
    let estimate = num_traits::ToPrimitive::to_u64(&region.width().mul_add(scale, 4.0).ceil())?
        .checked_mul(num_traits::ToPrimitive::to_u64(
            &region.height().mul_add(scale, 4.0).ceil(),
        )?)?;
    (estimate <= policy.mosaic_budget.group_pixels).then_some(RegionRequest {
        paint_index: paint.paint_index,
        region,
        dpi,
        gain,
        estimate,
    })
}

fn contains(outer: PageBox, inner: PageBox) -> bool {
    inner.x0() >= outer.x0()
        && inner.y0() >= outer.y0()
        && inner.x1() <= outer.x1()
        && inner.y1() <= outer.y1()
}

fn plan_regions(
    page: PageBox,
    paints: &[&ImagePaint],
    policy: PipelinePolicy,
) -> Vec<RegionRequest> {
    let mut candidates = paints
        .iter()
        .filter_map(|paint| request_region(paint, page, policy))
        .collect::<Vec<_>>();
    // Deterministic quality-first selection. Contained lower-resolution
    // requests are already covered by the chosen actual page raster.
    candidates.sort_by(|a, b| {
        b.gain
            .total_cmp(&a.gain)
            .then_with(|| a.paint_index.cmp(&b.paint_index))
    });
    let mut selected: Vec<RegionRequest> = Vec::new();
    let mut remaining = policy.mosaic_budget.page_pixels;
    for candidate in candidates {
        if selected.len() == policy.mosaic_budget.groups {
            break;
        }
        if candidate.estimate > remaining
            || selected
                .iter()
                .any(|req| req.dpi >= candidate.dpi && contains(req.region, candidate.region))
        {
            continue;
        }
        remaining -= candidate.estimate;
        selected.push(candidate);
    }
    selected
}

pub(crate) struct RegionPass<'a> {
    pub(crate) pdf: &'a Path,
    pub(crate) document: &'a PdfMetadata,
    pub(crate) paints: &'a PaintReport,
    pub(crate) routes: &'a BTreeMap<PageNumber, PageRoute>,
    pub(crate) policy: PipelinePolicy,
}

impl RegionPass<'_> {
    pub(crate) fn recognize(
        &self,
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        draft: &mut OcrDraft,
    ) -> Result<Vec<RegionAudit>, Error> {
        if self.policy.raster_mode == RasterMode::Off {
            return Ok(Vec::new());
        }
        let mut by_page = BTreeMap::<u32, Vec<&ImagePaint>>::new();
        for paint in &self.paints.placements {
            by_page.entry(paint.page).or_default().push(paint);
        }
        let mut audits = Vec::new();
        for draft_page in &mut draft.pages {
            let page = PageNumber::new(draft_page.page)?;
            if self.routes.get(&page) != Some(&PageRoute::PageFallback) {
                continue;
            }
            let Some(paints) = by_page.get(&page.get()) else {
                continue;
            };
            let bbox = self.document.media_boxes()[&page];
            let mut remaining = self.policy.mosaic_budget.page_pixels;
            for request in plan_regions(bbox, paints, self.policy) {
                let limit = remaining.min(self.policy.mosaic_budget.group_pixels);
                let outcome = match self.recognize_region(
                    engine,
                    draft_page,
                    page,
                    request,
                    limit,
                    &mut remaining,
                ) {
                    Ok(outcome) => outcome,
                    Err(error) => RegionOutcome::Failed {
                        reason: error.to_string(),
                    },
                };
                audits.push(RegionAudit {
                    page,
                    seed_paint_index: request.paint_index,
                    requested_region: [
                        request.region.x0(),
                        request.region.y0(),
                        request.region.x1(),
                        request.region.y1(),
                    ],
                    dpi: request.dpi,
                    outcome,
                });
            }
        }
        Ok(audits)
    }

    fn recognize_region(
        &self,
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        draft_page: &mut DraftPage,
        page: PageNumber,
        request: RegionRequest,
        limit: u64,
        remaining: &mut u64,
    ) -> Result<RegionOutcome, Error> {
        let crop = render_rgb_region(
            self.pdf,
            page,
            self.document.mapper(page)?,
            request.region,
            request.dpi,
            limit,
        )?;
        let dimensions = crop.rgb.dimensions();
        *remaining = remaining.saturating_sub(u64::from(dimensions.0) * u64::from(dimensions.1));
        let mapper = crop.pixel_mapper();
        let (w, h) = (f64::from(dimensions.0), f64::from(dimensions.1));
        let corners = [Point(0., 0.), Point(w, 0.), Point(w, h), Point(0., h)]
            .map(|point| mapper.to_pdf(point));
        let mut observations = Vec::new();
        let raster = engine.recognize_pixels(crop.rgb)?;
        let rgb_sha256 = raster.rgb_sha256;
        for pixel in raster.lines {
            let within_crop = area(&pixel.quad) > 1e-9
                && pixel
                    .quad
                    .iter()
                    .all(|&Point(x, y)| x >= 3.0 && y >= 3.0 && x <= w - 3.0 && y <= h - 3.0)
                && (0.0..=1.0).contains(&pixel.confidence);
            let detection_index = pixel.detection_index;
            let line = pixel.into_pdf(&|point| mapper.to_pdf(point));
            let admission = SourceAdmission::evaluate(
                &line,
                draft_page.lines.iter().map(|existing| {
                    (
                        &existing.quad,
                        existing.overlay_qualified(self.policy.minimum_confidence),
                    )
                }),
                &draft_page.native_text_regions,
                within_crop,
                self.policy.minimum_confidence,
            );
            if admission.included {
                draft_page.lines.push(line.clone());
            }
            observations.push(RegionObservation {
                detection_index,
                line,
                within_crop,
                page_overlap: admission.page_overlap,
                native_overlap: admission.native_overlap,
                included_as_extra: admission.included,
            });
        }
        let admitted_detection_indices = observations
            .iter()
            .filter(|o| o.included_as_extra)
            .map(|o| o.detection_index)
            .collect::<std::collections::BTreeSet<_>>();
        if !admitted_detection_indices.is_empty() {
            engine.record_character_mapping(RasterCharacterMapping {
                page,
                rgb_sha256: rgb_sha256.clone(),
                dimensions,
                scope: RasterMappingScope::Region {
                    seed_paint_index: request.paint_index,
                    admitted_detection_indices,
                },
                raster_corners: corners,
            });
        }
        Ok(RegionOutcome::Recognized {
            dimensions,
            rgb_sha256,
            corners,
            observations,
        })
    }
}
