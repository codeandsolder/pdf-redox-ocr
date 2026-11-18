//! Counterfactual PDF paint verification for native source text on mixed pages.
//!
//! All image observations begin in the source bitmap frame. Exactly one image
//! resource is masked in an isolated copy-on-write PDF, with all other PDF
//! objects and paint occurrences left intact. The original rendered page is
//! compared against that altered PDF in each candidate OCR polygon. A source
//! painted more than once on the same page cannot yield an occurrence-specific
//! proof without separate Do-invocation aliasing and remains on page fallback.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use image::RgbImage;
use pdf_redox::ObjectId;
use tempfile::NamedTempFile;

use crate::{
    Error, PageNumber,
    geometry::{DisplayMapper, Point, Quad},
    hybrid::{ImageObservation, VisibilityProof},
    paint_scan::{ImagePaint, PaintReport, PdfMatrix},
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    source_mask::mask_source_text,
    source_ocr::{SourceObservation, map_observations_to_paint},
};

/// A visibility audit measuring actual final PDF raster differences.
#[derive(Debug, Clone)]
pub struct SourceVisibilityAudit {
    /// One-based PDF page.
    pub page: PageNumber,
    /// Exact source `XObject` indirect ID.
    pub source_id: String,
    /// Actual expanded paint invocation.
    pub paint_index: u32,
    /// Number of source-local model detections considered.
    pub candidate_lines: usize,
    /// Number of source labels proven visible by differential rendering.
    pub visible_lines: usize,
}

/// Verified extra OCR observations plus the matching proof records.
#[derive(Debug, Default)]
pub struct MixedVisibility {
    /// Source-bound OCR labels whose visible page pixels have been proved.
    pub observations: Vec<ImageObservation>,
    /// Exact per-observation source/paint/page/quad identity proof.
    pub proofs: Vec<VisibilityProof>,
    /// Pages with completed proof comparisons.
    pub proven_pages: BTreeSet<PageNumber>,
    /// Detailed attempted unique-source treatments.
    pub audited: Vec<SourceVisibilityAudit>,
}

/// Build real differential source proofs for candidate mixed pages.
///
/// All missing/unsupported image decoders and duplicate occurrences remain
/// on the page OCR fallback. No proof is constructed from an absent render.
/// Source-wide masking is intentionally never attributed to a particular
/// repeated paint occurrence on the same page.
///
/// # Errors
/// Returns an error if a candidate PDF render cannot be completed, the masked
/// source has invalid geometry, or a PDF page cannot be identified.
pub fn prove_mixed_source_text(
    original_pdf: &Path,
    document: &PdfMetadata,
    painted: &PaintReport,
    recognized: &BTreeMap<String, SourceObservation>,
    selected: &[PageNumber],
    policy: RenderPolicy,
    minimum_confidence: f64,
) -> Result<MixedVisibility, Error> {
    if !minimum_confidence.is_finite() || !(0.0..=1.0).contains(&minimum_confidence) {
        return Err(Error::Invalid(
            "invalid visibility confidence threshold".into(),
        ));
    }
    let mut result = MixedVisibility::default();
    let approved = selected.iter().copied().collect::<BTreeSet<_>>();
    for &page in &approved {
        let relevant = painted
            .placements
            .iter()
            .filter(|paint| {
                paint.page == page.get()
                    && paint.unsafe_reasons.is_empty()
                    && recognized.contains_key(&paint.source_id)
            })
            .collect::<Vec<_>>();
        let unique = relevant
            .iter()
            .filter(|paint| {
                painted
                    .placements
                    .iter()
                    .filter(|other| other.page == page.get() && other.source_id == paint.source_id)
                    .count()
                    == 1
            })
            .copied()
            .collect::<Vec<_>>();
        if unique.is_empty() {
            continue;
        }
        let control = render_rgb_page(original_pdf, page, policy)?;
        let probe = PageProbe {
            document,
            policy,
            page,
            minimum_confidence,
            control: &control,
        };
        for paint in unique {
            let source = recognized
                .get(&paint.source_id)
                .ok_or_else(|| Error::Invalid("source recognition vanished during proof".into()))?;
            if let Some(proven) = probe.examine(paint, source)? {
                if !proven.proofs.is_empty() {
                    result.proven_pages.insert(page);
                }
                result.observations.extend(proven.observations);
                result.proofs.extend(proven.proofs);
                result.audited.push(proven.audit);
            }
        }
    }
    Ok(result)
}

struct ProbedSource {
    observations: Vec<ImageObservation>,
    proofs: Vec<VisibilityProof>,
    audit: SourceVisibilityAudit,
}

struct PageProbe<'a> {
    document: &'a PdfMetadata,
    control: &'a RgbImage,
    page: PageNumber,
    policy: RenderPolicy,
    minimum_confidence: f64,
}

impl PageProbe<'_> {
    fn examine(
        &self,
        paint: &ImagePaint,
        source: &SourceObservation,
    ) -> Result<Option<ProbedSource>, Error> {
        let lines = source
            .lines
            .iter()
            .filter(|line| {
                line.confidence >= self.minimum_confidence && line.text.trim().chars().count() >= 2
            })
            .cloned()
            .collect::<Vec<_>>();
        if lines.is_empty() {
            return Ok(None);
        }
        let source_object = ObjectId::new(
            i32::try_from(paint.object_number)
                .map_err(|_| Error::Invalid("source object number outside i32".into()))?,
            i32::from(paint.generation),
        );
        let Some(masked) = mask_source_text(
            self.document.original_bytes(),
            source_object,
            &lines,
            self.minimum_confidence,
        )?
        else {
            return Ok(None);
        };
        let temp = NamedTempFile::new()?;
        fs::write(temp.path(), &masked.bytes)?;
        let treatment = render_rgb_page(temp.path(), self.page, self.policy)?;
        if treatment.dimensions() != self.control.dimensions() {
            return Err(Error::Invalid(
                "counterfactual changed page raster size".into(),
            ));
        }
        let projection = SourceObservation {
            width: source.width,
            height: source.height,
            lines,
        };
        let projections = map_observations_to_paint(&projection, paint)?;
        if projections.len() != masked.metrics.len() {
            return Err(Error::Invalid("masked OCR metric/geometry mismatch".into()));
        }
        let display_mapper = self.document.mapper(self.page)?;
        let source_pdf_to_pixel = pixel_density(self.control, self.document, self.page)?;
        let expected_scale = PdfMatrix(paint.ctm).area_scale() * source_pdf_to_pixel
            / (f64::from(source.width) * f64::from(source.height));
        let mut observations = Vec::new();
        let mut proofs = Vec::new();
        for (line, metric) in projections.iter().zip(&masked.metrics) {
            if !paint.wholly_visible_in_clip(&line.quad) {
                continue;
            }
            let changed_pixels =
                changed_raster_pixels(self.control, &treatment, &line.quad, display_mapper);
            let expected = num_traits::ToPrimitive::to_f64(&metric.changed_source_pixels)
                .unwrap_or(f64::INFINITY)
                * expected_scale;
            let observed = num_traits::ToPrimitive::to_f64(&changed_pixels).unwrap_or(0.0);
            if metric.changed_source_pixels < 10 || observed < (expected * 0.35).max(8.0) {
                continue;
            }
            observations.push(ImageObservation {
                page: self.page.get(),
                source_id: Some(paint.source_id.clone()),
                paint_index: Some(paint.paint_index),
                line: line.clone(),
            });
            proofs.push(VisibilityProof {
                page: self.page.get(),
                source_id: Some(paint.source_id.clone()),
                paint_index: Some(paint.paint_index),
                text: line.text.clone(),
                quad: line.quad,
                visible: true,
            });
        }
        let audit = SourceVisibilityAudit {
            page: self.page,
            source_id: paint.source_id.clone(),
            paint_index: paint.paint_index,
            candidate_lines: projection.lines.len(),
            visible_lines: proofs.len(),
        };
        Ok(Some(ProbedSource {
            observations,
            proofs,
            audit,
        }))
    }
}

fn pixel_density(image: &RgbImage, document: &PdfMetadata, page: PageNumber) -> Result<f64, Error> {
    let page_box = document
        .media_boxes()
        .get(&page)
        .ok_or_else(|| Error::Invalid("page MediaBox absent".into()))?;
    Ok((f64::from(image.width()) * f64::from(image.height()))
        / (page_box.width() * page_box.height()))
}

fn changed_raster_pixels(
    before: &RgbImage,
    after: &RgbImage,
    polygon: &Quad,
    mapper: DisplayMapper,
) -> usize {
    let width = f64::from(before.width());
    let height = f64::from(before.height());
    let image_polygon = polygon.map(|vertex| {
        let Point(u, v) = mapper.to_raster(vertex);
        Point(u * width, v * height)
    });
    let xmin = image_polygon
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0);
    let xmax = image_polygon
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .min(width);
    let ymin = image_polygon
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0);
    let ymax = image_polygon
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .min(height);
    let to_index = |v: f64| num_traits::ToPrimitive::to_u32(&v).unwrap_or(0);
    let mut changed = 0;
    for y in to_index(ymin)..to_index(ymax) {
        for x in to_index(xmin)..to_index(xmax) {
            let pixel = Point(f64::from(x) + 0.5, f64::from(y) + 0.5);
            if !inside_convex(&image_polygon, pixel) {
                continue;
            }
            let control = before.get_pixel(x, y);
            let treatment = after.get_pixel(x, y);
            if control
                .0
                .iter()
                .zip(treatment.0)
                .any(|(old, new)| old.abs_diff(new) > 8)
            {
                changed += 1;
            }
        }
    }
    changed
}

fn inside_convex(polygon: &Quad, point: Point) -> bool {
    let mut negative = false;
    let mut positive = false;
    for index in 0..4 {
        let from = polygon[index];
        let to = polygon[(index + 1) % 4];
        let cross =
            (to.0 - from.0).mul_add(point.1 - from.1, -(to.1 - from.1) * (point.0 - from.0));
        if cross > 1e-6 {
            positive = true;
        }
        if cross < -1e-6 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::changed_raster_pixels;
    use crate::geometry::{DisplayMapper, PageBox, Point, Rotation};

    #[test]
    fn diff_detects_visible_but_not_fully_occluded_sources() -> Result<(), crate::Error> {
        let original = RgbImage::from_pixel(100, 100, Rgb([0, 0, 0]));
        let mut masked = original.clone();
        for y in 20..40 {
            for x in 20..60 {
                masked.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        let rect = [
            Point(20.0, 60.0),
            Point(60.0, 60.0),
            Point(60.0, 80.0),
            Point(20.0, 80.0),
        ];
        let mapper = DisplayMapper::new(PageBox::new([0.0, 0.0, 100.0, 100.0])?, Rotation::Deg0);
        assert_eq!(
            changed_raster_pixels(&original, &original, &rect, mapper),
            0
        );
        assert_eq!(
            changed_raster_pixels(&original, &masked, &rect, mapper),
            800
        );
        Ok(())
    }
}
