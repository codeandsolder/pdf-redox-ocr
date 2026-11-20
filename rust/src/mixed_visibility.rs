//! Counterfactual PDF paint verification for native source text on mixed pages.
//!
//! Every observation begins in the source bitmap frame. A single invocation
//! path is aliased through the enclosing Form chain into a copy-on-write PDF,
//! leaving other invocations of the same source unchanged. Final rendered page
//! pixels then prove visibility for that exact occurrence.

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
    draft::OcrLine,
    geometry::{DisplayMapper, Point, Quad},
    hybrid::{ImageObservation, SourceAdmission, VisibilityProof},
    paint_isolation::{PaintReplacement, isolate_source_paints},
    paint_scan::{ImagePaint, PaintReport, PdfMatrix},
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    source_mask::{MaskedSourceObject, mask_candidate, mask_source_object, mask_source_subset},
    source_ocr::{SourceObservation, map_observations_to_paint},
};

/// A visibility audit measuring actual final PDF raster differences.
#[derive(Debug, Clone, serde::Serialize)]
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
#[derive(Debug, Default, serde::Serialize)]
pub struct MixedVisibility {
    /// Source-bound OCR labels whose visible page pixels have been proved.
    pub observations: Vec<ImageObservation>,
    /// Exact per-observation source/paint/page/quad identity proof.
    pub proofs: Vec<VisibilityProof>,
    /// Pages with completed proof comparisons.
    pub proven_pages: BTreeSet<PageNumber>,
    /// Detailed attempted source-paint treatments.
    pub audited: Vec<SourceVisibilityAudit>,
}

impl MixedVisibility {
    pub(crate) fn extend(&mut self, other: Self) {
        self.observations.extend(other.observations);
        self.proofs.extend(other.proofs);
        self.proven_pages.extend(other.proven_pages);
        self.audited.extend(other.audited);
    }
}

/// Build real differential source proofs for candidate mixed pages.
///
/// Unsupported encodings remain on page OCR fallback. Every treatment changes
/// only its selected image paint path, including repeated Form invocations.
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
    let mut verifier = MixedVerifier::new(
        original_pdf,
        document,
        painted,
        recognized,
        policy,
        minimum_confidence,
    )?;
    let mut result = MixedVisibility::default();
    let mut ordered = selected.to_vec();
    ordered.sort_unstable();
    ordered.dedup();
    for page in ordered {
        result.extend(verifier.verify_page(page, None)?.evidence);
    }
    Ok(result)
}

/// A single-page residual alternative. Its suppression set is bound to the
/// same per-paint proof and admission evidence used by the final merge.
pub(crate) struct ResidualRaster {
    pub(crate) image: RgbImage,
    /// Page-local observation indices that must remain admitted by the merger.
    pub(crate) required: BTreeSet<usize>,
    /// Number of actual paint replacements, for diagnostics.
    pub(crate) paint_count: usize,
}

impl ResidualRaster {
    /// Transfer only the proved pixel changes onto the native-font-hidden
    /// control. If a font changed any of those pixels, discard the residual.
    pub(crate) fn transfer(mut self, original: &RgbImage, baseline: &RgbImage) -> Option<Self> {
        if original.dimensions() != baseline.dimensions()
            || self.image.dimensions() != baseline.dimensions()
        {
            return None;
        }
        for ((masked, old), base) in self
            .image
            .pixels_mut()
            .zip(original.pixels())
            .zip(baseline.pixels())
        {
            if *masked == *old {
                *masked = *base;
            } else if base != old {
                return None;
            }
        }
        Some(self)
    }
}

#[derive(Default)]
pub(crate) struct PageVerification {
    pub(crate) evidence: MixedVisibility,
    pub(crate) control: Option<RgbImage>,
    pub(crate) residual: Option<ResidualRaster>,
}

/// Page-local raster lifetime with job-wide source-sample mask reuse.
pub(crate) struct MixedVerifier<'a> {
    original_pdf: &'a Path,
    document: &'a PdfMetadata,
    relevant: BTreeMap<PageNumber, Vec<&'a ImagePaint>>,
    recognized: &'a BTreeMap<String, SourceObservation>,
    policy: RenderPolicy,
    minimum_confidence: f64,
    masked_cache: BTreeMap<String, Option<MaskedSourceObject>>,
}

impl<'a> MixedVerifier<'a> {
    pub(crate) fn new(
        original_pdf: &'a Path,
        document: &'a PdfMetadata,
        painted: &'a PaintReport,
        recognized: &'a BTreeMap<String, SourceObservation>,
        policy: RenderPolicy,
        minimum_confidence: f64,
    ) -> Result<Self, Error> {
        if !minimum_confidence.is_finite() || !(0.0..=1.0).contains(&minimum_confidence) {
            return Err(Error::Invalid(
                "invalid visibility confidence threshold".into(),
            ));
        }
        let mut relevant = BTreeMap::<PageNumber, Vec<&ImagePaint>>::new();
        for paint in &painted.placements {
            if paint.unsafe_reasons.is_empty() && recognized.contains_key(&paint.source_id) {
                relevant
                    .entry(PageNumber::new(paint.page)?)
                    .or_default()
                    .push(paint);
            }
        }
        Ok(Self {
            original_pdf,
            document,
            relevant,
            recognized,
            policy,
            minimum_confidence,
            masked_cache: BTreeMap::new(),
        })
    }

    pub(crate) fn verify_page(
        &mut self,
        page: PageNumber,
        native: Option<&[Quad]>,
    ) -> Result<PageVerification, Error> {
        let Some(relevant) = self.relevant.get(&page) else {
            return Ok(PageVerification::default());
        };
        let control = render_rgb_page(self.original_pdf, page, self.policy)?;
        let probe = PageProbe {
            document: self.document,
            policy: self.policy,
            page,
            minimum_confidence: self.minimum_confidence,
            control: &control,
        };
        let mut evidence = MixedVisibility::default();
        let mut residual = ResidualBuilder::default();
        let mut variants = BTreeMap::new();
        for &paint in relevant {
            let source = self
                .recognized
                .get(&paint.source_id)
                .ok_or_else(|| Error::Invalid("source recognition vanished during proof".into()))?;
            if source.width == 0
                || source.height == 0
                || source.width != paint.width
                || source.height != paint.height
            {
                continue;
            }
            let id = ObjectId::new(paint.object_number, i32::from(paint.generation));
            if !self.masked_cache.contains_key(&paint.source_id) {
                let masked = match mask_source_object(
                    self.document.source(),
                    id,
                    &source.lines,
                    self.minimum_confidence,
                ) {
                    Ok(masked) => masked,
                    Err(Error::Invalid(_)) => None,
                    Err(error) => return Err(error),
                };
                self.masked_cache.insert(paint.source_id.clone(), masked);
            }
            let Some(masked) = self
                .masked_cache
                .get(&paint.source_id)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            if let Some(proven) = probe.examine(paint, source, masked)? {
                if !proven.proofs.is_empty() {
                    evidence.proven_pages.insert(page);
                }
                let ProbedSource {
                    observations,
                    proofs,
                    audit,
                    treatment,
                    source_indices,
                } = proven;
                if let Some(native) = native {
                    // Simulate the real source order, including candidates that
                    // are not going to be suppressed. Retain only observations
                    // that can be admitted with an empty fallback.
                    let chosen = residual.plan(
                        &observations,
                        &source_indices,
                        native,
                        self.minimum_confidence,
                        evidence.observations.len(),
                    );
                    if !chosen.is_empty() {
                        let key = MaskKey {
                            source_id: paint.source_id.clone(),
                            indices: chosen.indices.clone(),
                        };
                        if chosen.indices.len() == audit.candidate_lines {
                            residual.consider(
                                paint,
                                key,
                                chosen.required,
                                &masked.object,
                                &control,
                                treatment,
                            );
                        } else if let Some((object, image)) =
                            probe.selective_treatment(paint, source, &key, &mut variants)
                        {
                            residual.consider(paint, key, chosen.required, object, &control, image);
                        }
                    }
                }
                evidence.observations.extend(observations);
                evidence.proofs.extend(proofs);
                evidence.audited.push(audit);
            }
        }
        let residual = residual.finish(&probe).unwrap_or(None);
        Ok(PageVerification {
            evidence,
            control: Some(control),
            residual,
        })
    }
}

/// A mask's identity includes the exact original native observation indices,
/// since reused images may be suppressed differently at different paint sites.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MaskKey {
    source_id: String,
    indices: Vec<usize>,
}

type MaskVariants = BTreeMap<MaskKey, Option<MaskedSourceObject>>;

#[derive(Default)]
struct ChosenLines {
    indices: Vec<usize>,
    required: BTreeSet<usize>,
}

impl ChosenLines {
    const fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

#[derive(Default)]
struct ResidualBuilder<'a> {
    image: Option<RgbImage>,
    groups: BTreeMap<MaskKey, PaintReplacement<'a>>,
    required: BTreeSet<usize>,
    planned_quads: Vec<Quad>,
    paint_count: usize,
}

impl<'a> ResidualBuilder<'a> {
    fn plan(
        &mut self,
        observations: &[ImageObservation],
        source_indices: &[usize],
        native: &[Quad],
        threshold: f64,
        offset: usize,
    ) -> ChosenLines {
        let mut chosen = ChosenLines::default();
        for (index, (observation, &source_index)) in
            observations.iter().zip(source_indices).enumerate()
        {
            if SourceAdmission::evaluate(
                &observation.line,
                self.planned_quads.iter().map(|quad| (quad, true)),
                native,
                true,
                threshold,
            )
            .included
            {
                self.planned_quads.push(observation.line.quad);
                chosen.indices.push(source_index);
                chosen.required.insert(offset + index);
            }
        }
        chosen
    }

    fn consider(
        &mut self,
        paint: &'a ImagePaint,
        key: MaskKey,
        required: BTreeSet<usize>,
        replacement: &pdf_redox::OwnedObject,
        control: &RgbImage,
        treatment: RgbImage,
    ) {
        if let Some(image) = &mut self.image {
            if !combine_disjoint_changes(control, image, &treatment) {
                return;
            }
        } else {
            self.image = Some(treatment);
        }
        self.groups
            .entry(key)
            .or_insert_with(|| PaintReplacement {
                image: replacement.clone(),
                paints: Vec::new(),
            })
            .paints
            .push(paint);
        self.required.extend(required);
        self.paint_count += 1;
    }

    fn finish(self, probe: &PageProbe<'_>) -> Result<Option<ResidualRaster>, Error> {
        let Some(image) = self.image else {
            return Ok(None);
        };
        if self.paint_count > 1 {
            let combined = probe.render_treatment(self.groups.into_values())?;
            if combined != image {
                return Ok(None);
            }
        }
        Ok(Some(ResidualRaster {
            image,
            required: self.required,
            paint_count: self.paint_count,
        }))
    }
}

/// Apply a second independent change atomically, rejecting overlapping writes.
fn combine_disjoint_changes(control: &RgbImage, combined: &mut RgbImage, next: &RgbImage) -> bool {
    if control.dimensions() != combined.dimensions()
        || control.dimensions() != next.dimensions()
        || control
            .pixels()
            .zip(combined.pixels())
            .zip(next.pixels())
            .any(|((old, a), b)| a != old && b != old)
    {
        return false;
    }
    for ((old, current), next) in control
        .pixels()
        .zip(combined.pixels_mut())
        .zip(next.pixels())
    {
        if next != old {
            *current = *next;
        }
    }
    true
}

struct ProbedSource {
    observations: Vec<ImageObservation>,
    proofs: Vec<VisibilityProof>,
    audit: SourceVisibilityAudit,
    treatment: RgbImage,
    source_indices: Vec<usize>,
}

struct PageProbe<'a> {
    document: &'a PdfMetadata,
    control: &'a RgbImage,
    page: PageNumber,
    policy: RenderPolicy,
    minimum_confidence: f64,
}

impl PageProbe<'_> {
    fn mask_pixel_margin(
        &self,
        paint: &ImagePaint,
        source: &SourceObservation,
    ) -> Result<f64, Error> {
        let page_box = self
            .document
            .media_boxes()
            .get(&self.page)
            .ok_or_else(|| Error::Invalid("page MediaBox missing".into()))?;
        let dpi_scale = (f64::from(self.control.width()) / page_box.width())
            .max(f64::from(self.control.height()) / page_box.height());
        let [a, b, c, d, _, _] = paint.ctm;
        Ok((5.0
            * (a.hypot(b) / f64::from(source.width) + c.hypot(d) / f64::from(source.height))
            * dpi_scale)
            .ceil()
            + 3.0)
    }

    fn render_treatment<'a>(
        &self,
        replacements: impl IntoIterator<Item = PaintReplacement<'a>>,
    ) -> Result<RgbImage, Error> {
        let temp = NamedTempFile::new()?;
        fs::write(
            temp.path(),
            isolate_source_paints(self.document, replacements)?,
        )?;
        render_rgb_page(temp.path(), self.page, self.policy)
    }

    /// Re-prove the exact subset, rather than using pixels from a treatment
    /// that also erased excluded/unverified candidates. Variants are cached
    /// within this page only, bounding resident combinations across documents.
    fn selective_treatment<'a>(
        &self,
        paint: &ImagePaint,
        source: &SourceObservation,
        key: &MaskKey,
        variants: &'a mut MaskVariants,
    ) -> Option<(&'a pdf_redox::OwnedObject, RgbImage)> {
        let lines = key
            .indices
            .iter()
            .map(|&index| source.lines.get(index).cloned())
            .collect::<Option<Vec<_>>>()?;
        let subset = SourceObservation {
            width: source.width,
            height: source.height,
            lines,
        };
        let masked = variants
            .entry(key.clone())
            .or_insert_with(|| {
                let id = ObjectId::new(paint.object_number, i32::from(paint.generation));
                mask_source_subset(self.document.source(), id, &source.lines, &key.indices)
                    .ok()
                    .flatten()
            })
            .as_ref()?;
        let proof = self.examine(paint, &subset, masked).ok().flatten()?;
        if proof.proofs.len() != subset.lines.len() {
            return None;
        }
        Some((&masked.object, proof.treatment))
    }

    fn examine(
        &self,
        paint: &ImagePaint,
        source: &SourceObservation,
        masked: &MaskedSourceObject,
    ) -> Result<Option<ProbedSource>, Error> {
        if source.width == 0 || source.height == 0 {
            return Err(Error::Invalid("invalid source OCR dimensions".into()));
        }
        let (indices, lines): (Vec<_>, Vec<_>) = source
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| mask_candidate(line, self.minimum_confidence))
            .map(|(index, line)| (index, line.clone()))
            .unzip();
        if lines.is_empty() {
            return Ok(None);
        }
        let treatment = self.render_treatment([PaintReplacement {
            image: masked.object.clone(),
            paints: vec![paint],
        }])?;
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
        let padding = self.mask_pixel_margin(paint, source)?;
        // A counterfactual is evidence only when *all* untargeted rendered
        // pixels remain byte-identical. Codec color transforms can otherwise
        // create false-positive visibility when an XObject is rewritten.
        if !unchanged_outside_source_masks(
            self.control,
            &treatment,
            &projections,
            display_mapper,
            padding,
        ) {
            return Ok(None);
        }
        let source_pdf_to_pixel = pixel_density(self.control, self.document, self.page)?;
        let expected_scale = PdfMatrix(paint.ctm).area_scale() * source_pdf_to_pixel
            / (f64::from(source.width) * f64::from(source.height));
        let mut observations = Vec::new();
        let mut proofs = Vec::new();
        let mut source_indices = Vec::new();
        for ((line, metric), index) in projections.iter().zip(&masked.metrics).zip(indices) {
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
            source_indices.push(index);
            observations.push(ImageObservation {
                page: self.page.get(),
                source_id: paint.source_id.clone(),
                paint_index: paint.paint_index,
                line: line.clone(),
            });
            proofs.push(VisibilityProof {
                page: self.page.get(),
                source_id: paint.source_id.clone(),
                paint_index: paint.paint_index,
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
            treatment,
            source_indices,
        }))
    }
}

/// Verify that all altered rendered pixels are confined to the selected
/// native OCR quads, with a conservative resampling/edge padding.
fn unchanged_outside_source_masks(
    before: &RgbImage,
    after: &RgbImage,
    regions: &[OcrLine],
    mapper: DisplayMapper,
    padding: f64,
) -> bool {
    let width = f64::from(before.width());
    let height = f64::from(before.height());
    let bounds = regions
        .iter()
        .map(|line| {
            let pixels = line.quad.map(|point| {
                let Point(x, y) = mapper.to_raster(point);
                Point(x * width, y * height)
            });
            let xmin = pixels.iter().map(|p| p.0).fold(f64::INFINITY, f64::min) - padding;
            let xmax = pixels.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max) + padding;
            let ymin = pixels.iter().map(|p| p.1).fold(f64::INFINITY, f64::min) - padding;
            let ymax = pixels.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) + padding;
            [xmin, ymin, xmax, ymax]
        })
        .collect::<Vec<_>>();
    for y in 0..before.height() {
        for x in 0..before.width() {
            let pixel = before.get_pixel(x, y);
            let candidate = after.get_pixel(x, y);
            if pixel.0 == candidate.0 {
                continue;
            }
            let center = Point(f64::from(x) + 0.5, f64::from(y) + 0.5);
            if !bounds
                .iter()
                .any(|b| (b[0]..=b[2]).contains(&center.0) && (b[1]..=b[3]).contains(&center.1))
            {
                return false;
            }
        }
    }
    true
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

    use super::{
        MixedVerifier, ResidualRaster, changed_raster_pixels, combine_disjoint_changes,
        unchanged_outside_source_masks,
    };
    use crate::{
        PageNumber, paint_scan::enumerate_image_paints, pdf_metadata::PdfMetadata,
        rasterize::RenderPolicy, source_ocr::SourceObservation,
    };
    use crate::{
        draft::OcrLine,
        geometry::{DisplayMapper, PageBox, Point, Rotation},
    };
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
    };

    #[test]
    fn pixel_composition_is_atomic_and_rejects_overlapping_changes() {
        let control = RgbImage::from_pixel(3, 1, Rgb([0; 3]));
        let mut first = control.clone();
        first.put_pixel(0, 0, Rgb([128; 3]));
        let mut second = control.clone();
        second.put_pixel(1, 0, Rgb([255; 3]));
        assert!(combine_disjoint_changes(&control, &mut first, &second));
        assert_eq!(first.as_raw(), &[128, 128, 128, 255, 255, 255, 0, 0, 0]);
        let before = first.clone();
        second.put_pixel(2, 0, Rgb([64; 3]));
        assert!(!combine_disjoint_changes(&control, &mut first, &second));
        assert_eq!(first, before);
    }

    #[test]
    fn native_font_transfer_only_applies_to_unchanged_control_pixels() -> Result<(), crate::Error> {
        let original = RgbImage::from_pixel(3, 1, Rgb([0; 3]));
        let mut treatment = original.clone();
        treatment.put_pixel(1, 0, Rgb([255; 3]));
        let residual = || ResidualRaster {
            image: treatment.clone(),
            required: BTreeSet::from([4]),
            paint_count: 1,
        };
        let mut baseline = original.clone();
        baseline.put_pixel(0, 0, Rgb([128; 3]));
        let result = residual()
            .transfer(&original, &baseline)
            .ok_or_else(|| crate::Error::Invalid("disjoint native change lost residual".into()))?;
        assert_eq!(
            result.image.as_raw(),
            &[128, 128, 128, 255, 255, 255, 0, 0, 0]
        );
        assert_eq!(result.required, BTreeSet::from([4]));
        baseline.put_pixel(1, 0, Rgb([128; 3]));
        assert!(residual().transfer(&original, &baseline).is_none());
        Ok(())
    }

    #[test]
    fn grouped_residual_reuses_disjoint_visible_paints_through_forms()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("visible.pdf");
        fs::write(
            &path,
            include_bytes!("../tests/fixtures/repeated-nested-paint-both-visible.pdf"),
        )?;
        let doc = PdfMetadata::open(&path)?;
        let pages = [
            PageNumber::new(1)?,
            PageNumber::new(2)?,
            PageNumber::new(3)?,
        ];
        let scan = enumerate_image_paints(&doc, &pages)?;
        let source = SourceObservation {
            width: 1200,
            height: 300,
            lines: vec![OcrLine {
                text: "CAPACITOR 47uF 16V".into(),
                confidence: 0.99,
                quad: [
                    Point(75.0, 120.0),
                    Point(1060.0, 120.0),
                    Point(1060.0, 200.0),
                    Point(75.0, 200.0),
                ],
                reviewed: false,
            }],
        };
        let recognized = BTreeMap::from([(scan.placements[0].source_id.clone(), source)]);
        let mut verifier = MixedVerifier::new(
            &path,
            &doc,
            &scan,
            &recognized,
            RenderPolicy {
                max_side: 1600,
                dpi: 72,
            },
            0.97,
        )?;
        for page in pages {
            let result = verifier.verify_page(page, Some(&[]))?;
            assert_eq!(
                result.evidence.proofs.len(),
                2,
                "{:#?}",
                result.evidence.audited
            );
            let residual = result.residual.ok_or("grouped treatment rejected")?;
            let control = result.control.ok_or("missing proof control")?;
            assert_eq!(residual.required, BTreeSet::from([0, 1]));
            assert_eq!(residual.paint_count, 2);
            let mut changed = [0, 0];
            for (i, (a, b)) in control.pixels().zip(residual.image.pixels()).enumerate() {
                if a != b {
                    let right = i % usize::try_from(control.width())?
                        >= usize::try_from(control.width())? / 2;
                    changed[usize::from(right)] += 1;
                }
            }
            assert!(changed.iter().all(|&n| n > 100), "{changed:?}");
        }
        Ok(())
    }

    #[test]
    fn counterfactual_rejects_any_untargeted_page_change() -> Result<(), crate::Error> {
        let before = RgbImage::from_pixel(100, 100, Rgb([0, 0, 0]));
        let mut after = before.clone();
        let quad = [
            Point(20.0, 60.0),
            Point(60.0, 60.0),
            Point(60.0, 80.0),
            Point(20.0, 80.0),
        ];
        let region = OcrLine {
            text: "bitmap text".into(),
            confidence: 0.99,
            quad,
            reviewed: false,
        };
        let mapper = DisplayMapper::new(PageBox::new([0.0, 0.0, 100.0, 100.0])?, Rotation::Deg0);
        after.put_pixel(30, 30, Rgb([255, 255, 255]));
        assert!(unchanged_outside_source_masks(
            &before,
            &after,
            std::slice::from_ref(&region),
            mapper,
            4.0
        ));
        after.put_pixel(80, 80, Rgb([10, 10, 10]));
        assert!(!unchanged_outside_source_masks(
            &before,
            &after,
            &[region],
            mapper,
            4.0
        ));
        Ok(())
    }

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
