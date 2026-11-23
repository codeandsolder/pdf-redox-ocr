//! Source-PDF line audit: bounded native-resolution crops and geometry-only
//! ink isolation, with every sample mapped back to the original PDF.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Error, PageNumber,
    boundary_ink::{BoundaryEvidence, audit_edge_extensions, measured_body_band, search_padding},
    draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
    foreign_ink::audit_strokes,
    fragment_ink::Sample,
    geometry::{DisplayMapper, PageBox, Point, Quad, QuadBounds},
    glyph_appearance::GlyphShape,
    glyph_isolation::{InkBand, InkGroup, InkIsolation, InkRun, isolate_ink},
    ink_polarity::{Polarity, PolarityEvidence},
    layout_leaders::{LeaderEdge, LeaderEvidence, detect_leaders},
    line_frame::{LineAxis, LineFrame, SampleWindow},
    pdf_metadata::PdfMetadata,
    pixel_regions::{InkBounds, PixelRegion},
    polygon::{area, clip_convex, convex},
    rasterize::{calculate_crop, render_crop},
};

pub use crate::foreign_ink::{
    ContractionEvidence, EdgeStroke, OrthogonalEdge, OrthogonalStroke, StripeProfile,
};
pub use crate::fragment_ink::{ComponentInk, InkAccounting, InkCounts};

/// Native source resolution and ink measurement settings.
#[derive(Clone, Debug)]
pub struct LineAuditPolicy {
    pub dpi: u32,
    pub max_pixels: u64,
    /// Total raster-pixel/region membership checks per audited line.
    pub max_ink_checks: u64,
    pub segmentation_threshold: u8,
    pub topology_thresholds: Vec<u8>,
    pub polarity: Polarity,
    /// Explicit left/top/right/bottom margins, or geometry-scaled margins when absent.
    pub padding_pixels: Option<[u32; 4]>,
}

impl Default for LineAuditPolicy {
    fn default() -> Self {
        Self {
            dpi: 600,
            max_pixels: 2_000_000,
            max_ink_checks: 32_000_000,
            segmentation_threshold: 192,
            topology_thresholds: vec![64, 128, 192, 224],
            polarity: Polarity::Auto,
            padding_pixels: Some([0; 4]),
        }
    }
}

impl LineAuditPolicy {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !(36..=2400).contains(&self.dpi)
            || self.max_pixels == 0
            || self.segmentation_threshold == 0
            || self.topology_thresholds.is_empty()
            || self.topology_thresholds.contains(&0)
        {
            return Err(Error::Invalid("invalid source line audit policy".into()));
        }
        Ok(())
    }
}

pub(crate) fn inside_page(display: DisplayMapper) -> impl Fn(Point) -> bool {
    move |point| {
        let Point(u, v) = display.to_raster(point);
        (-1e-8..=1.0 + 1e-8).contains(&u) && (-1e-8..=1.0 + 1e-8).contains(&v)
    }
}

fn envelope(quad: &Quad) -> Result<PageBox, Error> {
    let (mut l, mut b, mut r, mut t) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for &Point(x, y) in quad {
        l = l.min(x);
        r = r.max(x);
        b = b.min(y);
        t = t.max(y);
    }
    PageBox::new([l, b, r, t])
}

/// Geometry and topology for one isolated visible ink group.
#[derive(Debug, Serialize)]
pub struct MappedInkGroup {
    /// Half-open pixel bounds and per-threshold topology in the crop frame.
    #[serde(flatten)]
    pub group: InkGroup,
    /// Original PDF quad for the thresholded ink extent.
    pub ink_quad: Quad,
    /// Original PDF quad for the actual topology sampling window.
    pub crop_quad: Quad,
    /// Normalized source ink appearance, absent for clipped groups.
    pub appearance: Option<GlyphShape>,
}

/// A connected component with its original PDF-space bounding quad.
#[derive(Debug, Serialize)]
pub struct MappedComponent {
    #[serde(flatten)]
    pub component: PixelRegion,
    pub ink_quad: Quad,
}

/// Projection of components supported by the original observation.
#[derive(Debug, Serialize)]
pub struct MappedSupportedRun {
    #[serde(flatten)]
    pub run: InkRun,
    pub ink_quad: Quad,
}

/// Source-supported row band with independently mapped column runs.
#[derive(Debug, Serialize)]
pub struct MappedInkBand {
    #[serde(flatten)]
    pub band: MappedSupportedRun,
    pub column_runs: Vec<MappedSupportedRun>,
}

/// Geometry-only layout evidence; the source observations remain unchanged.
#[derive(Debug, Serialize)]
pub struct MappedLeader {
    #[serde(flatten)]
    pub evidence: LeaderEvidence,
    pub body_quad: Quad,
}

/// Exact peer observation potentially affected by an edge extension.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BoundaryPeer {
    OcrLine { line_index: usize },
    NativeText { region_index: usize },
}

/// Geometric competition, not a proof of pixel ownership or text identity.
#[derive(Debug, Serialize)]
pub struct BoundaryOverlap {
    pub peer: BoundaryPeer,
    /// Intersection divided by the newly added strip area.
    pub strip_fraction: f64,
    /// Intersection divided by the peer's entire quad area.
    pub peer_fraction: f64,
    /// Fraction of the peer quad covered by the entire proposed observation.
    pub expanded_peer_fraction: f64,
    /// Candidate component bounding quads intersecting the peer in the strip.
    pub component_indices: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct MappedBoundaryExtension {
    pub evidence: BoundaryEvidence,
    pub proposed_quad: Quad,
    /// Disjoint regions newly included by the proposed observation.
    pub added_strips: Vec<Quad>,
    /// Every peer overlapping the expanded observation, including original-only
    /// overlaps. An empty list
    /// does not establish that the new ink is a character.
    pub overlaps: Vec<BoundaryOverlap>,
    /// Actual connected foreground samples, indexed by the overlaps array.
    pub ink_accounting: InkAccounting,
}

#[derive(Debug, Serialize)]
pub struct MappedEdgeContraction {
    pub evidence: ContractionEvidence,
    pub proposed_quad: Quad,
}

#[derive(Debug, Serialize)]
pub struct MappedOrthogonalStroke {
    #[serde(flatten)]
    pub evidence: OrthogonalStroke,
    pub ink_quad: Quad,
}

#[derive(Debug, Serialize)]
pub struct MappedStrokeAudit {
    pub contractions: Vec<MappedEdgeContraction>,
    pub orthogonal: Vec<MappedOrthogonalStroke>,
}

/// Audit an original OCR observation without assigning text to individual ink groups.
#[derive(Debug, Serialize)]
pub struct AuditedLine {
    pub page: u32,
    pub line_index: usize,
    pub original_text: String,
    pub original_confidence: f64,
    pub original_quad: Quad,
    /// Sampling quad, including actual within-page source padding.
    pub sample_quad: Quad,
    pub observation_bounds: InkBounds,
    pub width: u32,
    pub height: u32,
    pub polarity: PolarityEvidence,
    /// SHA-256 of the actual rectified luminance raster after polarity normalization.
    pub normalized_sha256: String,
    pub groups: Vec<MappedInkGroup>,
    pub components: Vec<MappedComponent>,
    pub supported_runs: Vec<MappedSupportedRun>,
    pub supported_bands: Vec<MappedInkBand>,
    pub leaders: Vec<MappedLeader>,
    /// Candidate extensions in the original classified reading-order frame.
    pub boundary_extensions: Vec<MappedBoundaryExtension>,
    /// Measured by source line audit or reconciliation; None if not requested
    /// or the shared membership budget is exhausted. Never a deletion decision.
    pub foreign_strokes: Option<MappedStrokeAudit>,
}

/// Provenance-bound geometric evidence, not an OCR replacement draft.
#[derive(Debug, Serialize)]
pub struct LineAuditReport {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub draft_model: String,
    pub dpi: u32,
    pub segmentation_threshold: u8,
    pub topology_thresholds: Vec<u8>,
    pub requested_polarity: Polarity,
    pub requested_padding_pixels: Option<[u32; 4]>,
    pub max_ink_checks: u64,
    pub lines: Vec<AuditedLine>,
}

fn mapped_leaders(
    components: &[PixelRegion],
    window: SampleWindow,
    frame: LineFrame,
) -> Vec<MappedLeader> {
    detect_leaders(components, window.observation(frame))
        .into_iter()
        .map(|evidence| MappedLeader {
            body_quad: window.quad(frame, evidence.body),
            evidence,
        })
        .collect()
}

fn mapped_groups(
    groups: Vec<InkGroup>,
    normalized: &image::GrayImage,
    window: SampleWindow,
    frame: LineFrame,
) -> Result<Vec<MappedInkGroup>, Error> {
    groups
        .into_iter()
        .map(|group| {
            let appearance =
                if group.touches_line_edge || group.run.observation_support.outside_pixels > 0 {
                    None
                } else {
                    Some(GlyphShape::measure(normalized, group.run.ink)?)
                };
            Ok(MappedInkGroup {
                ink_quad: window.quad(frame, group.run.ink),
                crop_quad: window.quad(frame, group.crop),
                group,
                appearance,
            })
        })
        .collect()
}

fn mapped_run(run: InkRun, window: SampleWindow, frame: LineFrame) -> MappedSupportedRun {
    MappedSupportedRun {
        ink_quad: window.quad(frame, run.ink),
        run,
    }
}
fn mapped_runs(
    runs: Vec<InkRun>,
    window: SampleWindow,
    frame: LineFrame,
) -> Vec<MappedSupportedRun> {
    runs.into_iter()
        .map(|run| mapped_run(run, window, frame))
        .collect()
}
fn mapped_bands(bands: Vec<InkBand>, window: SampleWindow, frame: LineFrame) -> Vec<MappedInkBand> {
    bands
        .into_iter()
        .map(|InkBand { run, column_runs }| MappedInkBand {
            band: mapped_run(run, window, frame),
            column_runs: mapped_runs(column_runs, window, frame),
        })
        .collect()
}

pub(crate) struct SampledLine {
    pub audit: AuditedLine,
    pub(crate) normalized: image::GrayImage,
    #[cfg(feature = "native-inference")]
    pub(crate) rgb: image::RgbImage,
    pub(crate) frame: LineFrame,
    pub(crate) window: SampleWindow,
}

fn charge_sample_pixels(
    remaining: Option<&mut u64>,
    rendered: u64,
    (width, height): (u32, u32),
) -> Result<(), Error> {
    if let Some(remaining) = remaining {
        let cost = rendered
            .checked_add(u64::from(width) * u64::from(height))
            .ok_or_else(|| Error::Invalid("line audit pixel cost overflow".into()))?;
        *remaining = remaining
            .checked_sub(cost)
            .ok_or_else(|| Error::Invalid("fragment audit page pixel budget exhausted".into()))?;
    }
    Ok(())
}

pub(crate) fn audit_line(
    pdf: &Path,
    metadata: &PdfMetadata,
    page: PageNumber,
    index: usize,
    line: &OcrLine,
    policy: &LineAuditPolicy,
    remaining_pixels: Option<&mut u64>,
) -> Result<SampledLine, Error> {
    let frame = LineFrame::new(
        line.quad,
        |points| points * f64::from(policy.dpi) / 72.0,
        policy.max_pixels,
        LineAxis::ReadingOrder,
    )?;
    let display = metadata.mapper(page)?;
    let window = SampleWindow::new(
        frame,
        inside_page(display),
        policy
            .padding_pixels
            .unwrap_or_else(|| search_padding(frame.height)),
        policy.max_pixels,
    )?;
    let (width, height) = window
        .dimensions(frame)
        .ok_or_else(|| Error::Invalid("unrepresentable sample window".into()))?;
    let observation = window.observation(frame);
    let sample_quad = window.quad(
        frame,
        InkBounds {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        },
    );
    let request = calculate_crop(
        display,
        envelope(&sample_quad)?,
        policy.dpi,
        policy.max_pixels,
    )?;
    charge_sample_pixels(remaining_pixels, request.pixels(), (width, height))?;
    let crop = render_crop(pdf, page, &request)?;
    let mapper = crop.pixel_mapper();
    let rgb = window.rectify(
        frame,
        &crop.rgb,
        |point| mapper.to_raster(point),
        std::convert::identity,
    )?;
    let gray = image::GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        image::Pixel::to_luma(rgb.get_pixel(x, y))
    });
    let (normalized, polarity) = window.normalize_ink(frame, &gray, policy.polarity)?;
    let InkIsolation {
        components,
        groups,
        supported_runs,
        supported_bands,
    } = isolate_ink(
        &normalized,
        observation,
        policy.segmentation_threshold,
        &policy.topology_thresholds,
    )?;
    let groups = mapped_groups(groups, &normalized, window, frame)?;
    let leaders = mapped_leaders(&components, window, frame);
    let components = components
        .into_iter()
        .map(|component| MappedComponent {
            ink_quad: window.quad(frame, component.bounds),
            component,
        })
        .collect();
    let supported_runs = mapped_runs(supported_runs, window, frame);
    let supported_bands = mapped_bands(supported_bands, window, frame);
    let audit = AuditedLine {
        page: page.get(),
        line_index: index,
        original_text: line.text.clone(),
        original_confidence: line.confidence,
        original_quad: line.quad,
        sample_quad,
        observation_bounds: observation,
        width,
        height,
        polarity,
        normalized_sha256: hex::encode(Sha256::digest(normalized.as_raw())),
        groups,
        components,
        supported_runs,
        supported_bands,
        leaders,
        boundary_extensions: Vec::new(),
        foreign_strokes: None,
    };
    Ok(SampledLine {
        audit,
        normalized,
        #[cfg(feature = "native-inference")]
        rgb,
        frame,
        window,
    })
}

impl SampledLine {
    pub(crate) fn measure_strokes(
        &mut self,
        policy: &LineAuditPolicy,
        remaining_checks: &mut u64,
    ) -> Result<(), Error> {
        let components = self
            .audit
            .components
            .iter()
            .map(|c| &c.component)
            .collect::<Vec<_>>();
        let strokes = audit_strokes(
            &self.normalized,
            policy.segmentation_threshold,
            self.audit.observation_bounds,
            &components,
            remaining_checks,
        )?;
        self.audit.foreign_strokes = strokes.map(|audit| MappedStrokeAudit {
            contractions: audit
                .contractions
                .into_iter()
                .map(|evidence| MappedEdgeContraction {
                    proposed_quad: self.window.quad(self.frame, evidence.retained),
                    evidence,
                })
                .collect(),
            orthogonal: audit
                .orthogonal
                .into_iter()
                .map(|evidence| MappedOrthogonalStroke {
                    ink_quad: self.window.quad(self.frame, evidence.stroke.inside_bounds),
                    evidence,
                })
                .collect(),
        });
        Ok(())
    }

    pub(crate) fn measure_extensions(
        &mut self,
        page: &DraftPage,
        policy: &LineAuditPolicy,
        horizon: [u32; 2],
        remaining_checks: &mut u64,
    ) -> Result<(), Error> {
        self.audit.boundary_extensions = mapped_boundary_extensions(
            &self.audit,
            page,
            &Sample {
                image: &self.normalized,
                frame: self.frame,
                window: self.window,
                threshold: policy.segmentation_threshold,
            },
            horizon,
            remaining_checks,
        )?;
        Ok(())
    }
}

fn boundary_overlap(
    expanded: &Quad,
    strips: &[Quad],
    candidate_components: &[(usize, &MappedComponent)],
    peer: BoundaryPeer,
    peer_quad: &Quad,
) -> Option<BoundaryOverlap> {
    if !QuadBounds::new(expanded).intersects(QuadBounds::new(peer_quad)) {
        return None;
    }
    let expanded_overlap = area(&clip_convex(expanded, peer_quad));
    if !expanded_overlap.is_finite() || expanded_overlap <= 1e-9 {
        return None;
    }
    let intersections = strips
        .iter()
        .map(|strip| clip_convex(strip, peer_quad))
        .collect::<Vec<_>>();
    let overlap = intersections.iter().map(|i| area(i)).sum::<f64>();
    let strip_area = strips.iter().map(|q| area(q)).sum::<f64>();
    let peer_area = area(peer_quad);
    if !overlap.is_finite()
        || [strip_area, peer_area]
            .iter()
            .any(|a| !a.is_finite() || *a <= 1e-9)
    {
        return None;
    }
    let component_indices = candidate_components
        .iter()
        .filter(|(_, c)| {
            intersections
                .iter()
                .any(|i| area(&clip_convex(i, &c.ink_quad)) > 1e-9)
        })
        .map(|(i, _)| *i)
        .collect();
    Some(BoundaryOverlap {
        peer,
        strip_fraction: (overlap / strip_area).clamp(0.0, 1.0),
        peer_fraction: (overlap / peer_area).clamp(0.0, 1.0),
        expanded_peer_fraction: (expanded_overlap / peer_area).clamp(0.0, 1.0),
        component_indices,
    })
}

/// The reading-axis strip covers the full proposed band. Top/bottom
/// extensions are limited to the original reading-axis interval, so their
/// union is exactly expanded minus original without overlapping areas.
fn added_strips(original: InkBounds, expanded: InkBounds, edge: LeaderEdge) -> Vec<InkBounds> {
    let side = match edge {
        LeaderEdge::Left => InkBounds {
            right: original.left,
            ..expanded
        },
        LeaderEdge::Right => InkBounds {
            left: original.right,
            ..expanded
        },
    };
    let mut strips = vec![side];
    if expanded.top < original.top {
        strips.push(InkBounds {
            top: expanded.top,
            bottom: original.top,
            ..original
        });
    }
    if expanded.bottom > original.bottom {
        strips.push(InkBounds {
            top: original.bottom,
            bottom: expanded.bottom,
            ..original
        });
    }
    strips
}

fn boundary_candidates(
    line: &AuditedLine,
    sample: &Sample<'_>,
    horizon: [u32; 2],
) -> Result<Vec<BoundaryEvidence>, Error> {
    let original = line.observation_bounds;
    let components = line
        .components
        .iter()
        .map(|c| &c.component)
        .collect::<Vec<_>>();
    if let Some((band, measured)) =
        measured_body_band(sample.image, sample.threshold, original, &components)?
    {
        Ok(audit_edge_extensions(&measured, band, line.width, horizon))
    } else {
        Ok(audit_edge_extensions(
            components, original, line.width, horizon,
        ))
    }
}

fn mapped_boundary_extensions(
    line: &AuditedLine,
    page: &DraftPage,
    sample: &Sample<'_>,
    horizon: [u32; 2],
    remaining_checks: &mut u64,
) -> Result<Vec<MappedBoundaryExtension>, Error> {
    let observation = line.observation_bounds;
    let frame = sample.frame;
    let window = sample.window;
    let candidates = boundary_candidates(line, sample, horizon)?;
    if !candidates.is_empty()
        && page
            .lines
            .iter()
            .map(|l| &l.quad)
            .chain(&page.native_text_regions)
            .any(|q| !convex(q))
    {
        return Err(Error::Invalid(
            "invalid peer geometry in boundary audit".into(),
        ));
    }
    candidates
        .into_iter()
        .map(|evidence| {
            let proposed_quad = window.quad(frame, evidence.expanded);
            let added_strips = added_strips(observation, evidence.expanded, evidence.edge)
                .into_iter()
                .map(|b| window.quad(frame, b))
                .collect::<Vec<_>>();
            let components = evidence
                .component_indices
                .iter()
                .filter_map(|&i| line.components.get(i).map(|c| (i, c)))
                .collect::<Vec<_>>();
            let peers = page
                .lines
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != line.line_index)
                .map(|(line_index, l)| (BoundaryPeer::OcrLine { line_index }, &l.quad));
            let native = page
                .native_text_regions
                .iter()
                .enumerate()
                .map(|(region_index, quad)| (BoundaryPeer::NativeText { region_index }, quad));
            let (overlaps, peer_quads): (Vec<_>, Vec<_>) = peers
                .chain(native)
                .filter_map(|(peer, quad)| {
                    boundary_overlap(&proposed_quad, &added_strips, &components, peer, quad)
                        .map(|overlap| (overlap, *quad))
                })
                .unzip();
            let ink_accounting =
                sample.account(evidence.expanded, &peer_quads, remaining_checks)?;
            Ok(MappedBoundaryExtension {
                evidence,
                proposed_quad,
                added_strips,
                overlaps,
                ink_accounting,
            })
        })
        .collect()
}

/// Inspect explicitly selected line geometry in an unchanged, SHA-bound PDF.
///
/// Only the original visible PDF paint supplies pixels. The returned groups
/// are not individual Unicode characters, nor are they presumed associated
/// with the recognized text. There is no inference or modification of the draft.
///
/// # Errors
/// Rejects invalid targets, mismatched PDF hashes, unsupported geometry, or
/// source/rectified raster sizes exceeding the configured per-line budget.
pub fn audit_lines(
    pdf: &Path,
    draft: &OcrDraft,
    targets: &BTreeMap<PageNumber, BTreeSet<usize>>,
    policy: &LineAuditPolicy,
) -> Result<LineAuditReport, Error> {
    policy.validate()?;
    draft.verify_pdf(pdf)?;
    if targets.is_empty() {
        return Err(Error::Invalid(
            "line audit requires explicit targets".into(),
        ));
    }
    let metadata = PdfMetadata::open(pdf)?;
    let mut by_page = BTreeMap::new();
    for page in &draft.pages {
        let number = PageNumber::new(page.page)?;
        metadata.mapper(number)?;
        if by_page.insert(number, page).is_some() {
            return Err(Error::Invalid("duplicate source draft page".into()));
        }
    }
    let mut lines = Vec::new();
    for (&page, indices) in targets {
        let original = by_page
            .get(&page)
            .ok_or_else(|| Error::Invalid("audit page missing from draft".into()))?;
        if indices.is_empty() {
            return Err(Error::Invalid("empty line audit target set".into()));
        }
        for &index in indices {
            let line = original
                .lines
                .get(index)
                .ok_or_else(|| Error::Invalid("audit line index out of range".into()))?;
            if line.text.trim().is_empty() || !(0.0..=1.0).contains(&line.confidence) {
                return Err(Error::Invalid("invalid original OCR observation".into()));
            }
            let mut sampled = audit_line(pdf, &metadata, page, index, line, policy, None)?;
            let mut remaining_checks = policy.max_ink_checks;
            sampled.measure_extensions(original, policy, [0; 2], &mut remaining_checks)?;
            sampled.measure_strokes(policy, &mut remaining_checks)?;
            lines.push(sampled.audit);
        }
    }
    if sha256_file(pdf)? != draft.pdf_sha256 {
        return Err(Error::Invalid(
            "source PDF changed during line audit".into(),
        ));
    }
    Ok(LineAuditReport {
        schema_version: 16,
        pdf_sha256: draft.pdf_sha256.clone(),
        draft_model: draft.model.clone(),
        dpi: policy.dpi,
        segmentation_threshold: policy.segmentation_threshold,
        topology_thresholds: policy.topology_thresholds.clone(),
        requested_polarity: policy.polarity,
        requested_padding_pixels: policy.padding_pixels,
        max_ink_checks: policy.max_ink_checks,
        lines,
    })
}

#[cfg(test)]
mod tests {
    use super::{LineAuditPolicy, LineAxis, LineFrame, SampleWindow, inside_page};
    use crate::{
        geometry::{DisplayMapper, PageBox, Point, Rotation},
        pixel_regions::InkBounds,
    };

    #[test]
    fn quad_rectification_preserves_rotated_nonrectangular_geometry() -> Result<(), crate::Error> {
        let corners = [
            Point(10.0, 20.0),
            Point(30.0, 25.0),
            Point(28.0, 31.0),
            Point(8.0, 26.0),
        ];
        let frame = LineFrame::new(
            corners,
            |points| points * 600.0 / 72.0,
            2_000_000,
            LineAxis::Longest,
        )?;
        let bounds = InkBounds {
            left: 0,
            top: 0,
            right: frame.width,
            bottom: frame.height,
        };
        let window = SampleWindow { padding: [0; 4] };
        for (actual, expected) in window.quad(frame, bounds).iter().zip(corners) {
            assert!((actual.0 - expected.0).abs() < 1e-10 && (actual.1 - expected.1).abs() < 1e-10);
        }
        let center = frame.to_source(Point(
            f64::from(frame.width) / 2.0,
            f64::from(frame.height) / 2.0,
        ));
        assert!((center.0 - 19.0).abs() < 1e-10);
        assert!((center.1 - 25.5).abs() < 1e-10);
        let mut bad = corners;
        bad.swap(1, 2);
        assert!(
            LineFrame::new(
                bad,
                |points| points * 600.0 / 72.0,
                2_000_000,
                LineAxis::Longest
            )
            .is_err()
        );
        assert!(
            LineFrame::new(
                corners,
                |points| points * 600.0 / 72.0,
                1,
                LineAxis::Longest
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn sampling_padding_respects_page_boundary_and_preserves_original_mapping()
    -> Result<(), crate::Error> {
        let corners = [
            Point(30.0, 50.0),
            Point(50.0, 50.0),
            Point(50.0, 60.0),
            Point(30.0, 60.0),
        ];
        let policy = LineAuditPolicy::default();
        let frame = LineFrame::new(
            corners,
            |points| points * f64::from(policy.dpi) / 72.0,
            policy.max_pixels,
            LineAxis::Longest,
        )?;
        let page = PageBox::new([30.0, 40.0, 190.0, 160.0])?;
        for rotation in [
            Rotation::Deg0,
            Rotation::Deg90,
            Rotation::Deg180,
            Rotation::Deg270,
        ] {
            let display = DisplayMapper::new(page, rotation);
            let window = SampleWindow::new(frame, inside_page(display), [8; 4], policy.max_pixels)?;
            assert_eq!(window.padding, [0, 8, 8, 8]);
            let original = window.observation(frame);
            for (actual, expected) in window.quad(frame, original).iter().zip(corners) {
                assert!(
                    (actual.0 - expected.0).abs() < 1e-10 && (actual.1 - expected.1).abs() < 1e-10
                );
            }
            assert!(SampleWindow::new(frame, inside_page(display), [8; 4], 1).is_err());
        }
        Ok(())
    }

    #[test]
    fn search_padding_scales_with_line_axis_and_page_clip() -> Result<(), crate::Error> {
        let frame = LineFrame::new(
            [
                Point(30., 50.),
                Point(80., 50.),
                Point(80., 60.),
                Point(30., 60.),
            ],
            |n| n,
            2_000_000,
            LineAxis::ReadingOrder,
        )?;
        let page = PageBox::new([20., 20., 200., 200.])?;
        let request = super::search_padding(frame.height);
        assert_eq!(request, [23, 10, 23, 10]);
        let display = DisplayMapper::new(page, Rotation::Deg90);
        let window = SampleWindow::new(frame, inside_page(display), request, 2_000_000)?;
        assert_eq!(window.padding, [10, 10, 23, 10]);
        for (actual, expected) in window
            .quad(frame, window.observation(frame))
            .iter()
            .zip(frame.corners)
        {
            assert!((actual.0 - expected.0).abs() < 1e-10 && (actual.1 - expected.1).abs() < 1e-10);
        }
        Ok(())
    }

    #[test]
    fn vertical_axis_rotation_keeps_original_corner_identity() -> Result<(), crate::Error> {
        let corners = [
            Point(0.0, 0.0),
            Point(4.0, 0.0),
            Point(4.0, 20.0),
            Point(0.0, 20.0),
        ];
        let frame = LineFrame::new(
            corners,
            |points| points * 600.0 / 72.0,
            2_000_000,
            LineAxis::Longest,
        )?;
        assert!(frame.axis_swapped);
        assert!(frame.width > frame.height);
        assert_eq!(
            frame.corners,
            [corners[1], corners[2], corners[3], corners[0]]
        );
        Ok(())
    }
}
