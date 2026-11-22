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
    boundary_ink::{BoundaryEvidence, audit_edge_extensions},
    draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
    fragment_ink::Sample,
    geometry::{DisplayMapper, PageBox, Point, Quad, QuadBounds},
    glyph_appearance::GlyphShape,
    glyph_isolation::{InkBand, InkGroup, InkIsolation, InkRun, isolate_ink},
    ink_polarity::{Polarity, PolarityEvidence},
    layout_leaders::{LeaderEdge, LeaderEvidence, detect_leaders},
    line_frame::{LineAxis, LineFrame, SampleWindow, convex},
    pdf_metadata::PdfMetadata,
    pixel_regions::{InkBounds, PixelRegion},
    polygon::{area, clip_convex},
    rasterize::{calculate_crop, render_crop},
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
    pub padding_pixels: u32,
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
            padding_pixels: 0,
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

fn inside_page(display: DisplayMapper) -> impl Fn(Point) -> bool {
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
    pub added_strip: Quad,
    /// Every peer overlapping the expanded observation, including original-only
    /// overlaps. An empty list
    /// does not establish that the new ink is a character.
    pub overlaps: Vec<BoundaryOverlap>,
    /// Actual connected foreground samples, indexed by the overlaps array.
    pub ink_accounting: InkAccounting,
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
    pub requested_padding_pixels: u32,
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
        policy.padding_pixels,
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
    let map_run = |run: InkRun| MappedSupportedRun {
        ink_quad: window.quad(frame, run.ink),
        run,
    };
    let supported_runs = supported_runs.into_iter().map(map_run).collect();
    let supported_bands = supported_bands
        .into_iter()
        .map(|InkBand { run, column_runs }| MappedInkBand {
            band: map_run(run),
            column_runs: column_runs.into_iter().map(map_run).collect(),
        })
        .collect();
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
    pub(crate) fn measure_extensions(
        &mut self,
        page: &DraftPage,
        policy: &LineAuditPolicy,
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
            policy.max_ink_checks,
        )?;
        Ok(())
    }
}

fn boundary_overlap(
    expanded: &Quad,
    strip: &Quad,
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
    let intersection = clip_convex(strip, peer_quad);
    let overlap = area(&intersection);
    let strip_area = area(strip);
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
        .filter(|(_, c)| area(&clip_convex(&intersection, &c.ink_quad)) > 1e-9)
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

fn mapped_boundary_extensions(
    line: &AuditedLine,
    page: &DraftPage,
    sample: &Sample<'_>,
    mut remaining_checks: u64,
) -> Result<Vec<MappedBoundaryExtension>, Error> {
    let observation = line.observation_bounds;
    let frame = sample.frame;
    let window = sample.window;
    let candidates = audit_edge_extensions(
        line.components.iter().map(|c| &c.component),
        observation,
        line.width,
    );
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
            let strip = match evidence.edge {
                LeaderEdge::Left => InkBounds {
                    right: observation.left,
                    ..evidence.expanded
                },
                LeaderEdge::Right => InkBounds {
                    left: observation.right,
                    ..evidence.expanded
                },
            };
            let proposed_quad = window.quad(frame, evidence.expanded);
            let added_strip = window.quad(frame, strip);
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
                    boundary_overlap(&proposed_quad, &added_strip, &components, peer, quad)
                        .map(|overlap| (overlap, *quad))
                })
                .unzip();
            let ink_accounting =
                sample.account(evidence.expanded, &peer_quads, &mut remaining_checks)?;
            Ok(MappedBoundaryExtension {
                evidence,
                proposed_quad,
                added_strip,
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
            sampled.measure_extensions(original, policy)?;
            lines.push(sampled.audit);
        }
    }
    if sha256_file(pdf)? != draft.pdf_sha256 {
        return Err(Error::Invalid(
            "source PDF changed during line audit".into(),
        ));
    }
    Ok(LineAuditReport {
        schema_version: 9,
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
            let window = SampleWindow::new(frame, inside_page(display), 8, policy.max_pixels)?;
            assert_eq!(window.padding, [0, 8, 8, 8]);
            let original = window.observation(frame);
            for (actual, expected) in window.quad(frame, original).iter().zip(corners) {
                assert!(
                    (actual.0 - expected.0).abs() < 1e-10 && (actual.1 - expected.1).abs() < 1e-10
                );
            }
            assert!(SampleWindow::new(frame, inside_page(display), 8, 1).is_err());
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
