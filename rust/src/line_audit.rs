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
    draft::{OcrDraft, OcrLine, sha256_file},
    geometry::{DisplayMapper, PageBox, Point, Quad},
    glyph_appearance::GlyphShape,
    glyph_isolation::{InkBand, InkGroup, InkIsolation, InkRun, isolate_ink},
    ink_polarity::{Polarity, PolarityEvidence},
    layout_leaders::{LeaderEvidence, detect_leaders},
    line_frame::{LineAxis, LineFrame, SampleWindow},
    pdf_metadata::PdfMetadata,
    pixel_regions::{InkBounds, PixelRegion},
    rasterize::render_rgb_region,
};

/// Native source resolution and ink measurement settings.
#[derive(Clone, Debug)]
pub struct LineAuditPolicy {
    pub dpi: u32,
    pub max_pixels: u64,
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

/// Audit an original OCR observation without assigning text to individual ink groups.
#[derive(Debug, Serialize)]
pub struct AuditedLine {
    pub page: u32,
    pub line_index: usize,
    pub original_text: String,
    pub original_confidence: f64,
    pub original_quad: Quad,
    pub oriented_quad: Quad,
    /// Sampling quad, including actual within-page source padding.
    pub sample_quad: Quad,
    pub observation_bounds: InkBounds,
    pub width: u32,
    pub height: u32,
    pub axis_swapped: bool,
    pub polarity: PolarityEvidence,
    /// SHA-256 of the actual rectified luminance raster after polarity normalization.
    pub normalized_sha256: String,
    pub groups: Vec<MappedInkGroup>,
    pub components: Vec<MappedComponent>,
    pub supported_runs: Vec<MappedSupportedRun>,
    pub supported_bands: Vec<MappedInkBand>,
    pub leaders: Vec<MappedLeader>,
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

pub(crate) fn audit_line(
    pdf: &Path,
    metadata: &PdfMetadata,
    page: PageNumber,
    index: usize,
    line: &OcrLine,
    policy: &LineAuditPolicy,
) -> Result<AuditedLine, Error> {
    let frame = LineFrame::new(
        line.quad,
        |points| points * f64::from(policy.dpi) / 72.0,
        policy.max_pixels,
        LineAxis::Longest,
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
    let crop = render_rgb_region(
        pdf,
        page,
        display,
        envelope(&sample_quad)?,
        policy.dpi,
        policy.max_pixels,
    )?;
    let mapper = crop.pixel_mapper();
    let (normalized, polarity) = window.normalized_ink(
        frame,
        &crop.rgb,
        |point| mapper.to_raster(point),
        policy.polarity,
    )?;
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
    Ok(AuditedLine {
        page: page.get(),
        line_index: index,
        original_text: line.text.clone(),
        original_confidence: line.confidence,
        original_quad: line.quad,
        oriented_quad: frame.corners,
        sample_quad,
        observation_bounds: observation,
        width,
        height,
        axis_swapped: frame.axis_swapped,
        polarity,
        normalized_sha256: hex::encode(Sha256::digest(normalized.as_raw())),
        groups,
        components,
        supported_runs,
        supported_bands,
        leaders,
    })
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
            lines.push(audit_line(pdf, &metadata, page, index, line, policy)?);
        }
    }
    if sha256_file(pdf)? != draft.pdf_sha256 {
        return Err(Error::Invalid(
            "source PDF changed during line audit".into(),
        ));
    }
    Ok(LineAuditReport {
        schema_version: 6,
        pdf_sha256: draft.pdf_sha256.clone(),
        draft_model: draft.model.clone(),
        dpi: policy.dpi,
        segmentation_threshold: policy.segmentation_threshold,
        topology_thresholds: policy.topology_thresholds.clone(),
        requested_polarity: policy.polarity,
        requested_padding_pixels: policy.padding_pixels,
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
