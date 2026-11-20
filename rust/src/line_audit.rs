//! Source-PDF line audit: bounded native-resolution crops and geometry-only
//! ink isolation, with every sample mapped back to the original PDF.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use image::{GrayImage, Pixel};
use num_traits::ToPrimitive;
use serde::Serialize;

use crate::{
    Error, PageNumber,
    draft::{OcrDraft, OcrLine, sha256_file},
    geometry::{PageBox, Point, Quad},
    glyph_isolation::{InkBounds, InkGroup, isolate_ink},
    pdf_metadata::PdfMetadata,
    rasterize::{render_rgb_region, sample_whitened},
};

/// Native source resolution and ink measurement settings.
#[derive(Clone, Debug)]
pub struct LineAuditPolicy {
    pub dpi: u32,
    pub max_pixels: u64,
    pub segmentation_threshold: u8,
    pub topology_thresholds: Vec<u8>,
}

impl Default for LineAuditPolicy {
    fn default() -> Self {
        Self {
            dpi: 600,
            max_pixels: 2_000_000,
            segmentation_threshold: 192,
            topology_thresholds: vec![64, 128, 192, 224],
        }
    }
}

impl LineAuditPolicy {
    fn validate(&self) -> Result<(), Error> {
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

#[derive(Clone, Copy, Debug)]
struct LineFrame {
    corners: Quad,
    width: u32,
    height: u32,
    axis_swapped: bool,
}

fn distance(a: Point, b: Point) -> f64 {
    (b.0 - a.0).hypot(b.1 - a.1)
}

fn interpolate(a: Point, b: Point, t: f64) -> Point {
    Point((b.0 - a.0).mul_add(t, a.0), (b.1 - a.1).mul_add(t, a.1))
}

fn convex(quad: &Quad) -> bool {
    let mut positive = false;
    for index in 0..4 {
        let [a, b, c] = [quad[index], quad[(index + 1) % 4], quad[(index + 2) % 4]];
        let cross = (b.0 - a.0).mul_add(c.1 - b.1, -(b.1 - a.1) * (c.0 - b.0));
        if !cross.is_finite()
            || cross.abs() <= 1e-9
            || (index != 0 && cross.is_sign_positive() != positive)
        {
            return false;
        }
        positive = cross.is_sign_positive();
    }
    true
}

impl LineFrame {
    fn new(mut corners: Quad, policy: &LineAuditPolicy) -> Result<Self, Error> {
        if !convex(&corners) {
            return Err(Error::Invalid(
                "nonconvex or degenerate line audit quad".into(),
            ));
        }
        // Draft winding is BL/BR/TR/TL in the detector's frame.
        // For elongated vertical boxes, measure along the longer edge.
        // This establishes an axis, not the reading direction of the text.
        let horizontal = f64::midpoint(
            distance(corners[0], corners[1]),
            distance(corners[3], corners[2]),
        );
        let vertical = f64::midpoint(
            distance(corners[0], corners[3]),
            distance(corners[1], corners[2]),
        );
        let axis_swapped = vertical > 1.5 * horizontal;
        let (w, h) = if axis_swapped {
            corners.rotate_left(1);
            (vertical, horizontal)
        } else {
            (horizontal, vertical)
        };
        let pixels = |points: f64| {
            (points * f64::from(policy.dpi) / 72.0)
                .ceil()
                .to_u32()
                .filter(|&value| value > 0)
                .ok_or_else(|| Error::Invalid("unrepresentable line raster extent".into()))
        };
        let width = pixels(w)?;
        let height = pixels(h)?;
        if u64::from(width) * u64::from(height) > policy.max_pixels {
            return Err(Error::Invalid("oriented line exceeds raster budget".into()));
        }
        Ok(Self {
            corners,
            width,
            height,
            axis_swapped,
        })
    }

    /// Bilinear quad interpolation in pixel-corner coordinates.
    fn to_pdf(self, Point(x, y): Point) -> Point {
        let u = x / f64::from(self.width);
        let v = y / f64::from(self.height);
        let top = interpolate(self.corners[3], self.corners[2], u);
        let bottom = interpolate(self.corners[0], self.corners[1], u);
        interpolate(top, bottom, v)
    }

    fn quad(self, bounds: InkBounds) -> Quad {
        let l = f64::from(bounds.left);
        let r = f64::from(bounds.right);
        let t = f64::from(bounds.top);
        let b = f64::from(bounds.bottom);
        [Point(l, b), Point(r, b), Point(r, t), Point(l, t)].map(|point| self.to_pdf(point))
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
    pub width: u32,
    pub height: u32,
    pub axis_swapped: bool,
    pub groups: Vec<MappedInkGroup>,
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
    pub lines: Vec<AuditedLine>,
}

fn audit_line(
    pdf: &Path,
    metadata: &PdfMetadata,
    page: PageNumber,
    index: usize,
    line: &OcrLine,
    policy: &LineAuditPolicy,
) -> Result<AuditedLine, Error> {
    let frame = LineFrame::new(line.quad, policy)?;
    let display = metadata.mapper(page)?;
    for point in line.quad {
        let Point(u, v) = display.to_raster(point);
        if !(-1e-8..=1.0 + 1e-8).contains(&u) || !(-1e-8..=1.0 + 1e-8).contains(&v) {
            return Err(Error::Invalid(
                "line audit quad extends outside original page".into(),
            ));
        }
    }
    let crop = render_rgb_region(
        pdf,
        page,
        display,
        envelope(&line.quad)?,
        policy.dpi,
        policy.max_pixels,
    )?;
    let mapper = crop.pixel_mapper();
    let raster = GrayImage::from_fn(frame.width, frame.height, |x, y| {
        let source = frame.to_pdf(Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
        sample_whitened(&crop.rgb, mapper.to_raster(source)).to_luma()
    });
    let groups = isolate_ink(
        &raster,
        policy.segmentation_threshold,
        &policy.topology_thresholds,
    )?
    .into_iter()
    .map(|group| MappedInkGroup {
        ink_quad: frame.quad(group.ink),
        crop_quad: frame.quad(group.crop),
        group,
    })
    .collect();
    Ok(AuditedLine {
        page: page.get(),
        line_index: index,
        original_text: line.text.clone(),
        original_confidence: line.confidence,
        original_quad: line.quad,
        oriented_quad: frame.corners,
        width: frame.width,
        height: frame.height,
        axis_swapped: frame.axis_swapped,
        groups,
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
        schema_version: 1,
        pdf_sha256: draft.pdf_sha256.clone(),
        draft_model: draft.model.clone(),
        dpi: policy.dpi,
        segmentation_threshold: policy.segmentation_threshold,
        topology_thresholds: policy.topology_thresholds.clone(),
        lines,
    })
}

#[cfg(test)]
mod tests {
    use super::{LineAuditPolicy, LineFrame};
    use crate::{geometry::Point, glyph_isolation::InkBounds};

    #[test]
    fn quad_rectification_preserves_rotated_nonrectangular_geometry() -> Result<(), crate::Error> {
        let corners = [
            Point(10.0, 20.0),
            Point(30.0, 25.0),
            Point(28.0, 31.0),
            Point(8.0, 26.0),
        ];
        let frame = LineFrame::new(corners, &LineAuditPolicy::default())?;
        let bounds = InkBounds {
            left: 0,
            top: 0,
            right: frame.width,
            bottom: frame.height,
        };
        for (actual, expected) in frame.quad(bounds).iter().zip(corners) {
            assert!((actual.0 - expected.0).abs() < 1e-10 && (actual.1 - expected.1).abs() < 1e-10);
        }
        let center = frame.to_pdf(Point(
            f64::from(frame.width) / 2.0,
            f64::from(frame.height) / 2.0,
        ));
        assert!((center.0 - 19.0).abs() < 1e-10);
        assert!((center.1 - 25.5).abs() < 1e-10);
        let mut bad = corners;
        bad.swap(1, 2);
        assert!(LineFrame::new(bad, &LineAuditPolicy::default()).is_err());
        assert!(
            LineFrame::new(
                corners,
                &LineAuditPolicy {
                    max_pixels: 1,
                    ..LineAuditPolicy::default()
                }
            )
            .is_err()
        );
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
        let frame = LineFrame::new(corners, &LineAuditPolicy::default())?;
        assert!(frame.axis_swapped);
        assert!(frame.width > frame.height);
        assert_eq!(
            frame.corners,
            [corners[1], corners[2], corners[3], corners[0]]
        );
        Ok(())
    }
}
