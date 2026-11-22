//! Source-resolution OCR on PDF-painted image fragment mosaics.
//!
//! Reconstructing from the original cropped final PDF raster handles every
//! image codec, alpha mask and PDF clip that Poppler can render. The exact
//! display-to-source transform is carried through all upright rotations.

use std::{collections::HashSet, path::Path};

use image::{Rgb, RgbImage};

use crate::{
    Error,
    draft::OcrLine,
    geometry::{DisplayMapper, Point, Quad, QuadBounds},
    mosaic::{MosaicAxes, MosaicBudget, MosaicPlan},
    paint_scan::ImagePaint,
    rasterize::{CropPixelMapper, render_rgb_region, sample_whitened},
};

fn inside_convex(point: Point, polygon: &Quad) -> bool {
    if crate::polygon::area(polygon) <= 1e-9 {
        return false;
    }
    let (mut positive, mut negative) = (false, false);
    for index in 0..4 {
        let from = polygon[index];
        let to = polygon[(index + 1) % 4];
        let cross =
            (to.0 - from.0).mul_add(point.1 - from.1, -(to.1 - from.1) * (point.0 - from.0));
        if cross > 1e-5 {
            positive = true;
        } else if cross < -1e-5 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

fn disjoint_groups(plan: &MosaicPlan, paints: &[ImagePaint]) -> bool {
    let memberships = plan
        .groups
        .iter()
        .enumerate()
        .flat_map(|(group, item)| {
            item.paint_indices
                .iter()
                .copied()
                .map(move |index| (index, group))
        })
        .collect::<std::collections::HashMap<_, _>>();
    for (i, first) in paints.iter().enumerate() {
        for next in &paints[i + 1..] {
            if memberships.get(&first.paint_index) != memberships.get(&next.paint_index)
                && QuadBounds::new(&first.page_quad).intersects(QuadBounds::new(&next.page_quad))
            {
                return false;
            }
        }
    }
    true
}

fn paint_clip_mask(
    rgb: &mut RgbImage,
    mapper: CropPixelMapper,
    paints: &[&ImagePaint],
) -> Result<(), Error> {
    let width = rgb.width();
    let height = rgb.height();
    let pixels = u64::from(width) * u64::from(height);
    let len =
        usize::try_from(pixels).map_err(|_| Error::Invalid("mosaic mask too large".into()))?;
    let mut painted = vec![false; len];
    for paint in paints {
        let quad = paint.page_quad.map(|point| mapper.to_raster(point));
        let QuadBounds {
            min_x: mut left,
            min_y: mut top,
            max_x: mut right,
            max_y: mut bottom,
        } = QuadBounds::new(&quad);
        // Pixel-center masks may be a fraction of a pixel smaller than a
        // final painted image. One pixel guard preserves antialiased borders.
        left = (left.floor() - 1.0).max(0.0);
        top = (top.floor() - 1.0).max(0.0);
        right = (right.ceil() + 1.0).min(f64::from(width));
        bottom = (bottom.ceil() + 1.0).min(f64::from(height));
        let index = |value: f64| num_traits::ToPrimitive::to_u32(&value).unwrap_or(0);
        let left = index(left);
        let top = index(top);
        let right = index(right);
        let bottom = index(bottom);
        let clips = paint
            .clips
            .iter()
            .map(|quad| quad.map(|p| mapper.to_raster(p)))
            .collect::<Vec<_>>();
        for y in top..bottom {
            for x in left..right {
                let center = Point(f64::from(x) + 0.5, f64::from(y) + 0.5);
                if !inside_convex(center, &quad)
                    || clips.iter().any(|clip| !inside_convex(center, clip))
                {
                    continue;
                }
                let pos = usize::try_from(u64::from(y) * u64::from(width) + u64::from(x))
                    .map_err(|_| Error::Invalid("invalid native mosaic coverage".into()))?;
                painted[pos] = true;
            }
        }
    }
    for (pixel, &visible) in rgb.pixels_mut().zip(&painted) {
        if !visible {
            *pixel = Rgb([255, 255, 255]);
        }
    }
    Ok(())
}

fn group_dpi(pitch: f64, limit: u32) -> Option<u32> {
    if !pitch.is_finite() || pitch <= 0.0 {
        return None;
    }
    let target = (72.0 / pitch).round().max(36.0);
    let dpi = num_traits::ToPrimitive::to_u32(&target)?;
    (dpi <= limit).then_some(dpi)
}

/// Exact image-space mapping for the oriented mosaic raster.
///
/// Pixel positions refer to *corners*. The same affine transform creates
/// pixels and converts the OCR model's detected polygon into original PDF
/// space, avoiding a second independent transform calculation.
#[derive(Clone, Copy, Debug)]
struct OrientedFrame {
    axes: MosaicAxes,
    start: Point,
}

impl OrientedFrame {
    fn to_input(self, Point(x, y): Point) -> Point {
        self.axes
            .unproject(Point(self.start.0 + x, self.start.1 + y))
    }
}

fn oriented_bounds(
    paints: &[&ImagePaint],
    mapper: CropPixelMapper,
    axes: MosaicAxes,
) -> Option<(OrientedFrame, u32, u32)> {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for paint in paints {
        for point in paint.page_quad {
            let corner = axes.project(mapper.to_raster(point));
            min_x = min_x.min(corner.0);
            max_x = max_x.max(corner.0);
            min_y = min_y.min(corner.1);
            max_y = max_y.max(corner.1);
        }
    }
    if ![min_x, max_x, min_y, max_y]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }
    // Two source-pixel guards allow bilinear interpolation near the outer
    // paint boundary without cutting off antialiased glyph strokes.
    let x0 = min_x.floor() - 2.0;
    let y0 = min_y.floor() - 2.0;
    let x1 = max_x.ceil() + 2.0;
    let y1 = max_y.ceil() + 2.0;
    Some((
        OrientedFrame {
            axes,
            start: Point(x0, y0),
        },
        num_traits::ToPrimitive::to_u32(&(x1 - x0))?,
        num_traits::ToPrimitive::to_u32(&(y1 - y0))?,
    ))
}

/// Rectify a PDF-composited crop into a native-text-upright affine frame.
///
/// Rejects degenerate or out-of-budget resulting raster dimensions.
fn rectify_group(
    input: &RgbImage,
    mapper: CropPixelMapper,
    paints: &[&ImagePaint],
    axes: MosaicAxes,
    max_pixels: u64,
) -> Option<(RgbImage, OrientedFrame)> {
    let (frame, width, height) = oriented_bounds(paints, mapper, axes)?;
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > max_pixels {
        return None;
    }
    let mut output = RgbImage::new(width, height);
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let source = frame.to_input(Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
        *pixel = sample_whitened(input, source);
    }
    Some((output, frame))
}

/// Perform complete native-pitch OCR on a fragmented, image-only page.
///
/// The caller supplies OCR over mapped pixels; this also allows deterministic
/// renderer/coordinate tests without requiring ONNX model downloads.
///
/// Each group is rasterized from the actual PDF stack, but pixels outside
/// the selected group's image paints are cleared to prevent nearby groups
/// contaminating detections. One rejected group retains full-page fallback.
///
/// # Errors
/// Returns an error when native renderer, OCR model, or crop masks fail.
pub fn reconstruct_native_mosaics(
    pdf: &Path,
    page_mapper: DisplayMapper,
    plan: &MosaicPlan,
    paints: &[ImagePaint],
    budget: MosaicBudget,
    mut recognize: impl FnMut(usize, RgbImage, &dyn Fn(Point) -> Point) -> Result<Vec<OcrLine>, Error>,
) -> Result<Option<Vec<OcrLine>>, Error> {
    let budget = budget.validate()?;
    if !disjoint_groups(plan, paints) {
        return Ok(None);
    }
    let mut lines = Vec::new();
    let mut total_pixels = 0_u64;
    let mut seen = HashSet::new();
    for (group_index, group) in plan.groups.iter().enumerate() {
        let Some(dpi) = group_dpi(group.pitch, budget.dpi) else {
            return Ok(None);
        };
        let mut crop = render_rgb_region(
            pdf,
            group.page,
            page_mapper,
            group.region,
            dpi,
            budget.group_pixels,
        )?;
        total_pixels =
            total_pixels.saturating_add(u64::from(crop.rgb.width()) * u64::from(crop.rgb.height()));
        if total_pixels > budget.page_pixels {
            return Ok(None);
        }
        let members = paints
            .iter()
            .filter(|paint| group.paint_indices.contains(&paint.paint_index))
            .collect::<Vec<_>>();
        if members.len() != group.paint_indices.len()
            || members.iter().any(|paint| !seen.insert(paint.paint_index))
        {
            return Err(Error::Invalid(
                "native mosaic group omitted or duplicated image paint".into(),
            ));
        }
        let mapper = crop.pixel_mapper();
        paint_clip_mask(&mut crop.rgb, mapper, &members)?;
        let Some((oriented, frame)) =
            rectify_group(&crop.rgb, mapper, &members, group.axes, budget.group_pixels)
        else {
            return Ok(None);
        };
        let output_pixels = u64::from(oriented.width()) * u64::from(oriented.height());
        total_pixels = total_pixels.saturating_add(output_pixels);
        if total_pixels > budget.page_pixels {
            return Ok(None);
        }
        let pixel_to_pdf = |point| mapper.to_pdf(frame.to_input(point));
        let mut result = recognize(group_index, oriented, &pixel_to_pdf)?;
        lines.append(&mut result);
    }
    if seen.len() != paints.len() {
        return Err(Error::Invalid(
            "native mosaic did not reconstruct every paint".into(),
        ));
    }
    if lines.is_empty() {
        return Ok(None);
    }
    Ok(Some(lines))
}

#[cfg(test)]
mod tests {
    use super::{group_dpi, inside_convex};
    use crate::geometry::Point;

    #[test]
    fn pixel_pitch_to_dpi_is_bounded_without_truncating() {
        assert_eq!(group_dpi(0.24, 2400), Some(300));
        assert_eq!(group_dpi(0.30, 2400), Some(240));
        assert_eq!(group_dpi(0.01, 2400), None);
        assert_eq!(group_dpi(10.0, 2400), Some(36));
        assert_eq!(group_dpi(0.0, 2400), None);
        assert_eq!(group_dpi(f64::NAN, 2400), None);
    }

    #[test]
    fn winding_independent_coverage_test() {
        let rectangle = [
            Point(10.0, 10.0),
            Point(50.0, 10.0),
            Point(50.0, 40.0),
            Point(10.0, 40.0),
        ];
        assert!(inside_convex(Point(20.0, 20.0), &rectangle));
        assert!(!inside_convex(
            Point(20.0, 20.0),
            &rectangle.map(|_| Point(0.0, 0.0))
        ));
        assert!(!inside_convex(Point(60.0, 20.0), &rectangle));
        let reversed = [rectangle[3], rectangle[2], rectangle[1], rectangle[0]];
        assert!(inside_convex(Point(20.0, 20.0), &reversed));
    }
}
