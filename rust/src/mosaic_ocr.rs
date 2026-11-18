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
    geometry::{DisplayMapper, Point, Quad},
    inference::OcrInference,
    mosaic::{MosaicBudget, MosaicPlan, orient_raster, unrotate_pixel},
    paint_scan::ImagePaint,
    rasterize::{CropPixelMapper, render_rgb_region},
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

fn intersects(a: &Quad, b: &Quad) -> bool {
    let rect = |quad: &Quad| {
        [
            quad.iter().map(|p| p.0).fold(f64::INFINITY, f64::min),
            quad.iter().map(|p| p.1).fold(f64::INFINITY, f64::min),
            quad.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max),
            quad.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max),
        ]
    };
    let a = rect(a);
    let b = rect(b);
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
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
                && intersects(&first.page_quad, &next.page_quad)
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
        let mut left = quad.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let mut top = quad.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let mut right = quad.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let mut bottom = quad.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
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

fn group_dpi(pitch: f64, max_dpi: u32) -> Option<u32> {
    if !pitch.is_finite() || pitch <= 0.0 {
        return None;
    }
    let target = (72.0 / pitch).round().max(36.0);
    let dpi = num_traits::ToPrimitive::to_u32(&target)?;
    (dpi <= max_dpi).then_some(dpi)
}

/// Perform complete native-pitch OCR on a fragmented, image-only page.
///
/// Each group is rasterized from the actual PDF stack, but pixels outside
/// the selected group's image paints are cleared to prevent nearby groups
/// contaminating detections. One rejected group retains full-page fallback.
///
/// # Errors
/// Returns an error when native renderer, OCR model, or crop masks fail.
pub fn recognize_native_mosaics(
    engine: &OcrInference,
    pdf: &Path,
    page_mapper: DisplayMapper,
    plan: &MosaicPlan,
    paints: &[ImagePaint],
    budget: MosaicBudget,
) -> Result<Option<Vec<OcrLine>>, Error> {
    let budget = budget.validate()?;
    if !disjoint_groups(plan, paints) {
        return Ok(None);
    }
    let mut lines = Vec::new();
    let mut total_pixels = 0_u64;
    let mut seen = HashSet::new();
    for group in &plan.groups {
        let Some(dpi) = group_dpi(group.pitch, budget.max_dpi) else {
            return Ok(None);
        };
        let mut crop = render_rgb_region(
            pdf,
            group.page,
            page_mapper,
            group.region,
            dpi,
            budget.max_group_pixels,
        )?;
        total_pixels =
            total_pixels.saturating_add(u64::from(crop.rgb.width()) * u64::from(crop.rgb.height()));
        if total_pixels > budget.max_page_pixels {
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
        let (width, height) = (crop.rgb.width(), crop.rgb.height());
        let rotated = orient_raster(crop.rgb, group.clockwise_turns, group.flip_vertical);
        let mut result = engine.recognize_mapped(rotated, |point| {
            mapper.to_pdf(unrotate_pixel(
                point,
                width,
                height,
                group.clockwise_turns,
                group.flip_vertical,
            ))
        })?;
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
