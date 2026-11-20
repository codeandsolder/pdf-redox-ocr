//! Reading-frame geometry for the pinned OAR rectification contract.
//!
//! Coordinates describe pixel corners. OAR uses integer sample indices for
//! perspective interpolation; adjust by half a pixel when mapping those
//! samples' surrounding pixel cells.

use crate::{
    geometry::{Point, Quad},
    pixel_regions::InkBounds,
};
use serde::{Deserialize, Serialize};

/// One projective mapping from a recognizer crop to the parent raster.
/// The original crop dimensions and ordered corners are retained.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CropGeometry {
    pub parent_corners: Quad, // TL, TR, BR, BL before optional rotation
    pub unrotated_dimensions: (u32, u32),
    pub rotate270: bool,
    pub integer_crop: bool,
}

fn cross(a: Point, b: Point) -> f64 {
    a.0.mul_add(b.1, -a.1 * b.0)
}
fn sub(a: Point, b: Point) -> Point {
    Point(a.0 - b.0, a.1 - b.1)
}
fn add(a: Point, b: Point) -> Point {
    Point(a.0 + b.0, a.1 + b.1)
}
fn scale(a: Point, t: f64) -> Point {
    Point(a.0 * t, a.1 * t)
}

impl CropGeometry {
    /// Reconstruct the four-point OAR crop orientation and verify the actual
    /// returned dimensions. The four input points must be actual detector
    /// corners, not a polygon envelope.
    #[must_use]
    pub fn from_quad(mut points: Quad, parent: (u32, u32), crop: (u32, u32)) -> Option<Self> {
        if points.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
            return None;
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        if points[1].1 < points[0].1 {
            points.swap(0, 1);
        }
        if points[3].1 < points[2].1 {
            points.swap(2, 3);
        }
        points = [points[0], points[2], points[3], points[1]];
        let [top_left, top_right, bottom_right, bottom_left] = points;
        let edges = [
            sub(top_right, top_left),
            sub(bottom_right, top_right),
            sub(bottom_left, bottom_right),
            sub(top_left, bottom_left),
        ];
        if (0..4).any(|i| cross(edges[i], edges[(i + 1) % 4]) <= 1e-9) {
            return None;
        }
        let distance = |p: Point| p.0.hypot(p.1);
        let width = distance(edges[0]).max(distance(edges[2])).round();
        let height = distance(edges[1]).max(distance(edges[3])).round();
        // The actual output dimensions are authoritative (including f32
        // rounding in OAR). Reject disagreement rather than stretch top_left crop.
        let rotate270 = f64::from(crop.0) >= 1.5 * f64::from(crop.1)
            && height >= 1.5 * width
            && (width - f64::from(crop.1)).abs() <= 1.0
            && (height - f64::from(crop.0)).abs() <= 1.0;
        let dimensions = if rotate270 { (crop.1, crop.0) } else { crop };
        if dimensions.0 == 0
            || dimensions.1 == 0
            || (width - f64::from(dimensions.0)).abs() > 1.0
            || (height - f64::from(dimensions.1)).abs() > 1.0
        {
            return None;
        }
        let integer_crop = top_left.0 >= 0.0
            && top_left.1 >= 0.0
            && bottom_right.0 <= f64::from(parent.0)
            && bottom_right.1 <= f64::from(parent.1)
            && top_left == Point(top_left.0.round(), top_left.1.round())
            && top_right == Point(top_left.0 + f64::from(dimensions.0), top_left.1)
            && bottom_right == Point(top_right.0, top_left.1 + f64::from(dimensions.1))
            && bottom_left == Point(top_left.0, bottom_right.1);
        Some(Self {
            parent_corners: points,
            unrotated_dimensions: dimensions,
            rotate270,
            integer_crop,
        })
    }

    /// Return a mapped ink bounding rectangle in parent RGB pixel corners.
    #[must_use]
    pub fn ink_quad(self, ink: InkBounds) -> Option<Quad> {
        let bounds = [
            Point(f64::from(ink.left), f64::from(ink.bottom)),
            Point(f64::from(ink.right), f64::from(ink.bottom)),
            Point(f64::from(ink.right), f64::from(ink.top)),
            Point(f64::from(ink.left), f64::from(ink.top)),
        ];
        let mut result = [Point(0., 0.); 4];
        for (out, input) in result.iter_mut().zip(bounds) {
            *out = self.map(input)?;
        }
        Some(result)
    }

    fn map(self, point: Point) -> Option<Point> {
        let (width, height) = self.unrotated_dimensions;
        let (width, height) = (f64::from(width), f64::from(height));
        let unrotated = if self.rotate270 {
            Point(width - point.1, point.0)
        } else {
            point
        };
        let [top_left, top_right, bottom_right, bottom_left] = self.parent_corners;
        if self.integer_crop {
            return Some(add(top_left, unrotated));
        }
        // Unit square -> arbitrary convex quadrilateral, solved analytically.
        let uv = Point((unrotated.0 - 0.5) / width, (unrotated.1 - 0.5) / height);
        let delta = sub(add(top_left, bottom_right), add(top_right, bottom_left));
        let u_axis = sub(top_right, bottom_right);
        let v_axis = sub(bottom_left, bottom_right);
        let denominator = cross(u_axis, v_axis);
        if denominator.abs() < 1e-12 {
            return None;
        }
        let project_x = cross(delta, v_axis) / denominator;
        let project_y = cross(u_axis, delta) / denominator;
        let x_axis = sub(scale(top_right, 1.0 + project_x), top_left);
        let y_axis = sub(scale(bottom_left, 1.0 + project_y), top_left);
        let denom = project_x.mul_add(uv.0, project_y.mul_add(uv.1, 1.0));
        if denom <= 1e-12 {
            return None;
        }
        let mapped = scale(
            add(top_left, add(scale(x_axis, uv.0), scale(y_axis, uv.1))),
            1. / denom,
        );
        Some(add(mapped, Point(0.5, 0.5)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_and_rotated_crop_associations() -> Result<(), crate::Error> {
        let q = [
            Point(10., 20.),
            Point(50., 20.),
            Point(50., 30.),
            Point(10., 30.),
        ];
        let g = CropGeometry::from_quad(q, (100, 100), (40, 10))
            .ok_or_else(|| crate::Error::Invalid("fixture crop".into()))?;
        assert!(g.integer_crop);
        assert_eq!(g.map(Point(3., 4.)), Some(Point(13., 24.)));
        let q = [
            Point(10., 20.),
            Point(20., 20.),
            Point(20., 60.),
            Point(10., 60.),
        ];
        let g = CropGeometry::from_quad(q, (100, 100), (40, 10))
            .ok_or_else(|| crate::Error::Invalid("fixture rotation".into()))?;
        assert!(g.rotate270);
        assert_eq!(g.map(Point(3., 4.)), Some(Point(16., 23.)));
        Ok(())
    }
    #[test]
    fn projective_transform_hits_all_four_sample_corners() -> Result<(), crate::Error> {
        let q = [
            Point(10., 20.),
            Point(60., 23.),
            Point(50., 40.),
            Point(12., 36.),
        ];
        let g = CropGeometry::from_quad(q, (100, 100), (50, 20))
            .ok_or_else(|| crate::Error::Invalid("fixture perspective".into()))?;
        for (p, c) in [
            Point(0.5, 0.5),
            Point(50.5, 0.5),
            Point(50.5, 20.5),
            Point(0.5, 20.5),
        ]
        .into_iter()
        .zip(q)
        {
            let mapped = g
                .map(p)
                .ok_or_else(|| crate::Error::Invalid("fixture map".into()))?;
            assert!((mapped.0 - c.0 - 0.5).abs() < 1e-9);
            assert!((mapped.1 - c.1 - 0.5).abs() < 1e-9);
        }
        Ok(())
    }
}
