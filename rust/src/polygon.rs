//! Convex PDF paint clipping and OCR polygon intersection.

use crate::geometry::Point;

/// Signed polygon area. Its sign records winding.
#[must_use]
pub fn signed_area(points: &[Point]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut previous = points[points.len() - 1];
    for &current in points {
        sum += current.0.mul_add(-previous.1, previous.0 * current.1);
        previous = current;
    }
    sum / 2.0
}

/// Unsigned polygon area.
#[must_use]
pub fn area(points: &[Point]) -> f64 {
    signed_area(points).abs()
}

/// Clip a polygon by convex geometry, with winding-independent edge tests.
#[must_use]
pub fn clip_convex(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    if clip.len() < 3 {
        return Vec::new();
    }
    let sign = if signed_area(clip) >= 0.0 { 1.0 } else { -1.0 };
    let mut polygon = subject.to_vec();
    for index in 0..clip.len() {
        if polygon.is_empty() {
            break;
        }
        let start = clip[index];
        let end = clip[(index + 1) % clip.len()];
        let cross = |point: Point| {
            sign * (end.1 - start.1).mul_add(
                -(point.0 - start.0),
                (end.0 - start.0) * (point.1 - start.1),
            )
        };
        let mut next = Vec::with_capacity(polygon.len() + 2);
        let mut previous = polygon[polygon.len() - 1];
        for &current in &polygon {
            let previous_cross = cross(previous);
            let current_cross = cross(current);
            if (previous_cross >= -1e-7) != (current_cross >= -1e-7) {
                let denominator = previous_cross - current_cross;
                let fraction = if denominator.abs() < 1e-12 {
                    1.0
                } else {
                    previous_cross / denominator
                };
                next.push(Point(
                    (current.0 - previous.0).mul_add(fraction, previous.0),
                    (current.1 - previous.1).mul_add(fraction, previous.1),
                ));
            }
            if current_cross >= -1e-7 {
                next.push(current);
            }
            previous = current;
        }
        polygon = next;
    }
    polygon
}

/// Intersection area divided by the smaller polygon area, not `IoU`.
#[must_use]
pub fn overlap_fraction(first: &[Point], second: &[Point]) -> f64 {
    let minimum = area(first).min(area(second));
    if minimum <= 1e-9 {
        0.0
    } else {
        area(&clip_convex(first, second)) / minimum
    }
}

#[cfg(test)]
mod tests {
    use super::{area, overlap_fraction};
    use crate::geometry::Point;

    fn rect(x: f64, y: f64, w: f64, h: f64) -> [Point; 4] {
        [
            Point(x, y),
            Point(x + w, y),
            Point(x + w, y + h),
            Point(x, y + h),
        ]
    }

    #[test]
    fn matches_reference_overlap_cases() {
        let a = rect(20.0, 20.0, 60.0, 15.0);
        assert!((overlap_fraction(&a, &a) - 1.0).abs() < 1e-9);
        assert_eq!(overlap_fraction(&a, &rect(150.0, 20.0, 60.0, 15.0)), 0.0);
        assert!((overlap_fraction(&a, &rect(50.0, 20.0, 60.0, 15.0)) - 0.5).abs() < 1e-9);
        let rotated = [
            Point(0.0, 0.0),
            Point(0.0, 10.0),
            Point(-30.0, 10.0),
            Point(-30.0, 0.0),
        ];
        assert!(overlap_fraction(&rotated, &rect(-20.0, 2.0, 10.0, 6.0)) > 0.95);
        assert!((area(&rotated) - 300.0).abs() < 1e-9);
        assert!((overlap_fraction(&rect(50.0, 20.0, 60.0, 15.0), &a) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn degenerate_polygon_has_no_overlap() {
        assert_eq!(
            overlap_fraction(&[Point(0.0, 0.0); 4], &rect(0.0, 0.0, 1.0, 1.0)),
            0.0
        );
    }
}
