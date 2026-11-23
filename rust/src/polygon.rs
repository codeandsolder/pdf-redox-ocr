//! Convex PDF paint clipping and OCR polygon intersection.

#[cfg(any(feature = "pdf-integration", test))]
use crate::Error;
use crate::geometry::{Point, Quad};

pub(crate) fn convex(quad: &Quad) -> bool {
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

#[cfg(any(feature = "pdf-integration", test))]
pub(crate) struct ConvexQuad {
    pub(crate) corners: Quad,
    positive: bool,
    min: Point,
    max: Point,
}

#[cfg(any(feature = "pdf-integration", test))]
impl ConvexQuad {
    pub(crate) fn new(corners: Quad) -> Result<Self, Error> {
        if !convex(&corners) {
            return Err(Error::Invalid("invalid fragment accounting quad".into()));
        }
        let (min, max) = corners.iter().fold(
            (
                Point(f64::INFINITY, f64::INFINITY),
                Point(f64::NEG_INFINITY, f64::NEG_INFINITY),
            ),
            |(Point(l, t), Point(r, b)), &Point(x, y)| {
                (Point(l.min(x), t.min(y)), Point(r.max(x), b.max(y)))
            },
        );
        let [a, b, c, _] = corners;
        Ok(Self {
            positive: (b.0 - a.0).mul_add(c.1 - b.1, -(b.1 - a.1) * (c.0 - b.0)) > 0.0,
            corners,
            min,
            max,
        })
    }

    pub(crate) fn contains(&self, Point(x, y): Point) -> bool {
        if x < self.min.0 || x > self.max.0 || y < self.min.1 || y > self.max.1 {
            return false;
        }
        self.corners
            .iter()
            .zip(self.corners.iter().cycle().skip(1))
            .all(|(&a, &b)| {
                let cross = (b.0 - a.0).mul_add(y - a.1, -(b.1 - a.1) * (x - a.0));
                if self.positive {
                    cross >= 0.0
                } else {
                    cross <= 0.0
                }
            })
    }
}

/// Signed polygon area. Its sign records winding.
#[must_use = "use oriented area when testing polygon winding"]
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
#[must_use = "use polygon area when measuring ink coverage"]
pub fn area(points: &[Point]) -> f64 {
    signed_area(points).abs()
}

/// Clip a polygon by convex geometry, with winding-independent edge tests.
#[must_use = "use the clipped polygon to evaluate overlapping paint"]
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
#[must_use = "use the overlap fraction to decide source-paint attribution"]
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
