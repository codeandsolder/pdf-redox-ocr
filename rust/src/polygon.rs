//! Convex PDF paint clipping and OCR polygon intersection.

#[cfg(any(feature = "pdf-integration", test))]
use crate::Error;
use crate::geometry::{Point, Quad};

pub(crate) fn convex(quad: &Quad) -> bool {
    if !area(quad).is_finite() {
        return false;
    }
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
        if !x.is_finite()
            || !y.is_finite()
            || x < self.min.0
            || x > self.max.0
            || y < self.min.1
            || y > self.max.1
        {
            return false;
        }
        self.corners
            .iter()
            .zip(self.corners.iter().cycle().skip(1))
            .all(|(&a, &b)| {
                edge_cross(a, b, Point(x, y)).is_some_and(|cross| {
                    if self.positive {
                        cross >= 0.0
                    } else {
                        cross <= 0.0
                    }
                })
            })
    }
}

/// Signed polygon area. Its sign records winding. Translate to the first
/// vertex before forming products, so large absolute PDF origins do not
/// erase a small polygon's area through cancellation.
#[must_use = "use oriented area when testing polygon winding"]
pub fn signed_area(points: &[Point]) -> f64 {
    let Some((&origin, rest)) = points.split_first() else {
        return 0.0;
    };
    rest.windows(2)
        .map(|pair| {
            let a = Point(pair[0].0 - origin.0, pair[0].1 - origin.1);
            let b = Point(pair[1].0 - origin.0, pair[1].1 - origin.1);
            a.0.mul_add(b.1, -a.1 * b.0)
        })
        .sum::<f64>()
        / 2.0
}

/// Unsigned polygon area. Nonrepresentable arithmetic remains nonfinite.
#[must_use = "use polygon area when measuring ink coverage"]
pub fn area(points: &[Point]) -> f64 {
    signed_area(points).abs()
}

fn finite_points(points: &[Point]) -> bool {
    points.iter().all(|p| p.0.is_finite() && p.1.is_finite())
}

fn edge_cross(start: Point, end: Point, point: Point) -> Option<f64> {
    let cross =
        (end.0 - start.0).mul_add(point.1 - start.1, -(end.1 - start.1) * (point.0 - start.0));
    cross.is_finite().then_some(cross)
}

/// Validate a convex polygon by all its supporting half-planes, not just
/// local turn signs (which can also agree for self-intersecting polygons).
fn winding(points: &[Point]) -> Option<f64> {
    let signed = signed_area(points);
    if !signed.is_finite() || signed == 0.0 || !finite_points(points) {
        return None;
    }
    let sign = signed.signum();
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .all(|(&a, &b)| {
            points
                .iter()
                .all(|&p| edge_cross(a, b, p).is_some_and(|c| sign * c >= -1e-9))
        })
        .then_some(sign)
}

/// Every subject point must lie in the same validated convex half-planes.
pub(crate) fn contains_points(clip: &[Point], subject: &[Point], epsilon: f64) -> bool {
    let Some(sign) = winding(clip) else {
        return false;
    };
    finite_points(subject)
        && clip
            .iter()
            .zip(clip.iter().cycle().skip(1))
            .all(|(&a, &b)| {
                subject
                    .iter()
                    .all(|&p| edge_cross(a, b, p).is_some_and(|c| sign * c >= -epsilon))
            })
}

/// Locate an actual straddling segment's intersection with the edge. The
/// ratio avoids overflow in the sum of two finite, opposite-signed crosses.
fn crossing(previous: Point, current: Point, a: f64, b: f64) -> Option<Point> {
    let scale = a.abs().max(b.abs());
    let left = a.abs() / scale;
    let right = b.abs() / scale;
    let fraction = left / (left + right);
    let point = Point(
        (current.0 - previous.0).mul_add(fraction, previous.0),
        (current.1 - previous.1).mul_add(fraction, previous.1),
    );
    finite_points(&[point]).then_some(point)
}

/// Clip a finite subject by a validated convex polygon. An empty result
/// is a measured disjoint intersection, while None means the geometry or
/// arithmetic could not support a measurement.
#[must_use = "distinguish unmeasurable geometry from a disjoint intersection"]
pub fn clip_convex(subject: &[Point], clip: &[Point]) -> Option<Vec<Point>> {
    let sign = winding(clip)?;
    if !finite_points(subject) {
        return None;
    }
    let mut polygon = subject.to_vec();
    let mut next = Vec::with_capacity(polygon.len() + 2);
    for (&start, &end) in clip.iter().zip(clip.iter().cycle().skip(1)) {
        let Some(&last) = polygon.last() else {
            break;
        };
        let mut previous = last;
        let mut previous_cross = sign * edge_cross(start, end, previous)?;
        next.clear();
        for &current in &polygon {
            let current_cross = sign * edge_cross(start, end, current)?;
            if (previous_cross >= 0.0) != (current_cross >= 0.0) {
                next.push(crossing(previous, current, previous_cross, current_cross)?);
            }
            if current_cross >= 0.0 {
                next.push(current);
            }
            previous = current;
            previous_cross = current_cross;
        }
        std::mem::swap(&mut polygon, &mut next);
    }
    Some(polygon)
}

/// Finite intersection area, including zero for a measured disjoint pair.
#[must_use = "propagate unavailable overlap evidence"]
pub fn intersection_area(subject: &[Point], clip: &[Point]) -> Option<f64> {
    let subject_area = area(subject);
    if !subject_area.is_finite() || !finite_points(subject) {
        return None;
    }
    // A clipped intermediate may be an empty polygon, a point, or a segment.
    // It has no area and cannot contribute overlap.
    if subject.len() < 3 {
        winding(clip)?;
        return Some(0.0);
    }
    if subject_area == 0.0 {
        return None;
    }
    winding(subject)?;
    let measure = area(&clip_convex(subject, clip)?);
    measure.is_finite().then_some(measure)
}

/// Intersection area divided by the smaller polygon area, not `IoU`. Invalid
/// or degenerate input has no defined overlap fraction.
#[must_use = "distinguish unknown overlap from a measured overlap fraction"]
pub fn overlap_fraction(first: &[Point], second: &[Point]) -> Option<f64> {
    let first_area = area(first);
    let second_area = area(second);
    if !first_area.is_finite() || !second_area.is_finite() {
        return None;
    }
    let minimum = first_area.min(second_area);
    if minimum <= 1e-9 {
        return None;
    }
    Some((intersection_area(first, second)? / minimum).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::{area, clip_convex, overlap_fraction};
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
    fn matches_reference_overlap_cases() -> Result<(), crate::Error> {
        let measure = |a: &[Point], b: &[Point]| {
            overlap_fraction(a, b)
                .ok_or_else(|| crate::Error::Invalid("missing overlap measure".into()))
        };
        let a = rect(20.0, 20.0, 60.0, 15.0);
        assert!((measure(&a, &a)? - 1.0).abs() < 1e-9);
        assert_eq!(measure(&a, &rect(150.0, 20.0, 60.0, 15.0))?, 0.0);
        assert!((measure(&a, &rect(50.0, 20.0, 60.0, 15.0))? - 0.5).abs() < 1e-9);
        let rotated = [
            Point(0.0, 0.0),
            Point(0.0, 10.0),
            Point(-30.0, 10.0),
            Point(-30.0, 0.0),
        ];
        assert!(measure(&rotated, &rect(-20.0, 2.0, 10.0, 6.0))? > 0.95);
        assert!((area(&rotated) - 300.0).abs() < 1e-9);
        assert!((measure(&rect(50.0, 20.0, 60.0, 15.0), &a)? - 0.5).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn large_origins_and_invalid_geometry_are_distinguished() {
        let first = rect(1e13, -1e13, 60.0, 15.0);
        let mut second = rect(1e13 + 30.0, -1e13, 60.0, 15.0);
        assert_eq!(area(&first), 900.0);
        assert_eq!(overlap_fraction(&first, &second), Some(0.5));
        second.reverse();
        assert_eq!(overlap_fraction(&first, &second), Some(0.5));
        let disjoint = rect(1e13 + 100.0, -1e13, 20.0, 15.0);
        assert_eq!(overlap_fraction(&first, &disjoint), Some(0.0));
        let mut crossing = first;
        crossing.swap(1, 2);
        assert!(overlap_fraction(&first, &crossing).is_none());
        let mut invalid = first;
        invalid[0].0 = f64::NAN;
        assert!(clip_convex(&first, &invalid).is_none());
        let huge = rect(-1e200, -1e200, 1e200, 1e200);
        assert!(overlap_fraction(&first, &huge).is_none());
    }

    #[test]
    fn degenerate_polygon_has_no_measured_overlap() {
        assert_eq!(
            overlap_fraction(&[Point(0.0, 0.0); 4], &rect(0.0, 0.0, 1.0, 1.0)),
            None
        );
    }
}
