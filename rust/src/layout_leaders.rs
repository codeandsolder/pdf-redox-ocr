//! Geometric evidence of repeated dot leaders crossing an OCR line boundary.
//!
//! A compact periodic ink sequence that continues outside an observation is
//! distinguishable from ordinary punctuation inside it. Detection is diagnostic;
//! no observed Unicode or confidence is changed here.

use num_traits::ToPrimitive;
use serde::Serialize;

use crate::pixel_regions::{InkBounds, PixelRegion};

/// Edge of the original observation intersected by a leader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaderEdge {
    Left,
    Right,
}

/// One uniform compact-component chain and a crop separating it from body ink.
#[derive(Debug, Clone, Serialize)]
pub struct LeaderEvidence {
    pub edge: LeaderEdge,
    /// Indices in the original source-connected component array.
    pub component_indices: Vec<usize>,
    pub inside_dots: usize,
    pub outside_dots: usize,
    pub pitch_pixels: f64,
    /// Original observation with just the leader-side blank boundary moved
    /// inward, preserving every supported non-leader component.
    pub body: InkBounds,
}

#[derive(Clone, Copy)]
struct Dot {
    index: usize,
    bounds: InkBounds,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl Dot {
    fn compatible(self, other: Self) -> bool {
        let ratio = |a: f64, b: f64| a.min(b) >= 0.70 * a.max(b);
        ratio(self.width, other.width)
            && ratio(self.height, other.height)
            && (self.y - other.y).abs() <= (self.height.max(other.height) * 0.35).max(1.0)
    }
}

fn compact_dots(components: &[PixelRegion], observation: InkBounds) -> Vec<Dot> {
    let body_height = components
        .iter()
        .filter(|c| c.observation_support.inside_pixels > 0)
        .map(|c| {
            c.bounds
                .bottom
                .min(observation.bottom)
                .saturating_sub(c.bounds.top.max(observation.top))
        })
        .max()
        .unwrap_or(0);
    let limit = f64::from(body_height) * 0.40;
    let mut dots = components
        .iter()
        .enumerate()
        .filter_map(|(index, c)| {
            let b = c.bounds;
            let w = f64::from(b.right.checked_sub(b.left)?);
            let h = f64::from(b.bottom.checked_sub(b.top)?);
            if c.touches_line_edge
                || b.top < observation.top
                || b.bottom > observation.bottom
                || w <= 0.0
                || h <= 0.0
                || w.max(h) > limit
                || w.min(h) < 0.60 * w.max(h)
                || c.area_pixels.to_f64()? < 0.55 * w * h
            {
                return None;
            }
            Some(Dot {
                index,
                bounds: b,
                x: c.centroid[0],
                y: c.centroid[1],
                width: w,
                height: h,
            })
        })
        .collect::<Vec<_>>();
    dots.sort_by(|a, b| a.x.total_cmp(&b.x).then_with(|| a.index.cmp(&b.index)));
    dots
}

/// Restrict a chain to an actual observation edge with visible non-leader ink
/// on the other side. No other source-supported pixel component may be cut.
fn edge_evidence(
    chain: &[Dot],
    components: &[PixelRegion],
    observation: InkBounds,
    edge: LeaderEdge,
    pitch: f64,
) -> Option<LeaderEvidence> {
    const MIN_DOTS: usize = 5;
    const MIN_OUTSIDE: usize = 2;
    if chain.len() < MIN_DOTS {
        return None;
    }
    let outside_dots = chain
        .iter()
        .filter(|dot| match edge {
            LeaderEdge::Left => dot.bounds.right <= observation.left,
            LeaderEdge::Right => dot.bounds.left >= observation.right,
        })
        .count();
    let inside_dots = chain
        .iter()
        .filter(|dot| components[dot.index].observation_support.inside_pixels > 0)
        .count();
    if outside_dots < MIN_OUTSIDE || inside_dots == 0 {
        return None;
    }
    let indices = chain.iter().map(|dot| dot.index).collect::<Vec<_>>();
    let body = components
        .iter()
        .enumerate()
        .filter(|(index, c)| c.observation_support.inside_pixels > 0 && !indices.contains(index))
        .map(|(_, c)| c.bounds)
        .reduce(InkBounds::union)?;
    let mut crop = observation;
    match edge {
        LeaderEdge::Left => {
            let end = chain.last()?.bounds.right;
            if end >= body.left || body.left >= observation.right {
                return None;
            }
            crop.left = observation.left.max(end.midpoint(body.left));
        }
        LeaderEdge::Right => {
            let start = chain.first()?.bounds.left;
            if start <= body.right || body.right <= observation.left {
                return None;
            }
            crop.right = observation.right.min(start.midpoint(body.right));
        }
    }
    // Require genuine body ink visibly larger than the dot sequence.
    let tallest_body = components
        .iter()
        .enumerate()
        .filter(|(i, c)| c.observation_support.inside_pixels > 0 && !indices.contains(i))
        .map(|(_, c)| c.bounds.bottom - c.bounds.top)
        .max()?;
    if f64::from(tallest_body) < chain.iter().map(|dot| dot.height).fold(0.0_f64, f64::max) * 2.5 {
        return None;
    }
    Some(LeaderEvidence {
        edge,
        component_indices: indices,
        inside_dots,
        outside_dots,
        pitch_pixels: pitch,
        body: crop,
    })
}

fn extend_chain(dots: &[Dot], a: usize, b: usize) -> Vec<Dot> {
    let first = dots[a];
    let pitch = dots[b].x - first.x;
    let tolerance = (pitch * 0.18).max(1.0);
    let mut chain = vec![first, dots[b]];
    let mut next = b + 1;
    while let Some(last) = chain.last() {
        let target = last.x + pitch;
        let mut best: Option<(usize, f64)> = None;
        for (index, &dot) in dots.iter().enumerate().skip(next) {
            if dot.x > target + tolerance {
                break;
            }
            let error = (dot.x - target).abs();
            if error <= tolerance
                && first.compatible(dot)
                && best.is_none_or(|(_, previous)| error < previous)
            {
                best = Some((index, error));
            }
        }
        let Some((index, _)) = best else {
            break;
        };
        chain.push(dots[index]);
        next = index + 1;
    }
    chain
}

/// Detect repeated, regularly spaced compact ink crossing either edge of a
/// rectified observation.
///
/// At least five separate full dots, including two
/// fully outside the observation, are required. All other supported ink
/// must remain in the proposed body crop. This deliberately leaves ordinary
/// punctuation, internal decimal points and short ellipses unchanged.
///
/// The result is only evidence of layout ink; it does not classify glyphs or
/// authorize edits to recognized text.
#[must_use = "use leader detections when reconstructing layout structure"]
pub fn detect_leaders(components: &[PixelRegion], observation: InkBounds) -> Vec<LeaderEvidence> {
    let dots = compact_dots(components, observation);
    let mut best = [None::<LeaderEvidence>, None];
    for (a, &first) in dots.iter().enumerate() {
        let size = first.width.max(first.height);
        for b in a + 1..dots.len() {
            let second = dots[b];
            let pitch = second.x - first.x;
            if pitch > 4.0 * size {
                break;
            }
            if pitch < 1.25 * size || !first.compatible(second) {
                continue;
            }
            let chain = extend_chain(&dots, a, b);
            for (slot, edge) in [LeaderEdge::Left, LeaderEdge::Right]
                .into_iter()
                .enumerate()
            {
                if let Some(evidence) = edge_evidence(&chain, components, observation, edge, pitch)
                    && best[slot].as_ref().is_none_or(|old| {
                        evidence.component_indices.len() > old.component_indices.len()
                    })
                {
                    best[slot] = Some(evidence);
                }
            }
        }
    }
    best.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::{LeaderEdge, detect_leaders};
    use crate::{
        Error,
        pixel_regions::{InkBounds, PixelClass, regions},
    };
    use image::{GrayImage, Luma};

    fn rectangle(image: &mut GrayImage, x: u32, y: u32, w: u32, h: u32) {
        for dy in y..y + h {
            for dx in x..x + w {
                image.put_pixel(dx, dy, Luma([0]));
            }
        }
    }

    #[test]
    fn long_leaders_cross_observation_edges_but_preserve_body_punctuation() -> Result<(), Error> {
        let mut image = GrayImage::from_pixel(240, 80, Luma([255]));
        let obs = InkBounds {
            left: 50,
            top: 20,
            right: 180,
            bottom: 60,
        };
        rectangle(&mut image, 95, 23, 9, 29);
        rectangle(&mut image, 122, 25, 10, 26);
        // A decimal/full stop is supported ink, but not on the leader pitch.
        rectangle(&mut image, 110, 48, 4, 4);
        for x in [10, 21, 32, 43, 54, 65, 76] {
            rectangle(&mut image, x, 48, 6, 6);
        }
        for x in [151, 162, 173, 184, 195, 206, 217] {
            rectangle(&mut image, x, 48, 6, 6);
        }
        let components = regions(&image, 192, PixelClass::Ink, obs)?;
        let leaders = detect_leaders(&components, obs);
        assert_eq!(leaders.len(), 2, "{leaders:#?}");
        assert_eq!(leaders[0].edge, LeaderEdge::Left);
        assert_eq!(leaders[0].inside_dots, 3);
        assert_eq!(leaders[0].outside_dots, 4);
        assert_eq!(leaders[0].body.left, 88);
        assert_eq!(leaders[1].edge, LeaderEdge::Right);
        assert_eq!(leaders[1].inside_dots, 3);
        assert_eq!(leaders[1].outside_dots, 4);
        assert_eq!(leaders[1].body.right, 141);
        Ok(())
    }

    #[test]
    fn ellipsis_irregular_dots_and_nonleader_edge_ink_are_not_removed() -> Result<(), Error> {
        let obs = InkBounds {
            left: 30,
            top: 10,
            right: 100,
            bottom: 50,
        };
        for points in [
            vec![74, 85, 96],
            vec![74, 85, 96, 114, 130, 149],
            vec![74, 85, 96, 107, 118, 129],
        ] {
            let mut image = GrayImage::from_pixel(170, 60, Luma([255]));
            rectangle(&mut image, 40, 13, 12, 26);
            for &x in &points {
                rectangle(&mut image, x, 34, 6, 6);
            }
            // The last case has a real descender at the boundary, so moving
            // the right edge would cut nonleader ink.
            if points.last() == Some(&129) {
                rectangle(&mut image, 92, 15, 6, 16);
            }
            let components = regions(&image, 192, PixelClass::Ink, obs)?;
            assert!(detect_leaders(&components, obs).is_empty(), "{points:?}");
        }
        Ok(())
    }
}
