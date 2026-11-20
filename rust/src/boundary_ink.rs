//! Detect source ink omitted or clipped at the reading-axis edges of an OCR box.
//!
//! Connected ink is evidence for a larger recognition window, not a Unicode
//! label. Detached ink must be local to the existing body and fit within the
//! same orthogonal band; never merge an adjacent detected observation.

use serde::Serialize;

use crate::{
    layout_leaders::LeaderEdge,
    pixel_regions::{InkBounds, PixelRegion},
};

#[derive(Clone, Debug, Serialize)]
pub struct BoundaryEvidence {
    pub edge: LeaderEdge,
    pub component_indices: Vec<usize>,
    pub expanded: InkBounds,
    pub body_gap_pixels: u32,
}

const fn width(b: InkBounds) -> u32 {
    b.right - b.left
}
const fn height(b: InkBounds) -> u32 {
    b.bottom - b.top
}

const fn beyond(b: InkBounds, original: InkBounds, edge: LeaderEdge) -> bool {
    match edge {
        LeaderEdge::Left => b.left < original.left,
        LeaderEdge::Right => b.right > original.right,
    }
}

const fn distance(b: InkBounds, body: InkBounds, edge: LeaderEdge) -> u32 {
    match edge {
        LeaderEdge::Left => body.left.saturating_sub(b.right),
        LeaderEdge::Right => b.left.saturating_sub(body.right),
    }
}

/// Look for a fully sampled edge component in the same line band. A small
/// detached mark may sit just beyond the detector, or be inside its outer
/// margin but have insufficient background for the recognizer's resampling.
/// Long separated labels are not evidence of a missing edge character.
pub(crate) fn detect_boundaries(
    components: &[PixelRegion],
    original: InkBounds,
    sample_width: u32,
) -> Vec<BoundaryEvidence> {
    let h = height(original);
    if h < 6 {
        return vec![];
    }
    let body = components
        .iter()
        .filter(|c| {
            c.observation_support.inside_pixels > 0
                && c.bounds.top >= original.top
                && c.bounds.bottom <= original.bottom
                && height(c.bounds) * 2 >= h
        })
        .map(|c| c.bounds)
        .reduce(InkBounds::union);
    let Some(body) = body else {
        return vec![];
    };
    [LeaderEdge::Left, LeaderEdge::Right]
        .into_iter()
        .filter_map(|edge| {
            let near = h.div_ceil(5).max(2);
            let ink: Vec<_> = components
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    let b = c.bounds;
                    if c.touches_line_edge
                        || b.top < original.top
                        || b.bottom > original.bottom
                        || c.area_pixels < 2
                    {
                        return false;
                    }
                    // Expose a connected, clipped component; or a detached compact
                    // mark close to the edge, with plausible inter-character spacing.
                    let near_edge = match edge {
                        LeaderEdge::Left => b.left <= original.left.saturating_add(near),
                        LeaderEdge::Right => b.right >= original.right.saturating_sub(near),
                    };
                    let clipped =
                        beyond(b, original, edge) && c.observation_support.inside_pixels > 0;
                    let detached = width(b) * 3 <= h * 2
                        && height(b) * 3 <= h * 2
                        && distance(b, body, edge) * 4 <= h * 3
                        && match edge {
                            LeaderEdge::Left => b.right <= body.left,
                            LeaderEdge::Right => b.left >= body.right,
                        };
                    near_edge && (clipped || detached)
                })
                .collect();
            if ink.is_empty() {
                return None;
            }
            // Admit only one local cluster. A second outside mark elsewhere, a
            // repeated leader, or a neighboring region must not be swallowed.
            let bounds = ink.iter().map(|(_, c)| c.bounds).reduce(InkBounds::union)?;
            if width(bounds) > h {
                return None;
            }
            let pad = h.div_ceil(6).max(2);
            let mut expanded = original;
            match edge {
                LeaderEdge::Left => {
                    expanded.left = original.left.min(bounds.left.saturating_sub(pad));
                }
                LeaderEdge::Right => {
                    expanded.right = original
                        .right
                        .max(bounds.right.saturating_add(pad).min(sample_width));
                }
            }
            if expanded == original {
                return None;
            }
            // Reject any unrelated component newly intersected by the extension.
            let (strip_left, strip_right) = match edge {
                LeaderEdge::Left => (expanded.left, original.left),
                LeaderEdge::Right => (original.right, expanded.right),
            };
            if components.iter().enumerate().any(|(i, c)| {
                let b = c.bounds;
                let newly_covered = b.left < strip_right && b.right > strip_left;
                newly_covered
                    && b.top < original.bottom
                    && b.bottom > original.top
                    && !ink.iter().any(|(j, _)| *j == i)
            }) {
                return None;
            }
            Some(BoundaryEvidence {
                edge,
                component_indices: ink.iter().map(|(i, _)| *i).collect(),
                expanded,
                body_gap_pixels: distance(bounds, body, edge),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel_regions::ObservationSupport;
    fn mark(left: u32, top: u32, right: u32, bottom: u32, inside: u64) -> PixelRegion {
        PixelRegion {
            bounds: InkBounds {
                left,
                top,
                right,
                bottom,
            },
            area_pixels: 8,
            centroid: [0., 0.],
            touches_line_edge: false,
            observation_support: ObservationSupport {
                inside_pixels: inside,
                outside_pixels: 0,
                touches_observation_edge: false,
            },
        }
    }
    #[test]
    fn local_ink_not_neighboring_label_authorizes_extension() {
        let b = InkBounds {
            left: 20,
            top: 20,
            right: 90,
            bottom: 44,
        };
        let body = mark(31, 24, 75, 43, 50);
        let star = mark(10, 23, 19, 32, 0);
        let period = mark(85, 39, 89, 43, 8);
        let result = detect_boundaries(&[body.clone(), star, period], b, 120);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].expanded.left, 6);
        assert_eq!(result[1].expanded.right, 93);
        let adjacent = mark(2, 24, 18, 43, 0);
        assert!(detect_boundaries(&[body.clone(), adjacent], b, 120).is_empty());
        let clipped = mark(83, 26, 94, 43, 6);
        let result = detect_boundaries(&[body, clipped], b, 120);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].expanded.right, 98);
    }
}
