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

// Unlike production compact-mark repair, this research path can propose
// entire adjacent glyphs. It reports their geometry, not their ownership or
// text. The caller must resolve every competing observation.
struct EdgeRun {
    bounds: InkBounds,
    indices: Vec<usize>,
    inside: bool,
    outside: bool,
}

fn edge_runs(components: &[&PixelRegion], observation: InkBounds) -> Vec<EdgeRun> {
    let h = height(observation);
    let mut eligible = components
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            let b = c.bounds;
            !c.touches_line_edge
                && b.top >= observation.top
                && b.bottom <= observation.bottom
                && c.area_pixels >= 2
                && width(b) <= h
                && height(b) <= h
        })
        .collect::<Vec<_>>();
    eligible.sort_unstable_by_key(|&(index, c)| (c.bounds.left, c.bounds.right, index));
    let mut runs: Vec<EdgeRun> = Vec::new();
    for (index, component) in eligible {
        let inside = component.observation_support.inside_pixels > 0;
        let outside = component.observation_support.outside_pixels > 0;
        if let Some(run) = runs.last_mut()
            && component.bounds.left < run.bounds.right
        {
            run.bounds = run.bounds.union(component.bounds);
            run.indices.push(index);
            run.inside |= inside;
            run.outside |= outside;
        } else {
            runs.push(EdgeRun {
                bounds: component.bounds,
                indices: vec![index],
                inside,
                outside,
            });
        }
    }
    runs
}

// A positive outward axis for both edges, avoiding signed coordinate
// conversions. All bounds are in the very same measured sample frame.
const fn outward(bounds: InkBounds, edge: LeaderEdge, sample_width: u32) -> (u32, u32) {
    match edge {
        LeaderEdge::Left => (sample_width - bounds.right, sample_width - bounds.left),
        LeaderEdge::Right => (bounds.left, bounds.right),
    }
}

const fn edge_extent(height: u32) -> u32 {
    height.saturating_mul(2)
}
const fn edge_margin(height: u32) -> u32 {
    height.div_ceil(6)
}

/// Cover the same extent and background margin that the geometry search
/// can propose. Orthogonal padding allows fully sampling any glyph-sized
/// component intersecting the original band.
#[cfg(feature = "pdf-integration")]
#[must_use]
pub const fn search_padding(height: u32) -> [u32; 4] {
    let axis = edge_extent(height)
        .saturating_add(edge_margin(height))
        .saturating_add(1);
    [axis, height, axis, height]
}

fn extension_margin(
    components: &[&PixelRegion],
    selected: &std::collections::BTreeSet<usize>,
    observation: InkBounds,
    edge: LeaderEdge,
    sample_width: u32,
    far: u32,
) -> Option<u32> {
    let (_, original_edge) = outward(observation, edge, sample_width);
    let mut outer = far
        .saturating_add(edge_margin(height(observation)))
        .min(sample_width);
    for (index, c) in components.iter().enumerate() {
        if selected.contains(&index) || c.observation_support.inside_pixels > 0 {
            continue;
        }
        let near = match edge {
            LeaderEdge::Left => c.band_neighbors.left.map(|x| sample_width - x),
            LeaderEdge::Right => c.band_neighbors.right,
        };
        let Some(near) = near.filter(|&near| near >= original_edge) else {
            continue;
        };
        // A table frame can surround the observation without putting ink
        // anywhere in the extension. Only actual pixels in this band are
        // obstacles; a component's bounding rectangle is not foreground.
        if near < far {
            return None;
        }
        outer = outer.min(far + (near - far) / 2);
    }
    (outer > far).then_some(outer)
}

fn candidates_on_edge(
    components: &[&PixelRegion],
    runs: &[EdgeRun],
    observation: InkBounds,
    edge: LeaderEdge,
    sample_width: u32,
) -> Vec<BoundaryEvidence> {
    let h = height(observation);
    let (_, original_edge) = outward(observation, edge, sample_width);
    let Some(body_end) = runs
        .iter()
        .filter(|run| run.inside)
        .map(|run| outward(run.bounds, edge, sample_width).1)
        .max()
    else {
        return Vec::new();
    };
    let mut candidates = runs
        .iter()
        .filter(|run| outward(run.bounds, edge, sample_width).1 > original_edge)
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|run| outward(run.bounds, edge, sample_width));
    let mut previous = body_end.min(original_edge);
    let mut selected = std::collections::BTreeSet::new();
    let mut initial_gap = None;
    let mut result = Vec::new();
    for run in candidates {
        let (near, far) = outward(run.bounds, edge, sample_width);
        let gap = near.saturating_sub(previous);
        if gap > h - h.div_ceil(4) || far - original_edge > edge_extent(h) || width(run.bounds) > h
        {
            break;
        }
        if !run.outside {
            continue;
        }
        let body_gap_pixels = *initial_gap.get_or_insert(gap);
        selected.extend(run.indices.iter().copied());
        previous = far;
        let Some(outer) =
            extension_margin(components, &selected, observation, edge, sample_width, far)
        else {
            continue;
        };
        let mut expanded = observation;
        match edge {
            LeaderEdge::Left => expanded.left = sample_width - outer,
            LeaderEdge::Right => expanded.right = outer,
        }
        result.push(BoundaryEvidence {
            edge,
            component_indices: selected.iter().copied().collect(),
            expanded,
            body_gap_pixels,
        });
    }
    result
}

/// Enumerate local source-ink expansions on either measured reading-axis edge.
///
/// Every proposal is a contiguous chain of local, fully sampled components
/// with real outside ink, including full-size glyphs and detached marks.
/// It is geometry-only research evidence: ownership and recognition are
/// independent checks, and no expansion is automatically accepted.
///
/// Invalid or empty geometry yields no candidates.
#[must_use]
pub fn audit_edge_extensions<'a>(
    components: impl IntoIterator<Item = &'a PixelRegion>,
    observation: InkBounds,
    sample_width: u32,
) -> Vec<BoundaryEvidence> {
    let components = components.into_iter().collect::<Vec<_>>();
    if observation.left >= observation.right
        || observation.top >= observation.bottom
        || observation.right > sample_width
        || height(observation) < 6
        || components.iter().any(|c| {
            let b = c.bounds;
            b.left >= b.right || b.top >= b.bottom || b.right > sample_width
        })
    {
        return Vec::new();
    }
    let runs = edge_runs(&components, observation);
    if !runs
        .iter()
        .any(|run| run.inside && height(run.bounds) >= height(observation).div_ceil(2))
    {
        return Vec::new();
    }
    [LeaderEdge::Left, LeaderEdge::Right]
        .into_iter()
        .flat_map(|edge| candidates_on_edge(&components, &runs, observation, edge, sample_width))
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
            band_neighbors: crate::pixel_regions::BandNeighbors {
                left: Some(right),
                right: Some(left),
            },
            observation_support: ObservationSupport {
                inside_pixels: inside,
                outside_pixels: 8_u64.saturating_sub(inside),
                touches_observation_edge: false,
            },
        }
    }
    #[test]
    fn full_glyph_extensions_stop_before_unrelated_components() {
        let observation = InkBounds {
            left: 100,
            top: 100,
            right: 185,
            bottom: 160,
        };
        let components = [
            mark(118, 109, 135, 152, 8),
            mark(142, 109, 167, 152, 8),
            mark(177, 146, 182, 152, 8),
            mark(187, 109, 217, 152, 0),
            mark(223, 131, 285, 154, 0),
            mark(226, 119, 229, 122, 0),
        ];
        let candidates = audit_edge_extensions(&components, observation, 285);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].edge, LeaderEdge::Right);
        assert_eq!(candidates[0].component_indices, [3]);
        assert_eq!(candidates[0].expanded.right, 220);
        // Reflect every bound, preserving the same physical proposal and gap.
        let reflected = components.map(|mut c| {
            let (l, r) = (c.bounds.left, c.bounds.right);
            c.bounds.left = 285 - r;
            c.bounds.right = 285 - l;
            c.band_neighbors.left = Some(285 - l);
            c.band_neighbors.right = Some(285 - r);
            c
        });
        let reflected_observation = InkBounds {
            left: 100,
            right: 185,
            ..observation
        };
        let left = audit_edge_extensions(&reflected, reflected_observation, 285);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].edge, LeaderEdge::Left);
        assert_eq!(left[0].component_indices, [3]);
        assert_eq!(left[0].expanded.left, 65);
    }

    #[test]
    fn chain_preserves_multiple_edge_tokens_and_refuses_intervening_ink() {
        let observation = InkBounds {
            left: 100,
            top: 100,
            right: 181,
            bottom: 160,
        };
        let base = [
            mark(118, 112, 133, 157, 8),
            mark(141, 109, 170, 157, 8),
            mark(178, 151, 183, 157, 4),
            mark(191, 109, 220, 157, 0),
        ];
        let both = audit_edge_extensions(&base, observation, 281);
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].component_indices, [2]);
        assert_eq!(both[0].expanded.right, 187);
        assert_eq!(both[1].component_indices, [2, 3]);
        assert_eq!(both[1].expanded.right, 230);
        let mut blocked = base.to_vec();
        blocked.push(mark(186, 80, 189, 180, 0));
        let only_period = audit_edge_extensions(&blocked, observation, 281);
        assert_eq!(only_period.len(), 1);
        assert_eq!(only_period[0].component_indices, [2]);
        assert_eq!(only_period[0].expanded.right, 184);
    }

    #[test]
    fn enclosing_grid_does_not_obstruct_blank_band_but_real_stroke_does() -> Result<(), crate::Error>
    {
        let observation = InkBounds {
            left: 15,
            top: 10,
            right: 30,
            bottom: 30,
        };
        let raster = image::GrayImage::from_fn(60, 40, |x, y| {
            let frame = x == 0 || x == 59 || y == 0 || y == 39;
            let body = (18..26).contains(&x) && (12..28).contains(&y);
            let next = (32..40).contains(&x) && (12..28).contains(&y);
            image::Luma([if frame || body || next { 0 } else { 255 }])
        });
        let measure = |im: &image::GrayImage| {
            crate::pixel_regions::regions(
                im,
                128,
                crate::pixel_regions::PixelClass::Ink,
                observation,
            )
        };
        let result = audit_edge_extensions(&measure(&raster)?, observation, 60);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].edge, LeaderEdge::Right);
        assert_eq!(result[0].expanded.right, 44);
        let mut crossed = raster;
        for y in 0..40 {
            crossed.put_pixel(31, y, image::Luma([0]));
        }
        assert!(audit_edge_extensions(&measure(&crossed)?, observation, 60).is_empty());
        Ok(())
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
