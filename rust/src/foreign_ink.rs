//! Exact local cross-sections of drawing strokes that enter an OCR box.
//! Geometry alone never authorizes deleting a recognized character.

use crate::{
    Error,
    fragment_ink::membership_checks,
    layout_leaders::LeaderEdge,
    pixel_regions::{InkBounds, PixelClass, PixelRegion, regions_visit},
};
use image::GrayImage;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct StripeProfile {
    pub envelope: [u32; 2],
    pub common_rows: [u32; 2],
    pub pixels: u64,
}

#[derive(Debug, Serialize)]
pub struct EdgeStroke {
    pub component_index: usize,
    pub inside_bounds: InkBounds,
    pub inside: StripeProfile,
    pub exterior: StripeProfile,
    /// Exact half-open cross-section for each along-axis sample of the in-box tip.
    /// Orthogonal strokes use one source-X span per source-Y row.
    pub inside_sections: Vec<[u32; 2]>,
}

#[derive(Debug, Serialize)]
pub struct ContractionEvidence {
    pub edge: LeaderEdge,
    pub retained: InkBounds,
    pub strokes: Vec<EdgeStroke>,
    pub removed_pixels: u64,
    pub retained_pixels: u64,
}

/// The orthogonal profile uses source X as its cross-section coordinate.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OrthogonalEdge {
    Top,
    Bottom,
}

#[derive(Debug, Serialize)]
pub struct OrthogonalStroke {
    pub edge: OrthogonalEdge,
    pub stroke: EdgeStroke,
}

#[derive(Debug, Default)]
pub struct StrokeAudit {
    pub contractions: Vec<ContractionEvidence>,
    pub orthogonal: Vec<OrthogonalStroke>,
}

/// Work in the reading axis or its exact transpose, without copying a raster.
#[derive(Clone, Copy)]
enum Axis {
    Reading,
    Orthogonal,
}

impl Axis {
    const fn coordinates(self, x: u32, y: u32) -> (u32, u32) {
        match self {
            Self::Reading => (x, y),
            Self::Orthogonal => (y, x),
        }
    }

    const fn bounds(self, b: InkBounds) -> InkBounds {
        match self {
            Self::Reading => b,
            Self::Orthogonal => InkBounds {
                left: b.top,
                right: b.bottom,
                top: b.left,
                bottom: b.right,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct Column {
    top: u32,
    bottom: u32,
    pixels: u32,
}

impl Column {
    fn include(&mut self, y: u32) {
        self.top = self.top.min(y);
        self.bottom = self.bottom.max(y + 1);
        self.pixels += 1;
    }
}

/// A sustained stripe consists of nonempty contiguous columns that all
/// share at least one row. No morphological closing or synthesized ink.
fn profile(columns: &BTreeMap<u32, Column>, start: u32, end: u32) -> Option<StripeProfile> {
    if start >= end {
        return None;
    }
    let mut result = StripeProfile {
        envelope: [u32::MAX, 0],
        common_rows: [0, u32::MAX],
        pixels: 0,
    };
    for x in start..end {
        let c = columns.get(&x)?;
        if c.pixels != c.bottom - c.top {
            return None;
        }
        result.envelope[0] = result.envelope[0].min(c.top);
        result.envelope[1] = result.envelope[1].max(c.bottom);
        result.common_rows[0] = result.common_rows[0].max(c.top);
        result.common_rows[1] = result.common_rows[1].min(c.bottom);
        result.pixels += u64::from(c.pixels);
    }
    (result.common_rows[0] < result.common_rows[1]).then_some(result)
}

const fn candidate(c: &PixelRegion, o: InkBounds, axis: Axis) -> bool {
    let b = axis.bounds(c.bounds);
    let projected = axis.bounds(o);
    c.observation_support.inside_pixels > 0
        && c.observation_support.outside_pixels > c.observation_support.inside_pixels
        && b.right - b.left > o.bottom - o.top
        && (b.left < projected.left || b.right > projected.right)
}

struct ComponentColumns {
    columns: BTreeMap<u32, Column>,
}

impl ComponentColumns {
    fn record(&mut self, x: u32, y: u32) {
        self.columns
            .entry(x)
            .and_modify(|c| c.include(y))
            .or_insert(Column {
                top: y,
                bottom: y + 1,
                pixels: 1,
            });
    }

    fn stripe(
        &self,
        edge: LeaderEdge,
        o: InkBounds,
        b: InkBounds,
        component_index: usize,
        expected: u64,
    ) -> Option<EdgeStroke> {
        let width = b.right - b.left;
        let inside = profile(&self.columns, b.left, b.right)?;
        if inside.pixels != expected {
            return None;
        }
        // Compare the whole observed tip to an equally long immediately
        // adjacent exterior segment, not the entire drawing's bounding box.
        let exterior = match edge {
            LeaderEdge::Left if b.left == o.left => {
                profile(&self.columns, b.left.checked_sub(width)?, b.left)?
            }
            LeaderEdge::Right if b.right == o.right => {
                profile(&self.columns, b.right, b.right.checked_add(width)?)?
            }
            _ => return None,
        };
        if inside.envelope[0] < exterior.envelope[0]
            || inside.envelope[1] > exterior.envelope[1]
            || inside.common_rows[0].max(exterior.common_rows[0])
                >= inside.common_rows[1].min(exterior.common_rows[1])
        {
            return None;
        }
        let inside_sections = (b.left..b.right)
            .map(|x| self.columns.get(&x).map(|c| [c.top, c.bottom]))
            .collect::<Option<Vec<_>>>()?;
        Some(EdgeStroke {
            component_index,
            inside_bounds: b,
            inside,
            exterior,
            inside_sections,
        })
    }
}

fn contraction(
    edge: LeaderEdge,
    observation: InkBounds,
    strokes: Vec<EdgeStroke>,
    inside: &[Option<InkBounds>],
    components: &[&PixelRegion],
) -> Option<ContractionEvidence> {
    let removed = strokes
        .iter()
        .map(|s| s.component_index)
        .collect::<BTreeSet<_>>();
    let removed_bounds = strokes
        .iter()
        .map(|s| s.inside_bounds)
        .reduce(InkBounds::union)?;
    let mut retained = observation;
    let mut kept_pixels = 0_u64;
    let mut body = None;
    let mut has_glyph_height = false;
    for (i, bounds) in inside.iter().enumerate() {
        let Some(b) = *bounds else {
            continue;
        };
        if removed.contains(&i) {
            continue;
        }
        kept_pixels += components[i].observation_support.inside_pixels;
        has_glyph_height |= b.bottom - b.top >= (observation.bottom - observation.top).div_ceil(2);
        body = Some(body.map_or(b, |old: InkBounds| old.union(b)));
    }
    let body = body.filter(|_| has_glyph_height)?;
    match edge {
        LeaderEdge::Left if removed_bounds.right <= body.left => {
            retained.left = removed_bounds.right + (body.left - removed_bounds.right) / 2;
        }
        LeaderEdge::Right if body.right <= removed_bounds.left => {
            retained.right = body.right + (removed_bounds.left - body.right).div_ceil(2);
        }
        _ => return None,
    }
    let removed_pixels = strokes.iter().map(|s| s.inside.pixels).sum();
    Some(ContractionEvidence {
        edge,
        retained,
        strokes,
        removed_pixels,
        retained_pixels: kept_pixels,
    })
}

/// Both orientations accumulate exact cross-sections in one flood traversal.
struct Profiles {
    axis: Axis,
    columns: BTreeMap<usize, ComponentColumns>,
}

impl Profiles {
    fn new(axis: Axis, components: &[&PixelRegion], observation: InkBounds) -> Self {
        let columns = components
            .iter()
            .enumerate()
            .filter(|(_, c)| candidate(c, observation, axis))
            .map(|(i, _)| {
                (
                    i,
                    ComponentColumns {
                        columns: BTreeMap::new(),
                    },
                )
            })
            .collect();
        Self { axis, columns }
    }

    fn record(&mut self, i: usize, x: u32, y: u32, observation: InkBounds) {
        let (x, y) = self.axis.coordinates(x, y);
        let b = self.axis.bounds(observation);
        if (b.top..b.bottom).contains(&y)
            && let Some(profile) = self.columns.get_mut(&i)
        {
            profile.record(x, y);
        }
    }

    fn edge(
        &self,
        edge: LeaderEdge,
        observation: InkBounds,
        inside: &[Option<InkBounds>],
        components: &[&PixelRegion],
    ) -> Vec<EdgeStroke> {
        self.columns
            .iter()
            .filter_map(|(&i, c)| {
                let mut stroke = c.stripe(
                    edge,
                    self.axis.bounds(observation),
                    self.axis.bounds(inside[i]?),
                    i,
                    components[i].observation_support.inside_pixels,
                )?;
                stroke.inside_bounds = self.axis.bounds(stroke.inside_bounds);
                Some(stroke)
            })
            .collect()
    }
}

/// Measure separable exterior stripes on every observation edge, preserving
/// actual component membership. Only reading-axis, edge-separable strokes
/// can propose a crop contraction. Orthogonal strokes are evidence for
/// subsequent counterfactual analysis, never automatic text deletions.
/// Returns None when the shared membership-check budget is exhausted.
///
/// # Errors
/// Rejects inconsistent raster component identities or bounds.
pub fn audit_strokes(
    image: &GrayImage,
    threshold: u8,
    observation: InkBounds,
    components: &[&PixelRegion],
    remaining_checks: &mut u64,
) -> Result<Option<StrokeAudit>, Error> {
    let mut reading = Profiles::new(Axis::Reading, components, observation);
    let mut orthogonal = Profiles::new(Axis::Orthogonal, components, observation);
    if reading.columns.is_empty() && orthogonal.columns.is_empty() {
        return Ok(Some(StrokeAudit::default()));
    }
    let required = membership_checks(image.width(), image.height(), 0);
    let Some(remaining) = remaining_checks.checked_sub(required) else {
        return Ok(None);
    };
    *remaining_checks = remaining;
    let mut inside = vec![None::<InkBounds>; components.len()];
    let measured = regions_visit(image, threshold, PixelClass::Ink, observation, |i, x, y| {
        reading.record(i, x, y, observation);
        orthogonal.record(i, x, y, observation);
        if observation.contains(x, y)
            && let Some(cell) = inside.get_mut(i)
        {
            let b = InkBounds {
                left: x,
                top: y,
                right: x + 1,
                bottom: y + 1,
            };
            *cell = Some(cell.map_or(b, |old| old.union(b)));
        }
    })?;
    if measured.len() != components.len()
        || measured.iter().zip(components).any(|(new, old)| {
            new.bounds != old.bounds
                || new.area_pixels != old.area_pixels
                || new.observation_support.inside_pixels != old.observation_support.inside_pixels
        })
    {
        return Err(Error::Invalid(
            "foreign stroke component identity mismatch".into(),
        ));
    }
    let contractions = [LeaderEdge::Left, LeaderEdge::Right]
        .into_iter()
        .filter_map(|edge| {
            let strokes = reading.edge(edge, observation, &inside, components);
            contraction(edge, observation, strokes, &inside, components)
        })
        .collect();
    let orthogonal = [
        (LeaderEdge::Left, OrthogonalEdge::Top),
        (LeaderEdge::Right, OrthogonalEdge::Bottom),
    ]
    .into_iter()
    .flat_map(|(edge, orthogonal_edge)| {
        orthogonal
            .edge(edge, observation, &inside, components)
            .into_iter()
            .map(move |stroke| OrthogonalStroke {
                edge: orthogonal_edge,
                stroke,
            })
    })
    .collect();
    Ok(Some(StrokeAudit {
        contractions,
        orthogonal,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    fn measure(image: &GrayImage, observation: InkBounds) -> Result<StrokeAudit, Error> {
        let components = crate::pixel_regions::regions(image, 128, PixelClass::Ink, observation)?;
        let mut remaining = u64::MAX;
        audit_strokes(
            image,
            128,
            observation,
            &components.iter().collect::<Vec<_>>(),
            &mut remaining,
        )?
        .ok_or_else(|| Error::Invalid("unexpected contraction budget exhaustion".into()))
    }
    #[test]
    fn sustained_foreign_tip_is_removed_but_attached_glyph_is_not() -> Result<(), Error> {
        let o = InkBounds {
            left: 40,
            top: 10,
            right: 85,
            bottom: 40,
        };
        let mut image = GrayImage::from_fn(110, 50, |x, y| {
            let stroke = (0..46).contains(&x) && (24..27).contains(&y);
            let body = (49..57).contains(&x) && (14..36).contains(&y);
            let real_sign = (60..67).contains(&x) && (24..27).contains(&y);
            let final_glyph = (70..79).contains(&x) && (14..36).contains(&y);
            Luma([if stroke || body || real_sign || final_glyph {
                0
            } else {
                255
            }])
        });
        let result = measure(&image, o)?.contractions;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].edge, LeaderEdge::Left);
        assert_eq!(result[0].removed_pixels, 18);
        assert_eq!(result[0].retained_pixels, 395);
        assert_eq!(result[0].retained.left, 47);
        let mirrored = image::imageops::flip_horizontal(&image);
        let mirrored_o = InkBounds {
            left: 110 - o.right,
            right: 110 - o.left,
            ..o
        };
        let right = measure(&mirrored, mirrored_o)?.contractions;
        assert_eq!(right.len(), 1);
        assert_eq!(right[0].edge, LeaderEdge::Right);
        assert_eq!(right[0].retained.right, 110 - result[0].retained.left);
        // Connect a genuine glyph to the external stripe: all its ink
        // belongs to the same component, but the in-box cross-section
        // no longer continues the external line.
        for x in 46..49 {
            for y in 24..27 {
                image.put_pixel(x, y, Luma([0]));
            }
        }
        assert!(measure(&image, o)?.contractions.is_empty());
        Ok(())
    }

    #[test]
    fn contraction_charges_shared_membership_budget() -> Result<(), Error> {
        let o = InkBounds {
            left: 40,
            top: 10,
            right: 80,
            bottom: 40,
        };
        let image = GrayImage::from_fn(100, 50, |x, y| {
            Luma([
                if (x < 45 && (24..27).contains(&y))
                    || ((50..60).contains(&x) && (14..36).contains(&y))
                {
                    0
                } else {
                    255
                },
            ])
        });
        let components = crate::pixel_regions::regions(&image, 128, PixelClass::Ink, o)?;
        let refs = components.iter().collect::<Vec<_>>();
        let charge = membership_checks(image.width(), image.height(), 0);
        let mut remaining = charge - 1;
        assert!(audit_strokes(&image, 128, o, &refs, &mut remaining)?.is_none());
        assert_eq!(remaining, charge - 1);
        remaining = charge;
        assert_eq!(
            audit_strokes(&image, 128, o, &refs, &mut remaining)?
                .ok_or_else(|| Error::Invalid("missing contraction evidence".into()))?
                .contractions
                .len(),
            1
        );
        assert_eq!(remaining, 0);
        Ok(())
    }

    #[test]
    fn perpendicular_stubs_preserve_intervening_glyphs_and_attached_ink() -> Result<(), Error> {
        let o = InkBounds {
            left: 10,
            top: 40,
            right: 80,
            bottom: 70,
        };
        let mut image = GrayImage::from_fn(100, 110, |x, y| {
            let top = (41..44).contains(&x) && y < 45;
            let bottom = (41..44).contains(&x) && y >= 65;
            let digit = (20..30).contains(&x) && (44..66).contains(&y);
            let period = (45..48).contains(&x) && (61..64).contains(&y);
            let next = (55..65).contains(&x) && (44..66).contains(&y);
            Luma([if top || bottom || digit || period || next {
                0
            } else {
                255
            }])
        });
        let r = measure(&image, o)?;
        assert!(r.contractions.is_empty());
        assert_eq!(r.orthogonal.len(), 2);
        assert!(matches!(r.orthogonal[0].edge, OrthogonalEdge::Top));
        assert!(matches!(r.orthogonal[1].edge, OrthogonalEdge::Bottom));
        assert_eq!(r.orthogonal[0].stroke.inside.pixels, 15);
        assert_eq!(r.orthogonal[1].stroke.inside.pixels, 15);
        assert_eq!(
            r.orthogonal[0].stroke.inside_bounds,
            InkBounds {
                left: 41,
                right: 44,
                top: 40,
                bottom: 45
            }
        );
        // Attach a real glyph to the incoming stroke. Its full measured
        // in-box section no longer continues the stripe.
        for x in 43..56 {
            for y in 43..47 {
                image.put_pixel(x, y, Luma([0]));
            }
        }
        let r = measure(&image, o)?;
        assert_eq!(r.orthogonal.len(), 1);
        assert!(matches!(r.orthogonal[0].edge, OrthogonalEdge::Bottom));
        Ok(())
    }

    #[test]
    fn exact_half_open_cut_keeps_adjacent_projection_without_a_gap() -> Result<(), Error> {
        let o = InkBounds {
            left: 30,
            top: 10,
            right: 75,
            bottom: 35,
        };
        let image = GrayImage::from_fn(90, 45, |x, y| {
            let stroke = x < 35 && (29..31).contains(&y);
            let body = (35..40).contains(&x) && (12..24).contains(&y);
            let second = (45..50).contains(&x) && (20..33).contains(&y);
            Luma([if stroke || body || second { 0 } else { 255 }])
        });
        let r = measure(&image, o)?.contractions;
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].retained.left, 35);
        assert_eq!(r[0].removed_pixels, 10);
        Ok(())
    }
}
