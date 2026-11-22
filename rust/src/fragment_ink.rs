//! Count actual thresholded foreground support in one common source raster.
//! Pixel membership is measured at rectified sample centers. This is not
//! semantic ownership, and partially sampled peers are explicitly identified.

use std::collections::BTreeMap;

use image::GrayImage;
use serde::Serialize;

use crate::{
    Error,
    geometry::{Point, Quad},
    line_frame::{LineFrame, SampleWindow, convex},
    pixel_regions::{InkBounds, PixelClass, regions_visit},
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct InkCounts {
    pub original: u64,
    /// Newly included foreground beyond the original reading-axis edges.
    pub axis_added: u64,
    /// Newly included body foreground above/below the original observation.
    pub orthogonal_added: u64,
    pub outside_expansion: u64,
}

#[derive(Clone, Copy)]
enum Part {
    Original,
    AxisAdded,
    OrthogonalAdded,
    Outside,
}

fn classify_pixel(original: InkBounds, expanded: InkBounds, x: u32, y: u32) -> Part {
    if original.contains(x, y) {
        Part::Original
    } else if !expanded.contains(x, y) {
        Part::Outside
    } else if (original.left..original.right).contains(&x) {
        Part::OrthogonalAdded
    } else {
        Part::AxisAdded
    }
}

impl InkCounts {
    const fn record(&mut self, part: Part) {
        match part {
            Part::Original => self.original += 1,
            Part::AxisAdded => self.axis_added += 1,
            Part::OrthogonalAdded => self.orthogonal_added += 1,
            Part::Outside => self.outside_expansion += 1,
        }
    }

    const fn add(&mut self, other: Self) {
        self.original += other.original;
        self.axis_added += other.axis_added;
        self.orthogonal_added += other.orthogonal_added;
        self.outside_expansion += other.outside_expansion;
    }

    /// Total newly included foreground, regardless of direction.
    #[must_use]
    pub const fn added(self) -> u64 {
        self.axis_added + self.orthogonal_added
    }

    #[cfg(feature = "native-inference")]
    const fn included(self) -> Self {
        Self {
            outside_expansion: 0,
            ..self
        }
    }

    const fn any(self) -> bool {
        self.original != 0 || self.added() != 0 || self.outside_expansion != 0
    }
}

/// Counts in four disjoint parts of the shared sampling frame.
#[derive(Debug, Default, Serialize)]
pub struct ComponentInk {
    pub component_index: usize,
    pub sample: InkCounts,
    /// Each pixel inside one or more peers is counted once.
    pub peer_union: InkCounts,
    /// Keys index the candidate's peer list. Multiple peers may cover one pixel.
    pub peers: BTreeMap<usize, InkCounts>,
}

impl ComponentInk {
    const fn relevant(&self) -> bool {
        self.sample.original != 0 || self.sample.added() != 0 || self.peer_union.any()
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum InkAccounting {
    Measured {
        /// Whether each complete peer quad lies inside the sampled source window.
        peers_fully_sampled: Vec<bool>,
        /// Counts over the entire sampled raster, not just selected components.
        sample: InkCounts,
        peer_union: InkCounts,
        peers: BTreeMap<usize, InkCounts>,
        /// Original connected-component indices, never bounding-box pixel counts.
        components: Vec<ComponentInk>,
        charged_checks: u64,
    },
    BudgetExceeded {
        required_checks: u64,
        remaining_checks: u64,
    },
}

/// Exact pixel membership of one run in the recognized expanded crop.
#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
#[derive(Debug, Default, Serialize)]
pub struct RunInk {
    pub run: usize,
    pub sample: InkCounts,
    pub peers: BTreeMap<usize, InkCounts>,
}

#[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
#[derive(Debug, Serialize)]
pub struct RunAccounting {
    pub runs: Vec<RunInk>,
    pub charged_checks: u64,
}

struct ConvexQuad {
    corners: Quad,
    positive: bool,
    min: Point,
    max: Point,
}

impl ConvexQuad {
    fn new(corners: Quad) -> Result<Self, Error> {
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

    fn contains(&self, Point(x, y): Point) -> bool {
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

struct MeasuredInk {
    sample: InkCounts,
    peer_union: InkCounts,
    peers: BTreeMap<usize, InkCounts>,
    components: Vec<ComponentInk>,
}

/// A single immutable rectified source raster, and its exact page transform.
pub struct Sample<'a> {
    pub image: &'a GrayImage,
    pub frame: LineFrame,
    pub window: SampleWindow,
    pub threshold: u8,
}

impl Sample<'_> {
    /// Count the foreground, including residual peer ink outside the expansion.
    /// Work is bounded by raster area times the number of tested regions.
    pub fn account(
        &self,
        expanded: InkBounds,
        peers: &[Quad],
        remaining_checks: &mut u64,
    ) -> Result<InkAccounting, Error> {
        let observation = self.window.observation(self.frame);
        let (width, height) = self.image.dimensions();
        if self.window.dimensions(self.frame) != Some((width, height))
            || self.threshold == 0
            || expanded.left > observation.left
            || expanded.right < observation.right
            || expanded.top > observation.top
            || expanded.bottom < observation.bottom
            || expanded.right > width
            || expanded.bottom > height
        {
            return Err(Error::Invalid(
                "inconsistent fragment accounting bounds".into(),
            ));
        }
        let required_checks = u64::from(width)
            .saturating_mul(u64::from(height))
            .saturating_mul(
                u64::try_from(peers.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(1),
            );
        let Some(remaining) = remaining_checks.checked_sub(required_checks) else {
            return Ok(InkAccounting::BudgetExceeded {
                required_checks,
                remaining_checks: *remaining_checks,
            });
        };
        *remaining_checks = remaining;
        let sample_quad = ConvexQuad::new(self.window.quad(
            self.frame,
            InkBounds {
                left: 0,
                top: 0,
                right: width,
                bottom: height,
            },
        ))?;
        let peers = peers
            .iter()
            .copied()
            .map(ConvexQuad::new)
            .collect::<Result<Vec<_>, _>>()?;
        let peers_fully_sampled = peers
            .iter()
            .map(|peer| peer.corners.iter().all(|&p| sample_quad.contains(p)))
            .collect();
        let accounting = self.measure(expanded, observation, &peers)?;
        Ok(InkAccounting::Measured {
            peers_fully_sampled,
            sample: accounting.sample,
            peer_union: accounting.peer_union,
            peers: accounting.peers,
            components: accounting.components,
            charged_checks: required_checks,
        })
    }

    /// Measure recognized projection runs in the *same* thresholded raster
    /// as the boundary extension. Their coordinates are relative to expanded.
    #[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
    pub(crate) fn account_runs(
        &self,
        expanded: InkBounds,
        groups: &[crate::glyph_isolation::InkGroup],
        peer_quads: &[Quad],
        accounting: &InkAccounting,
        remaining_checks: &mut u64,
    ) -> Result<Option<RunAccounting>, Error> {
        let InkAccounting::Measured { sample, peers, .. } = accounting else {
            return Ok(None);
        };
        let width = expanded.right - expanded.left;
        let height = expanded.bottom - expanded.top;
        let charged_checks = u64::from(width)
            .saturating_mul(u64::from(height))
            .saturating_mul(
                u64::try_from(peer_quads.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(1),
            );
        let Some(remaining) = remaining_checks.checked_sub(charged_checks) else {
            return Ok(None);
        };
        *remaining_checks = remaining;
        let peer_quads = peer_quads
            .iter()
            .copied()
            .map(ConvexQuad::new)
            .collect::<Result<Vec<_>, _>>()?;
        let observation = self.window.observation(self.frame);
        let mut previous_right = 0;
        let mut runs = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let bounds = group.run.ink;
            if bounds.left < previous_right
                || bounds.left >= bounds.right
                || bounds.top >= bounds.bottom
                || bounds.right > width
                || bounds.bottom > height
            {
                return Err(Error::Invalid(
                    "invalid shared recognition run bounds".into(),
                ));
            }
            previous_right = bounds.right;
            let mut run = RunInk {
                run: index,
                ..RunInk::default()
            };
            for y in bounds.top..bounds.bottom {
                for x in bounds.left..bounds.right {
                    let (x, y) = (x + expanded.left, y + expanded.top);
                    if self.image.get_pixel(x, y).0[0] >= self.threshold {
                        continue;
                    }
                    let part = classify_pixel(observation, expanded, x, y);
                    run.sample.record(part);
                    let point = self
                        .window
                        .to_source(self.frame, Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
                    for (peer, quad) in peer_quads.iter().enumerate() {
                        if quad.contains(point) {
                            run.peers.entry(peer).or_default().record(part);
                        }
                    }
                }
            }
            if run.sample.original + run.sample.added()
                != group.run.observation_support.inside_pixels
            {
                return Err(Error::Invalid(
                    "recognized run differs from shared raster ink".into(),
                ));
            }
            runs.push(run);
        }
        let mut covered = InkCounts::default();
        let mut covered_peers = BTreeMap::<usize, InkCounts>::new();
        for run in &runs {
            covered.add(run.sample);
            for (&peer, &count) in &run.peers {
                covered_peers.entry(peer).or_default().add(count);
            }
        }
        if sample.included() != covered
            || peers.iter().any(|(peer, count)| {
                let covered = covered_peers.get(peer).copied().unwrap_or_default();
                covered != count.included()
            })
            || covered_peers.keys().any(|peer| !peers.contains_key(peer))
        {
            return Err(Error::Invalid(
                "recognition runs do not conserve measured source ink".into(),
            ));
        }
        Ok(Some(RunAccounting {
            runs,
            charged_checks,
        }))
    }

    fn measure(
        &self,
        expanded: InkBounds,
        observation: InkBounds,
        peers: &[ConvexQuad],
    ) -> Result<MeasuredInk, Error> {
        let mut components = Vec::<ComponentInk>::new();
        let regions = regions_visit(
            self.image,
            self.threshold,
            PixelClass::Ink,
            observation,
            |index, x, y| {
                if components.len() == index {
                    components.push(ComponentInk {
                        component_index: index,
                        ..ComponentInk::default()
                    });
                }
                let c = &mut components[index];
                let part = classify_pixel(observation, expanded, x, y);
                c.sample.record(part);
                let point = self
                    .window
                    .to_source(self.frame, Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
                let mut covered = false;
                for (i, peer) in peers.iter().enumerate() {
                    if peer.contains(point) {
                        c.peers.entry(i).or_default().record(part);
                        covered = true;
                    }
                }
                if covered {
                    c.peer_union.record(part);
                }
            },
        )?;
        let mut sample = InkCounts::default();
        let mut peer_union = InkCounts::default();
        let mut peer_counts = BTreeMap::<usize, InkCounts>::new();
        for (region, component) in regions.iter().zip(&components) {
            if u64::try_from(region.area_pixels).ok()
                != Some(
                    component.sample.original
                        + component.sample.added()
                        + component.sample.outside_expansion,
                )
                || region.observation_support.inside_pixels != component.sample.original
            {
                return Err(Error::Invalid(
                    "fragment accounting component mismatch".into(),
                ));
            }
            sample.add(component.sample);
            peer_union.add(component.peer_union);
            for (&i, &counts) in &component.peers {
                peer_counts.entry(i).or_default().add(counts);
            }
        }
        components.retain(ComponentInk::relevant);
        Ok(MeasuredInk {
            sample,
            peer_union,
            peers: peer_counts,
            components,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn rect(l: f64, t: f64, r: f64, b: f64) -> Quad {
        [Point(l, b), Point(r, b), Point(r, t), Point(l, t)]
    }

    #[test]
    fn count_pixels_not_component_boxes_or_duplicate_peer_membership() -> Result<(), Error> {
        let mut image = GrayImage::from_pixel(9, 5, Luma([255]));
        for (x, y) in [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (4, 2),
            (5, 2),
            (6, 2),
            (6, 3),
            (8, 4),
        ] {
            image.put_pixel(x, y, Luma([0]));
        }
        let sample = Sample {
            image: &image,
            frame: LineFrame {
                corners: rect(1., 0., 4., 5.),
                width: 3,
                height: 5,
                axis_swapped: false,
            },
            window: SampleWindow {
                padding: [1, 0, 5, 0],
            },
            threshold: 128,
        };
        let expanded = InkBounds {
            left: 1,
            top: 0,
            right: 6,
            bottom: 5,
        };
        let peers = [rect(3., 0., 7., 4.), rect(4., 0., 10., 4.)];
        let mut budget = 135;
        let measured = sample.account(expanded, &peers, &mut budget)?;
        let InkAccounting::Measured {
            sample,
            peer_union,
            peers,
            components,
            peers_fully_sampled,
            charged_checks,
        } = measured
        else {
            return Err(Error::Invalid("expected measured".into()));
        };
        assert_eq!(budget, 0);
        assert_eq!(charged_checks, 135);
        assert_eq!(peers_fully_sampled, [true, false]);
        assert_eq!(
            sample,
            InkCounts {
                original: 3,
                axis_added: 3,
                orthogonal_added: 0,
                outside_expansion: 3
            }
        );
        assert_eq!(
            peer_union,
            InkCounts {
                original: 1,
                axis_added: 3,
                orthogonal_added: 0,
                outside_expansion: 2
            }
        );
        assert_eq!(peers[&0], peer_union);
        assert_eq!(
            peers[&1],
            InkCounts {
                original: 0,
                axis_added: 3,
                orthogonal_added: 0,
                outside_expansion: 2
            }
        );
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].component_index, 0);
        Ok(())
    }

    #[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
    #[test]
    fn exact_recognition_run_partition_conserves_shared_peer_ink() -> Result<(), Error> {
        let mut image = GrayImage::from_pixel(9, 5, Luma([255]));
        for (x, y) in [(1, 1), (2, 1), (4, 2), (5, 2), (6, 2), (8, 4)] {
            image.put_pixel(x, y, Luma([0]));
        }
        let sample = Sample {
            image: &image,
            frame: LineFrame {
                corners: rect(1., 0., 5., 5.),
                width: 4,
                height: 5,
                axis_swapped: false,
            },
            window: SampleWindow {
                padding: [1, 0, 4, 0],
            },
            threshold: 128,
        };
        let expanded = InkBounds {
            left: 1,
            top: 0,
            right: 7,
            bottom: 5,
        };
        let peers = [rect(4., 0., 7., 5.), rect(5., 0., 9., 5.)];
        let mut budget = 135 + 90;
        let accounting = sample.account(expanded, &peers, &mut budget)?;
        let crop = image::imageops::crop_imm(&image, 1, 0, 6, 5).to_image();
        let isolation = crate::glyph_isolation::isolate_ink(
            &crop,
            InkBounds {
                left: 0,
                top: 0,
                right: 6,
                bottom: 5,
            },
            128,
            &[128],
        )?;
        let runs = sample
            .account_runs(
                expanded,
                &isolation.groups,
                &peers,
                &accounting,
                &mut budget,
            )?
            .ok_or_else(|| Error::Invalid("expected full run accounting".into()))?;
        assert_eq!(budget, 0);
        assert_eq!(runs.charged_checks, 90);
        assert_eq!(runs.runs.len(), 2);
        assert_eq!(
            runs.runs[0].sample,
            InkCounts {
                original: 2,
                axis_added: 0,
                orthogonal_added: 0,
                outside_expansion: 0
            }
        );
        assert_eq!(
            runs.runs[1].sample,
            InkCounts {
                original: 1,
                axis_added: 2,
                orthogonal_added: 0,
                outside_expansion: 0
            }
        );
        assert_eq!(runs.runs[1].peers[&0], runs.runs[1].sample);
        assert_eq!(
            runs.runs[1].peers[&1],
            InkCounts {
                original: 0,
                axis_added: 2,
                orthogonal_added: 0,
                outside_expansion: 0
            }
        );
        assert!(
            sample
                .account_runs(
                    expanded,
                    &isolation.groups,
                    &peers,
                    &accounting,
                    &mut budget
                )?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn orthogonal_salvage_is_not_axis_added_foreground() -> Result<(), Error> {
        let mut image = GrayImage::from_pixel(8, 6, Luma([255]));
        for (x, y) in [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (5, 1),
            (5, 2),
            (5, 3),
            (6, 4),
            (7, 5),
        ] {
            image.put_pixel(x, y, Luma([0]));
        }
        let sample = Sample {
            image: &image,
            frame: LineFrame {
                corners: rect(1., 2., 4., 4.),
                width: 3,
                height: 2,
                axis_swapped: false,
            },
            window: SampleWindow {
                padding: [1, 2, 4, 2],
            },
            threshold: 128,
        };
        let expanded = InkBounds {
            left: 1,
            top: 1,
            right: 7,
            bottom: 5,
        };
        let peers = [rect(1., 0., 4., 2.), rect(4., 0., 7., 5.)];
        let result = sample.account(expanded, &peers, &mut 1_000_000)?;
        let InkAccounting::Measured {
            sample: count,
            peers,
            ..
        } = result
        else {
            return Err(Error::Invalid("expected measured".into()));
        };
        assert_eq!(
            count,
            InkCounts {
                original: 2,
                axis_added: 4,
                orthogonal_added: 2,
                outside_expansion: 1
            }
        );
        assert_eq!(peers[&0].orthogonal_added, 1);
        assert_eq!(peers[&0].axis_added, 0);
        assert_eq!(peers[&1].axis_added, 4);
        assert_eq!(peers[&1].orthogonal_added, 0);
        Ok(())
    }

    #[test]
    fn rotated_mapping_and_budget_are_explicit() -> Result<(), Error> {
        let image = GrayImage::from_pixel(4, 3, Luma([0]));
        let rotate = |Point(x, y)| Point(10.0 - y, 20.0 + x);
        let frame = LineFrame {
            corners: rect(0., 0., 2., 3.).map(rotate),
            width: 2,
            height: 3,
            axis_swapped: false,
        };
        let sample = Sample {
            image: &image,
            frame,
            window: SampleWindow {
                padding: [0, 0, 2, 0],
            },
            threshold: 128,
        };
        let expanded = InkBounds {
            left: 0,
            top: 0,
            right: 3,
            bottom: 3,
        };
        let peer = rect(2., 0., 4., 3.).map(rotate);
        let mut budget = 23;
        assert!(matches!(
            sample.account(expanded, &[peer], &mut budget)?,
            InkAccounting::BudgetExceeded {
                required_checks: 24,
                remaining_checks: 23
            }
        ));
        assert_eq!(budget, 23);
        let mut budget = 24;
        let result = sample.account(expanded, &[peer], &mut budget)?;
        assert!(
            matches!(result,InkAccounting::Measured {peer_union:InkCounts {original:0,axis_added:3,orthogonal_added:0,outside_expansion:3},peers_fully_sampled,..} if peers_fully_sampled == [true])
        );
        Ok(())
    }
}
