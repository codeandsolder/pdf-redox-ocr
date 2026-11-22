//! Shared connected-region traversal for foreground ink and background holes.

use image::GrayImage;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::Error;

/// Half-open bounds in a source raster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InkBounds {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl InkBounds {
    pub(crate) fn contains(self, x: u32, y: u32) -> bool {
        (self.left..self.right).contains(&x) && (self.top..self.bottom).contains(&y)
    }

    pub(crate) const fn union(self, other: Self) -> Self {
        Self {
            left: if self.left < other.left {
                self.left
            } else {
                other.left
            },
            top: if self.top < other.top {
                self.top
            } else {
                other.top
            },
            right: if self.right > other.right {
                self.right
            } else {
                other.right
            },
            bottom: if self.bottom > other.bottom {
                self.bottom
            } else {
                other.bottom
            },
        }
    }

    pub(crate) const fn touches_edge(self, width: u32, height: u32) -> bool {
        self.left == 0 || self.top == 0 || self.right == width || self.bottom == height
    }
}

/// Pixel support inside and outside the supplied observation.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ObservationSupport {
    pub inside_pixels: u64,
    pub outside_pixels: u64,
    pub touches_observation_edge: bool,
}

impl ObservationSupport {
    fn record(&mut self, x: u32, y: u32, observation: InkBounds) {
        if observation.contains(x, y) {
            self.inside_pixels += 1;
            self.touches_observation_edge |= x == observation.left
                || y == observation.top
                || x + 1 == observation.right
                || y + 1 == observation.bottom;
        } else {
            self.outside_pixels += 1;
        }
    }

    pub(crate) const fn add(&mut self, other: Self) {
        self.inside_pixels += other.inside_pixels;
        self.outside_pixels += other.outside_pixels;
        self.touches_observation_edge |= other.touches_observation_edge;
    }
}

/// Nearest actual component pixels beyond either reading-axis edge, within
/// the observation's orthogonal band. Left is exclusive; right is inclusive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct BandNeighbors {
    pub left: Option<u32>,
    pub right: Option<u32>,
}

impl BandNeighbors {
    fn record(&mut self, x: u32, y: u32, observation: InkBounds) {
        if !(observation.top..observation.bottom).contains(&y) {
            return;
        }
        if x < observation.left {
            self.left = Some(self.left.map_or(x + 1, |old| old.max(x + 1)));
        } else if x >= observation.right {
            self.right = Some(self.right.map_or(x, |old| old.min(x)));
        }
    }
}

/// A threshold-connected set of actual raster pixels.
#[derive(Clone, Debug, Serialize)]
pub struct PixelRegion {
    pub bounds: InkBounds,
    pub area_pixels: usize,
    pub centroid: [f64; 2],
    pub touches_line_edge: bool,
    pub observation_support: ObservationSupport,
    pub band_neighbors: BandNeighbors,
}

/// Dual digital topology: eight-connected ink, four-connected background.
#[derive(Clone, Copy)]
pub(crate) enum PixelClass {
    Ink,
    Background,
}

impl PixelClass {
    const fn accepts(self, value: u8, threshold: u8) -> bool {
        match self {
            Self::Ink => value < threshold,
            Self::Background => value >= threshold,
        }
    }
}

struct Flood<'a> {
    image: &'a GrayImage,
    width: usize,
    height: usize,
    class: PixelClass,
    threshold: u8,
    visited: Vec<bool>,
    stack: Vec<usize>,
}

impl Flood<'_> {
    fn enqueue(&mut self, index: usize) {
        if !self.visited[index]
            && self
                .class
                .accepts(self.image.as_raw()[index], self.threshold)
        {
            self.visited[index] = true;
            self.stack.push(index);
        }
    }

    fn neighbors(&mut self, index: usize) {
        let (x, y) = (index % self.width, index / self.width);
        let (left, right, above, below) = (x > 0, x + 1 < self.width, y > 0, y + 1 < self.height);
        if left {
            self.enqueue(index - 1);
        }
        if right {
            self.enqueue(index + 1);
        }
        if above {
            self.enqueue(index - self.width);
        }
        if below {
            self.enqueue(index + self.width);
        }
        if matches!(self.class, PixelClass::Ink) {
            if left && above {
                self.enqueue(index - self.width - 1);
            }
            if right && above {
                self.enqueue(index - self.width + 1);
            }
            if left && below {
                self.enqueue(index + self.width - 1);
            }
            if right && below {
                self.enqueue(index + self.width + 1);
            }
        }
    }

    fn region(
        &mut self,
        seed: usize,
        observation: InkBounds,
        mut visit: impl FnMut(u32, u32),
    ) -> Result<PixelRegion, Error> {
        self.enqueue(seed);
        let mut bounds = InkBounds {
            left: self.image.width(),
            top: self.image.height(),
            right: 0,
            bottom: 0,
        };
        let mut area = 0_usize;
        let mut sums = [0.0, 0.0];
        let mut support = ObservationSupport::default();
        let mut band_neighbors = BandNeighbors::default();
        while let Some(index) = self.stack.pop() {
            let x = u32::try_from(index % self.width)
                .map_err(|_| Error::Invalid("unrepresentable region X".into()))?;
            let y = u32::try_from(index / self.width)
                .map_err(|_| Error::Invalid("unrepresentable region Y".into()))?;
            bounds.left = bounds.left.min(x);
            bounds.top = bounds.top.min(y);
            bounds.right = bounds.right.max(x + 1);
            bounds.bottom = bounds.bottom.max(y + 1);
            area += 1;
            sums[0] += f64::from(x);
            sums[1] += f64::from(y);
            support.record(x, y, observation);
            band_neighbors.record(x, y, observation);
            visit(x, y);
            self.neighbors(index);
        }
        let denominator = area
            .to_f64()
            .ok_or_else(|| Error::Invalid("unrepresentable region area".into()))?;
        Ok(PixelRegion {
            bounds,
            area_pixels: area,
            centroid: sums.map(|sum| sum / denominator),
            touches_line_edge: bounds.touches_edge(self.image.width(), self.image.height()),
            observation_support: support,
            band_neighbors,
        })
    }
}

/// Extract connected regions without altering the original pixel topology.
pub(crate) fn regions(
    image: &GrayImage,
    threshold: u8,
    class: PixelClass,
    observation: InkBounds,
) -> Result<Vec<PixelRegion>, Error> {
    regions_visit(image, threshold, class, observation, |_, _, _| {})
}

/// Visit every member pixel exactly once with its resulting component index.
/// The index is identical to that returned by the ordinary regions traversal.
pub(crate) fn regions_visit(
    image: &GrayImage,
    threshold: u8,
    class: PixelClass,
    observation: InkBounds,
    mut visit: impl FnMut(usize, u32, u32),
) -> Result<Vec<PixelRegion>, Error> {
    let (width, height) = image.dimensions();
    let total = usize::try_from(u64::from(width) * u64::from(height))
        .ok()
        .filter(|&n| n > 0 && n == image.as_raw().len())
        .ok_or_else(|| Error::Invalid("invalid connected raster dimensions".into()))?;
    if observation.left >= observation.right
        || observation.top >= observation.bottom
        || observation.right > width
        || observation.bottom > height
    {
        return Err(Error::Invalid("invalid raster observation".into()));
    }
    let mut flood = Flood {
        image,
        width: usize::try_from(width)
            .map_err(|_| Error::Invalid("unrepresentable width".into()))?,
        height: usize::try_from(height)
            .map_err(|_| Error::Invalid("unrepresentable height".into()))?,
        threshold,
        class,
        visited: vec![false; total],
        stack: Vec::new(),
    };
    let mut result = Vec::new();
    for (seed, &value) in image.as_raw().iter().enumerate() {
        if !flood.visited[seed] && class.accepts(value, threshold) {
            let index = result.len();
            result.push(flood.region(seed, observation, |x, y| visit(index, x, y))?);
        }
    }
    Ok(result)
}
