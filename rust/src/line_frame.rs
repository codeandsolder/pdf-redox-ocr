//! Shared rectified quadrilateral frame for source audits and retained-raster OCR.

use crate::{
    Error,
    geometry::{Point, Quad},
    pixel_regions::InkBounds,
};
use crate::{
    ink_polarity::{Polarity, PolarityEvidence, estimate},
    rasterize::sample_whitened,
};
use image::{GrayImage, Pixel, RgbImage, imageops};
use num_traits::ToPrimitive;

/// Choose between undirected geometry inspection and already oriented text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineAxis {
    Longest,
    #[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
    ReadingOrder,
}

#[derive(Clone, Copy, Debug)]
pub struct LineFrame {
    pub corners: Quad,
    pub width: u32,
    pub height: u32,
    pub axis_swapped: bool,
}

fn distance(a: Point, b: Point) -> f64 {
    (b.0 - a.0).hypot(b.1 - a.1)
}

fn interpolate(a: Point, b: Point, t: f64) -> Point {
    Point((b.0 - a.0).mul_add(t, a.0), (b.1 - a.1).mul_add(t, a.1))
}

pub fn convex(quad: &Quad) -> bool {
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

impl LineFrame {
    pub fn new(
        mut corners: Quad,
        scale: impl Fn(f64) -> f64,
        max_pixels: u64,
        axis: LineAxis,
    ) -> Result<Self, Error> {
        if !convex(&corners) {
            return Err(Error::Invalid(
                "nonconvex or degenerate line audit quad".into(),
            ));
        }
        // Draft winding is BL/BR/TR/TL in the detector's frame.
        // For elongated vertical boxes, measure along the longer edge.
        // This establishes an axis, not the reading direction of the text.
        let horizontal = f64::midpoint(
            distance(corners[0], corners[1]),
            distance(corners[3], corners[2]),
        );
        let vertical = f64::midpoint(
            distance(corners[0], corners[3]),
            distance(corners[1], corners[2]),
        );
        let axis_swapped = axis == LineAxis::Longest && vertical > 1.5 * horizontal;
        let (w, h) = if axis_swapped {
            corners.rotate_left(1);
            (vertical, horizontal)
        } else {
            (horizontal, vertical)
        };
        let pixels = |points: f64| {
            scale(points)
                .ceil()
                .to_u32()
                .filter(|&value| value > 0)
                .ok_or_else(|| Error::Invalid("unrepresentable line raster extent".into()))
        };
        let width = pixels(w)?;
        let height = pixels(h)?;
        if u64::from(width) * u64::from(height) > max_pixels {
            return Err(Error::Invalid("oriented line exceeds raster budget".into()));
        }
        Ok(Self {
            corners,
            width,
            height,
            axis_swapped,
        })
    }

    /// Bilinear quad interpolation in pixel-corner coordinates.
    pub fn to_source(self, Point(x, y): Point) -> Point {
        let u = x / f64::from(self.width);
        let v = y / f64::from(self.height);
        let top = interpolate(self.corners[3], self.corners[2], u);
        let bottom = interpolate(self.corners[0], self.corners[1], u);
        interpolate(top, bottom, v)
    }
}

/// The extended sampling rectangle in the original line's pixel coordinate
/// system. Negative positions are valid, but all four mapped corners must
/// remain within the source raster or PDF page.
#[derive(Clone, Copy, Debug)]
pub struct SampleWindow {
    pub padding: [u32; 4], // left, top, right, bottom
}

impl SampleWindow {
    pub fn dimensions(self, frame: LineFrame) -> Option<(u32, u32)> {
        let [left, top, right, bottom] = self.padding;
        Some((
            frame.width.checked_add(left)?.checked_add(right)?,
            frame.height.checked_add(top)?.checked_add(bottom)?,
        ))
    }

    pub const fn observation(self, frame: LineFrame) -> InkBounds {
        let [left, top, _, _] = self.padding;
        InkBounds {
            left,
            top,
            right: left + frame.width,
            bottom: top + frame.height,
        }
    }

    pub fn to_source(self, frame: LineFrame, point: Point) -> Point {
        frame.to_source(Point(
            point.0 - f64::from(self.padding[0]),
            point.1 - f64::from(self.padding[1]),
        ))
    }

    pub fn quad(self, frame: LineFrame, bounds: InkBounds) -> Quad {
        let l = f64::from(bounds.left);
        let r = f64::from(bounds.right);
        let t = f64::from(bounds.top);
        let b = f64::from(bounds.bottom);
        [Point(l, b), Point(r, b), Point(r, t), Point(l, t)].map(|p| self.to_source(frame, p))
    }

    /// Rectify the same bounded source window and normalize ink using only the original observation.
    pub fn normalized_ink(
        self,
        frame: LineFrame,
        image: &RgbImage,
        to_pixel: impl Fn(Point) -> Point,
        polarity: Polarity,
    ) -> Result<(GrayImage, PolarityEvidence), Error> {
        let (width, height) = self
            .dimensions(frame)
            .ok_or_else(|| Error::Invalid("unrepresentable sample window".into()))?;
        let raster = GrayImage::from_fn(width, height, |x, y| {
            let source = self.to_source(frame, Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
            sample_whitened(image, to_pixel(source)).to_luma()
        });
        let observation = self.observation(frame);
        let original = imageops::crop_imm(
            &raster,
            observation.left,
            observation.top,
            frame.width,
            frame.height,
        )
        .to_image();
        let polarity = estimate(&original, polarity)?;
        Ok((polarity.normalize(&raster), polarity))
    }

    fn valid(self, frame: LineFrame, inside: &impl Fn(Point) -> bool) -> bool {
        let Some((width, height)) = self.dimensions(frame) else {
            return false;
        };
        let quad = self.quad(
            frame,
            InkBounds {
                left: 0,
                top: 0,
                right: width,
                bottom: height,
            },
        );
        convex(&quad) && quad.into_iter().all(inside)
    }

    /// Grow each side independently, keeping the complete mapped rectangle
    /// inside the page and rejecting folded extrapolations.
    pub fn new(
        frame: LineFrame,
        inside: impl Fn(Point) -> bool,
        requested: u32,
        max_pixels: u64,
    ) -> Result<Self, Error> {
        let mut result = Self { padding: [0; 4] };
        if !result.valid(frame, &inside) {
            return Err(Error::Invalid(
                "line audit quad extends outside source extent".into(),
            ));
        }
        for side in 0..4 {
            let (mut low, mut high) = (0, requested);
            while low < high {
                let candidate = low + (high - low).div_ceil(2);
                result.padding[side] = candidate;
                if result.valid(frame, &inside) {
                    low = candidate;
                } else {
                    high = candidate - 1;
                }
            }
            result.padding[side] = low;
        }
        let (width, height) = result
            .dimensions(frame)
            .ok_or_else(|| Error::Invalid("unrepresentable padded line".into()))?;
        if u64::from(width) * u64::from(height) > max_pixels {
            return Err(Error::Invalid("padded line exceeds raster budget".into()));
        }
        Ok(result)
    }
}
