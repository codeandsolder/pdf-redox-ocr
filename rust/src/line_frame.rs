//! Shared rectified quadrilateral frame for source audits and retained-raster OCR.

use crate::{
    Error,
    geometry::{Point, Quad},
    pixel_regions::InkBounds,
    polygon::convex,
};
use crate::{
    ink_polarity::{Polarity, PolarityEvidence, estimate},
    rasterize::sample_whitened,
};
use image::{GrayImage, ImageBuffer, Pixel, Rgb, RgbImage, imageops};
use num_traits::ToPrimitive;

/// Choose between undirected geometry inspection and already oriented text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineAxis {
    Longest,
    #[cfg(feature = "pdf-integration")]
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

    /// Rectify the exact bounded sampling window once, preserving RGB.
    pub fn rectify<P: Pixel<Subpixel = u8>>(
        self,
        frame: LineFrame,
        image: &RgbImage,
        to_pixel: impl Fn(Point) -> Point,
        convert: impl Fn(Rgb<u8>) -> P,
    ) -> Result<ImageBuffer<P, Vec<u8>>, Error> {
        let (width, height) = self
            .dimensions(frame)
            .ok_or_else(|| Error::Invalid("unrepresentable sample window".into()))?;
        Ok(ImageBuffer::from_fn(width, height, |x, y| {
            let source = self.to_source(frame, Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
            convert(sample_whitened(image, to_pixel(source)))
        }))
    }

    /// Normalize the shared rectified raster using only the original observation.
    pub fn normalize_ink(
        self,
        frame: LineFrame,
        raster: &GrayImage,
        polarity: Polarity,
    ) -> Result<(GrayImage, PolarityEvidence), Error> {
        if self.dimensions(frame) != Some(raster.dimensions()) {
            return Err(Error::Invalid(
                "inconsistent rectified sample dimensions".into(),
            ));
        }
        let observation = self.observation(frame);
        let original = imageops::crop_imm(
            raster,
            observation.left,
            observation.top,
            frame.width,
            frame.height,
        )
        .to_image();
        let polarity = estimate(&original, polarity)?;
        Ok((polarity.normalize(raster), polarity))
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

    fn grow_side(
        &mut self,
        frame: LineFrame,
        inside: &impl Fn(Point) -> bool,
        side: usize,
        requested: u32,
    ) {
        let (mut low, mut high) = (self.padding[side], requested.max(self.padding[side]));
        while low < high {
            let candidate = low + (high - low).div_ceil(2);
            self.padding[side] = candidate;
            if self.valid(frame, inside) {
                low = candidate;
            } else {
                high = candidate - 1;
            }
        }
        self.padding[side] = low;
    }

    /// Minimum reading-axis extent enclosing a peer, plus a background
    /// margin, while retaining the existing orthogonal and opposite margins.
    /// Bounds are in original-frame pixels, not a projected page AABB.
    #[cfg(all(feature = "pdf-integration", feature = "native-inference"))]
    pub fn cover_peer(
        self,
        frame: LineFrame,
        inside: &impl Fn(Point) -> bool,
        edge: crate::layout_leaders::LeaderEdge,
        peer: &Quad,
        max_pixels: u64,
    ) -> Option<(Self, u32)> {
        use crate::polygon::ConvexQuad;
        let side = match edge {
            crate::layout_leaders::LeaderEdge::Left => 0,
            crate::layout_leaders::LeaderEdge::Right => 2,
        };
        let (w, h) = self.dimensions(frame)?;
        let base = w.checked_sub(self.padding[side])?;
        let maximum = u32::try_from(max_pixels / u64::from(h))
            .unwrap_or(u32::MAX)
            .saturating_sub(base);
        if maximum < self.padding[side] {
            return None;
        }
        let mut largest = self;
        largest.grow_side(frame, inside, side, maximum);
        let contains = |window: Self| {
            let Some((width, height)) = window.dimensions(frame) else {
                return false;
            };
            ConvexQuad::new(window.quad(
                frame,
                InkBounds {
                    left: 0,
                    top: 0,
                    right: width,
                    bottom: height,
                },
            ))
            .is_ok_and(|quad| peer.iter().all(|&p| quad.contains(p)))
        };
        if !contains(largest) {
            return None;
        }
        let (mut low, mut high) = (0, largest.padding[side]);
        while low < high {
            let mid = low + (high - low) / 2;
            let mut window = self;
            window.padding[side] = mid;
            if contains(window) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        let required = low;
        let mut result = self;
        // orthogonal body repair may increase the band height; use the
        // measured full orthogonal sampling dimension as an upper bound.
        let margin = crate::boundary_ink::edge_margin(h).saturating_add(1);
        result.padding[side] =
            largest.padding[side].min(required.saturating_add(margin).max(self.padding[side]));
        Some((result, required))
    }

    /// Grow each side independently, keeping the complete mapped rectangle
    /// inside the page and rejecting folded extrapolations.
    pub fn new(
        frame: LineFrame,
        inside: impl Fn(Point) -> bool,
        requested: [u32; 4],
        max_pixels: u64,
    ) -> Result<Self, Error> {
        let mut result = Self { padding: [0; 4] };
        if !result.valid(frame, &inside) {
            return Err(Error::Invalid(
                "line audit quad extends outside source extent".into(),
            ));
        }
        for (side, requested) in requested.into_iter().enumerate() {
            result.grow_side(frame, &inside, side, requested);
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

#[cfg(all(test, feature = "pdf-integration", feature = "native-inference"))]
mod peer_tests {
    use super::{LineFrame, SampleWindow};
    use crate::{geometry::Point, layout_leaders::LeaderEdge};

    #[test]
    fn peer_extent_follows_rotated_frame_and_respects_page_and_budget() -> Result<(), crate::Error>
    {
        for rotate in [false, true] {
            let map = |Point(x, y)| {
                if rotate {
                    Point(200. - y, x)
                } else {
                    Point(x, y)
                }
            };
            let frame = LineFrame {
                corners: [
                    Point(50., 70.),
                    Point(100., 70.),
                    Point(100., 50.),
                    Point(50., 50.),
                ]
                .map(map),
                width: 50,
                height: 20,
                axis_swapped: false,
            };
            let inside = |Point(x, y)| (0. ..=200.).contains(&x) && (0. ..=200.).contains(&y);
            let window = SampleWindow::new(frame, inside, [20, 10, 45, 10], 100_000)?;
            let peer = [
                Point(100., 70.),
                Point(165., 70.),
                Point(165., 50.),
                Point(100., 50.),
            ]
            .map(map);
            let (right, extent) = window
                .cover_peer(frame, &inside, LeaderEdge::Right, &peer, 100_000)
                .ok_or_else(|| crate::Error::Invalid("expected right peer extent".into()))?;
            assert_eq!(extent, 65);
            assert_eq!(right.padding, [20, 10, 73, 10]);
            assert!(
                window
                    .cover_peer(frame, &inside, LeaderEdge::Right, &peer, 5_000)
                    .is_none()
            );
            let other = [
                Point(10., 70.),
                Point(50., 70.),
                Point(50., 50.),
                Point(10., 50.),
            ]
            .map(map);
            let (left, extent) = window
                .cover_peer(frame, &inside, LeaderEdge::Left, &other, 100_000)
                .ok_or_else(|| crate::Error::Invalid("expected left peer extent".into()))?;
            assert_eq!(extent, 40);
            assert_eq!(left.padding, [48, 10, 45, 10]);
            let outside = peer.map(|Point(x, y)| {
                if rotate {
                    Point(x, y + 100.)
                } else {
                    Point(x + 100., y)
                }
            });
            assert!(
                window
                    .cover_peer(frame, &inside, LeaderEdge::Right, &outside, 100_000)
                    .is_none()
            );
        }
        Ok(())
    }
}
