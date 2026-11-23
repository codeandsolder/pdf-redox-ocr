//! Page display geometry matching Poppler MediaBox/CropBox raster coordinates.

use crate::Error;
use serde::{Deserialize, Serialize};

/// PDF point in user-space coordinates. JSON encodes it as an X/Y pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point(pub f64, pub f64);

/// Four vertices in PDF user-space coordinates.
pub type Quad = [Point; 4];

/// Axis-aligned extent of a fixed-size polygon. Unlike `PageBox`, this may
/// be degenerate; neither coordinate axis assumes a particular orientation.
#[derive(Clone, Copy)]
pub(crate) struct QuadBounds {
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) max_x: f64,
    pub(crate) max_y: f64,
}

impl QuadBounds {
    pub(crate) fn new(quad: &Quad) -> Self {
        let mut result = Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        };
        for &Point(x, y) in quad {
            result.min_x = result.min_x.min(x);
            result.min_y = result.min_y.min(y);
            result.max_x = result.max_x.max(x);
            result.max_y = result.max_y.max(y);
        }
        result
    }

    pub(crate) const fn intersects(self, other: Self) -> bool {
        self.min_x < other.max_x
            && other.min_x < self.max_x
            && self.min_y < other.max_y
            && other.min_y < self.max_y
    }
}

/// Validated PDF page box with potentially nonzero origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageBox {
    /// Left.
    pub x0: f64,
    /// Bottom.
    pub y0: f64,
    /// Right.
    pub x1: f64,
    /// Top.
    pub y1: f64,
}

impl PageBox {
    /// Reject non-finite and degenerate rectangles.
    ///
    /// # Errors
    /// Returns an error when a rectangle contains nonfinite coordinates or has zero area.
    pub fn new(values: [f64; 4]) -> Result<Self, Error> {
        let [x0, y0, x1, y1] = values;
        if !values.iter().all(|v| v.is_finite()) || x1 <= x0 || y1 <= y0 {
            return Err(Error::Invalid("invalid PDF page box".into()));
        }
        Ok(Self { x0, y0, x1, y1 })
    }

    /// Interpret a PDF rectangle as two corner coordinates in either order.
    ///
    /// PDF files encountered in practice include Form `BBox` coordinates in
    /// descending X or Y order, even when their `MediaBox` is canonical.
    /// A zero-area rectangle contains no usable clipping region.
    ///
    /// # Errors
    /// Rejects non-finite coordinates instead of hiding malformed content.
    pub fn from_unordered_corners(values: [f64; 4]) -> Result<Option<Self>, Error> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("non-finite PDF rectangle".into()));
        }
        let [x0, y0, x1, y1] = values;
        let (left, right) = (x0.min(x1), x0.max(x1));
        let (bottom, top) = (y0.min(y1), y0.max(y1));
        if right <= left || top <= bottom {
            return Ok(None);
        }
        Self::new([left, bottom, right, top]).map(Some)
    }

    /// Width.
    #[must_use = "use the bounding-box width when positioning PDF geometry"]
    pub fn width(self) -> f64 {
        self.x1 - self.x0
    }

    /// Height.
    #[must_use = "use the bounding-box height when positioning PDF geometry"]
    pub fn height(self) -> f64 {
        self.y1 - self.y0
    }
}

/// PDF page rotation, limited to quarter turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    /// No rotation.
    Deg0,
    /// One clockwise quarter turn.
    Deg90,
    /// Half turn.
    Deg180,
    /// Three clockwise quarter turns.
    Deg270,
}

impl TryFrom<i32> for Rotation {
    type Error = Error;

    fn try_from(degrees: i32) -> Result<Self, Self::Error> {
        match degrees.rem_euclid(360) {
            0 => Ok(Self::Deg0),
            90 => Ok(Self::Deg90),
            180 => Ok(Self::Deg180),
            270 => Ok(Self::Deg270),
            _ => Err(Error::Invalid(format!(
                "unsupported page rotation {degrees}"
            ))),
        }
    }
}

/// Maps normalized rendered coordinates into and out of PDF page space.
///
/// Poppler uses the `MediaBox` by default, even if the `CropBox` differs.
/// The caller must supply the same box used for its rasterization.
#[derive(Clone, Copy, Debug)]
pub struct DisplayMapper {
    box_: PageBox,
    rotation: Rotation,
}

impl DisplayMapper {
    /// Create a bidirectional mapper.
    #[must_use = "retain the page transform for subsequent coordinate conversions"]
    pub const fn new(box_: PageBox, rotation: Rotation) -> Self {
        Self { box_, rotation }
    }

    /// The displayed page width and height in PDF user-space units.
    ///
    /// The page rotation swaps dimensions for 90°/270°. Deriving this
    /// directly from the mapped box avoids repeated inverse transforms and
    /// keeps native image rectification and Poppler crop scaling consistent.
    #[must_use = "use the page display dimensions for raster placement"]
    pub fn display_dimensions(self) -> Point {
        match self.rotation {
            Rotation::Deg0 | Rotation::Deg180 => Point(self.box_.width(), self.box_.height()),
            Rotation::Deg90 | Rotation::Deg270 => Point(self.box_.height(), self.box_.width()),
        }
    }

    /// Convert normalized raster U (right), V (down) into PDF user-space.
    #[must_use = "use the converted PDF coordinates when locating source elements"]
    pub fn to_pdf(self, raster_u: f64, raster_v: f64) -> Point {
        let page_box = self.box_;
        let width = page_box.width();
        let height = page_box.height();
        match self.rotation {
            Rotation::Deg0 => Point(
                raster_u.mul_add(width, page_box.x0),
                raster_v.mul_add(-height, page_box.y1),
            ),
            Rotation::Deg90 => Point(
                raster_v.mul_add(width, page_box.x0),
                raster_u.mul_add(height, page_box.y0),
            ),
            Rotation::Deg180 => Point(
                raster_u.mul_add(-width, page_box.x1),
                raster_v.mul_add(height, page_box.y0),
            ),
            Rotation::Deg270 => Point(
                raster_v.mul_add(-width, page_box.x1),
                raster_u.mul_add(-height, page_box.y1),
            ),
        }
    }

    /// Convert a PDF user-space point into normalized raster coordinates.
    #[must_use = "use the converted raster coordinates when sampling source pixels"]
    pub fn to_raster(self, Point(pdf_x, pdf_y): Point) -> Point {
        let page_box = self.box_;
        let width = page_box.width();
        let height = page_box.height();
        match self.rotation {
            Rotation::Deg0 => Point(
                (pdf_x - page_box.x0) / width,
                (page_box.y1 - pdf_y) / height,
            ),
            Rotation::Deg90 => Point(
                (pdf_y - page_box.y0) / height,
                (pdf_x - page_box.x0) / width,
            ),
            Rotation::Deg180 => Point(
                (page_box.x1 - pdf_x) / width,
                (pdf_y - page_box.y0) / height,
            ),
            Rotation::Deg270 => Point(
                (page_box.y1 - pdf_y) / height,
                (page_box.x1 - pdf_x) / width,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DisplayMapper, PageBox, Point, Rotation};

    #[test]
    fn all_quarter_turns_roundtrip_nonzero_origin() -> Result<(), crate::Error> {
        let box_ = PageBox::new([30.0, 40.0, 830.0, 640.0])?;
        for angle in [0, 90, 180, 270, 360, -90] {
            let mapper = DisplayMapper::new(box_, Rotation::try_from(angle)?);
            for [u, v] in [[0.0, 0.0], [1.0, 1.0], [0.3, 0.4], [0.25, 0.875]] {
                let Point(back_u, back_v) = mapper.to_raster(mapper.to_pdf(u, v));
                assert!((u - back_u).abs() < 1e-12, "{angle}: U mismatch");
                assert!((v - back_v).abs() < 1e-12, "{angle}: V mismatch");
            }
        }
        Ok(())
    }

    #[test]
    fn display_dimensions_follow_page_rotation() -> Result<(), crate::Error> {
        let box_ = PageBox::new([30.0, 40.0, 830.0, 640.0])?;
        for (rotation, expected) in [
            (Rotation::Deg0, Point(800.0, 600.0)),
            (Rotation::Deg90, Point(600.0, 800.0)),
            (Rotation::Deg180, Point(800.0, 600.0)),
            (Rotation::Deg270, Point(600.0, 800.0)),
        ] {
            let mapper = DisplayMapper::new(box_, rotation);
            assert_eq!(mapper.display_dimensions(), expected);
            let origin = mapper.to_pdf(0.0, 0.0);
            let horizontal = mapper.to_pdf(1.0, 0.0);
            let vertical = mapper.to_pdf(0.0, 1.0);
            assert!(
                ((horizontal.0 - origin.0).hypot(horizontal.1 - origin.1) - expected.0).abs()
                    < 1e-10
            );
            assert!(
                ((vertical.0 - origin.0).hypot(vertical.1 - origin.1) - expected.1).abs() < 1e-10
            );
        }
        Ok(())
    }

    #[test]
    fn unordered_form_bbox_corners_normalize_and_empty_rectangles_are_distinguished()
    -> Result<(), crate::Error> {
        let sorted = PageBox::new([10.0, 20.0, 70.0, 120.0])?;
        for candidate in [
            [10.0, 20.0, 70.0, 120.0],
            [70.0, 120.0, 10.0, 20.0],
            [70.0, 20.0, 10.0, 120.0],
            [10.0, 120.0, 70.0, 20.0],
        ] {
            assert_eq!(PageBox::from_unordered_corners(candidate)?, Some(sorted));
        }
        assert_eq!(
            PageBox::from_unordered_corners([0.0, 0.0, 0.0, 80.0])?,
            None
        );
        assert!(PageBox::from_unordered_corners([0.0, 0.0, f64::NAN, 80.0]).is_err());
        assert!(PageBox::from_unordered_corners([0.0, 0.0, f64::INFINITY, 80.0]).is_err());
        Ok(())
    }

    #[test]
    fn invalid_boxes_and_rotations_are_rejected() {
        assert!(PageBox::new([0.0, 0.0, f64::NAN, 1.0]).is_err());
        assert!(PageBox::new([1.0, 0.0, 1.0, 5.0]).is_err());
        assert!(Rotation::try_from(42).is_err());
    }
}
