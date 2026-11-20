//! Resolution-normalized appearance evidence. Image similarity is not an OCR
//! probability, and a shape never supplies its own semantic label.

use crate::{Error, pixel_regions::InkBounds};
use image::{
    GrayImage,
    imageops::{self, FilterType},
};
use serde::{Deserialize, Serialize};

const SIDE: u32 = 24;
const CELLS: usize = 576;

/// Tight dark-ink bounds and a fixed-resolution coverage image. The original
/// aspect ratio remains separate from the normalized two-dimensional shape.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "ShapeData", into = "ShapeData")]
pub struct GlyphShape {
    width: u32,
    height: u32,
    coverage: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct ShapeData {
    width: u32,
    height: u32,
    ink_hex: String,
}

impl From<GlyphShape> for ShapeData {
    fn from(shape: GlyphShape) -> Self {
        Self {
            width: shape.width,
            height: shape.height,
            ink_hex: hex::encode(shape.coverage),
        }
    }
}

impl TryFrom<ShapeData> for GlyphShape {
    type Error = Error;
    fn try_from(data: ShapeData) -> Result<Self, Error> {
        let coverage = hex::decode(data.ink_hex)
            .map_err(|_| Error::Invalid("invalid glyph coverage encoding".into()))?;
        if data.width == 0 || data.height == 0 || coverage.len() != CELLS {
            return Err(Error::Invalid("invalid glyph shape dimensions".into()));
        }
        Ok(Self {
            width: data.width,
            height: data.height,
            coverage,
        })
    }
}

impl GlyphShape {
    /// Measure the actual grayscale values on a known ink rectangle. The
    /// normalized crop is not thresholded or stretched back into PDF space.
    ///
    /// # Errors
    /// Rejects empty or out-of-image rectangles.
    pub fn measure(image: &GrayImage, ink: InkBounds) -> Result<Self, Error> {
        if ink.left >= ink.right
            || ink.top >= ink.bottom
            || ink.right > image.width()
            || ink.bottom > image.height()
        {
            return Err(Error::Invalid("invalid glyph shape bounds".into()));
        }
        let width = ink.right - ink.left;
        let height = ink.bottom - ink.top;
        let crop = imageops::crop_imm(image, ink.left, ink.top, width, height);
        let mut coverage = imageops::resize(&*crop, SIDE, SIDE, FilterType::Triangle).into_raw();
        for value in &mut coverage {
            *value = 255 - *value;
        }
        Ok(Self {
            width,
            height,
            coverage,
        })
    }

    /// Symmetric, scale-normalized distance including original aspect ratio.
    /// A one-cell translation search accounts for rasterization and tight-box
    /// rounding without permitting arbitrary deformation or rotation.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let aspect = (f64::from(self.width) * f64::from(other.height)
            / (f64::from(self.height) * f64::from(other.width)))
        .ln()
        .abs()
        .min(1.0);
        let mass = |ink: &[u8]| ink.iter().map(|&v| u32::from(v)).sum::<u32>();
        let total = mass(&self.coverage) + mass(&other.coverage);
        let mut best_overlap = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                best_overlap =
                    best_overlap.max(shifted_overlap(&self.coverage, &other.coverage, dx, dy));
            }
        }
        0.2_f64.mul_add(
            aspect,
            f64::from(total - 2 * best_overlap) / (255.0 * f64::from(SIDE * SIDE)),
        )
    }
}

fn shifted_overlap(first: &[u8], second: &[u8], dx: i32, dy: i32) -> u32 {
    const N: usize = 24;
    let (a_x, b_x) = (
        usize::try_from((-dx).max(0)).unwrap_or_default(),
        usize::try_from(dx.max(0)).unwrap_or_default(),
    );
    let (a_y, b_y) = (
        usize::try_from((-dy).max(0)).unwrap_or_default(),
        usize::try_from(dy.max(0)).unwrap_or_default(),
    );
    let width = N - a_x.max(b_x);
    let height = N - a_y.max(b_y);
    (0..height)
        .map(|row| {
            let a = (row + a_y) * N + a_x;
            let b = (row + b_y) * N + b_x;
            first[a..a + width]
                .iter()
                .zip(&second[b..b + width])
                .map(|(&u, &v)| u32::from(u.min(v)))
                .sum::<u32>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn full(image: &GrayImage) -> InkBounds {
        InkBounds {
            left: 0,
            top: 0,
            right: image.width(),
            bottom: image.height(),
        }
    }
    #[test]
    fn shape_distance_is_symmetric_and_preserves_aspect() -> Result<(), Error> {
        let image =
            GrayImage::from_fn(12, 12, |x, y| Luma([if x < 3 || y >= 9 { 0 } else { 255 }]));
        let shape = GlyphShape::measure(&image, full(&image))?;
        let scaled = imageops::resize(&image, 24, 24, FilterType::Nearest);
        let same = GlyphShape::measure(&scaled, full(&scaled))?;
        let wide = imageops::resize(&image, 24, 12, FilterType::Nearest);
        let different = GlyphShape::measure(&wide, full(&wide))?;
        assert!(shape.distance(&shape).abs() < 1e-12);
        assert!(shape.distance(&same) < 0.04);
        assert!(shape.distance(&different) > 0.1);
        assert!((shape.distance(&different) - different.distance(&shape)).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn serialized_shapes_reject_truncated_coverage() -> Result<(), Box<dyn std::error::Error>> {
        let shape = GlyphShape::measure(
            &GrayImage::new(3, 5),
            InkBounds {
                left: 0,
                top: 0,
                right: 3,
                bottom: 5,
            },
        )?;
        let encoded = serde_json::to_string(&shape)?;
        let decoded: GlyphShape = serde_json::from_str(&encoded)?;
        assert!(shape.distance(&decoded).abs() < 1e-12);
        assert!(
            serde_json::from_str::<GlyphShape>(r#"{"width":3,"height":5,"ink_hex":"ff"}"#).is_err()
        );
        Ok(())
    }
}
