//! Model-independent luminance polarity and contrast of a source line crop.
//!
//! A two-class histogram partition estimates the visible background from the
//! crop perimeter. This is an auditable hypothesis, not a glyph classification.

use image::{GrayImage, Luma};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::Error;

/// Requested orientation of the foreground luminance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    #[default]
    Auto,
    Dark,
    Light,
}

/// The actual foreground luminance relative to the estimated background.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Foreground {
    Dark,
    Light,
    Uniform,
}

/// Observable histogram evidence, including low-contrast and weak-border cases.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolarityEvidence {
    pub foreground: Foreground,
    /// Last luminance in the lower histogram class (Otsu partition).
    pub split: Option<u8>,
    /// Median luminances of the two classes, if both exist.
    pub class_medians: Option<[u8; 2]>,
    /// Number of pixels in the low and high classes respectively.
    pub class_pixels: [u64; 2],
    /// Number of distinct perimeter pixels in the low and high classes.
    pub perimeter_pixels: [u64; 2],
}

fn histogram(image: &GrayImage) -> [u64; 256] {
    let mut counts = [0; 256];
    for &value in image.as_raw() {
        counts[usize::from(value)] += 1;
    }
    counts
}

fn median(counts: &[u64], start: u8, total: u64) -> u8 {
    let mut cumulative = 0;
    for (offset, &count) in counts.iter().enumerate() {
        cumulative += count;
        if cumulative > (total - 1) / 2 {
            return start.saturating_add(u8::try_from(offset).unwrap_or(u8::MAX));
        }
    }
    start
}

fn split_histogram(counts: &[u64; 256]) -> Option<u8> {
    let total = counts.iter().sum::<u64>().to_f64()?;
    let sum = counts
        .iter()
        .enumerate()
        .try_fold(0.0_f64, |acc, (index, &count)| {
            Some(index.to_f64()?.mul_add(count.to_f64()?, acc))
        })?;
    let (mut lower_count, mut lower_sum, mut best_score) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut best = None;
    for (index, &count) in counts.iter().take(255).enumerate() {
        let n = count.to_f64()?;
        lower_count += n;
        lower_sum = index.to_f64()?.mul_add(n, lower_sum);
        let upper_count = total - lower_count;
        if lower_count <= 0.0 || upper_count <= 0.0 {
            continue;
        }
        let separation = lower_sum / lower_count - (sum - lower_sum) / upper_count;
        let score = lower_count * upper_count * separation * separation;
        if score > best_score {
            best_score = score;
            best = u8::try_from(index).ok();
        }
    }
    best
}

fn perimeter(image: &GrayImage, split: u8) -> [u64; 2] {
    let mut counts = [0, 0];
    let (width, height) = image.dimensions();
    for (x, y, pixel) in image.enumerate_pixels() {
        if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
            counts[usize::from(pixel[0] > split)] += 1;
        }
    }
    counts
}

/// Estimate polarity without looking at OCR text.
///
/// Ties in perimeter support fall back to whole-crop occupancy.
/// The raw counts allow callers to identify ambiguous or mixed backgrounds.
///
/// # Errors
/// Rejects an empty crop.
pub fn estimate(image: &GrayImage, requested: Polarity) -> Result<PolarityEvidence, Error> {
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::Invalid("empty polarity crop".into()));
    }
    let counts = histogram(image);
    let Some(split) = split_histogram(&counts) else {
        return Ok(PolarityEvidence {
            foreground: Foreground::Uniform,
            split: None,
            class_medians: None,
            class_pixels: [counts.iter().sum(), 0],
            perimeter_pixels: perimeter(image, 255),
        });
    };
    let (lower, upper) = counts.split_at(usize::from(split) + 1);
    let class_pixels = [lower.iter().sum(), upper.iter().sum()];
    let perimeter_pixels = perimeter(image, split);
    let foreground = match requested {
        Polarity::Dark => Foreground::Dark,
        Polarity::Light => Foreground::Light,
        Polarity::Auto => {
            if (perimeter_pixels[0], class_pixels[0]) > (perimeter_pixels[1], class_pixels[1]) {
                Foreground::Light
            } else {
                Foreground::Dark
            }
        }
    };
    Ok(PolarityEvidence {
        foreground,
        split: Some(split),
        class_medians: Some([
            median(lower, 0, class_pixels[0]),
            median(upper, split + 1, class_pixels[1]),
        ]),
        class_pixels,
        perimeter_pixels,
    })
}

impl PolarityEvidence {
    /// Orient ink and normalize the measured background to white.
    ///
    /// The foreground uses the physical luminance endpoint, not its class
    /// median. This preserves antialias coverage on a white background and
    /// avoids broadening faint strokes by stretching their median to black.
    /// A uniform observation has no measurable ink.
    #[must_use = "use the polarity-normalized image for downstream recognition"]
    pub fn normalize(&self, image: &GrayImage) -> GrayImage {
        let Some([low, high]) = self.class_medians else {
            return GrayImage::from_pixel(image.width(), image.height(), Luma([255]));
        };
        let invert = self.foreground == Foreground::Light;
        let background = if invert { 255 - low } else { high };
        if background == 0 || low >= high {
            return GrayImage::from_pixel(image.width(), image.height(), Luma([255]));
        }
        let denominator = u16::from(background);
        GrayImage::from_fn(image.width(), image.height(), |x, y| {
            let value = image.get_pixel(x, y)[0];
            let oriented = if invert { 255 - value } else { value };
            let normalized =
                (u16::from(oriented.min(background)) * 255 + denominator / 2) / denominator;
            Luma([u8::try_from(normalized).unwrap_or(255)])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Foreground, Polarity, estimate};
    use image::{GrayImage, Luma};

    #[test]
    fn reverse_and_nonwhite_background_preserve_same_ink() -> Result<(), crate::Error> {
        let dark = GrayImage::from_fn(17, 9, |x, y| {
            Luma([if (3..7).contains(&x) && (2..7).contains(&y) {
                30
            } else {
                180
            }])
        });
        let light = GrayImage::from_fn(17, 9, |x, y| {
            Luma([if dark.get_pixel(x, y)[0] == 30 {
                225
            } else {
                75
            }])
        });
        let a = estimate(&dark, Polarity::Auto)?;
        let b = estimate(&light, Polarity::Auto)?;
        assert_eq!(a.foreground, Foreground::Dark);
        assert_eq!(b.foreground, Foreground::Light);
        assert_eq!(a.normalize(&dark), b.normalize(&light));
        assert_eq!(a.class_medians, Some([30, 180]));
        assert_eq!(b.class_medians, Some([75, 225]));
        assert_eq!(
            estimate(&light, Polarity::Dark)?.foreground,
            Foreground::Dark
        );
        Ok(())
    }

    #[test]
    fn white_background_preserves_every_gray_level_even_for_faint_ink() -> Result<(), crate::Error>
    {
        let image = GrayImage::from_fn(256, 4, |x, y| {
            Luma([if y == 1 {
                u8::try_from(x).unwrap_or(255)
            } else {
                255
            }])
        });
        let evidence = estimate(&image, Polarity::Auto)?;
        assert_eq!(evidence.foreground, Foreground::Dark);
        assert_eq!(evidence.normalize(&image), image);
        Ok(())
    }

    #[test]
    fn uniform_and_thin_crops_report_only_real_perimeter_pixels() -> Result<(), crate::Error> {
        for (width, height) in [(1, 1), (1, 9), (9, 1), (7, 8)] {
            let image = GrayImage::from_pixel(width, height, Luma([132]));
            let evidence = estimate(&image, Polarity::Auto)?;
            assert_eq!(evidence.foreground, Foreground::Uniform);
            assert_eq!(
                usize::try_from(evidence.perimeter_pixels[0]).ok(),
                Some(
                    image
                        .enumerate_pixels()
                        .filter(|(x, y, _)| *x == 0
                            || *y == 0
                            || *x + 1 == width
                            || *y + 1 == height)
                        .count()
                )
            );
            assert!(
                evidence
                    .normalize(&image)
                    .as_raw()
                    .iter()
                    .all(|&p| p == 255)
            );
        }
        Ok(())
    }
}
