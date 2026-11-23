//! Exact, reproducible counterfactual mask of independently measured
//! perpendicular continuation stripes. No new foreground is synthesized.

use std::collections::BTreeMap;

use image::{GrayImage, Luma, Rgb, RgbImage};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Error,
    foreign_ink::{EdgeStroke, OrthogonalStroke},
    ink_polarity::{Foreground, PolarityEvidence},
    pixel_regions::InkBounds,
};

#[derive(Debug, Serialize)]
pub struct MaskEvidence {
    pub component_pixels: BTreeMap<usize, u64>,
    /// SHA-256 of observation-sized row-major mask bytes (0/1).
    pub mask_sha256: String,
    pub removed_pixels: u64,
    pub foreground_before: u64,
    pub foreground_after: u64,
    /// Luminance used in RGB replacement samples, from original background class.
    pub background_luminance: u8,
}

pub struct Mask {
    pub pixels: GrayImage,
    pub evidence: MaskEvidence,
}

fn invalid() -> Error {
    Error::Invalid("inconsistent measured orthogonal stroke mask".into())
}

fn unique_strokes<'a>(
    strokes: impl IntoIterator<Item = &'a OrthogonalStroke>,
) -> Result<BTreeMap<usize, &'a EdgeStroke>, Error> {
    let mut unique: BTreeMap<usize, &EdgeStroke> = BTreeMap::new();
    for s in strokes {
        let s = &s.stroke;
        if let Some(old) = unique.insert(s.component_index, s)
            && (old.inside_bounds != s.inside_bounds
                || old.inside_sections != s.inside_sections
                || old.inside.pixels != s.inside.pixels)
        {
            return Err(invalid());
        }
    }
    Ok(unique)
}

fn mark(
    pixels: &mut GrayImage,
    normalized: &GrayImage,
    threshold: u8,
    observation: InkBounds,
    stroke: &EdgeStroke,
) -> Result<u64, Error> {
    let b = stroke.inside_bounds;
    if b.top < observation.top
        || b.bottom > observation.bottom
        || b.left < observation.left
        || b.right > observation.right
        || b.top >= b.bottom
        || b.left >= b.right
        || usize::try_from(b.bottom - b.top).ok() != Some(stroke.inside_sections.len())
    {
        return Err(invalid());
    }
    let mut count = 0;
    for (y, &[left, right]) in (b.top..b.bottom).zip(&stroke.inside_sections) {
        if left < b.left || right > b.right || left >= right {
            return Err(invalid());
        }
        let local_y = y - observation.top;
        for x in left..right {
            let local_x = x - observation.left;
            if normalized.get_pixel(local_x, local_y).0[0] >= threshold
                || pixels.get_pixel(local_x, local_y).0[0] != 0
            {
                return Err(invalid());
            }
            pixels.put_pixel(local_x, local_y, Luma([1]));
            count += 1;
        }
    }
    if count != stroke.inside.pixels {
        return Err(invalid());
    }
    Ok(count)
}

/// Only the exact, already measured source foreground membership is
/// eligible to be masked. No clipping or dilation is performed.
/// Returns no mask if no foreground would remain or nothing would change.
///
/// # Errors
/// Rejects inconsistent span/count/geometry/polarity evidence.
pub fn build_mask<'a>(
    normalized: &GrayImage,
    threshold: u8,
    observation: InkBounds,
    strokes: impl IntoIterator<Item = &'a OrthogonalStroke>,
    polarity: &PolarityEvidence,
) -> Result<Option<Mask>, Error> {
    let (w, h) = normalized.dimensions();
    if observation.right.checked_sub(observation.left) != Some(w)
        || observation.bottom.checked_sub(observation.top) != Some(h)
    {
        return Err(invalid());
    }
    let [low, high] = polarity.class_medians.ok_or_else(invalid)?;
    let background_luminance = match polarity.foreground {
        Foreground::Dark => high,
        Foreground::Light => low,
        Foreground::Uniform => return Err(invalid()),
    };
    let mut pixels = GrayImage::new(w, h);
    let component_pixels = unique_strokes(strokes)?
        .into_iter()
        .map(|(id, stroke)| {
            Ok((
                id,
                mark(&mut pixels, normalized, threshold, observation, stroke)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, Error>>()?;
    let removed_pixels = component_pixels.values().sum::<u64>();
    let foreground_before = u64::try_from(
        normalized
            .as_raw()
            .iter()
            .filter(|&&v| v < threshold)
            .count(),
    )
    .map_err(|_| invalid())?;
    let foreground_after = foreground_before
        .checked_sub(removed_pixels)
        .ok_or_else(invalid)?;
    if removed_pixels == 0 || foreground_after == 0 {
        return Ok(None);
    }
    let evidence = MaskEvidence {
        component_pixels,
        mask_sha256: hex::encode(Sha256::digest(pixels.as_raw())),
        removed_pixels,
        foreground_before,
        foreground_after,
        background_luminance,
    };
    Ok(Some(Mask { pixels, evidence }))
}

impl Mask {
    /// Replace precisely the mask samples by the measured background class.
    /// The rest of the original RGB buffer is byte-identical.
    ///
    /// # Errors
    /// Rejects inconsistent dimensions, foreground membership or counts.
    pub fn apply(
        &self,
        rgb: &mut RgbImage,
        normalized: &mut GrayImage,
        threshold: u8,
    ) -> Result<(), Error> {
        if rgb.dimensions() != self.pixels.dimensions()
            || normalized.dimensions() != self.pixels.dimensions()
        {
            return Err(invalid());
        }
        for ((raw, gray), flag) in rgb
            .pixels_mut()
            .zip(normalized.pixels_mut())
            .zip(self.pixels.pixels())
        {
            if flag.0[0] == 1 {
                if gray.0[0] >= threshold {
                    return Err(invalid());
                }
                *raw = Rgb([self.evidence.background_luminance; 3]);
                *gray = Luma([255]);
            }
        }
        let after = u64::try_from(
            normalized
                .as_raw()
                .iter()
                .filter(|&&p| p < threshold)
                .count(),
        )
        .map_err(|_| invalid())?;
        if after != self.evidence.foreground_after {
            return Err(invalid());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        foreign_ink::audit_strokes,
        ink_polarity::{Polarity, estimate},
        pixel_regions::{PixelClass, regions},
    };

    #[test]
    fn mask_changes_only_proven_stripe_pixels_and_deduplicates_both_edges() -> Result<(), Error> {
        let observation = InkBounds {
            left: 10,
            top: 35,
            right: 60,
            bottom: 55,
        };
        let sample = GrayImage::from_fn(70, 90, |x, y| {
            let crossing = (31..33).contains(&x);
            let digit = (20..25).contains(&x) && (38..52).contains(&y);
            let period = (34..37).contains(&x) && (48..51).contains(&y);
            Luma([if crossing || digit || period { 0 } else { 255 }])
        });
        let components = regions(&sample, 128, PixelClass::Ink, observation)?;
        let mut checks = u64::MAX;
        let audit = audit_strokes(
            &sample,
            128,
            observation,
            &components.iter().collect::<Vec<_>>(),
            &mut checks,
        )?
        .ok_or_else(invalid)?;
        assert_eq!(audit.orthogonal.len(), 2);
        assert_eq!(
            audit.orthogonal[0].stroke.component_index,
            audit.orthogonal[1].stroke.component_index
        );
        let mut original = image::imageops::crop_imm(&sample, 10, 35, 50, 20).to_image();
        let polarity = estimate(&original, Polarity::Auto)?;
        let mut rgb = RgbImage::from_fn(50, 20, |x, y| Rgb([original.get_pixel(x, y).0[0]; 3]));
        // Include a non-gray background pixel to verify untouched RGB values.
        rgb.put_pixel(0, 0, Rgb([252, 251, 250]));
        let old_rgb = rgb.clone();
        let old_gray = original.clone();
        let mask = build_mask(&original, 128, observation, &audit.orthogonal, &polarity)?
            .ok_or_else(invalid)?;
        assert_eq!(mask.evidence.component_pixels.len(), 1);
        assert_eq!(mask.evidence.removed_pixels, 40);
        assert_eq!(mask.evidence.foreground_after, 79);
        mask.apply(&mut rgb, &mut original, 128)?;
        for (x, y, flag) in mask.pixels.enumerate_pixels() {
            if flag.0[0] == 1 {
                assert_eq!(original.get_pixel(x, y).0, [255]);
                assert_eq!(rgb.get_pixel(x, y).0, [255; 3]);
            } else {
                assert_eq!(original.get_pixel(x, y), old_gray.get_pixel(x, y));
                assert_eq!(rgb.get_pixel(x, y), old_rgb.get_pixel(x, y));
            }
        }
        Ok(())
    }
    #[test]
    fn entirely_foreign_observation_abstains() -> Result<(), Error> {
        let observation = InkBounds {
            left: 10,
            top: 35,
            right: 60,
            bottom: 55,
        };
        let sample = GrayImage::from_fn(70, 90, |x, _| {
            Luma([if (31..33).contains(&x) { 0 } else { 255 }])
        });
        let components = regions(&sample, 128, PixelClass::Ink, observation)?;
        let mut checks = u64::MAX;
        let audit = audit_strokes(
            &sample,
            128,
            observation,
            &components.iter().collect::<Vec<_>>(),
            &mut checks,
        )?
        .ok_or_else(invalid)?;
        assert_eq!(audit.orthogonal.len(), 2);
        let original = image::imageops::crop_imm(&sample, 10, 35, 50, 20).to_image();
        let polarity = estimate(&original, Polarity::Auto)?;
        assert!(build_mask(&original, 128, observation, &audit.orthogonal, &polarity)?.is_none());
        Ok(())
    }
}
