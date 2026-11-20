//! Source-raster glyph topology, independent of OCR model text probabilities.
//!
//! Reports connected *background holes* in a grayscale crop. This supplies
//! structural evidence but never assigns Unicode or rewrites recognized text.

use image::GrayImage;
use num_traits::ToPrimitive;
use serde::Serialize;

use crate::Error;

/// One enclosed 4-connected background region of a dark glyph.
#[derive(Clone, Debug, Serialize)]
pub struct EnclosedRegion {
    /// Number of background pixels within the region.
    pub area_pixels: usize,
    /// Geometric center in original crop pixel coordinates.
    pub centroid: [f64; 2],
    /// Inclusive pixel bounds in order `min_x`, `min_y`, `max_x`, `max_y`.
    pub bounds: [usize; 4],
}

/// Topology of a thresholded glyph crop, without any character interpretation.
#[derive(Clone, Debug, Serialize)]
pub struct GlyphTopology {
    pub width: u32,
    pub height: u32,
    /// Pixels below this luminance were classified as foreground ink.
    pub threshold: u8,
    pub foreground_pixels: usize,
    pub enclosed_background_regions: Vec<EnclosedRegion>,
}

/// Extract all enclosed light regions inside dark ink using 4-neighbor flood fill.
///
/// Edge-connected background is excluded, and individual holes are reported
/// even if too small to be meaningful. Callers can inspect their area. Input
/// should include a white margin: a crop that cuts a glyph at its border
/// cannot prove the existence or absence of a closed region.
///
/// # Errors
/// Rejects empty images or image dimensions not representable by the host.
pub fn analyze(image: &GrayImage, threshold: u8) -> Result<GlyphTopology, Error> {
    let width = usize::try_from(image.width())
        .map_err(|_| Error::Invalid("glyph width not representable".into()))?;
    let height = usize::try_from(image.height())
        .map_err(|_| Error::Invalid("glyph height not representable".into()))?;
    let total = width
        .checked_mul(height)
        .filter(|&size| size > 0)
        .ok_or_else(|| Error::Invalid("empty or unrepresentable glyph crop".into()))?;
    let pixels = image.as_raw();
    if pixels.len() != total {
        return Err(Error::Invalid(
            "glyph crop has inconsistent pixel count".into(),
        ));
    }

    let foreground_pixels = pixels.iter().filter(|&&value| value < threshold).count();
    let mut visited = vec![false; total];
    let mut regions = Vec::new();
    let mut stack = Vec::new();

    for seed in 0..total {
        if visited[seed] || pixels[seed] < threshold {
            continue;
        }
        visited[seed] = true;
        stack.push(seed);
        let mut area = 0_usize;
        let (mut sum_x, mut sum_y) = (0_f64, 0_f64);
        let (mut min_x, mut min_y) = (width, height);
        let (mut max_x, mut max_y) = (0, 0);
        let mut exterior = false;

        while let Some(index) = stack.pop() {
            let (x, y) = (index % width, index / width);
            area += 1;
            sum_x += x
                .to_f64()
                .ok_or_else(|| Error::Invalid("glyph X unrepresentable".into()))?;
            sum_y += y
                .to_f64()
                .ok_or_else(|| Error::Invalid("glyph Y unrepresentable".into()))?;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            exterior |= x == 0 || y == 0 || x + 1 == width || y + 1 == height;

            // Only 4-connected background regions count as enclosed;
            // diagonal touches do not create a path through a 1px ink wall.
            let neighbors = [
                (x > 0).then_some(index.saturating_sub(1)),
                (x + 1 < width).then_some(index + 1),
                (y > 0).then_some(index.saturating_sub(width)),
                (y + 1 < height).then_some(index + width),
            ];
            for adjacent in neighbors.into_iter().flatten() {
                if !visited[adjacent] && pixels[adjacent] >= threshold {
                    visited[adjacent] = true;
                    stack.push(adjacent);
                }
            }
        }

        if !exterior {
            let denominator = area
                .to_f64()
                .ok_or_else(|| Error::Invalid("glyph region area unrepresentable".into()))?;
            regions.push(EnclosedRegion {
                area_pixels: area,
                centroid: [sum_x / denominator, sum_y / denominator],
                bounds: [min_x, min_y, max_x, max_y],
            });
        }
    }

    regions.sort_by(|a, b| {
        a.centroid[1]
            .total_cmp(&b.centroid[1])
            .then_with(|| a.centroid[0].total_cmp(&b.centroid[0]))
    });
    Ok(GlyphTopology {
        width: image.width(),
        height: image.height(),
        threshold,
        foreground_pixels,
        enclosed_background_regions: regions,
    })
}

#[cfg(test)]
mod tests {
    use super::analyze;
    use image::{GrayImage, Luma};

    fn from_lines(rows: &[&str]) -> GrayImage {
        GrayImage::from_fn(
            u32::try_from(rows[0].len()).unwrap_or_default(),
            u32::try_from(rows.len()).unwrap_or_default(),
            |x, y| {
                if rows[usize::try_from(y).unwrap_or_default()].as_bytes()
                    [usize::try_from(x).unwrap_or_default()]
                    == b'#'
                {
                    Luma([0])
                } else {
                    Luma([255])
                }
            },
        )
    }

    #[test]
    fn arbitrary_enclosed_regions_exclude_edge_background() -> Result<(), crate::Error> {
        let image = from_lines(&[
            "...........",
            ".#########.",
            ".#...#...#.",
            ".#...#...#.",
            ".#########.",
            "...........",
        ]);
        let report = analyze(&image, 128)?;
        assert_eq!(report.enclosed_background_regions.len(), 2);
        assert_eq!(report.foreground_pixels, 24);
        assert_eq!(report.enclosed_background_regions[0].area_pixels, 6);
        assert_eq!(report.enclosed_background_regions[1].area_pixels, 6);
        assert!(analyze(&GrayImage::new(0, 0), 128).is_err());
        Ok(())
    }

    #[test]
    fn real_pdf_crop_holes_separate_diameter_zero_and_eight()
    -> Result<(), Box<dyn std::error::Error>> {
        let cases = [
            (
                include_bytes!("../tests/fixtures/glyph-topology/defond28-diameter-20.png")
                    .as_slice(),
                2,
                true,
            ),
            (
                include_bytes!("../tests/fixtures/glyph-topology/defond28-diameter-23.png")
                    .as_slice(),
                2,
                true,
            ),
            (
                include_bytes!("../tests/fixtures/glyph-topology/defond28-zero-20.png").as_slice(),
                1,
                false,
            ),
            (
                include_bytes!("../tests/fixtures/glyph-topology/defond28-zero-23.png").as_slice(),
                1,
                false,
            ),
            (
                include_bytes!("../tests/fixtures/glyph-topology/defond28-eight-18.png").as_slice(),
                2,
                false,
            ),
            (
                include_bytes!("../tests/fixtures/glyph-topology/defond28-zero-18.png").as_slice(),
                1,
                false,
            ),
        ];
        for (bytes, expected_holes, diagonal) in cases {
            let image = image::load_from_memory(bytes)?.to_luma8();
            for threshold in [64, 128, 192, 224] {
                let report = analyze(&image, threshold)?;
                // The original 600-DPI vector-derived print stays stable under
                // the entire tested luminance interval.
                let meaningful = report
                    .enclosed_background_regions
                    .iter()
                    .filter(|hole| hole.area_pixels >= 4)
                    .collect::<Vec<_>>();
                assert_eq!(meaningful.len(), expected_holes);
                if meaningful.len() == 2 {
                    let dx = (meaningful[0].centroid[0] - meaningful[1].centroid[0]).abs();
                    let dy = (meaningful[0].centroid[1] - meaningful[1].centroid[1]).abs();
                    assert!(dy > 4.0);
                    if diagonal {
                        assert!(dx > 4.0, "diameter enclosed regions are offset diagonally");
                    } else {
                        assert!(dx < 2.0, "digit eight has vertically stacked holes");
                    }
                }
            }
        }
        Ok(())
    }
}
