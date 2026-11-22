//! Source-raster glyph topology, independent of OCR model text probabilities.
//!
//! Reports connected *background holes* in a grayscale crop. This supplies
//! structural evidence but never assigns Unicode or rewrites recognized text.

use image::GrayImage;
use serde::{Deserialize, Serialize};

use crate::{
    Error,
    pixel_regions::{InkBounds, PixelClass, regions},
};

/// One enclosed 4-connected background region of a dark glyph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnclosedRegion {
    /// Number of background pixels within the region.
    pub area_pixels: usize,
    /// Geometric center in original crop pixel coordinates.
    pub centroid: [f64; 2],
    /// Inclusive pixel bounds in order `min_x`, `min_y`, `max_x`, `max_y`.
    pub bounds: [u32; 4],
}

/// Topology of a thresholded glyph crop, without any character interpretation.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    let bounds = InkBounds {
        left: 0,
        top: 0,
        right: image.width(),
        bottom: image.height(),
    };
    let foreground_pixels = image
        .as_raw()
        .iter()
        .filter(|&&value| value < threshold)
        .count();
    let mut regions = regions(image, threshold, PixelClass::Background, bounds)?
        .into_iter()
        .filter(|region| !region.touches_line_edge)
        .map(|region| {
            let b = region.bounds;
            EnclosedRegion {
                area_pixels: region.area_pixels,
                centroid: region.centroid,
                bounds: [b.left, b.top, b.right - 1, b.bottom - 1],
            }
        })
        .collect::<Vec<_>>();

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
