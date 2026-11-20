//! Geometry-only isolation of visible ink along an OCR line's supplied quad.
//!
//! A segment is a run of occupied columns, not an inferred character. Touching
//! glyphs remain joined; detached marks sharing columns stay together.

use image::{GrayImage, imageops};
use serde::Serialize;

use crate::{
    Error,
    glyph_topology::{GlyphTopology, analyze},
};

/// Half-open pixel bounds in the oriented line crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct InkBounds {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

/// One isolated ink group, with diagnostic topology at each requested threshold.
#[derive(Debug, Serialize)]
pub struct InkGroup {
    /// Ink bounds at the segmentation threshold.
    pub ink: InkBounds,
    /// Actual sampled window, including available neighboring background.
    pub crop: InkBounds,
    /// The ink reaches the supplied line image boundary.
    pub touches_line_edge: bool,
    pub topology: Vec<GlyphTopology>,
}

fn column_runs(image: &GrayImage, threshold: u8) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    let mut start = None;
    for x in 0..image.width() {
        let occupied = (0..image.height()).any(|y| image.get_pixel(x, y)[0] < threshold);
        match (start, occupied) {
            (None, true) => start = Some(x),
            (Some(left), false) => {
                ranges.push((left, x));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(left) = start {
        ranges.push((left, image.width()));
    }
    ranges
}

fn group_bounds(image: &GrayImage, left: u32, right: u32, threshold: u8) -> InkBounds {
    let mut top = image.height();
    let mut bottom = 0;
    for y in 0..image.height() {
        if (left..right).any(|x| image.get_pixel(x, y)[0] < threshold) {
            top = top.min(y);
            bottom = y + 1;
        }
    }
    InkBounds {
        left,
        top,
        right,
        bottom,
    }
}

/// Separate runs of ink only at observed blank columns.
///
/// Sample topology inside each run's actual source window, with up to one
/// pixel of real surrounding margin. No artificial outline or closing is
/// applied, and neither recognition text nor a character count is required.
///
/// # Errors
/// Rejects empty images, degenerate thresholds, or topology extraction failures.
pub fn isolate_ink(
    image: &GrayImage,
    threshold: u8,
    topology_thresholds: &[u8],
) -> Result<Vec<InkGroup>, Error> {
    if image.width() == 0
        || image.height() == 0
        || threshold == 0
        || topology_thresholds.is_empty()
        || topology_thresholds.contains(&0)
    {
        return Err(Error::Invalid("invalid glyph isolation inputs".into()));
    }
    let mut result = Vec::new();
    for (left, right) in column_runs(image, threshold) {
        let ink = group_bounds(image, left, right, threshold);
        let crop = InkBounds {
            left: left.saturating_sub(1),
            right: right.saturating_add(1).min(image.width()),
            top: ink.top.saturating_sub(1),
            bottom: ink.bottom.saturating_add(1).min(image.height()),
        };
        let pixels = imageops::crop_imm(
            image,
            crop.left,
            crop.top,
            crop.right - crop.left,
            crop.bottom - crop.top,
        )
        .to_image();
        let topology = topology_thresholds
            .iter()
            .map(|&cutoff| analyze(&pixels, cutoff))
            .collect::<Result<Vec<_>, _>>()?;
        result.push(InkGroup {
            ink,
            crop,
            touches_line_edge: ink.left == 0
                || ink.top == 0
                || ink.right == image.width()
                || ink.bottom == image.height(),
            topology,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{InkBounds, isolate_ink};
    use image::{GrayImage, Luma};

    #[test]
    fn touching_characters_stay_joined_and_detached_marks_share_columns() -> Result<(), crate::Error>
    {
        let mut image = GrayImage::from_pixel(15, 9, Luma([255]));
        // Two joined rectangular rings; a separate dotted stroke.
        for y in 2..7 {
            for x in 1..8 {
                if y == 2 || y == 6 || matches!(x, 1 | 4 | 7) {
                    image.put_pixel(x, y, Luma([0]));
                }
            }
        }
        image.put_pixel(11, 0, Luma([0]));
        for y in 3..7 {
            image.put_pixel(11, y, Luma([0]));
        }
        let groups = isolate_ink(&image, 128, &[64, 128, 192])?;
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups[0].ink,
            InkBounds {
                left: 1,
                top: 2,
                right: 8,
                bottom: 7
            }
        );
        assert!(!groups[0].touches_line_edge);
        for topology in &groups[0].topology {
            assert_eq!(topology.enclosed_background_regions.len(), 2);
        }
        assert_eq!(groups[1].ink.top, 0);
        assert!(groups[1].touches_line_edge);
        Ok(())
    }

    #[test]
    fn empty_line_has_no_inferred_characters() -> Result<(), crate::Error> {
        let image = GrayImage::from_pixel(12, 7, Luma([255]));
        assert!(isolate_ink(&image, 128, &[128])?.is_empty());
        assert!(isolate_ink(&image, 0, &[128]).is_err());
        assert!(isolate_ink(&image, 128, &[]).is_err());
        Ok(())
    }
}
