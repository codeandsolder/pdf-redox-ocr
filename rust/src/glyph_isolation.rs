//! Geometry-only isolation of visible ink along an OCR line.

use image::{GrayImage, Pixel, RgbImage, imageops};
use serde::{Deserialize, Serialize};

use crate::{
    Error,
    glyph_topology::{GlyphTopology, analyze},
    ink_polarity::{Polarity, PolarityEvidence, estimate},
    pixel_regions::{InkBounds, ObservationSupport, PixelClass, PixelRegion, regions},
};

/// A projection group and its exact constituent connected components.
#[derive(Debug, Serialize, Deserialize)]
pub struct InkRun {
    pub ink: InkBounds,
    /// Indices into the line's connected-component array.
    pub component_indices: Vec<usize>,
    pub observation_support: ObservationSupport,
}

/// One isolated ink group, with diagnostic topology at each requested threshold.
#[derive(Debug, Serialize, Deserialize)]
pub struct InkGroup {
    #[serde(flatten)]
    pub run: InkRun,
    /// Actual sampled window, including available neighboring background.
    pub crop: InkBounds,
    pub touches_line_edge: bool,
    pub topology: Vec<GlyphTopology>,
}

/// A row-projection band with its independent column projection.
#[derive(Debug, Serialize)]
pub struct InkBand {
    #[serde(flatten)]
    pub run: InkRun,
    pub column_runs: Vec<InkRun>,
}

/// Independent components and complementary projections of the same source evidence.
#[derive(Debug)]
pub struct InkIsolation {
    pub components: Vec<PixelRegion>,
    pub groups: Vec<InkGroup>,
    /// Projection of only components with pixels inside the observation.
    /// Entire connected boundary continuations remain included.
    pub supported_runs: Vec<InkRun>,
    /// Orthogonal projection of source-supported components. Detached marks
    /// may occupy their own band; neither bands nor runs imply characters.
    pub supported_bands: Vec<InkBand>,
}

/// Ordinary recognition-crop ink measurement shared by greedy and candidate
/// geometry. Shape extraction can remain lazy for individual associated runs.
pub(crate) struct MeasuredInk {
    pub normalized: GrayImage,
    pub polarity: PolarityEvidence,
    pub isolation: InkIsolation,
}

impl MeasuredInk {
    pub(crate) fn from_rgb(image: &RgbImage) -> Result<Self, Error> {
        let dimensions = image.dimensions();
        let gray = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
            image.get_pixel(x, y).to_luma()
        });
        let polarity = estimate(&gray, Polarity::Auto)?;
        let normalized = polarity.normalize(&gray);
        let isolation = isolate_ink(
            &normalized,
            InkBounds {
                left: 0,
                top: 0,
                right: dimensions.0,
                bottom: dimensions.1,
            },
            192,
            &[128, 192, 224],
        )?;
        Ok(Self {
            normalized,
            polarity,
            isolation,
        })
    }
}

#[derive(Clone, Copy)]
enum Axis {
    Columns,
    Rows,
}

impl Axis {
    const fn interval(self, bounds: InkBounds) -> (u32, u32) {
        match self {
            Self::Columns => (bounds.left, bounds.right),
            Self::Rows => (bounds.top, bounds.bottom),
        }
    }
}

fn projection_runs(
    components: &[PixelRegion],
    indices: impl IntoIterator<Item = usize>,
    axis: Axis,
) -> Vec<InkRun> {
    let mut indices = indices.into_iter().collect::<Vec<_>>();
    indices.sort_unstable_by_key(|&index| {
        let bounds = components[index].bounds;
        let other = match axis {
            Axis::Columns => Axis::Rows,
            Axis::Rows => Axis::Columns,
        };
        let (start, end) = axis.interval(bounds);
        let (other_start, other_end) = other.interval(bounds);
        (start, other_start, end, other_end, index)
    });
    let mut runs: Vec<InkRun> = Vec::new();
    for index in indices {
        let component = &components[index];
        if let Some(run) = runs
            .last_mut()
            .filter(|run| axis.interval(component.bounds).0 <= axis.interval(run.ink).1)
        {
            run.ink = run.ink.union(component.bounds);
            run.component_indices.push(index);
            run.observation_support.add(component.observation_support);
        } else {
            runs.push(InkRun {
                ink: component.bounds,
                component_indices: vec![index],
                observation_support: component.observation_support,
            });
        }
    }
    runs
}

fn measure_run(image: &GrayImage, run: InkRun, thresholds: &[u8]) -> Result<InkGroup, Error> {
    let ink = run.ink;
    let crop = InkBounds {
        left: ink.left.saturating_sub(1),
        right: ink.right.saturating_add(1).min(image.width()),
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
    let topology = thresholds
        .iter()
        .map(|&cutoff| analyze(&pixels, cutoff))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InkGroup {
        run,
        crop,
        touches_line_edge: ink.touches_edge(image.width(), image.height()),
        topology,
    })
}

/// Isolate eight-connected ink and derive exact occupied-column projections.
///
/// No OCR text or expected character count is used. Diagonal neighbors are
/// connected, detached strokes with overlapping column ranges may share a run,
/// and touching glyphs are not split. The full projection retains all sampled
/// source ink; a second projection excludes only whole components that have
/// no pixels in the original observation. Topology is measured on unmodified
/// source crops, not on a synthetically cleaned mask.
///
/// # Errors
/// Rejects invalid crop/observation bounds or degenerate thresholds.
pub fn isolate_ink(
    image: &GrayImage,
    observation: InkBounds,
    threshold: u8,
    topology_thresholds: &[u8],
) -> Result<InkIsolation, Error> {
    if threshold == 0 || topology_thresholds.is_empty() || topology_thresholds.contains(&0) {
        return Err(Error::Invalid("invalid glyph isolation thresholds".into()));
    }
    let components = regions(image, threshold, PixelClass::Ink, observation)?;
    let groups = projection_runs(&components, 0..components.len(), Axis::Columns)
        .into_iter()
        .map(|run| measure_run(image, run, topology_thresholds))
        .collect::<Result<Vec<_>, _>>()?;
    let supported = || {
        (0..components.len())
            .filter(|&index| components[index].observation_support.inside_pixels > 0)
    };
    let supported_runs = projection_runs(&components, supported(), Axis::Columns);
    let supported_bands = projection_runs(&components, supported(), Axis::Rows)
        .into_iter()
        .map(|run| InkBand {
            column_runs: projection_runs(
                &components,
                run.component_indices.iter().copied(),
                Axis::Columns,
            ),
            run,
        })
        .collect();
    Ok(InkIsolation {
        components,
        groups,
        supported_runs,
        supported_bands,
    })
}

#[cfg(test)]
mod tests {
    use super::isolate_ink;
    use crate::pixel_regions::InkBounds;
    use image::{GrayImage, Luma};

    fn full(image: &GrayImage) -> InkBounds {
        InkBounds {
            left: 0,
            top: 0,
            right: image.width(),
            bottom: image.height(),
        }
    }

    #[test]
    fn touching_characters_stay_joined_and_detached_marks_share_columns() -> Result<(), crate::Error>
    {
        let mut image = GrayImage::from_pixel(15, 9, Luma([255]));
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
        let audit = isolate_ink(&image, full(&image), 128, &[64, 128, 192])?;
        assert_eq!(audit.components.len(), 3);
        assert_eq!(audit.groups.len(), 2);
        assert_eq!(
            audit.groups[0].run.ink,
            InkBounds {
                left: 1,
                top: 2,
                right: 8,
                bottom: 7
            }
        );
        assert!(!audit.groups[0].touches_line_edge);
        for topology in &audit.groups[0].topology {
            assert_eq!(topology.enclosed_background_regions.len(), 2);
        }
        assert_eq!(audit.groups[1].run.component_indices.len(), 2);
        assert!(audit.groups[1].touches_line_edge);
        assert_eq!(audit.supported_runs.len(), 2);
        Ok(())
    }

    #[test]
    fn outside_only_rule_can_merge_projection_but_not_supported_components()
    -> Result<(), crate::Error> {
        let mut image = GrayImage::from_pixel(17, 10, Luma([255]));
        let original = InkBounds {
            left: 2,
            top: 3,
            right: 15,
            bottom: 9,
        };
        for x in 0..17 {
            image.put_pixel(x, 0, Luma([0]));
        }
        for y in 4..8 {
            image.put_pixel(4, y, Luma([0]));
            image.put_pixel(12, y, Luma([0]));
        }
        // A disconnected diagonal pair inside the observation.
        image.put_pixel(7, 5, Luma([0]));
        image.put_pixel(8, 6, Luma([0]));
        let audit = isolate_ink(&image, original, 128, &[128])?;
        assert_eq!(audit.components.len(), 4);
        assert_eq!(audit.groups.len(), 1);
        assert_eq!(audit.supported_runs.len(), 3);
        assert_eq!(audit.groups[0].run.observation_support.inside_pixels, 10);
        assert_eq!(audit.groups[0].run.observation_support.outside_pixels, 17);
        assert_eq!(audit.components[0].observation_support.inside_pixels, 0);
        assert!(
            audit
                .supported_runs
                .iter()
                .all(|run| run.observation_support.outside_pixels == 0)
        );
        assert!(audit.components.iter().any(|c| c.area_pixels == 2));
        Ok(())
    }

    #[test]
    fn component_projection_equals_pixel_projection_for_all_small_bitmaps()
    -> Result<(), Box<dyn std::error::Error>> {
        use super::{Axis, projection_runs};
        use crate::pixel_regions::{PixelClass, regions};
        for bits in 0_u16..4096 {
            let image = GrayImage::from_fn(4, 3, |x, y| {
                Luma([if bits & (1 << (y * 4 + x)) != 0 {
                    0
                } else {
                    255
                }])
            });
            let components = regions(&image, 128, PixelClass::Ink, full(&image))?;
            let runs = projection_runs(&components, 0..components.len(), Axis::Columns);
            let mut actual = [false; 4];
            for run in &runs {
                for x in run.ink.left..run.ink.right {
                    actual[usize::try_from(x)?] = true;
                }
            }
            for (x, occupied) in actual.into_iter().enumerate() {
                assert_eq!(occupied, (0..3).any(|y| bits & (1 << (y * 4 + x)) != 0));
            }
            assert!(
                runs.windows(2)
                    .all(|pair| pair[0].ink.right < pair[1].ink.left)
            );
            let bands = projection_runs(&components, 0..components.len(), Axis::Rows);
            let mut actual_rows = [false; 3];
            let mut members = Vec::new();
            for band in &bands {
                for y in band.ink.top..band.ink.bottom {
                    actual_rows[usize::try_from(y)?] = true;
                }
                let columns = projection_runs(
                    &components,
                    band.component_indices.iter().copied(),
                    Axis::Columns,
                );
                members.extend(columns.into_iter().flat_map(|run| run.component_indices));
            }
            members.sort_unstable();
            assert_eq!(members, (0..components.len()).collect::<Vec<_>>());
            for (y, occupied) in actual_rows.into_iter().enumerate() {
                assert_eq!(occupied, (0..4).any(|x| bits & (1 << (y * 4 + x)) != 0));
            }
            assert_eq!(
                components
                    .iter()
                    .map(|component| component.area_pixels)
                    .sum::<usize>(),
                usize::try_from(bits.count_ones())?
            );
        }
        Ok(())
    }

    #[test]
    fn orthogonal_bands_preserve_in_observation_rules_and_detached_marks()
    -> Result<(), crate::Error> {
        let mut image = GrayImage::from_pixel(17, 12, Luma([255]));
        for x in 0..17 {
            image.put_pixel(x, 10, Luma([0]));
        }
        image.put_pixel(4, 0, Luma([0]));
        for y in 3..8 {
            image.put_pixel(4, y, Luma([0]));
            image.put_pixel(12, y, Luma([0]));
        }
        let audit = isolate_ink(&image, full(&image), 128, &[128])?;
        assert_eq!(audit.supported_runs.len(), 1);
        assert_eq!(audit.supported_bands.len(), 3);
        let bands = &audit.supported_bands;
        assert_eq!(bands[0].run.component_indices.len(), 1);
        assert_eq!(bands[1].column_runs.len(), 2);
        assert_eq!(bands[2].run.observation_support.inside_pixels, 17);
        assert_eq!(bands[2].column_runs.len(), 1);
        Ok(())
    }

    #[test]
    fn empty_line_has_no_inferred_characters() -> Result<(), crate::Error> {
        let image = GrayImage::from_pixel(12, 7, Luma([255]));
        let bounds = full(&image);
        assert!(isolate_ink(&image, bounds, 128, &[128])?.groups.is_empty());
        assert!(isolate_ink(&image, bounds, 0, &[128]).is_err());
        assert!(isolate_ink(&image, bounds, 128, &[]).is_err());
        Ok(())
    }
}
