//! CTC emission columns and independent glyph geometry in the *actual*
//! rectified RGB recognition crop.
//!
//! CTC peaks are localization hints, not
//! character bounding boxes, confidence estimates or proof of a Unicode label.

use crate::{
    Error,
    crop_geometry::CropGeometry,
    crop_repair::BodyPrediction,
    disjoint_set::DisjointSet,
    glyph_confirmation::{ConfirmationDecision, ConfirmationPolicy},
    glyph_isolation::{InkGroup, isolate_ink},
    ink_polarity::{Polarity, PolarityEvidence, estimate},
    pixel_regions::{InkBounds, PixelRegion},
    recognition::PixelLine,
};
use image::{Pixel, RgbImage};
use num_traits::ToPrimitive;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Explicit recognition preprocessing shared with the model adapter.
pub const INPUT_SHAPE: [usize; 3] = [3, 48, 320];
pub const MAX_INPUT_WIDTH: usize = 3200;

/// Actual padded-width contract of the pinned CRNN adapter. Both batch
/// maximum and min base width participate; the individual crop is ceil-resized.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct BatchGeometry {
    pub tensor_width: usize,
}

impl BatchGeometry {
    /// # Errors
    /// Rejects empty/unrepresentable crop dimensions.
    pub fn new(dimensions: impl IntoIterator<Item = (u32, u32)>) -> Result<Self, Error> {
        let [_, height, width] = INPUT_SHAPE;
        let mut max_ratio = ratio(width, height)?;
        for (w, h) in dimensions {
            max_ratio = max_ratio.max(ratio(w, h)?);
        }
        let tensor_width = (height.to_f32().ok_or_else(invalid_geometry)? * max_ratio)
            .to_usize()
            .ok_or_else(invalid_geometry)?
            .min(MAX_INPUT_WIDTH);
        Ok(Self { tensor_width })
    }

    fn resized_width(self, (w, h): (u32, u32)) -> Result<usize, Error> {
        Ok(
            (INPUT_SHAPE[1].to_f32().ok_or_else(invalid_geometry)? * ratio(w, h)?)
                .ceil()
                .to_usize()
                .ok_or_else(invalid_geometry)?
                .min(self.tensor_width),
        )
    }
}

fn ratio<T: ToPrimitive + Copy>(w: T, h: T) -> Result<f32, Error> {
    let w = w.to_f32().ok_or_else(invalid_geometry)?;
    let h = h.to_f32().ok_or_else(invalid_geometry)?;
    if w <= 0. || h <= 0. {
        return Err(invalid_geometry());
    }
    Ok(w / h)
}
fn invalid_geometry() -> Error {
    Error::Invalid("invalid CTC crop geometry".into())
}

/// One observed CTC token. Its signed pixel coordinate is intentionally
/// unclamped; an emission in right padding is not a source glyph.
#[derive(Debug, Serialize)]
pub struct CharacterAnchor {
    pub character: char,
    pub column: usize,
    pub crop_x: f64,
    pub candidate_runs: Vec<usize>,
    pub alignment: RunAlignment,
}

/// Ordered matching describes geometric compatibility, not character identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RunAlignment {
    Whitespace,
    Padding,
    NoInk,
    /// More characters share the ink than can be separately identified.
    NoInjectiveAssignment,
    /// At least two ordered, injective assignments are possible.
    Ambiguous {
        first: usize,
        last: usize,
    },
    /// Same run in every feasible ordered, injective assignment in the component.
    Unique {
        run: usize,
    },
}

/// Exact crop-frame evidence.
///
/// The line quad is in parent RGB raster space;
/// all run coordinates are in the rectified recognizer crop, not that quad
/// or the original PDF. No invented affine inverse for perspective crops.
#[derive(Debug, Serialize)]
pub struct CharacterLineAudit {
    pub detection_index: usize,
    pub original: PixelLine,
    pub crop_sha256: String,
    pub crop_dimensions: (u32, u32),
    pub geometry: Option<CropGeometry>,
    pub tensor_width: usize,
    pub resized_width: usize,
    pub sequence_length: usize,
    pub polarity: PolarityEvidence,
    pub components: Vec<PixelRegion>,
    pub runs: Vec<InkGroup>,
    pub anchors: Vec<CharacterAnchor>,
}

#[derive(Debug, Serialize)]
pub struct ConfirmedPrediction {
    pub prediction: BodyPrediction,
    pub tensor_width: usize,
    pub resized_width: usize,
    pub sequence_length: usize,
    pub anchors: Vec<CharacterAnchor>,
    pub decision: ConfirmationDecision,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ConfirmationOutcome {
    Predicted { result: ConfirmedPrediction },
    InferenceFailed { error: String },
}

impl ConfirmationOutcome {
    #[must_use]
    pub const fn accepted(&self) -> bool {
        matches!(self,Self::Predicted{result} if result.decision.accepted())
    }
    #[must_use]
    pub const fn failed(&self) -> bool {
        matches!(self, Self::InferenceFailed { .. })
    }
}

#[derive(Debug, Serialize)]
pub struct CharacterConfirmation {
    pub detection_index: usize,
    pub original: PixelLine,
    pub crop_sha256: String,
    pub outcome: ConfirmationOutcome,
}

#[derive(Debug, Default, Serialize)]
pub struct CharacterAuditReport {
    pub lines: Vec<CharacterLineAudit>,
    pub skipped_budget: usize,
    pub invalid_metadata: usize,
    pub sampled_pixels: u64,
    pub confirmation_batches: usize,
    pub confirmations: Vec<CharacterConfirmation>,
}

const MAX_SAMPLE_PIXELS: u64 = 16_000_000;

/// Match each CTC emission to ink groups with a local reading-axis cell.
/// Cells are divided by neighboring nonspace peaks. This gives evidence
/// for ambiguous joins/splits instead of manufacturing a one-to-one mapping.
fn anchor_candidates(cells: &[(char, usize, f64)], runs: &[InkGroup]) -> Vec<CharacterAnchor> {
    cells
        .iter()
        .enumerate()
        .map(|(i, &(character, column, x))| {
            let left = cells[..i]
                .iter()
                .rev()
                .find(|(c, _, _)| !c.is_whitespace())
                .map_or(f64::NEG_INFINITY, |(_, _, p)| x.midpoint(*p));
            let right = cells[i + 1..]
                .iter()
                .find(|(c, _, _)| !c.is_whitespace())
                .map_or(f64::INFINITY, |(_, _, p)| x.midpoint(*p));
            let candidate_runs = if character.is_whitespace() {
                vec![]
            } else {
                runs.iter()
                    .enumerate()
                    .filter_map(|(j, run)| {
                        let b = run.run.ink;
                        (f64::from(b.left) < right && f64::from(b.right) > left).then_some(j)
                    })
                    .collect()
            };
            CharacterAnchor {
                character,
                column,
                crop_x: x,
                candidate_runs,
                alignment: RunAlignment::NoInk,
            }
        })
        .collect()
}

// Greedy earliest/latest assignments give pointwise extrema for strictly
// increasing ordered matching. Skipped ink runs are permitted. If the
// earliest scan fails, no assignment can exist: advancing a previous match
// can only reduce options for later tokens. Reverse scan is its dual.
fn extreme_assignment(
    anchors: &[CharacterAnchor],
    indices: &[usize],
    reverse: bool,
) -> Option<Vec<usize>> {
    let mut previous = None;
    let mut result = Vec::with_capacity(indices.len());
    for step in 0..indices.len() {
        let index = if reverse {
            indices[indices.len() - 1 - step]
        } else {
            indices[step]
        };
        let runs = &anchors[index].candidate_runs;
        let next = if reverse {
            runs.iter()
                .rev()
                .copied()
                .find(|&r| previous.is_none_or(|p| r < p))
        } else {
            runs.iter()
                .copied()
                .find(|&r| previous.is_none_or(|p| r > p))
        }?;
        previous = Some(next);
        result.push(next);
    }
    if reverse {
        result.reverse();
    }
    Some(result)
}

/// Group mutually competing tokens by shared ink runs. An unresolvable
/// touching-glyph cluster cannot poison unrelated glyph associations in the
/// same line. Ordered extrema solve each independent candidate component.
fn align_runs(anchors: &mut [CharacterAnchor], run_count: usize, crop_width: u32) {
    let mut components = DisjointSet::new(anchors.len());
    let mut owner = vec![None; run_count];
    for (index, anchor) in anchors.iter_mut().enumerate() {
        if anchor.character.is_whitespace() {
            anchor.alignment = RunAlignment::Whitespace;
            continue;
        }
        if !(0.0..f64::from(crop_width)).contains(&anchor.crop_x) {
            anchor.candidate_runs.clear();
            anchor.alignment = RunAlignment::Padding;
            continue;
        }
        if anchor.candidate_runs.is_empty() {
            anchor.alignment = RunAlignment::NoInk;
            continue;
        }
        anchor.alignment = RunAlignment::NoInjectiveAssignment;
        for &run in &anchor.candidate_runs {
            if let Some(first) = owner[run] {
                components.connect(index, first);
            } else {
                owner[run] = Some(index);
            }
        }
    }
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, anchor) in anchors.iter().enumerate() {
        if !anchor.candidate_runs.is_empty() {
            groups
                .entry(components.root(index))
                .or_default()
                .push(index);
        }
    }
    for indices in groups.values() {
        let Some(early) = extreme_assignment(anchors, indices, false) else {
            continue;
        };
        let Some(late) = extreme_assignment(anchors, indices, true) else {
            continue;
        };
        for (&index, (first, last)) in indices.iter().zip(early.into_iter().zip(late)) {
            anchors[index].alignment = if first == last {
                RunAlignment::Unique { run: first }
            } else {
                RunAlignment::Ambiguous { first, last }
            };
        }
    }
}

fn prediction_anchors(
    text: &str,
    dimensions: (u32, u32),
    batch: BatchGeometry,
    columns: &[usize],
    sequence_length: usize,
    runs: &[InkGroup],
) -> Result<Vec<CharacterAnchor>, Error> {
    let resized_width = batch.resized_width(dimensions)?;
    let characters = text.chars().collect::<Vec<_>>();
    if sequence_length == 0
        || columns.len() != characters.len()
        || columns.iter().any(|&column| column >= sequence_length)
        || columns.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(Error::Invalid("inconsistent CTC emission columns".into()));
    }
    let scale = batch.tensor_width.to_f64().ok_or_else(invalid_geometry)?
        / sequence_length.to_f64().ok_or_else(invalid_geometry)?
        * f64::from(dimensions.0)
        / resized_width.to_f64().ok_or_else(invalid_geometry)?;
    let cells = characters
        .into_iter()
        .zip(columns.iter().copied())
        .map(|(character, column)| {
            Ok((
                character,
                column,
                (column.to_f64().ok_or_else(invalid_geometry)? + 0.5) * scale,
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut anchors = anchor_candidates(&cells, runs);
    align_runs(&mut anchors, runs.len(), dimensions.0);
    Ok(anchors)
}

impl CharacterLineAudit {
    /// Project a second actual prediction on the very same source ink.
    /// No image resampling or new segmentation is required.
    ///
    /// # Errors
    /// Returns invalid CTC metadata errors.
    pub fn confirmation(
        &self,
        prediction: BodyPrediction,
        batch: BatchGeometry,
        columns: &[usize],
        length: usize,
        policy: ConfirmationPolicy,
    ) -> Result<ConfirmedPrediction, Error> {
        let anchors = prediction_anchors(
            &prediction.text,
            self.crop_dimensions,
            batch,
            columns,
            length,
            &self.runs,
        )?;
        let decision = crate::glyph_confirmation::decide(
            self.original.confidence,
            prediction.confidence,
            policy,
            &self.anchors,
            &anchors,
            self.runs.len(),
        );
        Ok(ConfirmedPrediction {
            prediction,
            tensor_width: batch.tensor_width,
            resized_width: batch.resized_width(self.crop_dimensions)?,
            sequence_length: length,
            anchors,
            decision,
        })
    }

    fn measure(
        original: PixelLine,
        crop: &RgbImage,
        batch: BatchGeometry,
        columns: &[usize],
        sequence_length: usize,
    ) -> Result<Self, Error> {
        let dimensions = crop.dimensions();
        let resized_width = batch.resized_width(dimensions)?;
        let gray = image::GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
            crop.get_pixel(x, y).to_luma()
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
        let anchors = prediction_anchors(
            &original.text,
            dimensions,
            batch,
            columns,
            sequence_length,
            &isolation.groups,
        )?;
        Ok(Self {
            detection_index: 0,
            original,
            crop_sha256: hex::encode(Sha256::digest(crop.as_raw())),
            crop_dimensions: dimensions,
            geometry: None,
            tensor_width: batch.tensor_width,
            resized_width,
            sequence_length,
            polarity,
            components: isolation.components,
            runs: isolation.groups,
            anchors,
        })
    }
}

impl CharacterAuditReport {
    /// Audit each crop at most once, after actual primary recognition.
    /// The immutable observation and exact crop are retained as metadata only.
    pub fn observe(
        &mut self,
        detection_index: usize,
        original: &PixelLine,
        crop: &RgbImage,
        batch: BatchGeometry,
        columns: &[usize],
        sequence_length: usize,
    ) -> Option<&mut CharacterLineAudit> {
        let pixels = u64::from(crop.width()) * u64::from(crop.height());
        if pixels > MAX_SAMPLE_PIXELS - self.sampled_pixels {
            self.skipped_budget += 1;
            return None;
        }
        self.sampled_pixels += pixels;
        if let Ok(mut entry) =
            CharacterLineAudit::measure(original.clone(), crop, batch, columns, sequence_length)
        {
            entry.detection_index = detection_index;
            self.lines.push(entry);
            self.lines.last_mut()
        } else {
            self.invalid_metadata += 1;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchGeometry, CharacterAnchor, CharacterLineAudit, RunAlignment, align_runs};
    use crate::{Error, geometry::Point, recognition::PixelLine};
    use image::{Rgb, RgbImage};

    #[test]
    fn padded_batch_positions_are_corrected_before_associating_ink() -> Result<(), Error> {
        let image = RgbImage::from_fn(100, 20, |x, y| {
            if (10..25).contains(&x) && (4..17).contains(&y)
                || (70..85).contains(&x) && (4..17).contains(&y)
            {
                Rgb([0; 3])
            } else {
                Rgb([255; 3])
            }
        });
        let observation = PixelLine {
            text: "A B".into(),
            confidence: 0.99,
            quad: [
                Point(0., 20.),
                Point(100., 20.),
                Point(100., 0.),
                Point(0., 0.),
            ],
        };
        let batch = BatchGeometry::new([(100, 20), (400, 20)])?;
        assert_eq!(batch.tensor_width, 960);
        let audit =
            CharacterLineAudit::measure(observation.clone(), &image, batch, &[10, 30, 46], 240)?;
        assert_eq!(audit.resized_width, 240);
        assert_eq!(audit.runs.len(), 2);
        assert_eq!(audit.anchors[0].candidate_runs, vec![0]);
        assert_eq!(audit.anchors[1].candidate_runs.len(), 0);
        assert_eq!(audit.anchors[2].candidate_runs, vec![1]);
        assert!((audit.anchors[2].crop_x - 77.5).abs() < 1e-9);
        assert!(
            CharacterLineAudit::measure(observation, &image, batch, &[10, 10, 46], 240).is_err()
        );
        Ok(())
    }
    fn anchor(character: char, x: f64, choices: &[usize]) -> CharacterAnchor {
        CharacterAnchor {
            character,
            column: 0,
            crop_x: x,
            candidate_runs: choices.to_vec(),
            alignment: RunAlignment::NoInk,
        }
    }

    #[test]
    fn independent_competing_groups_do_not_poison_each_other() {
        let mut anchors = vec![
            anchor('a', 10., &[0, 1]),
            anchor('b', 20., &[1, 2]),
            anchor('c', 30., &[2]),
            anchor('d', 40., &[3]),
            anchor('e', 45., &[3]),
            anchor(' ', 50., &[]),
            anchor('f', 60., &[4, 5]),
            anchor('g', 70., &[6]),
            anchor('h', 80., &[]),
            anchor('i', 101., &[7]),
        ];
        align_runs(&mut anchors, 8, 100);
        assert_eq!(
            anchors.iter().map(|a| a.alignment).collect::<Vec<_>>(),
            [
                RunAlignment::Unique { run: 0 },
                RunAlignment::Unique { run: 1 },
                RunAlignment::Unique { run: 2 },
                RunAlignment::NoInjectiveAssignment,
                RunAlignment::NoInjectiveAssignment,
                RunAlignment::Whitespace,
                RunAlignment::Ambiguous { first: 4, last: 5 },
                RunAlignment::Unique { run: 6 },
                RunAlignment::NoInk,
                RunAlignment::Padding
            ]
        );
    }

    #[test]
    fn ordered_extrema_agree_with_exhaustive_short_assignment_search() {
        // Exhaust every nonempty interval of four run indices for three
        // ordered anchors. Compare the greedy extrema with all actual paths.
        let intervals = (0..4)
            .flat_map(|lo| (lo..4).map(move |hi| (lo..=hi).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        for a in &intervals {
            for b in &intervals {
                for c in &intervals {
                    let anchors = [anchor('a', 1., a), anchor('b', 2., b), anchor('c', 3., c)];
                    let paths = a
                        .iter()
                        .flat_map(|&x| {
                            b.iter().filter(move |&&y| y > x).flat_map(move |&y| {
                                c.iter()
                                    .filter(move |&&z| z > y)
                                    .map(move |&z| vec![x, y, z])
                            })
                        })
                        .collect::<Vec<_>>();
                    let indices = [0, 1, 2];
                    let lo = super::extreme_assignment(&anchors, &indices, false);
                    let hi = super::extreme_assignment(&anchors, &indices, true);
                    assert_eq!(lo, paths.first().cloned());
                    assert_eq!(hi, paths.last().cloned());
                }
            }
        }
    }

    #[test]
    fn cap_and_base_width_are_both_accounted_for() -> Result<(), Error> {
        let small = BatchGeometry::new([(21, 48)])?;
        assert_eq!(small.tensor_width, 320);
        assert_eq!(small.resized_width((21, 48))?, 21);
        let wide = BatchGeometry::new([(10000, 48), (100, 48)])?;
        assert_eq!(wide.tensor_width, 3200);
        assert_eq!(wide.resized_width((10000, 48))?, 3200);
        Ok(())
    }
}
