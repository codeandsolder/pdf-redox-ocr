//! Re-recognize only source-supported text bodies, using the retained input raster.
//!
//! The geometry establishes where layout ink ends; a second recognition must
//! independently confirm the complete body. It cannot introduce other edits.

use image::RgbImage;
use serde::Serialize;

use crate::{
    Error,
    geometry::{Point, Quad},
    ink_polarity::Polarity,
    layout_leaders::{LeaderEdge, LeaderEvidence, detect_leaders},
    line_frame::{LineFrame, SampleWindow},
    pixel_regions::{InkBounds, PixelClass, regions},
    recognition::PixelLine,
};

const MAX_LINE_PIXELS: u64 = 2_000_000;
const MAX_INPUT_SAMPLE_PIXELS: u64 = 16_000_000;
const BATCH_SIZE: usize = 4;

#[derive(Clone, Debug, Serialize)]
pub struct BodyPrediction {
    pub text: String,
    pub confidence: f64,
}

/// Evidence in the original recognition raster, not PDF occurrence coordinates.
#[derive(Clone, Debug, Serialize)]
pub struct LeaderRepair {
    pub line_index: usize,
    pub original: PixelLine,
    pub body_quad: Quad,
    pub sample_quad: Quad,
    pub sample_dimensions: (u32, u32),
    pub observation: InkBounds,
    pub leaders: Vec<LeaderEvidence>,
    /// Bounds in the sample frame, indexed by the geometry evidence.
    pub components: Vec<InkBounds>,
    pub outcome: RepairOutcome,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RepairOutcome {
    Accepted { prediction: BodyPrediction },
    Rejected { prediction: BodyPrediction },
    InferenceFailed { error: String },
}

impl RepairOutcome {
    #[must_use]
    pub const fn accepted(&self) -> bool {
        matches!(self, Self::Accepted { .. })
    }

    #[must_use]
    pub const fn failed(&self) -> bool {
        matches!(self, Self::InferenceFailed { .. })
    }
}

#[derive(Default)]
pub struct RepairReport {
    pub repairs: Vec<LeaderRepair>,
    pub batches: usize,
}

const fn dot(ch: char) -> bool {
    matches!(ch, '.' | '…' | '⋯' | '．')
}

fn strip_edge(text: &str, edge: LeaderEdge) -> &str {
    let trim = |ch: char| dot(ch) || ch.is_whitespace();
    match edge {
        LeaderEdge::Left => text.trim_start_matches(trim),
        LeaderEdge::Right => text.trim_end_matches(trim),
    }
}

fn has_leader_text(text: &str, edge: LeaderEdge) -> bool {
    let text = text.trim();
    match edge {
        LeaderEdge::Left => text.starts_with(dot),
        LeaderEdge::Right => text.ends_with(dot),
    }
}

struct CropEvidence {
    quad: Quad,
    sample_quad: Quad,
    dimensions: (u32, u32),
    observation: InkBounds,
    leaders: Vec<LeaderEvidence>,
    components: Vec<InkBounds>,
}

fn body_evidence(image: &RgbImage, line: &PixelLine, budget: &mut u64) -> Option<CropEvidence> {
    if ![LeaderEdge::Left, LeaderEdge::Right]
        .into_iter()
        .any(|edge| has_leader_text(&line.text, edge))
    {
        return None;
    }
    let mut frame = LineFrame::new(line.quad, |points| points, MAX_LINE_PIXELS).ok()?;
    // OAR rotates tall crops counterclockwise, so their reading axis grows
    // along the original downward image axis. The diagnostic frame alone
    // has no reading-direction contract.
    if frame.axis_swapped {
        frame.corners.rotate_left(2);
    }
    let inside = |Point(x, y): Point| {
        (0.0..=f64::from(image.width())).contains(&x)
            && (0.0..=f64::from(image.height())).contains(&y)
    };
    let window = SampleWindow::new(
        frame,
        inside,
        frame.height.saturating_mul(2),
        MAX_LINE_PIXELS,
    )
    .ok()?;
    let dimensions = window.dimensions(frame)?;
    let count = u64::from(dimensions.0) * u64::from(dimensions.1);
    *budget = budget.checked_sub(count)?;
    let observation = window.observation(frame);
    let (normalized, _) = window
        .normalized_ink(frame, image, |point| point, Polarity::Auto)
        .ok()?;
    let components = regions(&normalized, 192, PixelClass::Ink, observation).ok()?;
    let leaders = detect_leaders(&components, observation)
        .into_iter()
        .filter(|l| has_leader_text(&line.text, l.edge))
        .collect::<Vec<_>>();
    if leaders.is_empty() {
        return None;
    }
    let mut body = observation;
    for leader in &leaders {
        body.left = body.left.max(leader.body.left);
        body.right = body.right.min(leader.body.right);
    }
    if body.left >= body.right {
        return None;
    }
    // The frame may swap axes for vertical lines. The recognition-only
    // predictor still has to confirm the exact body on the proved edge.
    // Restore original detector corner identity when mapping back.
    let mut quad = window.quad(frame, body);
    if frame.axis_swapped {
        quad.rotate_left(1);
    }
    Some(CropEvidence {
        quad,
        sample_quad: window.quad(
            frame,
            InkBounds {
                left: 0,
                top: 0,
                right: dimensions.0,
                bottom: dimensions.1,
            },
        ),
        dimensions,
        observation,
        leaders,
        components: components.into_iter().map(|c| c.bounds).collect(),
    })
}

fn accepted(
    original: &str,
    leaders: &[LeaderEvidence],
    prediction: &BodyPrediction,
    threshold: f64,
) -> bool {
    let mut body = original.trim();
    for leader in leaders {
        body = strip_edge(body, leader.edge);
    }
    let text = prediction.text.trim();
    !body.is_empty()
        && body != original.trim()
        && text == body
        && (threshold..=1.0).contains(&prediction.confidence)
}

fn repair_batch(
    batch: Vec<(usize, CropEvidence)>,
    lines: &mut [PixelLine],
    threshold: f64,
    recognize: &mut impl FnMut(&[Quad]) -> Result<Vec<BodyPrediction>, Error>,
    report: &mut RepairReport,
) {
    if batch.is_empty() {
        return;
    }
    let quads = batch.iter().map(|(_, crop)| crop.quad).collect::<Vec<_>>();
    report.batches += 1;
    let predictions = match recognize(&quads) {
        Ok(predictions) if predictions.len() == batch.len() => Ok(predictions),
        Ok(_) => Err(Error::Inference(
            "leader recognition returned wrong batch size".into(),
        )),
        Err(error) => Err(error),
    };
    let outcomes = predictions.map_or_else(
        |error| {
            vec![
                RepairOutcome::InferenceFailed {
                    error: error.to_string()
                };
                batch.len()
            ]
        },
        |predictions| {
            predictions
                .into_iter()
                .zip(&batch)
                .map(|(prediction, (index, crop))| {
                    if accepted(&lines[*index].text, &crop.leaders, &prediction, threshold) {
                        RepairOutcome::Accepted { prediction }
                    } else {
                        RepairOutcome::Rejected { prediction }
                    }
                })
                .collect()
        },
    );
    for ((index, crop), outcome) in batch.into_iter().zip(outcomes) {
        let line = &mut lines[index];
        let original = line.clone();
        if let RepairOutcome::Accepted { prediction } = &outcome {
            prediction.text.trim().clone_into(&mut line.text);
            line.confidence = prediction.confidence;
            line.quad = crop.quad;
        }
        report.repairs.push(LeaderRepair {
            line_index: index,
            original,
            body_quad: crop.quad,
            sample_quad: crop.sample_quad,
            sample_dimensions: crop.dimensions,
            observation: crop.observation,
            leaders: crop.leaders,
            components: crop.components,
            outcome,
        });
    }
}

/// Use only bounded crop batches; retain all proposed and rejected evidence.
/// A failed batch leaves its original lines intact and allows subsequent retry.
pub fn repair_leaders(
    image: &RgbImage,
    lines: &mut [PixelLine],
    threshold: f64,
    mut recognize: impl FnMut(&[Quad]) -> Result<Vec<BodyPrediction>, Error>,
) -> RepairReport {
    let mut budget = MAX_INPUT_SAMPLE_PIXELS;
    let mut candidates = Vec::with_capacity(BATCH_SIZE);
    let mut report = RepairReport::default();
    for index in 0..lines.len() {
        if let Some(crop) = body_evidence(image, &lines[index], &mut budget) {
            candidates.push((index, crop));
            if candidates.len() == BATCH_SIZE {
                repair_batch(
                    std::mem::replace(&mut candidates, Vec::with_capacity(BATCH_SIZE)),
                    lines,
                    threshold,
                    &mut recognize,
                    &mut report,
                );
            }
        }
    }
    repair_batch(candidates, lines, threshold, &mut recognize, &mut report);
    report
}

#[cfg(test)]
mod tests {
    use super::{BodyPrediction, repair_leaders};
    use crate::{Error, geometry::Point, recognition::PixelLine};
    use image::{Rgb, RgbImage};

    fn sample() -> (RgbImage, PixelLine) {
        let mut image = RgbImage::from_pixel(240, 100, Rgb([255; 3]));
        for (x, y, w, h) in [(35, 37, 10, 22), (50, 53, 4, 4), (62, 37, 12, 22)] {
            for dy in y..y + h {
                for dx in x..x + w {
                    image.put_pixel(dx, dy, Rgb([0; 3]));
                }
            }
        }
        for x in [88, 98, 108, 118, 128, 138, 148, 158] {
            for y in 53..59 {
                for dx in x..x + 6 {
                    image.put_pixel(dx, y, Rgb([0; 3]));
                }
            }
        }
        (
            image,
            PixelLine {
                text: "A.B ..".into(),
                confidence: 0.98,
                quad: [
                    Point(30.0, 65.0),
                    Point(112.0, 65.0),
                    Point(112.0, 30.0),
                    Point(30.0, 30.0),
                ],
            },
        )
    }

    #[test]
    fn geometry_and_independent_text_must_agree() {
        let (image, original) = sample();
        for (text, confidence, expected) in [
            ("A.B", 0.99, true),
            ("AB", 0.999, false),
            ("A.C", 0.999, false),
            ("A.B", 0.96, false),
            ("A.B ..", 0.999, false),
        ] {
            let mut lines = vec![original.clone()];
            let report = repair_leaders(&image, &mut lines, 0.97, |quads| {
                assert_eq!(quads.len(), 1);
                assert!(quads[0][1].0 < 88.0 && quads[0][1].0 > 74.0);
                Ok(vec![BodyPrediction {
                    text: text.into(),
                    confidence,
                }])
            });
            assert_eq!(report.repairs.len(), 1, "{text}");
            assert_eq!(report.repairs[0].original, original);
            assert_eq!(report.repairs[0].outcome.accepted(), expected);
            if expected {
                assert_eq!(lines[0].text, "A.B");
                assert_eq!(lines[0].confidence, 0.99);
            } else {
                assert_eq!(lines[0], original);
            }
        }
    }

    #[test]
    fn vertical_crop_uses_same_reading_direction_and_restores_detector_corners() {
        let (image, line) = sample();
        let rotated = image::imageops::rotate90(&image);
        let mut original = line;
        original.quad = original
            .quad
            .map(|Point(x, y)| Point(f64::from(image.height()) - y, x));
        original.quad.rotate_left(1);
        let mut lines = vec![original.clone()];
        let report = repair_leaders(&rotated, &mut lines, 0.97, |quads| {
            assert_eq!(quads.len(), 1);
            assert!(quads[0][0].1 > 74.0 && quads[0][0].1 < 88.0);
            assert_eq!(quads[0][2], original.quad[2]);
            assert_eq!(quads[0][3], original.quad[3]);
            Ok(vec![BodyPrediction {
                text: "A.B".into(),
                confidence: 0.99,
            }])
        });
        assert_eq!(report.repairs.len(), 1);
        assert!(report.repairs[0].outcome.accepted());
        assert_eq!(lines[0].text, "A.B");
    }

    #[test]
    fn transient_repair_failure_does_not_pollute_exact_rgb_cache() -> Result<(), Error> {
        use crate::recognition::{
            RasterRecognition, RecognitionBackend, RecognitionCacheBudget, RecognitionSession,
        };
        use std::cell::Cell;
        struct Backend {
            original: PixelLine,
            calls: Cell<usize>,
        }
        impl RecognitionBackend for Backend {
            fn model_profile(&self) -> &str {
                "leader fixture"
            }
            fn recognize_pixels(&self, rgb: RgbImage) -> Result<RasterRecognition, Error> {
                let call = self.calls.get();
                self.calls.set(call + 1);
                let mut lines = vec![self.original.clone()];
                let leader_repair = repair_leaders(&rgb, &mut lines, 0.97, |_| {
                    if call == 0 {
                        return Err(Error::Inference("transient body inference".into()));
                    }
                    Ok(vec![BodyPrediction {
                        text: "A.B".into(),
                        confidence: 0.99,
                    }])
                });
                Ok(RasterRecognition {
                    lines,
                    leader_repair,
                })
            }
        }
        let (image, original) = sample();
        let backend = Backend {
            original: original.clone(),
            calls: Cell::new(0),
        };
        let mut session = RecognitionSession::new(&backend, RecognitionCacheBudget::default());
        assert_eq!(session.recognize_pixels(image.clone())?, vec![original]);
        let accepted = session.recognize_pixels(image.clone())?;
        assert_eq!(accepted[0].text, "A.B");
        assert_eq!(session.recognize_pixels(image)?, accepted);
        let stats = session.stats();
        assert_eq!(stats.inference_calls, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.leader_batches, 2);
        assert_eq!(stats.leader_failures, 1);
        assert_eq!(stats.leader_accepted, 1);
        let audits = session.take_repair_audits();
        assert_eq!(audits.len(), 2);
        assert_eq!(audits[0].rgb_sha256, audits[1].rgb_sha256);
        Ok(())
    }

    #[test]
    fn no_candidate_no_model_and_failed_batch_preserves_observations() {
        let (image, original) = sample();
        let mut unpunctuated = original.clone();
        unpunctuated.text = "A.B".into();
        let mut lines = vec![unpunctuated.clone()];
        let report = repair_leaders(&image, &mut lines, 0.97, |_| {
            Err(Error::Inference("unexpected".into()))
        });
        assert_eq!(report.batches, 0);
        assert_eq!(lines[0], unpunctuated);
        let mut lines = vec![original.clone(); 9];
        let report = repair_leaders(&image, &mut lines, 0.97, |_| {
            Err(Error::Inference("fixture".into()))
        });
        assert_eq!(report.batches, 3);
        assert!(report.repairs.iter().all(|r| r.outcome.failed()));
        assert_eq!(lines, vec![original; 9]);
    }
}
