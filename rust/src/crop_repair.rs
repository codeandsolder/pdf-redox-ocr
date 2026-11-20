//! Re-recognize geometry-supported crop revisions on the retained RGB raster.
//!
//! Layout-leader removal and boundary extension share sampling, inference,
//! admission, batching and provenance. No unsupported interior edits.

use image::RgbImage;
use serde::Serialize;

use crate::{
    Error,
    boundary_ink::{BoundaryEvidence, detect_boundaries},
    geometry::{Point, Quad, QuadBounds},
    ink_polarity::Polarity,
    layout_leaders::{LeaderEdge, LeaderEvidence, detect_leaders},
    line_frame::{LineFrame, SampleWindow},
    pixel_regions::{InkBounds, PixelClass, regions},
    polygon::overlap_fraction,
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
pub struct CropRepair {
    pub line_index: usize,
    pub original: PixelLine,
    pub proposed_quad: Quad,
    pub sample_quad: Quad,
    pub sample_dimensions: (u32, u32),
    pub observation: InkBounds,
    pub reason: CropReason,
    /// Bounds in the sample frame, indexed by the geometry evidence.
    pub components: Vec<InkBounds>,
    pub outcome: RepairOutcome,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CropReason {
    PeriodicLeader { leaders: Vec<LeaderEvidence> },
    BoundaryInk { edges: Vec<BoundaryEvidence> },
}

#[derive(Clone, Copy, Debug)]
pub struct RepairPolicy {
    pub leaders: bool,
    pub boundaries: bool,
    pub minimum_confidence: f64,
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
    pub repairs: Vec<CropRepair>,
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

fn leader_candidate(line: &PixelLine, policy: RepairPolicy) -> bool {
    policy.leaders
        && [LeaderEdge::Left, LeaderEdge::Right]
            .into_iter()
            .any(|edge| has_leader_text(&line.text, edge))
}

struct CropEvidence {
    quad: Quad,
    sample_quad: Quad,
    dimensions: (u32, u32),
    observation: InkBounds,
    reason: CropReason,
    components: Vec<InkBounds>,
    added_strips: Vec<Quad>,
}

fn significant_overlap(first: &Quad, second: &Quad) -> bool {
    QuadBounds::new(first).intersects(QuadBounds::new(second))
        && overlap_fraction(first, second) > 0.01
}

fn added_strip(
    window: SampleWindow,
    frame: LineFrame,
    observation: InkBounds,
    edge: &BoundaryEvidence,
) -> Quad {
    let bounds = match edge.edge {
        LeaderEdge::Left => InkBounds {
            right: observation.left,
            ..edge.expanded
        },
        LeaderEdge::Right => InkBounds {
            left: observation.right,
            ..edge.expanded
        },
    };
    window.quad(frame, bounds)
}

fn crop_evidence(
    image: &RgbImage,
    lines: &[PixelLine],
    index: usize,
    policy: RepairPolicy,
    budget: &mut u64,
) -> Option<CropEvidence> {
    let line = &lines[index];
    if !policy.boundaries && !leader_candidate(line, policy) {
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
    let all_leaders = detect_leaders(&components, observation);
    let leaders = all_leaders
        .iter()
        .filter(|l| policy.leaders && has_leader_text(&line.text, l.edge))
        .cloned()
        .collect::<Vec<_>>();
    let (body, reason, added_strips) = if !leaders.is_empty() {
        let mut body = observation;
        for leader in &leaders {
            body.left = body.left.max(leader.body.left);
            body.right = body.right.min(leader.body.right);
        }
        if body.left >= body.right {
            return None;
        }
        (body, CropReason::PeriodicLeader { leaders }, vec![])
    } else if policy.boundaries && all_leaders.is_empty() {
        let resolved = detect_boundaries(&components, observation, dimensions.0)
            .into_iter()
            .filter_map(|edge| {
                let quad = added_strip(window, frame, observation, &edge);
                (!lines
                    .iter()
                    .enumerate()
                    .any(|(i, peer)| i != index && significant_overlap(&quad, &peer.quad)))
                .then_some((edge, quad))
            })
            .collect::<Vec<_>>();
        if resolved.is_empty() {
            return None;
        }
        let (edges, strips): (Vec<_>, Vec<_>) = resolved.into_iter().unzip();
        let body = edges
            .iter()
            .map(|e| e.expanded)
            .fold(observation, InkBounds::union);
        (body, CropReason::BoundaryInk { edges }, strips)
    } else {
        return None;
    };
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
        reason,
        added_strips,
        components: components.into_iter().map(|c| c.bounds).collect(),
    })
}

fn accepted(
    original: &str,
    reason: &CropReason,
    prediction: &BodyPrediction,
    threshold: f64,
) -> bool {
    let original = original.trim();
    let text = prediction.text.trim();
    if original.is_empty()
        || !(threshold..=1.0).contains(&prediction.confidence)
        || original == text
    {
        return false;
    }
    match reason {
        CropReason::PeriodicLeader { leaders } => {
            let body = leaders
                .iter()
                .fold(original, |body, leader| strip_edge(body, leader.edge));
            !body.is_empty() && body != original && text == body
        }
        CropReason::BoundaryInk { edges } => {
            let capacity = |edge| {
                edges
                    .iter()
                    .filter(|e| e.edge == edge)
                    .map(|e| e.component_indices.len())
                    .sum::<usize>()
            };
            let (left_capacity, right_capacity) =
                (capacity(LeaderEdge::Left), capacity(LeaderEdge::Right));
            // Each independently supported edge has its own bound. Do not
            // transfer left-side evidence to authorize extra right-side text.
            text.match_indices(original).any(|(start, _)| {
                let left = text[..start].trim().chars().count();
                let right = text[start + original.len()..].trim().chars().count();
                left <= left_capacity && right <= right_capacity && left + right > 0
            })
        }
    }
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
            "crop recognition returned wrong batch size".into(),
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
                    if accepted(&lines[*index].text, &crop.reason, &prediction, threshold) {
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
        report.repairs.push(CropRepair {
            line_index: index,
            original,
            proposed_quad: crop.quad,
            sample_quad: crop.sample_quad,
            sample_dimensions: crop.dimensions,
            observation: crop.observation,
            reason: crop.reason,
            components: crop.components,
            outcome,
        });
    }
}

/// Use only bounded crop batches; retain all proposed and rejected evidence.
/// A failed batch leaves its original lines intact and allows subsequent retry.
pub fn repair_crops(
    image: &RgbImage,
    lines: &mut [PixelLine],
    policy: RepairPolicy,
    mut recognize: impl FnMut(&[Quad]) -> Result<Vec<BodyPrediction>, Error>,
) -> RepairReport {
    let mut budget = MAX_INPUT_SAMPLE_PIXELS;
    // Preserve the existing leader scan priority and inference batch shape,
    // even when boundary repair is also requested. Every line is sampled
    // at most once and both kinds still share the same total pixel budget.
    let (priority, remaining): (Vec<_>, Vec<_>) =
        (0..lines.len()).partition(|&i| leader_candidate(&lines[i], policy));
    let candidates = priority
        .into_iter()
        .chain(remaining)
        .filter_map(|index| {
            crop_evidence(image, lines, index, policy, &mut budget).map(|crop| (index, crop))
        })
        .collect::<Vec<_>>();
    let mut conflicts = vec![false; candidates.len()];
    for (i, (_, crop)) in candidates.iter().enumerate() {
        if crop.added_strips.is_empty() {
            continue;
        }
        for (j, (_, other)) in candidates.iter().enumerate().skip(i + 1) {
            if crop
                .added_strips
                .iter()
                .any(|a| other.added_strips.iter().any(|b| significant_overlap(a, b)))
            {
                conflicts[i] = true;
                conflicts[j] = true;
            }
        }
    }
    let mut report = RepairReport::default();
    let (leaders, boundaries): (Vec<_>, Vec<_>) = candidates
        .into_iter()
        .zip(conflicts)
        .filter_map(|(candidate, conflict)| (!conflict).then_some(candidate))
        .partition(|(_, crop)| matches!(crop.reason, CropReason::PeriodicLeader { .. }));
    let kinds: [Vec<_>; 2] = (leaders, boundaries).into();
    for batch_kind in kinds {
        let mut pending = batch_kind.into_iter();
        loop {
            let batch = pending.by_ref().take(BATCH_SIZE).collect::<Vec<_>>();
            if batch.is_empty() {
                break;
            }
            repair_batch(
                batch,
                lines,
                policy.minimum_confidence,
                &mut recognize,
                &mut report,
            );
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::{BodyPrediction, RepairPolicy, repair_crops};
    const LEADERS: RepairPolicy = RepairPolicy {
        leaders: true,
        boundaries: false,
        minimum_confidence: 0.97,
    };
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
            let report = repair_crops(&image, &mut lines, LEADERS, |quads| {
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
        let report = repair_crops(&rotated, &mut lines, LEADERS, |quads| {
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
            fn model_profile(&self) -> &'static str {
                "leader fixture"
            }
            fn recognize_pixels(&self, rgb: RgbImage) -> Result<RasterRecognition, Error> {
                let call = self.calls.get();
                self.calls.set(call + 1);
                let mut lines = vec![self.original.clone()];
                let crop_repair = repair_crops(&rgb, &mut lines, LEADERS, |_| {
                    if call == 0 {
                        return Err(Error::Inference("transient body inference".into()));
                    }
                    Ok(vec![BodyPrediction {
                        text: "A.B".into(),
                        confidence: 0.99,
                    }])
                });
                Ok(RasterRecognition { lines, crop_repair })
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
        assert_eq!(stats.crop_batches, 2);
        assert_eq!(stats.crop_failures, 1);
        assert_eq!(stats.leader_accepted, 1);
        let audits = session.take_repair_audits();
        assert_eq!(audits.len(), 2);
        assert_eq!(audits[0].rgb_sha256, audits[1].rgb_sha256);
        Ok(())
    }

    #[test]
    fn outside_mark_repair_preserves_body_and_respects_detector_ownership() {
        let mut image = RgbImage::from_pixel(150, 100, Rgb([255; 3]));
        for (x, y, w, h) in [
            (20, 43, 8, 8),
            (40, 42, 14, 24),
            (60, 42, 14, 24),
            (80, 42, 14, 24),
        ] {
            for dy in y..y + h {
                for dx in x..x + w {
                    image.put_pixel(dx, dy, Rgb([0; 3]));
                }
            }
        }
        let original = PixelLine {
            text: "ABC".into(),
            confidence: 0.99,
            quad: [
                Point(30., 70.),
                Point(100., 70.),
                Point(100., 35.),
                Point(30., 35.),
            ],
        };
        let policy = RepairPolicy {
            leaders: false,
            boundaries: true,
            minimum_confidence: 0.97,
        };
        for (text, score, ok) in [
            ("* ABC", 0.99, true),
            ("ABC*", 0.999, false),
            ("* ABD", 0.999, false),
            ("*# ABC", 0.999, false),
            ("* ABC", 0.96, false),
        ] {
            let mut lines = vec![original.clone()];
            let report = repair_crops(&image, &mut lines, policy, |quads| {
                assert_eq!(quads.len(), 1);
                assert!(quads[0][0].0 < 20.);
                Ok(vec![BodyPrediction {
                    text: text.into(),
                    confidence: score,
                }])
            });
            assert_eq!(report.repairs.len(), 1, "{text}");
            assert_eq!(report.repairs[0].outcome.accepted(), ok, "{text}");
            assert_eq!(lines[0].text, if ok { text } else { "ABC" });
        }
        let peer = PixelLine {
            text: "*".into(),
            confidence: 0.99,
            quad: [
                Point(17., 55.),
                Point(29., 55.),
                Point(29., 40.),
                Point(17., 40.),
            ],
        };
        let mut lines = vec![original.clone(), peer.clone()];
        let report = repair_crops(&image, &mut lines, policy, |_| {
            Err(Error::Inference("should not recognize owned ink".into()))
        });
        assert!(report.repairs.is_empty());
        assert_eq!(lines, vec![original, peer]);
    }

    #[test]
    fn boundary_repair_cannot_change_leader_batches() {
        let (leader_image, leader) = sample();
        let mut image = RgbImage::from_pixel(240, 250, Rgb([255; 3]));
        for (x, y, pixel) in leader_image.enumerate_pixels() {
            image.put_pixel(x, y, *pixel);
        }
        for (x, y, w, h) in [
            (20, 193, 8, 8),
            (40, 192, 14, 24),
            (60, 192, 14, 24),
            (80, 192, 14, 24),
        ] {
            for dy in y..y + h {
                for dx in x..x + w {
                    image.put_pixel(dx, dy, Rgb([0; 3]));
                }
            }
        }
        let boundary = PixelLine {
            text: "ABC".into(),
            confidence: 0.99,
            quad: [
                Point(30., 220.),
                Point(100., 220.),
                Point(100., 185.),
                Point(30., 185.),
            ],
        };
        let mut lines = vec![boundary, leader];
        let policy = RepairPolicy {
            boundaries: true,
            ..LEADERS
        };
        let mut batches = vec![];
        let report = repair_crops(&image, &mut lines, policy, |quads| {
            assert_eq!(quads.len(), 1);
            let is_leader = quads[0][0].1 < 100.;
            batches.push(is_leader);
            Ok(vec![BodyPrediction {
                text: if is_leader { "A.B" } else { "* ABC" }.into(),
                confidence: 0.99,
            }])
        });
        assert_eq!(batches, [true, false]);
        assert_eq!(report.batches, 2);
        assert_eq!(lines[0].text, "* ABC");
        assert_eq!(lines[1].text, "A.B");
    }

    #[test]
    fn no_candidate_no_model_and_failed_batch_preserves_observations() {
        let (image, original) = sample();
        let mut unpunctuated = original.clone();
        unpunctuated.text = "A.B".into();
        let mut lines = vec![unpunctuated.clone()];
        let report = repair_crops(&image, &mut lines, LEADERS, |_| {
            Err(Error::Inference("unexpected".into()))
        });
        assert_eq!(report.batches, 0);
        assert_eq!(lines[0], unpunctuated);
        let mut lines = vec![original.clone(); 9];
        let report = repair_crops(&image, &mut lines, LEADERS, |_| {
            Err(Error::Inference("fixture".into()))
        });
        assert_eq!(report.batches, 3);
        assert!(report.repairs.iter().all(|r| r.outcome.failed()));
        assert_eq!(lines, vec![original; 9]);
    }
}
