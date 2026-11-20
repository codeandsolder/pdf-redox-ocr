//! Evaluate SHA-bound OCR drafts against manually transcribed PDF-space anchors.
//! Missing observations contribute full reference length to character error.

mod alignment;

use std::collections::{BTreeMap, BTreeSet};

use alignment::alignment;

use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::{
    Error,
    draft::{OcrDraft, OcrLine},
    geometry::{Point, Quad},
};

/// Human-curated labels for one exact original PDF.
#[derive(Debug, Clone, Deserialize)]
pub struct GroundTruth {
    pub schema_version: u32,
    pub source_pdf_sha256: String,
    pub provenance: String,
    pub coordinate_system: String,
    pub samples: Vec<ReferenceSample>,
}

/// One manually located reference line.
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceSample {
    pub page: u32,
    pub center: Point,
    pub text: String,
    pub kind: String,
}

/// Exact and whitespace/NFKC-insensitive quality metrics.
#[derive(Debug, Clone, Serialize)]
pub struct QualityMetrics {
    pub samples: usize,
    pub matched: usize,
    pub missing: usize,
    pub exact: usize,
    pub normalized_exact: usize,
    pub character_edits: usize,
    pub normalized_character_edits: usize,
    pub reference_characters: usize,
    pub normalized_reference_characters: usize,
    pub character_error_rate: f64,
    pub normalized_character_error_rate: f64,
}

/// Audit for a reference that may or may not have a corresponding OCR line.
#[derive(Debug, Clone, Serialize)]
pub struct QualitySample {
    pub page: u32,
    pub center: Point,
    pub kind: String,
    pub reference: String,
    pub predicted: Option<String>,
    /// Zero-based index in the original, unreordered draft page; absent for unmatched anchors.
    pub line_index: Option<usize>,
    /// Exact selected observation geometry, retained even with overlapping candidates.
    pub line_quad: Option<Quad>,
    pub confidence: Option<f64>,
    pub candidate_count: usize,
    pub exact: bool,
    pub normalized_exact: bool,
    pub character_edits: usize,
    pub normalized_character_edits: usize,
}

/// Observed quality when admitting every matched line at or above this score.
///
/// These are retrospective, *labeled* sample statistics, not proof that the
/// model's numeric confidence is a calibrated probability.
#[derive(Debug, Clone, Serialize)]
pub struct ConfidencePoint {
    pub minimum_confidence: f64,
    pub admitted: usize,
    pub admitted_normalized_exact: usize,
    pub admitted_errors: usize,
    pub reference_coverage: f64,
    pub empirical_error_rate: f64,
}

/// One Unicode substitution, insertion, or deletion found in manual OCR text.
/// A missing side is a genuine insertion/deletion, not an empty character.
#[derive(Debug, Clone, Serialize)]
pub struct CharacterConfusion {
    pub reference: Option<char>,
    pub predicted: Option<char>,
    pub count: usize,
    pub highest_line_confidence: Option<f64>,
}

/// Auditable report bound to the exact original PDF hash.
#[derive(Debug, Clone, Serialize)]
pub struct QualityReport {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub reference_provenance: String,
    pub draft_model: String,
    pub overall: QualityMetrics,
    pub by_kind: BTreeMap<String, QualityMetrics>,
    pub confidence_curve: Vec<ConfidencePoint>,
    pub character_confusions: Vec<CharacterConfusion>,
    pub samples: Vec<QualitySample>,
}

#[derive(Default)]
struct Counters {
    samples: usize,
    matched: usize,
    exact: usize,
    normalized_exact: usize,
    character_edits: usize,
    normalized_character_edits: usize,
    reference_characters: usize,
    normalized_reference_characters: usize,
}

impl Counters {
    fn add(&mut self, sample: &QualitySample) {
        self.samples += 1;
        self.matched += usize::from(sample.predicted.is_some());
        self.exact += usize::from(sample.exact);
        self.normalized_exact += usize::from(sample.normalized_exact);
        self.character_edits += sample.character_edits;
        self.normalized_character_edits += sample.normalized_character_edits;
        self.reference_characters += sample.reference.chars().count();
        self.normalized_reference_characters += normalized(&sample.reference).chars().count();
    }

    fn metrics(self) -> QualityMetrics {
        QualityMetrics {
            samples: self.samples,
            matched: self.matched,
            missing: self.samples - self.matched,
            exact: self.exact,
            normalized_exact: self.normalized_exact,
            character_edits: self.character_edits,
            normalized_character_edits: self.normalized_character_edits,
            reference_characters: self.reference_characters,
            normalized_reference_characters: self.normalized_reference_characters,
            character_error_rate: proportion(self.character_edits, self.reference_characters),
            normalized_character_error_rate: proportion(
                self.normalized_character_edits,
                self.normalized_reference_characters,
            ),
        }
    }
}

pub(crate) fn normalized(s: &str) -> String {
    s.nfkc().filter(|c| !c.is_whitespace()).collect()
}

/// A ratio over observed counts, with no implied statistical confidence.
fn proportion(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator
        .to_f64()
        .zip(denominator.to_f64())
        .map_or(0.0, |(numerator, denominator)| numerator / denominator)
}

/// Full empirical risk/coverage curve over distinct model confidence scores.
///
/// A missing observation has no model confidence and never counts as admitted.
/// Tied confidence values are grouped, so a threshold cannot select an
/// arbitrary subset of equally confident predictions.
#[must_use]
pub fn confidence_curve(samples: &[QualitySample]) -> Vec<ConfidencePoint> {
    let mut scores = samples
        .iter()
        .filter_map(|sample| {
            sample
                .confidence
                .map(|score| (score, sample.normalized_exact))
        })
        .collect::<Vec<_>>();
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut curve = Vec::new();
    let mut index = 0;
    let mut admitted = 0;
    let mut correct = 0;
    while index < scores.len() {
        let threshold = scores[index].0;
        while index < scores.len() && scores[index].0.total_cmp(&threshold).is_eq() {
            admitted += 1;
            correct += usize::from(scores[index].1);
            index += 1;
        }
        curve.push(ConfidencePoint {
            minimum_confidence: threshold,
            admitted,
            admitted_normalized_exact: correct,
            admitted_errors: admitted - correct,
            reference_coverage: proportion(admitted, samples.len()),
            empirical_error_rate: proportion(admitted - correct, admitted),
        });
    }
    curve
}

/// Aggregate independently aligned Unicode mistakes without losing insertions
/// or deletions. All counts refer to normalized manual labels.
#[must_use]
pub fn character_confusions(samples: &[QualitySample]) -> Vec<CharacterConfusion> {
    let mut counts = BTreeMap::<(Option<char>, Option<char>), (usize, Option<f64>)>::new();
    for sample in samples {
        let reference = normalized(&sample.reference);
        let prediction = normalized(sample.predicted.as_deref().unwrap_or(""));
        for (expected, observed) in alignment(&reference, &prediction) {
            if expected == observed {
                continue;
            }
            let entry = counts.entry((expected, observed)).or_default();
            entry.0 += 1;
            if let Some(score) = sample.confidence {
                entry.1 = Some(entry.1.map_or(score, |previous| previous.max(score)));
            }
        }
    }
    let mut errors = counts
        .into_iter()
        .map(
            |((reference, predicted), (count, highest_line_confidence))| CharacterConfusion {
                reference,
                predicted,
                count,
                highest_line_confidence,
            },
        )
        .collect::<Vec<_>>();
    errors.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.reference.cmp(&b.reference))
            .then_with(|| a.predicted.cmp(&b.predicted))
    });
    errors
}

/// Levenshtein distance over Unicode scalar values, not UTF-8 bytes.
#[must_use]
pub fn character_edits(reference: &str, prediction: &str) -> usize {
    let first = reference.chars().collect::<Vec<_>>();
    let second = prediction.chars().collect::<Vec<_>>();
    let (short, long) = if first.len() <= second.len() {
        (first, second)
    } else {
        (second, first)
    };
    alignment::last_row(&long, &short)[short.len()]
}

/// Half-plane membership for an arbitrary convex OCR quadrilateral.
fn contains(quad: &Quad, point: Point) -> bool {
    if !point.0.is_finite()
        || !point.1.is_finite()
        || quad.iter().any(|p| !p.0.is_finite() || !p.1.is_finite())
    {
        return false;
    }
    let (mut positive, mut negative, mut signed_area) = (false, false, 0.0_f64);
    let origin = quad[0];
    for (index, &a) in quad.iter().enumerate() {
        let b = quad[(index + 1) % quad.len()];
        let cross = (b.0 - a.0).mul_add(point.1 - a.1, -(b.1 - a.1) * (point.0 - a.0));
        positive |= cross > 1e-9;
        negative |= cross < -1e-9;
        let ax = a.0 - origin.0;
        let ay = a.1 - origin.1;
        let bx = b.0 - origin.0;
        let by = b.1 - origin.1;
        signed_area += ax.mul_add(by, -bx * ay);
    }
    !(positive && negative) && signed_area.is_finite() && signed_area.abs() > 1e-10
}

fn validate(reference: &GroundTruth, draft: &OcrDraft) -> Result<(), Error> {
    if reference.schema_version != 1
        || reference.coordinate_system != "PDF user space"
        || reference.source_pdf_sha256 != draft.pdf_sha256
        || reference.source_pdf_sha256.len() != 64
        || !reference
            .source_pdf_sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
        || reference.provenance.trim().is_empty()
        || reference.samples.is_empty()
        || draft.schema_version != 3
    {
        return Err(Error::Invalid(
            "invalid or source-mismatched manual OCR reference".into(),
        ));
    }
    let mut centers = BTreeSet::new();
    for sample in &reference.samples {
        if sample.page == 0
            || !sample.center.0.is_finite()
            || !sample.center.1.is_finite()
            || sample.text.trim().is_empty()
            || sample.kind.trim().is_empty()
            || !centers.insert((
                sample.page,
                sample.center.0.to_bits(),
                sample.center.1.to_bits(),
            ))
        {
            return Err(Error::Invalid(
                "invalid or duplicate reference anchor".into(),
            ));
        }
    }
    let mut pages = BTreeSet::new();
    for page in &draft.pages {
        if page.page == 0 || !pages.insert(page.page) {
            return Err(Error::Invalid("invalid or duplicate draft page".into()));
        }
        for line in &page.lines {
            if !line.confidence.is_finite() || !(0.0..=1.0).contains(&line.confidence) {
                return Err(Error::Invalid("invalid OCR confidence".into()));
            }
        }
    }
    Ok(())
}

/// Kuhn augmenting path: maximum cardinality even with overlapping detections.
fn claim(
    reference: usize,
    options: &[Vec<usize>],
    owners: &mut [Option<usize>],
    visited: &mut [bool],
) -> bool {
    for &candidate in &options[reference] {
        if visited[candidate] {
            continue;
        }
        visited[candidate] = true;
        if owners[candidate].is_none_or(|previous| claim(previous, options, owners, visited)) {
            owners[candidate] = Some(reference);
            return true;
        }
    }
    false
}

fn candidate_lines<'a>(
    reference: &GroundTruth,
    draft: &'a OcrDraft,
) -> (Vec<(usize, &'a OcrLine)>, Vec<Vec<usize>>) {
    let all = draft
        .pages
        .iter()
        .flat_map(|page| {
            page.lines
                .iter()
                .enumerate()
                .filter(|(_, line)| !line.text.trim().is_empty())
                .map(move |(index, line)| (page.page, index, line))
        })
        .collect::<Vec<_>>();
    let mut options = Vec::with_capacity(reference.samples.len());
    for sample in &reference.samples {
        let mut found = all
            .iter()
            .enumerate()
            .filter(|(_, (page, _, line))| {
                *page == sample.page && contains(&line.quad, sample.center)
            })
            .map(|(index, (_, _, line))| {
                let center_x = line.quad.iter().map(|p| p.0).sum::<f64>() / 4.0;
                let center_y = line.quad.iter().map(|p| p.1).sum::<f64>() / 4.0;
                let dx = center_x - sample.center.0;
                let dy = center_y - sample.center.1;
                (index, dx.mul_add(dx, dy * dy))
            })
            .collect::<Vec<_>>();
        found.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        options.push(found.into_iter().map(|(index, _)| index).collect());
    }
    (
        all.into_iter()
            .map(|(_, index, line)| (index, line))
            .collect(),
        options,
    )
}

/// Evaluate the reference against existing, unchanged OCR observations.
///
/// The caller must verify the input draft against the original PDF first.
///
/// # Errors
/// Rejects mismatched PDF hashes, invalid anchors, scores and duplicate pages.
pub fn evaluate(reference: &GroundTruth, draft: &OcrDraft) -> Result<QualityReport, Error> {
    validate(reference, draft)?;
    let (lines, options) = candidate_lines(reference, draft);
    let mut owners = vec![None; lines.len()];
    for index in 0..reference.samples.len() {
        let mut visited = vec![false; lines.len()];
        claim(index, &options, &mut owners, &mut visited);
    }
    let mut matched = vec![None; reference.samples.len()];
    for (line, owner) in owners.into_iter().enumerate() {
        if let Some(index) = owner {
            matched[index] = Some(line);
        }
    }
    let mut overall = Counters::default();
    let mut kinds = BTreeMap::<String, Counters>::new();
    let mut samples = Vec::with_capacity(reference.samples.len());
    for (index, label) in reference.samples.iter().enumerate() {
        let selected = matched[index].map(|index| lines[index]);
        let line = selected.map(|(_, line)| line);
        let predicted = line.map_or("", |line| line.text.as_str());
        let sample = QualitySample {
            page: label.page,
            center: label.center,
            kind: label.kind.clone(),
            reference: label.text.clone(),
            predicted: line.map(|line| line.text.clone()),
            line_index: selected.map(|(index, _)| index),
            line_quad: line.map(|line| line.quad),
            confidence: line.map(|line| line.confidence),
            candidate_count: options[index].len(),
            exact: line.is_some_and(|_| label.text == predicted),
            normalized_exact: line
                .is_some_and(|_| normalized(&label.text) == normalized(predicted)),
            character_edits: character_edits(&label.text, predicted),
            normalized_character_edits: character_edits(
                &normalized(&label.text),
                &normalized(predicted),
            ),
        };
        overall.add(&sample);
        kinds.entry(label.kind.clone()).or_default().add(&sample);
        samples.push(sample);
    }
    let curve = confidence_curve(&samples);
    let errors = character_confusions(&samples);
    let metrics = overall.metrics();
    debug_assert_eq!(
        errors.iter().map(|entry| entry.count).sum::<usize>(),
        metrics.normalized_character_edits
    );
    Ok(QualityReport {
        schema_version: 3,
        pdf_sha256: draft.pdf_sha256.clone(),
        reference_provenance: reference.provenance.clone(),
        draft_model: draft.model.clone(),
        overall: metrics,
        by_kind: kinds
            .into_iter()
            .map(|(kind, metrics)| (kind, metrics.metrics()))
            .collect(),
        confidence_curve: curve,
        character_confusions: errors,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        GroundTruth, QualitySample, ReferenceSample, character_confusions, character_edits,
        contains, evaluate,
    };
    use crate::{
        draft::{DraftPage, OcrDraft, OcrLine},
        geometry::Point,
    };

    fn quad(x: f64) -> [Point; 4] {
        [
            Point(x, 0.0),
            Point(x + 10.0, 0.0),
            Point(x + 10.0, 3.0),
            Point(x, 3.0),
        ]
    }

    fn draft() -> OcrDraft {
        OcrDraft::new(
            "a".repeat(64),
            "test".into(),
            vec![DraftPage {
                page: 3,
                lines: vec![
                    OcrLine {
                        text: "10A 250VAC".into(),
                        confidence: 0.9,
                        quad: quad(1.0),
                        reviewed: false,
                    },
                    OcrLine {
                        text: "温度 +125°C".into(),
                        confidence: 0.95,
                        quad: quad(15.0),
                        reviewed: false,
                    },
                ],
                native_text_regions: vec![],
            }],
        )
    }

    fn truth() -> GroundTruth {
        GroundTruth {
            schema_version: 1,
            source_pdf_sha256: "a".repeat(64),
            provenance: "visually verified PDF".into(),
            coordinate_system: "PDF user space".into(),
            samples: vec![
                ReferenceSample {
                    page: 3,
                    center: Point(5.0, 1.5),
                    text: "10A 250VAC".into(),
                    kind: "technical".into(),
                },
                ReferenceSample {
                    page: 3,
                    center: Point(20.0, 1.5),
                    text: "温度 +125℃".into(),
                    kind: "mixed".into(),
                },
                ReferenceSample {
                    page: 3,
                    center: Point(50.0, 1.5),
                    text: "20A 125VAC".into(),
                    kind: "technical".into(),
                },
            ],
        }
    }

    #[test]
    fn checked_in_manual_reference_manifests_parse() -> Result<(), serde_json::Error> {
        let defond: GroundTruth = serde_json::from_str(include_str!(
            "../tests/fixtures/quality/defond-manual-groundtruth.json"
        ))?;
        let walsin: GroundTruth = serde_json::from_str(include_str!(
            "../tests/fixtures/quality/walsin-manual-groundtruth.json"
        ))?;
        assert_eq!(defond.samples.len(), 86);
        assert_eq!(walsin.samples.len(), 10);
        let taping: GroundTruth = serde_json::from_str(include_str!(
            "../tests/fixtures/quality/walsin-taping-groundtruth.json"
        ))?;
        assert_eq!(taping.samples.len(), 3);
        assert_eq!(taping.source_pdf_sha256, walsin.source_pdf_sha256);
        assert!(
            taping
                .samples
                .iter()
                .all(|sample| sample.page == 3 && sample.text.matches('Ø').count() == 2)
        );
        assert_ne!(defond.source_pdf_sha256, walsin.source_pdf_sha256);
        assert_eq!(defond.coordinate_system, "PDF user space");
        assert_eq!(walsin.coordinate_system, "PDF user space");
        Ok(())
    }

    #[test]
    fn unicode_distances_count_scalars() {
        assert_eq!(character_edits("温度", "温"), 1);
        assert_eq!(character_edits("μ", "µ"), 1);
        assert_eq!(character_edits("ABC", "AXC"), 1);
        assert_eq!(character_edits("ABC", ""), 3);
        assert_eq!(character_edits("AB", "XYAB"), 2);
    }

    #[test]
    fn skewed_quads_do_not_match_rectangular_envelopes() {
        let q = [
            Point(0.0, 1.0),
            Point(1.0, 0.0),
            Point(2.0, 1.0),
            Point(1.0, 2.0),
        ];
        assert!(contains(&q, Point(1.0, 1.0)));
        assert!(!contains(&q, Point(0.1, 0.1)));
        assert!(!contains(&q, Point(f64::NAN, 1.0)));
        assert!(!contains(&[Point(1.0, 1.0); 4], Point(1.0, 1.0)));
        let offset = q.map(|p| Point(p.0 + 1.0e13, p.1 - 1.0e13));
        assert!(contains(&offset, Point(1.0e13 + 1.0, -1.0e13 + 1.0)));
    }

    #[test]
    fn missing_lines_count_toward_cer() -> Result<(), crate::Error> {
        let report = evaluate(&truth(), &draft())?;
        assert_eq!(report.overall.samples, 3);
        assert_eq!(report.overall.matched, 2);
        assert_eq!(report.overall.missing, 1);
        assert_eq!(report.overall.exact, 1);
        assert_eq!(report.overall.normalized_exact, 2);
        assert_eq!(report.overall.character_edits, 12);
        assert_eq!(report.overall.normalized_character_edits, 9);
        assert_eq!(report.by_kind["technical"].missing, 1);
        assert_eq!(report.samples[2].predicted, None);
        Ok(())
    }

    #[test]
    fn blank_draft_lines_do_not_count_as_observed() -> Result<(), crate::Error> {
        let mut source = draft();
        source.pages[0].lines[0].text = "   ".into();
        let report = evaluate(&truth(), &source)?;
        assert_eq!(report.overall.matched, 1);
        assert_eq!(report.overall.missing, 2);
        assert_eq!(report.samples[0].predicted, None);
        assert_eq!(report.samples[0].line_index, None);
        assert_eq!(report.samples[0].line_quad, None);
        assert_eq!(report.samples[1].line_index, Some(1));
        assert_eq!(
            report.samples[1].line_quad,
            Some(source.pages[0].lines[1].quad)
        );
        Ok(())
    }

    #[test]
    fn one_detection_cannot_satisfy_two_anchors() -> Result<(), crate::Error> {
        let mut manual = truth();
        manual.samples[1].center = Point(5.5, 1.5);
        let result = evaluate(&manual, &draft())?;
        assert_eq!(result.overall.matched, 1);
        assert_eq!(result.overall.missing, 2);
        Ok(())
    }

    #[test]
    fn augmenting_path_finds_maximum_unique_matching() -> Result<(), crate::Error> {
        let mut source = draft();
        source.pages[0].lines = vec![
            OcrLine {
                text: "A".into(),
                confidence: 0.9,
                quad: quad(1.0),
                reviewed: false,
            },
            OcrLine {
                text: "B".into(),
                confidence: 0.9,
                quad: quad(3.0),
                reviewed: false,
            },
        ];
        let mut manual = truth();
        manual.samples = vec![
            ReferenceSample {
                page: 3,
                center: Point(5.0, 1.5),
                text: "B".into(),
                kind: "technical".into(),
            },
            ReferenceSample {
                page: 3,
                center: Point(1.5, 1.5),
                text: "A".into(),
                kind: "technical".into(),
            },
        ];
        let report = evaluate(&manual, &source)?;
        assert_eq!(report.overall.matched, 2);
        assert_eq!(report.overall.exact, 2);
        assert_eq!(report.samples[0].candidate_count, 2);
        assert_eq!(report.samples[0].line_index, Some(1));
        assert_eq!(report.samples[1].line_index, Some(0));
        Ok(())
    }

    #[test]
    fn confidence_curve_exposes_high_confidence_errors_and_missing_anchors()
    -> Result<(), crate::Error> {
        let mut source = draft();
        source.pages[0].lines[1].text = "wrong".into();
        source.pages[0].lines[1].confidence = 0.999;
        let report = evaluate(&truth(), &source)?;
        assert_eq!(report.overall.missing, 1);
        assert_eq!(report.confidence_curve.len(), 2);
        let strict = &report.confidence_curve[0];
        assert_eq!(strict.minimum_confidence, 0.999);
        assert_eq!(strict.admitted, 1);
        assert_eq!(strict.admitted_normalized_exact, 0);
        assert_eq!(strict.admitted_errors, 1);
        assert!((strict.reference_coverage - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(strict.empirical_error_rate, 1.0);

        let permissive = &report.confidence_curve[1];
        assert_eq!(permissive.minimum_confidence, 0.9);
        assert_eq!(permissive.admitted, 2);
        assert_eq!(permissive.admitted_normalized_exact, 1);
        assert_eq!(permissive.admitted_errors, 1);
        assert_eq!(permissive.empirical_error_rate, 0.5);

        source.pages[0].lines[0].confidence = 0.999;
        let ties = evaluate(&truth(), &source)?;
        assert_eq!(ties.confidence_curve.len(), 1);
        assert_eq!(ties.confidence_curve[0].admitted, 2);
        assert_eq!(ties.confidence_curve[0].admitted_errors, 1);
        Ok(())
    }

    #[test]
    fn every_unicode_confusion_is_counted_with_source_confidence() {
        let make = |reference: &str, predicted: Option<&str>, confidence| QualitySample {
            page: 1,
            center: Point(1.0, 1.0),
            kind: "technical".into(),
            reference: reference.into(),
            predicted: predicted.map(str::to_owned),
            line_index: None,
            line_quad: None,
            confidence,
            candidate_count: 1,
            exact: false,
            normalized_exact: false,
            character_edits: 0,
            normalized_character_edits: 0,
        };
        let samples = vec![
            make("Ø20.0", Some("020.0"), Some(0.999)),
            make("Ø23.0", Some("023.0"), Some(0.998)),
            make("温度", Some("温"), Some(0.8)),
            make("A", Some("AB"), Some(0.5)),
            make("A", None, None),
            make("℃", Some("°C"), Some(0.99)),
        ];
        let errors = character_confusions(&samples);
        assert_eq!(errors.iter().map(|error| error.count).sum::<usize>(), 5);
        let diameter = &errors[0];
        assert_eq!(
            (diameter.reference, diameter.predicted),
            (Some('Ø'), Some('0'))
        );
        assert_eq!(diameter.count, 2);
        assert_eq!(diameter.highest_line_confidence, Some(0.999));
        assert!(
            errors
                .iter()
                .any(|error| { error.reference == Some('度') && error.predicted.is_none() })
        );
        assert!(
            errors
                .iter()
                .any(|error| { error.reference.is_none() && error.predicted == Some('B') })
        );
        assert!(errors.iter().any(|error| {
            error.reference == Some('A')
                && error.predicted.is_none()
                && error.highest_line_confidence.is_none()
        }));
    }

    #[test]
    fn refuses_mismatched_or_malformed_sources() {
        let mut manual = truth();
        manual.source_pdf_sha256 = "0".repeat(64);
        assert!(evaluate(&manual, &draft()).is_err());
        let mut manual = truth();
        manual.samples.push(manual.samples[0].clone());
        assert!(evaluate(&manual, &draft()).is_err());
        let mut source = draft();
        source.pages.push(source.pages[0].clone());
        assert!(evaluate(&truth(), &source).is_err());
        source.pages.pop();
        source.pages[0].lines[0].confidence = f64::NAN;
        assert!(evaluate(&truth(), &source).is_err());
    }
}
