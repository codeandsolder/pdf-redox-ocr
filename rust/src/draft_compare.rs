//! Compare alternative recognition drafts without treating either as ground truth.
//!
//! Spatially connected bipartite components preserve split/merge ambiguities:
//! neither model's line order determines the association or the result.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    Error,
    disjoint_set::DisjointSet,
    draft::{DraftPage, OcrDraft, OcrLine},
    geometry::{Point, Quad, QuadBounds},
    polygon::overlap_fraction,
    quality_eval::{character_edits, normalized},
};

/// Spatial/text comparison category, independent of confidence admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LineAgreement {
    Exact,
    Normalized,
    DifferentText,
    SegmentationDifference,
    BaselineOnly,
    CandidateOnly,
}

/// Original line identity within the unreordered input page.
#[derive(Debug, Serialize)]
pub struct DraftObservation<'a> {
    pub index: usize,
    pub line: &'a OcrLine,
}

/// All observations associated by original PDF-space overlap.
#[derive(Debug, Serialize)]
pub struct SpatialGroup<'a> {
    pub agreement: LineAgreement,
    pub baseline: Vec<DraftObservation<'a>>,
    pub candidate: Vec<DraftObservation<'a>>,
    /// Only one-to-one groups have an unambiguous line-level edit distance.
    pub character_edits: Option<usize>,
    pub normalized_character_edits: Option<usize>,
    pub overlap_of_smaller_quad: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct PageComparison<'a> {
    pub page: u32,
    pub baseline_lines: usize,
    pub candidate_lines: usize,
    pub baseline_threshold_qualified: usize,
    pub candidate_threshold_qualified: usize,
    pub baseline_native_words: usize,
    pub candidate_native_words: usize,
    /// Maximum vertex-coordinate difference in original native-word order;
    /// absent when the number of native words differs.
    pub native_ordered_max_coordinate_difference: Option<f64>,
    pub counts: BTreeMap<LineAgreement, usize>,
    pub groups: Vec<SpatialGroup<'a>>,
}

/// Source-bound disagreement report, not a measured ground-truth error rate.
#[derive(Debug, Serialize)]
pub struct DraftComparison<'a> {
    pub schema_version: u32,
    pub pdf_sha256: &'a str,
    pub baseline_model: &'a str,
    pub candidate_model: &'a str,
    pub minimum_overlap: f64,
    pub confidence_threshold: f64,
    pub counts: BTreeMap<LineAgreement, usize>,
    pub pages: Vec<PageComparison<'a>>,
}

fn pages(draft: &OcrDraft) -> Result<BTreeMap<u32, &DraftPage>, Error> {
    let mut indexed = BTreeMap::new();
    for page in &draft.pages {
        if page.page == 0 || indexed.insert(page.page, page).is_some() {
            return Err(Error::Invalid("invalid or duplicate draft page".into()));
        }
        let finite = |quad: &Quad| {
            quad.iter()
                .all(|Point(x, y)| x.is_finite() && y.is_finite())
        };
        if page
            .lines
            .iter()
            .any(|line| !(0.0..=1.0).contains(&line.confidence) || !finite(&line.quad))
            || page.native_text_regions.iter().any(|quad| !finite(quad))
        {
            return Err(Error::Invalid(
                "invalid comparison draft coordinates or confidence".into(),
            ));
        }
    }
    Ok(indexed)
}

fn components<'a>(
    baseline: &'a [OcrLine],
    candidate: &'a [OcrLine],
    minimum_overlap: f64,
) -> Vec<SpatialGroup<'a>> {
    let mut components = DisjointSet::new(baseline.len() + candidate.len());
    let bounds = candidate
        .iter()
        .map(|line| QuadBounds::new(&line.quad))
        .collect::<Vec<_>>();
    for (left, line) in baseline.iter().enumerate() {
        let region = QuadBounds::new(&line.quad);
        for (right, other) in candidate.iter().enumerate() {
            if region.intersects(bounds[right])
                && overlap_fraction(&line.quad, &other.quad) >= minimum_overlap
            {
                components.connect(left, baseline.len() + right);
            }
        }
    }
    let mut grouped = BTreeMap::<usize, (Vec<_>, Vec<_>)>::new();
    for (index, line) in baseline.iter().enumerate() {
        grouped
            .entry(components.root(index))
            .or_default()
            .0
            .push(DraftObservation { index, line });
    }
    for (index, line) in candidate.iter().enumerate() {
        grouped
            .entry(components.root(baseline.len() + index))
            .or_default()
            .1
            .push(DraftObservation { index, line });
    }
    grouped
        .into_values()
        .map(|(left, right)| classify(left, right))
        .collect()
}

fn classify<'a>(
    baseline: Vec<DraftObservation<'a>>,
    candidate: Vec<DraftObservation<'a>>,
) -> SpatialGroup<'a> {
    let mut edits = None;
    let mut normalized_edits = None;
    let mut overlap = None;
    let agreement = match (baseline.as_slice(), candidate.as_slice()) {
        ([], _) => LineAgreement::CandidateOnly,
        (_, []) => LineAgreement::BaselineOnly,
        ([a], [b]) => {
            let left = &a.line.text;
            let right = &b.line.text;
            let raw = character_edits(left, right);
            let canonical = character_edits(&normalized(left), &normalized(right));
            edits = Some(raw);
            normalized_edits = Some(canonical);
            overlap = Some(overlap_fraction(&a.line.quad, &b.line.quad));
            if raw == 0 {
                LineAgreement::Exact
            } else if canonical == 0 {
                LineAgreement::Normalized
            } else {
                LineAgreement::DifferentText
            }
        }
        _ => LineAgreement::SegmentationDifference,
    };
    SpatialGroup {
        agreement,
        baseline,
        candidate,
        character_edits: edits,
        normalized_character_edits: normalized_edits,
        overlap_of_smaller_quad: overlap,
    }
}

fn native_difference(baseline: &[Quad], candidate: &[Quad]) -> Option<f64> {
    (baseline.len() == candidate.len()).then(|| {
        baseline
            .iter()
            .zip(candidate)
            .flat_map(|(a, b)| a.iter().zip(b))
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
            .fold(0.0_f64, f64::max)
    })
}

/// Compare two source-bound drafts in their common original PDF coordinate frame.
///
/// Overlap is intersection divided by the smaller polygon's area, so a split
/// line and its merged counterpart can be associated without relying on text.
/// Every input observation appears exactly once. Different segmentation is
/// reported explicitly, not counted as a recognition error.
///
/// # Errors
/// Rejects unequal source identities, page sets, invalid thresholds or input
/// coordinates/confidence. The caller must verify the source PDF independently.
pub fn compare<'a>(
    baseline: &'a OcrDraft,
    candidate: &'a OcrDraft,
    minimum_overlap: f64,
    confidence_threshold: f64,
) -> Result<DraftComparison<'a>, Error> {
    if baseline.schema_version != 3
        || candidate.schema_version != 3
        || baseline.pdf_sha256 != candidate.pdf_sha256
        || !(0.0..=1.0).contains(&minimum_overlap)
        || minimum_overlap == 0.0
        || !(0.0..=1.0).contains(&confidence_threshold)
    {
        return Err(Error::Invalid(
            "incompatible draft source or comparison policy".into(),
        ));
    }
    let left = pages(baseline)?;
    let right = pages(candidate)?;
    if left.keys().ne(right.keys()) {
        return Err(Error::Invalid(
            "comparison requires the same selected PDF pages".into(),
        ));
    }
    let mut counts = BTreeMap::new();
    let mut output = Vec::with_capacity(left.len());
    let qualified = |lines: &[OcrLine]| {
        lines
            .iter()
            .filter(|line| line.overlay_qualified(confidence_threshold))
            .count()
    };
    for (page, a) in left {
        let b = right[&page];
        let groups = components(&a.lines, &b.lines, minimum_overlap);
        let mut page_counts = BTreeMap::new();
        for group in &groups {
            *page_counts.entry(group.agreement).or_insert(0) += 1;
            *counts.entry(group.agreement).or_insert(0) += 1;
        }
        output.push(PageComparison {
            page,
            baseline_lines: a.lines.len(),
            candidate_lines: b.lines.len(),
            baseline_threshold_qualified: qualified(&a.lines),
            candidate_threshold_qualified: qualified(&b.lines),
            baseline_native_words: a.native_text_regions.len(),
            candidate_native_words: b.native_text_regions.len(),
            native_ordered_max_coordinate_difference: native_difference(
                &a.native_text_regions,
                &b.native_text_regions,
            ),
            counts: page_counts,
            groups,
        });
    }
    Ok(DraftComparison {
        schema_version: 1,
        pdf_sha256: &baseline.pdf_sha256,
        baseline_model: &baseline.model,
        candidate_model: &candidate.model,
        minimum_overlap,
        confidence_threshold,
        counts,
        pages: output,
    })
}

#[cfg(test)]
mod tests {
    use super::{LineAgreement, compare};
    use crate::{
        Error,
        draft::{DraftPage, OcrDraft, OcrLine},
        geometry::Point,
    };

    fn line(text: &str, x: f64, y: f64, w: f64) -> OcrLine {
        OcrLine {
            text: text.into(),
            confidence: 0.99,
            quad: [
                Point(x, y),
                Point(x + w, y),
                Point(x + w, y + 10.0),
                Point(x, y + 10.0),
            ],
            reviewed: false,
        }
    }

    fn draft(lines: Vec<OcrLine>) -> OcrDraft {
        OcrDraft::new(
            "same-source".into(),
            "fixture".into(),
            vec![DraftPage {
                page: 1,
                lines,
                native_text_regions: vec![],
            }],
        )
    }

    #[test]
    fn spatial_groups_preserve_reordering_splits_and_unmatched_lines() -> Result<(), Error> {
        let baseline = draft(vec![
            line("EXACT", 0.0, 0.0, 100.0),
            line("A B", 0.0, 20.0, 100.0),
            line("C", 0.0, 40.0, 100.0),
            line("SPLIT LABEL", 0.0, 60.0, 100.0),
            line("MISSING", 0.0, 80.0, 100.0),
        ]);
        let candidate = draft(vec![
            line("LABEL", 50.0, 60.0, 50.0),
            line("EXACT", 0.0, 0.0, 100.0),
            line("EXTRA", 0.0, 100.0, 100.0),
            line("ＡＢ", 0.0, 20.0, 100.0),
            line("SPLIT", 0.0, 60.0, 50.0),
            line("D", 0.0, 40.0, 100.0),
        ]);
        let report = compare(&baseline, &candidate, 0.5, 0.97)?;
        for agreement in [
            LineAgreement::Exact,
            LineAgreement::Normalized,
            LineAgreement::DifferentText,
            LineAgreement::SegmentationDifference,
            LineAgreement::BaselineOnly,
            LineAgreement::CandidateOnly,
        ] {
            assert_eq!(report.counts[&agreement], 1);
        }
        let groups = &report.pages[0].groups;
        assert_eq!(groups[0].candidate[0].index, 1);
        assert_eq!(groups[2].character_edits, Some(1));
        assert_eq!(groups[3].candidate.len(), 2);
        assert_eq!(groups[3].character_edits, None);
        assert_eq!(groups.iter().map(|g| g.baseline.len()).sum::<usize>(), 5);
        assert_eq!(groups.iter().map(|g| g.candidate.len()).sum::<usize>(), 6);
        Ok(())
    }

    #[test]
    fn different_source_or_page_partition_is_rejected() {
        let baseline = draft(vec![]);
        let mut other = baseline.clone();
        other.pdf_sha256 = "different".into();
        assert!(compare(&baseline, &other, 0.5, 0.97).is_err());
        other = baseline.clone();
        other.pages[0].page = 2;
        assert!(compare(&baseline, &other, 0.5, 0.97).is_err());
        assert!(compare(&baseline, &baseline, 0.0, 0.97).is_err());
    }
}
