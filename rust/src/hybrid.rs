//! Provenance-preserving merge of independent native-image and page OCR.
//!
//! No native-image observation may enter a mixed page unless a matching
//! source-and-paint-specific differential rendering has proven visibility.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    Error, PageNumber,
    draft::{DraftPage, OcrDraft, OcrLine},
    geometry::Quad,
    polygon::overlap_fraction,
};

/// Independent native-raster OCR observation and its paint identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageObservation {
    /// One-based page number.
    pub page: u32,
    /// Required original source object identifier.
    pub source_id: String,
    /// Required zero-based expanded PDF paint invocation ordinal.
    pub paint_index: u32,
    /// OCR line in the page's PDF user-space frame.
    pub line: OcrLine,
}

/// Result of an isolated render-visibility comparison for one source paint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisibilityProof {
    /// One-based page number.
    pub page: u32,
    /// Exact source-object identity.
    pub source_id: String,
    /// Exact expanded paint invocation.
    pub paint_index: u32,
    /// Recognized text, bound to a specific visible polygon.
    pub text: String,
    /// Recognized PDF-space polygon.
    pub quad: Quad,
    /// Whether source-specific removal changed the rendered final pixels.
    pub visible: bool,
}

/// Reports whether a raster page can safely skip ordinary page OCR.
#[derive(Clone, Debug)]
pub struct RasterPageEvidence {
    /// One-based page number.
    pub page: u32,
    /// All visual paint on the page is eligible standalone raster paint.
    pub image_only_after_native: bool,
    /// Source image paint operations.
    pub paints: Vec<ImagePaintEvidence>,
    /// Number of nonempty native-image OCR lines.
    pub recognized_lines: usize,
}

/// Image source paint proof for standalone raster pages.
#[derive(Clone, Debug)]
pub struct ImagePaintEvidence {
    /// Bounding quad of the expanded image paint.
    pub page_quad: Quad,
    /// The source was decoded and recognized at native resolution.
    pub recognized: bool,
}

/// Account of why a raster page must use page-level fallback.
#[derive(Clone, Debug, Serialize)]
pub struct RasterPageDecision {
    /// Page index.
    pub page: u32,
    /// Eligibility of a pure native-image result.
    pub approved: bool,
    /// All reasons ordinary page OCR is needed.
    pub fallback_reasons: Vec<&'static str>,
}

/// Determine which pages are safe for source-image-only OCR.
///
/// An opaque image-only page is not automatically safe: every source must
/// have been recognized, and overlapping images require compositing.
#[must_use]
pub fn approve_raster_only_pages(pages: &[RasterPageEvidence]) -> Vec<RasterPageDecision> {
    pages
        .iter()
        .map(|page| {
            let mut reasons = Vec::new();
            if !page.image_only_after_native {
                reasons.push("mixed_pdf_visual_content_or_unsupported_paint");
            }
            if page.paints.is_empty() {
                reasons.push("no_extracted_image_placements");
            }
            if page.paints.iter().any(|paint| !paint.recognized) {
                reasons.push("unrecognized_or_unsupported_raster_placement");
            }
            if page.recognized_lines == 0 {
                reasons.push("no_positive_raster_text_evidence");
            }
            if page.paints.iter().enumerate().any(|(index, paint)| {
                page.paints[index + 1..]
                    .iter()
                    .any(|other| overlap_fraction(&paint.page_quad, &other.page_quad) > 1e-9)
            }) {
                reasons.push("overlapping_image_layers_require_page_compositing");
            }
            RasterPageDecision {
                page: page.page,
                approved: reasons.is_empty(),
                fallback_reasons: reasons,
            }
        })
        .collect()
}

/// A per-observation merge admission audit.
#[derive(Clone, Debug, Serialize)]
pub struct MergeProposal {
    /// Target page.
    pub page: u32,
    /// Candidate source text.
    pub raster_text: String,
    /// Required candidate source image identity.
    pub source_id: String,
    /// Expanded paint invocation identity.
    pub paint_index: u32,
    /// Raw recognition confidence.
    pub raster_confidence: f64,
    /// Maximum page OCR overlap fraction.
    pub page_overlap: f64,
    /// Maximum overlap with existing native searchable text.
    pub native_overlap: f64,
    /// Source-specific visibility verified.
    pub verified_visible_source: bool,
    /// Whether the candidate enters the importer draft.
    pub included_as_extra: bool,
}

/// Summary of the source/page merge.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct MergeStats {
    /// Pages where source OCR replaces full page OCR.
    pub image_source_only_pages: usize,
    /// OCR lines emitted directly from source-only pages.
    pub image_source_lines: usize,
    /// Pages that require ordinary full-page OCR.
    pub page_fallback_pages: usize,
    /// Original full-page recognized lines.
    pub original_page_lines: usize,
    /// Admitted additional source-image lines.
    pub nonoverlapping_image_lines_added: usize,
    /// Image candidates rejected but kept in the audit.
    pub raster_candidates_kept_in_audit_only: usize,
}

/// Index by exact page, source object, paint invocation, and Unicode text.
/// Geometry remains a second-stage match to preserve fractional-pixel proofs.
type ProofIndex<'a> = BTreeMap<(u32, &'a str, u32, &'a str), Vec<&'a VisibilityProof>>;

fn proof_geometry_matches(first: &Quad, second: &Quad) -> bool {
    first.iter().zip(second).all(|(a, b)| {
        ((a.0 * 10_000.0).round_ties_even() - (b.0 * 10_000.0).round_ties_even()).abs() < 1e-9
            && ((a.1 * 10_000.0).round_ties_even() - (b.1 * 10_000.0).round_ties_even()).abs()
                < 1e-9
    })
}

fn proof_matches(image: &ImageObservation, proof: &VisibilityProof) -> bool {
    proof.visible
        && !image.source_id.is_empty()
        && proof.page == image.page
        && proof.source_id == image.source_id
        && proof.paint_index == image.paint_index
        && proof.text == image.line.text
        && proof_geometry_matches(&proof.quad, &image.line.quad)
}

/// Original candidate order is preserved within each page.
type ObservationIndex<'a> = BTreeMap<PageNumber, Vec<&'a ImageObservation>>;

/// Validate admission evidence once and construct narrow lookups for merging.
fn index_merge_evidence<'a>(
    observations: &'a [ImageObservation],
    visibility_proofs: &'a [VisibilityProof],
    selected_pages: &BTreeSet<PageNumber>,
) -> Result<(ObservationIndex<'a>, ProofIndex<'a>), Error> {
    let mut observations_by_page = ObservationIndex::new();
    for image in observations {
        let page = PageNumber::new(image.page)?;
        if !selected_pages.contains(&page) {
            return Err(Error::Invalid(
                "image observation for unselected page".into(),
            ));
        }
        if image.source_id.is_empty() {
            return Err(Error::Invalid(
                "image observation missing source identity".into(),
            ));
        }
        observations_by_page.entry(page).or_default().push(image);
    }
    let mut proofs = ProofIndex::new();
    for proof in visibility_proofs {
        if !selected_pages.contains(&PageNumber::new(proof.page)?) || proof.source_id.is_empty() {
            return Err(Error::Invalid(
                "visibility proof has invalid page or source identity".into(),
            ));
        }
        if proof.visible {
            proofs
                .entry((
                    proof.page,
                    proof.source_id.as_str(),
                    proof.paint_index,
                    proof.text.as_str(),
                ))
                .or_default()
                .push(proof);
        }
    }
    Ok((observations_by_page, proofs))
}

/// Exact inputs and source-specific provenance for one hybrid OCR merge.
#[derive(Debug, Clone, Copy)]
pub struct MergeInput<'a> {
    /// Original immutable input PDF.
    pub pdf: &'a Path,
    /// Selected one-based pages in caller-requested order.
    pub selected: &'a [PageNumber],
    /// Native-raster OCR draft for all selected pages.
    pub raster: &'a OcrDraft,
    /// Source-bound observations from each painted image.
    pub image_observations: &'a [ImageObservation],
    /// Full-page OCR draft for all pages not approved as raster-only.
    pub baseline: &'a OcrDraft,
    /// Pages proven to contain only independent, visible raster paint.
    pub raster_only: &'a BTreeSet<PageNumber>,
    /// Pages whose differential mask-verification completed.
    pub proven_pages: &'a BTreeSet<PageNumber>,
    /// Source and paint-specific visibility observations.
    pub visibility_proofs: &'a [VisibilityProof],
    /// Minimum confidence required for a *supplementary* image line.
    pub threshold: f64,
}

/// Merge raster-only pages and eligible visible image labels into page OCR.
///
/// The verifier passes *only* proofs from pages for which the source-masked
/// render comparison completed. Source and paint IDs cannot be dropped.
/// No optimizer/importer text layer may be created from an unverified line.
///
/// # Errors
/// Returns an error for a changed PDF digest, missing pages, duplicate page
/// observations, differing native text exclusions, or invalid thresholds.
pub fn merge_ocr(
    input: MergeInput<'_>,
) -> Result<(OcrDraft, MergeStats, Vec<MergeProposal>), Error> {
    let MergeInput {
        pdf,
        selected,
        raster,
        image_observations,
        baseline,
        raster_only,
        proven_pages,

        threshold,
        ..
    } = input;
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Error::Invalid(
            "invalid additional raster confidence threshold".into(),
        ));
    }
    raster.verify_pdf(pdf)?;
    baseline.verify_pdf(pdf)?;
    if raster.pdf_sha256 != baseline.pdf_sha256 {
        return Err(Error::Invalid(
            "page fallback was run on a different PDF".into(),
        ));
    }
    let selected_pages = selected.iter().copied().collect::<BTreeSet<_>>();
    if selected_pages.len() != selected.len()
        || selected_pages.is_empty()
        || !raster_only.is_subset(&selected_pages)
        || !proven_pages.is_subset(&selected_pages)
    {
        return Err(Error::Invalid("invalid selected-page partition".into()));
    }
    let raster_map = pages_by_number(&raster.pages)?;
    let baseline_map = pages_by_number(&baseline.pages)?;
    if raster_map.keys().copied().collect::<BTreeSet<_>>() != selected_pages
        || baseline_map.keys().copied().collect::<BTreeSet<_>>()
            != selected_pages.difference(raster_only).copied().collect()
    {
        return Err(Error::Invalid(
            "raster/page fallback coverage incomplete".into(),
        ));
    }
    let (observations_by_page, proofs) =
        index_merge_evidence(image_observations, input.visibility_proofs, &selected_pages)?;
    let mut merged_pages = Vec::with_capacity(selected.len());
    let mut stats = MergeStats::default();
    let mut proposals = Vec::new();
    for page in selected {
        let raster_page = raster_map
            .get(page)
            .ok_or_else(|| Error::Invalid(format!("missing raster page {}", page.get())))?;
        if raster_only.contains(page) {
            merged_pages.push((*raster_page).clone());
            stats.image_source_only_pages += 1;
            stats.image_source_lines += raster_page.lines.len();
            continue;
        }
        let baseline_page = baseline_map
            .get(page)
            .ok_or_else(|| Error::Invalid(format!("missing fallback page {}", page.get())))?;
        if baseline_page.native_text_regions != raster_page.native_text_regions {
            return Err(Error::Invalid(
                "inconsistent source native text coordinates".into(),
            ));
        }
        let (output_page, page_stats, page_proposals) = merge_fallback_page(
            &input,
            *page,
            baseline_page,
            observations_by_page.get(page).map_or(&[], Vec::as_slice),
            &proofs,
        );
        stats.page_fallback_pages += 1;
        stats.original_page_lines += page_stats.original_page_lines;
        stats.nonoverlapping_image_lines_added += page_stats.nonoverlapping_image_lines_added;
        stats.raster_candidates_kept_in_audit_only +=
            page_stats.raster_candidates_kept_in_audit_only;
        proposals.extend(page_proposals);
        merged_pages.push(output_page);
    }
    Ok((
        OcrDraft::new(
            raster.pdf_sha256.clone(),
            format!("{}; fallback {}", raster.model, baseline.model),
            merged_pages,
        ),
        stats,
        proposals,
    ))
}

fn merge_fallback_page(
    input: &MergeInput<'_>,
    page: PageNumber,
    baseline_page: &DraftPage,
    observations: &[&ImageObservation],
    proofs: &ProofIndex<'_>,
) -> (DraftPage, MergeStats, Vec<MergeProposal>) {
    let mut output_page = (*baseline_page).clone();
    let mut stats = MergeStats {
        original_page_lines: baseline_page.lines.len(),
        ..MergeStats::default()
    };
    let mut proposals = Vec::new();
    for &image in observations {
        // Include lines already admitted from other native-image paints:
        // two individually visible sources can still overlap at page level.
        let page_overlap = output_page
            .lines
            .iter()
            .map(|line| overlap_fraction(&image.line.quad, &line.quad))
            .fold(0.0_f64, f64::max);
        let native_overlap = baseline_page
            .native_text_regions
            .iter()
            .map(|quad| overlap_fraction(&image.line.quad, quad))
            .fold(0.0_f64, f64::max);
        let visible = input.proven_pages.contains(&page)
            && proofs
                .get(&(
                    image.page,
                    image.source_id.as_str(),
                    image.paint_index,
                    image.line.text.as_str(),
                ))
                .is_some_and(|candidates| {
                    candidates.iter().any(|proof| proof_matches(image, proof))
                });
        let eligible = visible
            && page_overlap < 0.30
            && native_overlap < 0.20
            && image.line.confidence >= input.threshold
            && image.line.text.trim().chars().count() >= 2;
        proposals.push(MergeProposal {
            page: page.get(),
            raster_text: image.line.text.clone(),
            source_id: image.source_id.clone(),
            paint_index: image.paint_index,
            raster_confidence: image.line.confidence,
            page_overlap: round4(page_overlap),
            native_overlap: round4(native_overlap),
            verified_visible_source: visible,
            included_as_extra: eligible,
        });
        if eligible {
            output_page.lines.push(image.line.clone());
            stats.nonoverlapping_image_lines_added += 1;
        } else {
            stats.raster_candidates_kept_in_audit_only += 1;
        }
    }
    (output_page, stats, proposals)
}

fn pages_by_number(pages: &[DraftPage]) -> Result<BTreeMap<PageNumber, &DraftPage>, Error> {
    let mut output = BTreeMap::new();
    for page in pages {
        if output.insert(PageNumber::new(page.page)?, page).is_some() {
            return Err(Error::Invalid("duplicate OCR page".into()));
        }
    }
    Ok(output)
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round_ties_even() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::{
        ImageObservation, ImagePaintEvidence, MergeInput, RasterPageEvidence, VisibilityProof,
        approve_raster_only_pages, merge_ocr,
    };
    use crate::{
        PageNumber,
        draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
        geometry::Point,
    };
    use std::{collections::BTreeSet, fs, path::PathBuf};

    fn rect(x: f64, y: f64, width: f64, height: f64) -> [Point; 4] {
        [
            Point(x, y),
            Point(x + width, y),
            Point(x + width, y + height),
            Point(x, y + height),
        ]
    }

    fn line(text: &str, quad: [Point; 4]) -> OcrLine {
        OcrLine {
            text: text.into(),
            confidence: 0.99,
            quad,
            reviewed: false,
        }
    }

    #[test]
    fn raster_only_rejects_overlapping_paints() {
        let paints = vec![
            ImagePaintEvidence {
                page_quad: rect(10.0, 10.0, 100.0, 100.0),
                recognized: true,
            },
            ImagePaintEvidence {
                page_quad: rect(10.0, 10.0, 100.0, 100.0),
                recognized: true,
            },
        ];
        let decisions = approve_raster_only_pages(&[RasterPageEvidence {
            page: 1,
            image_only_after_native: true,
            paints,
            recognized_lines: 2,
        }]);
        assert!(!decisions[0].approved);
        assert!(
            decisions[0]
                .fallback_reasons
                .contains(&"overlapping_image_layers_require_page_compositing")
        );
    }

    fn merge_fixture(digest: String) -> (OcrDraft, OcrDraft, Vec<ImageObservation>) {
        let first_page = DraftPage {
            page: 1,
            lines: vec![line("CAPACITOR 47uF", rect(20.0, 20.0, 120.0, 18.0))],
            native_text_regions: vec![],
        };
        let second_page = DraftPage {
            page: 2,
            lines: vec![
                line("New image text", rect(20.0, 20.0, 100.0, 18.0)),
                line("Possible duplicate", rect(40.0, 80.0, 90.0, 12.0)),
            ],
            native_text_regions: vec![],
        };
        let raster = OcrDraft::new(
            digest.clone(),
            "source".into(),
            vec![first_page, second_page.clone()],
        );
        let baseline = OcrDraft::new(
            digest,
            "page".into(),
            vec![DraftPage {
                page: 2,
                lines: vec![line("Existing text", rect(40.0, 80.0, 90.0, 12.0))],
                native_text_regions: vec![],
            }],
        );
        let observations = second_page
            .lines
            .iter()
            .map(|observation| ImageObservation {
                page: 2,
                source_id: "8:0".into(),
                paint_index: 1,
                line: observation.clone(),
            })
            .collect::<Vec<_>>();
        (raster, baseline, observations)
    }

    #[test]
    fn independently_verified_overlapping_source_labels_are_deduplicated()
    -> Result<(), Box<dyn std::error::Error>> {
        let sample = tempfile::NamedTempFile::new()?;
        fs::write(
            sample.path(),
            b"distinct source identity, same visible OCR region",
        )?;
        let (raster, baseline, observations) = merge_fixture(sha256_file(sample.path())?);
        let original = observations[0].clone();
        let mut other = original.clone();
        other.source_id = "37:0".into();
        other.paint_index = 12;
        let candidates = [original, other];
        let proofs = candidates
            .iter()
            .map(|candidate| VisibilityProof {
                page: candidate.page,
                source_id: candidate.source_id.clone(),
                paint_index: candidate.paint_index,
                text: candidate.line.text.clone(),
                quad: candidate.line.quad,
                visible: true,
            })
            .collect::<Vec<_>>();
        let selected = [PageNumber::new(1)?, PageNumber::new(2)?];
        let (merged, stats, audit) = merge_ocr(MergeInput {
            pdf: sample.path(),
            selected: &selected,
            raster: &raster,
            image_observations: &candidates,
            baseline: &baseline,
            raster_only: &BTreeSet::from([selected[0]]),
            proven_pages: &BTreeSet::from([selected[1]]),
            visibility_proofs: &proofs,
            threshold: 0.97,
        })?;
        assert_eq!(stats.nonoverlapping_image_lines_added, 1);
        assert_eq!(stats.raster_candidates_kept_in_audit_only, 1);
        assert_eq!(audit.len(), 2);
        assert!(audit[0].included_as_extra);
        assert!(!audit[1].included_as_extra);
        assert_eq!(merged.pages[1].lines.len(), 2);
        Ok(())
    }

    #[test]
    fn source_proof_gates_mixed_page_merge() -> Result<(), Box<dyn std::error::Error>> {
        let path: PathBuf = std::env::temp_dir().join(format!(
            "pdf-redox-ocr-rust-merge-{}-{}.pdf",
            std::process::id(),
            line!()
        ));
        fs::write(&path, b"pdf sample bound to exact digest")?;
        let result = (|| {
            let (raster, baseline, observations) = merge_fixture(sha256_file(&path)?);
            let selected = [PageNumber::new(1)?, PageNumber::new(2)?];
            let only = BTreeSet::from([selected[0]]);
            let (ignored, stats, _) = merge_ocr(MergeInput {
                pdf: &path,
                selected: &selected,
                raster: &raster,
                image_observations: &observations,
                baseline: &baseline,
                raster_only: &only,
                proven_pages: &BTreeSet::new(),
                visibility_proofs: &[],
                threshold: 0.97,
            })?;
            assert_eq!(ignored.pages[1].lines.len(), 1);
            assert_eq!(stats.nonoverlapping_image_lines_added, 0);
            let proof = observations
                .iter()
                .map(|item| VisibilityProof {
                    page: item.page,
                    source_id: item.source_id.clone(),
                    paint_index: item.paint_index,
                    text: item.line.text.clone(),
                    quad: item.line.quad,
                    visible: true,
                })
                .collect::<Vec<_>>();
            let (merged, stats, audit) = merge_ocr(MergeInput {
                pdf: &path,
                selected: &selected,
                raster: &raster,
                image_observations: &observations,
                baseline: &baseline,
                raster_only: &only,
                proven_pages: &BTreeSet::from([selected[1]]),
                visibility_proofs: &proof,
                threshold: 0.97,
            })?;
            assert_eq!(stats.image_source_only_pages, 1);
            assert_eq!(stats.page_fallback_pages, 1);
            assert_eq!(stats.nonoverlapping_image_lines_added, 1);
            assert_eq!(stats.raster_candidates_kept_in_audit_only, 1);
            assert_eq!(audit.len(), 2);
            assert_eq!(merged.pages[1].lines.len(), 2);
            assert_eq!(merged.pages[1].lines[1].text, "New image text");
            let mut wrong_source_proof = proof;
            wrong_source_proof[0].source_id = "9:0".into();
            let (rejected, _, _) = merge_ocr(MergeInput {
                pdf: &path,
                selected: &selected,
                raster: &raster,
                image_observations: &observations,
                baseline: &baseline,
                raster_only: &only,
                proven_pages: &BTreeSet::from([selected[1]]),
                visibility_proofs: &wrong_source_proof,
                threshold: 0.97,
            })?;
            assert_eq!(rejected.pages[1].lines.len(), 1);
            Ok::<(), Box<dyn std::error::Error>>(())
        })();
        fs::remove_file(path)?;
        result
    }

    #[test]
    fn missing_or_mismatched_paint_provenance_never_admits_text()
    -> Result<(), Box<dyn std::error::Error>> {
        let sample = tempfile::NamedTempFile::new()?;
        fs::write(sample.path(), b"test image paint evidence identity")?;
        let (raster, baseline, observations) = merge_fixture(sha256_file(sample.path())?);
        let selected = [PageNumber::new(1)?, PageNumber::new(2)?];
        let source_only = BTreeSet::from([selected[0]]);
        let proven_pages = BTreeSet::from([selected[1]]);
        let original = observations[0].clone();
        let proof = VisibilityProof {
            page: original.page,
            source_id: original.source_id.clone(),
            paint_index: original.paint_index,
            text: original.line.text.clone(),
            quad: original.line.quad,
            visible: true,
        };
        let check = |observations: &[ImageObservation], proofs: &[VisibilityProof]| {
            merge_ocr(MergeInput {
                pdf: sample.path(),
                selected: &selected,
                raster: &raster,
                image_observations: observations,
                baseline: &baseline,
                raster_only: &source_only,
                proven_pages: &proven_pages,
                visibility_proofs: proofs,
                threshold: 0.97,
            })
        };
        let mut unknown = original.clone();
        unknown.source_id.clear();
        assert!(check(&[unknown], std::slice::from_ref(&proof)).is_err());
        let mut unbound_proof = proof.clone();
        unbound_proof.source_id.clear();
        assert!(check(std::slice::from_ref(&original), &[unbound_proof]).is_err());
        let mut wrong_paint = proof.clone();
        wrong_paint.paint_index += 1;
        let (unchanged, stats, audit) = check(std::slice::from_ref(&original), &[wrong_paint])?;
        assert_eq!(unchanged.pages[1].lines.len(), 1);
        assert_eq!(stats.nonoverlapping_image_lines_added, 0);
        assert!(!audit[0].verified_visible_source);

        let mut wrong_geometry = proof.clone();
        wrong_geometry.quad[0].0 += 0.001;
        let mut wrong_text = proof.clone();
        wrong_text.text.push('!');
        let mut invisible = proof.clone();
        invisible.visible = false;
        for rejected_proof in [wrong_geometry, wrong_text, invisible] {
            let (rejected, stats, audit) =
                check(std::slice::from_ref(&original), &[rejected_proof])?;
            assert_eq!(rejected.pages[1].lines.len(), 1);
            assert_eq!(stats.nonoverlapping_image_lines_added, 0);
            assert!(!audit[0].verified_visible_source);
        }

        let (accepted, stats, audit) = check(&[original], &[proof])?;
        assert_eq!(accepted.pages[1].lines.len(), 2);
        assert_eq!(stats.nonoverlapping_image_lines_added, 1);
        assert!(audit[0].verified_visible_source);
        Ok(())
    }
}
