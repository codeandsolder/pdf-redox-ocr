//! Complete structure-aware OCR orchestration, independent of inference and CLI I/O.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::{
    Error, PageNumber,
    draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
    geometry::Quad,
    hybrid::{
        MergeInput, MergeProposal, MergeStats, PageRoute, masked_source_conflicts, merge_ocr,
    },
    mixed_visibility::{MixedVerifier, MixedVisibility, PageVerification},
    mosaic::{
        MosaicBudget, MosaicPlan, contains_native_fragments, mosaic_page_eligible,
        plan_raster_mosaics,
    },
    mosaic_ocr::reconstruct_native_mosaics,
    native_font_mask::mask_native_fonts,
    native_images::{NativeImageBudget, native_source_images},
    native_words::native_words_for_pages,
    page_selection::parse_page_selection,
    paint_scan::{PaintReport, enumerate_image_paints},
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    recognition::RecognitionBackend,
    source_ocr::{
        SourceObservation, SourceOnlyResult, source_image_mapper, source_only_candidates,
        source_only_pages,
    },
    source_selection::{ImagePlacement, SourcePolicy, choose_mixed_raster_sources},
};

/// Independent source raster, mosaic and fallback policy for an OCR job.
#[derive(Clone, Copy, Debug)]
pub struct PipelinePolicy {
    pub source_budget: NativeImageBudget,
    pub mosaic_budget: MosaicBudget,
    pub render: RenderPolicy,
    pub minimum_confidence: f64,
    pub native_chunk: usize,
    pub keep_native_text: bool,
    /// Reuse verified source-masked rasters instead of ordinary full-page input.
    pub residual_ocr: bool,
}

impl Default for PipelinePolicy {
    fn default() -> Self {
        Self {
            source_budget: NativeImageBudget::default(),
            mosaic_budget: MosaicBudget::default(),
            render: RenderPolicy::default(),
            minimum_confidence: 0.97,
            native_chunk: 24,
            keep_native_text: false,
            residual_ocr: true,
        }
    }
}

impl PipelinePolicy {
    fn validate(self) -> Result<Self, Error> {
        self.source_budget.validate()?;
        self.mosaic_budget.validate()?;
        self.render.validate()?;
        if !(0.0..=1.0).contains(&self.minimum_confidence) || self.native_chunk == 0 {
            return Err(Error::Invalid("invalid OCR pipeline policy".into()));
        }
        Ok(self)
    }
}

/// Finished, source-bound draft and source-admission evidence.
#[derive(Debug)]
pub struct PipelineResult {
    pub draft: OcrDraft,
    pub stats: MergeStats,
    /// The actual recognition path taken on every selected page.
    pub routes: BTreeMap<PageNumber, PageRoute>,
    pub proposals: Vec<MergeProposal>,
    /// Original per-paint differential evidence, including occluded paints.
    pub visibility: MixedVisibility,
    /// Fallback pages recognized from verified residual rasters.
    pub residual_pages: BTreeSet<PageNumber>,
}

/// Immutable source metadata and native word exclusion for a validated OCR request.
///
/// Parsing and input validation complete before a model needs to be initialized.
/// Every recognition verifies that the source bytes still match this snapshot.
pub struct PreparedOcr {
    pdf: PathBuf,
    doc: PdfMetadata,
    selected: Vec<PageNumber>,
    natives: BTreeMap<PageNumber, Vec<Quad>>,
    digest: String,
    policy: PipelinePolicy,
}

struct NativePreparation {
    source_only: SourceOnlyResult,
    scanned: PaintReport,
    recognized: BTreeMap<String, SourceObservation>,
    routes: BTreeMap<PageNumber, PageRoute>,
    fallback_pages: Vec<PageNumber>,
}

#[derive(Default)]
struct RecognizedPages {
    raster: Vec<DraftPage>,
    fallback: Vec<DraftPage>,
    visibility: MixedVisibility,
    residual_pages: BTreeSet<PageNumber>,
}

type FragmentPlans = (BTreeSet<PageNumber>, BTreeMap<PageNumber, MosaicPlan>);

/// Extract only from pages known to paint a wanted source. Keep all such
/// occurrences in document order: a later repetition can still decode when
/// an earlier page exceeds the extraction budget or has an unsupported codec.
struct NativeSourcePlan {
    ids: BTreeSet<String>,
    pages: Vec<PageNumber>,
}

impl NativeSourcePlan {
    fn from_paints(ids: BTreeSet<String>, scanned: &PaintReport) -> Result<Self, Error> {
        let pages = scanned
            .placements
            .iter()
            .filter(|paint| ids.contains(&paint.source_id))
            .map(|paint| PageNumber::new(paint.page))
            .collect::<Result<BTreeSet<_>, _>>()?
            .into_iter()
            .collect();
        Ok(Self { ids, pages })
    }
}

fn model_provenance(profile: &str, stage: &str, render: RenderPolicy) -> String {
    format!(
        "{profile} {stage} page-raster-max-side={} page-raster-dpi={}",
        render.max_side, render.dpi
    )
}

impl PreparedOcr {
    /// Validate policies, bind the source PDF, and extract native word exclusions.
    ///
    /// # Errors
    /// Returns an error for invalid policies, PDF geometry, page selection, native
    /// text extraction, or a source that changes during preparation.
    pub fn open(pdf: &Path, pages: Option<&str>, policy: PipelinePolicy) -> Result<Self, Error> {
        let policy = policy.validate()?;
        let pdf = std::fs::canonicalize(pdf)?;
        let doc = PdfMetadata::open(&pdf)?;
        let selected = parse_page_selection(pages, doc.page_count())?;
        if selected.is_empty() {
            return Err(Error::Invalid("empty OCR source".into()));
        }
        let digest = hex::encode(Sha256::digest(doc.original_bytes()));
        let transforms = doc.select_mappers(&selected)?;
        let natives = native_words_for_pages(&pdf, &transforms, policy.native_chunk)?;
        let result = Self {
            pdf,
            doc,
            selected,
            natives,
            digest,
            policy,
        };
        result.verify_source()?;
        Ok(result)
    }

    /// Exact original bytes for a subsequent semantic overlay.
    #[must_use]
    pub fn original_bytes(&self) -> &[u8] {
        self.doc.original_bytes()
    }

    fn verify_source(&self) -> Result<(), Error> {
        if sha256_file(&self.pdf)? != self.digest {
            return Err(Error::Invalid("OCR source PDF changed".into()));
        }
        Ok(())
    }

    /// Execute the native source, mosaic, visibility and full-page paths.
    /// Progress messages are delivered to the caller, not printed by the library.
    ///
    /// # Errors
    /// Returns an error for modified inputs, invalid source evidence, or failure
    /// of the required full-page recognizer or renderer.
    pub fn recognize(
        &self,
        engine: &impl RecognitionBackend,
        mut progress: impl FnMut(String),
    ) -> Result<PipelineResult, Error> {
        self.verify_source()?;
        let prepared = self.prepare_native_sources(engine, &mut progress)?;
        let temporary_mask = self.prepare_raster_pdf(&prepared.fallback_pages, &mut progress)?;
        let raster_pdf = temporary_mask
            .as_ref()
            .map_or(self.pdf.as_path(), NamedTempFile::path);
        let RecognizedPages {
            raster,
            fallback,
            visibility,
            residual_pages,
        } = self.recognize_pages(engine, &prepared, raster_pdf, &mut progress)?;
        let policy = self.policy.render;
        let raster_draft = OcrDraft::new(
            self.digest.clone(),
            model_provenance(engine.model_profile(), "native-source", policy),
            raster,
        );
        let fallback_draft = OcrDraft::new(
            self.digest.clone(),
            model_provenance(
                engine.model_profile(),
                if residual_pages.is_empty() {
                    "page-fallback"
                } else {
                    "page-fallback-with-verified-residuals"
                },
                policy,
            ),
            fallback,
        );
        let (draft, stats, proposals) = merge_ocr(MergeInput {
            pdf: &self.pdf,
            selected: &self.selected,
            raster: &raster_draft,
            image_observations: &visibility.observations,
            baseline: &fallback_draft,
            routes: &prepared.routes,
            proven_pages: &visibility.proven_pages,
            visibility_proofs: &visibility.proofs,
            threshold: self.policy.minimum_confidence,
        })?;
        progress(format!(
            "OCR merge: {} native-only pages, {} source-only pages, {} fallback pages, {} verified native labels added",
            stats.native_only_pages,
            stats.image_source_only_pages,
            stats.page_fallback_pages,
            stats.nonoverlapping_image_lines_added,
        ));
        progress(format!(
            "verified residual rasters used on {} pages",
            residual_pages.len()
        ));
        self.verify_source()?;
        Ok(PipelineResult {
            draft,
            stats,
            routes: prepared.routes,
            proposals,
            visibility,
            residual_pages,
        })
    }

    fn recognize_pages(
        &self,
        engine: &impl RecognitionBackend,
        prepared: &NativePreparation,
        raster_pdf: &Path,
        progress: &mut dyn FnMut(String),
    ) -> Result<RecognizedPages, Error> {
        let policy = self.policy.render;
        let mut result = RecognizedPages {
            raster: Vec::with_capacity(self.selected.len()),
            ..RecognizedPages::default()
        };
        let mut inspector = MixedVerifier::new(
            &self.pdf,
            &self.doc,
            &prepared.scanned,
            &prepared.recognized,
            policy,
            self.policy.minimum_confidence,
        )?;
        for &page in &self.selected {
            let native_text_regions = self.natives.get(&page).cloned().ok_or_else(|| {
                Error::Invalid("Poppler native text omitted selected page".into())
            })?;
            let native_lines = prepared
                .source_only
                .pages
                .get(&page)
                .cloned()
                .unwrap_or_default();
            result.raster.push(DraftPage {
                page: page.get(),
                lines: native_lines,
                native_text_regions: native_text_regions.clone(),
            });
            if prepared.routes[&page] == PageRoute::PageFallback {
                let page_evidence = match inspector.verify_page(
                    page,
                    self.policy
                        .residual_ocr
                        .then_some(native_text_regions.as_slice()),
                ) {
                    Ok(evidence) => evidence,
                    Err(error) => {
                        progress(format!(
                            "page {} verification unavailable: {error}; using page OCR",
                            page.get()
                        ));
                        PageVerification::default()
                    }
                };
                let (lines, residual, evidence) = self.recognize_fallback(
                    engine,
                    raster_pdf,
                    page,
                    &native_text_regions,
                    page_evidence,
                    progress,
                )?;
                if residual {
                    result.residual_pages.insert(page);
                }
                result.visibility.extend(evidence);
                progress(format!(
                    "page {} (fallback): {} OCR lines, {} native words",
                    page.get(),
                    lines.len(),
                    native_text_regions.len()
                ));
                result.fallback.push(DraftPage {
                    page: page.get(),
                    lines,
                    native_text_regions,
                });
            }
        }
        Ok(result)
    }

    fn recognize_fallback(
        &self,
        engine: &impl RecognitionBackend,
        raster_pdf: &Path,
        page: PageNumber,
        native: &[Quad],
        verified: PageVerification,
        progress: &mut dyn FnMut(String),
    ) -> Result<(Vec<OcrLine>, bool, MixedVisibility), Error> {
        let PageVerification {
            evidence,
            control,
            residual,
        } = verified;
        let (ordinary, alternative) = match control {
            Some(control) if raster_pdf == self.pdf => (control, residual),
            Some(control) => {
                let baseline = render_rgb_page(raster_pdf, page, self.policy.render)?;
                let alternative = residual.and_then(|r| r.transfer(&control, &baseline));
                (baseline, alternative)
            }
            None => (render_rgb_page(raster_pdf, page, self.policy.render)?, None),
        };
        let mapper = self.doc.mapper(page)?;
        if let Some(alternative) = alternative {
            let crate::mixed_visibility::ResidualRaster {
                image,
                required,
                paint_count,
            } = alternative;
            let mut candidate = DraftPage {
                page: page.get(),
                lines: Vec::new(),
                native_text_regions: native.to_vec(),
            };
            let conflicts = |draft: &DraftPage| {
                masked_source_conflicts(
                    draft,
                    &evidence.observations,
                    &evidence.proofs,
                    &required,
                    self.policy.minimum_confidence,
                )
            };
            // First check the source/native exclusions without spending inference.
            if conflicts(&candidate)?.is_empty() {
                match engine.recognize_page(image, mapper) {
                    Ok(lines) => {
                        candidate.lines = lines;
                        let rejected = conflicts(&candidate)?;
                        if rejected.is_empty() {
                            progress(format!(
                                "page {}: verified residual OCR accepted ({} paints, {} source labels)",
                                page.get(),
                                paint_count,
                                required.len()
                            ));
                            return Ok((candidate.lines, true, evidence));
                        }
                        for (index, reason) in rejected {
                            let source = &evidence.observations[index];
                            let conflicting = candidate
                                .lines
                                .iter()
                                .filter(|line| {
                                    crate::polygon::overlap_fraction(&source.line.quad, &line.quad)
                                        >= 0.30
                                })
                                .map(|line| line.text.as_str())
                                .collect::<Vec<_>>();
                            progress(format!(
                                "page {} residual source {:?} rejected: page overlap {}, native overlap {}, residual lines {:?}",
                                page.get(),
                                reason.raster_text,
                                reason.page_overlap,
                                reason.native_overlap,
                                conflicting
                            ));
                        }
                    }
                    Err(error) => progress(format!(
                        "page {} residual recognition failed: {error}",
                        page.get()
                    )),
                }
            }
            progress(format!(
                "page {}: residual merge conflict; recognizing ordinary page",
                page.get()
            ));
        }
        Ok((engine.recognize_page(ordinary, mapper)?, false, evidence))
    }

    fn plan_native_fragment_pages(&self, scanned: &PaintReport) -> Result<FragmentPlans, Error> {
        let doc = &self.doc;
        let budget = self.policy.mosaic_budget;
        let mut fragmented = BTreeSet::new();
        let mut plans = BTreeMap::new();
        for summary in &scanned.pages {
            let page = PageNumber::new(summary.page)?;
            let paints = scanned
                .placements
                .iter()
                .filter(|p| p.page == summary.page)
                .cloned()
                .collect::<Vec<_>>();
            if !mosaic_page_eligible(summary, &paints) {
                continue;
            }
            let box_ = *doc
                .media_boxes()
                .get(&page)
                .ok_or_else(|| Error::Invalid("selected page missing MediaBox".into()))?;
            let mapper = doc.mapper(page)?;
            if !contains_native_fragments(&paints, box_, mapper) {
                continue;
            }
            fragmented.insert(page);
            if summary.native_searchable_shows == 0
                && let Some(plan) = plan_raster_mosaics(page, &paints, box_, mapper, budget)?
            {
                plans.insert(page, plan);
            }
        }
        Ok((fragmented, plans))
    }

    fn plan_native_sources(
        &self,
        scanned: &PaintReport,
        fragmented: &BTreeSet<PageNumber>,
        progress: &mut dyn FnMut(String),
    ) -> Result<NativeSourcePlan, Error> {
        let doc = &self.doc;
        let source_budget = self.policy.source_budget;
        let render = self.policy.render;
        let mut source_ids = source_only_candidates(scanned, source_budget.max_source_pixels);
        let sampling = scanned
            .placements
            .iter()
            .map(|paint| ImagePlacement {
                page: paint.page,
                source_id: paint.source_id.clone(),
                width: paint.width,
                height: paint.height,
                ctm: paint.ctm,
                unsafe_reasons: paint
                    .unsafe_reasons
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let (mixed, report) = choose_mixed_raster_sources(
            &sampling,
            doc.media_boxes(),
            SourcePolicy {
                max_side: render.max_side,
                max_inference_pixels: source_budget.max_total_pixels,
                max_sources: source_budget.max_sources,
                minimum_source_pixels: 20_000,
                min_gain: 2.25,
            },
        )?;
        progress(format!(
            "source candidates: {} high-detail sources, {} native-only sources",
            report.native_resolution_candidates,
            source_ids.len()
        ));
        source_ids.extend(mixed);
        // Native source OCR is insufficient on a page where a text line can cross
        // neighboring fragments; recognize the complete native-pitch group instead.
        let eligible_elsewhere = scanned
            .placements
            .iter()
            .filter(|paint| !fragmented.iter().any(|page| page.get() == paint.page))
            .map(|paint| paint.source_id.as_str())
            .collect::<BTreeSet<_>>();
        source_ids.retain(|id| eligible_elsewhere.contains(id.as_str()));
        let plan = NativeSourcePlan::from_paints(source_ids, scanned)?;
        progress(format!(
            "native source extraction planned: {} sources across {} candidate pages",
            plan.ids.len(),
            plan.pages.len()
        ));
        Ok(plan)
    }

    fn recognize_native_sources(
        &self,
        engine: &impl RecognitionBackend,
        plan: &NativeSourcePlan,
        progress: &mut dyn FnMut(String),
    ) -> Result<BTreeMap<String, SourceObservation>, Error> {
        let decoded = match native_source_images(
            &self.pdf,
            &plan.pages,
            &plan.ids,
            self.policy.source_budget,
        ) {
            Ok(images) => images,
            Err(error) => {
                progress(format!(
                    "Native source extraction unavailable: {error}; using page OCR"
                ));
                BTreeMap::new()
            }
        };
        let mut recognized = BTreeMap::new();
        for (source_id, rgb) in decoded {
            let (width, height) = (rgb.width(), rgb.height());
            let mapper = source_image_mapper(width, height)?;
            match engine.recognize_page(rgb, mapper) {
                Ok(lines) => {
                    progress(format!(
                        "source {source_id}: {} OCR observations",
                        lines.len()
                    ));
                    recognized.insert(
                        source_id,
                        SourceObservation {
                            width,
                            height,
                            lines,
                        },
                    );
                }
                Err(error) => progress(format!("source {source_id}: {error}; using page OCR")),
            }
        }
        Ok(recognized)
    }

    fn add_reconstructed_mosaics(
        &self,
        engine: &impl RecognitionBackend,
        scanned: &PaintReport,
        plans: &BTreeMap<PageNumber, MosaicPlan>,
        source_only: &mut SourceOnlyResult,
        progress: &mut dyn FnMut(String),
    ) -> Result<usize, Error> {
        let mut count = 0;
        for (&page, plan) in plans {
            let paints = scanned
                .placements
                .iter()
                .filter(|paint| paint.page == page.get())
                .cloned()
                .collect::<Vec<_>>();
            match reconstruct_native_mosaics(
                &self.pdf,
                self.doc.mapper(page)?,
                plan,
                &paints,
                self.policy.mosaic_budget,
                |rgb, mapper| engine.recognize_mapped(rgb, mapper),
            ) {
                Ok(Some(lines)) => {
                    progress(format!(
                        "page {} native mosaic: {} OCR lines from {} groups",
                        page.get(),
                        lines.len(),
                        plan.groups.len()
                    ));
                    source_only.approved_pages.insert(page);
                    source_only.pages.insert(page, lines);
                    count += 1;
                }
                Ok(None) => progress(format!(
                    "page {} native mosaic incomplete; using page OCR",
                    page.get()
                )),
                Err(error) => progress(format!(
                    "page {} native mosaic rejected: {error}; using page OCR",
                    page.get()
                )),
            }
        }
        Ok(count)
    }

    /// Decode and recognize original source images, source-resolution fragment
    /// mosaics, and verify exact mixed-page source paint visibility.
    fn prepare_native_sources(
        &self,
        engine: &impl RecognitionBackend,
        progress: &mut dyn FnMut(String),
    ) -> Result<NativePreparation, Error> {
        let scanned = enumerate_image_paints(&self.doc, &self.selected)?;
        let (fragmented, plans) = self.plan_native_fragment_pages(&scanned)?;
        let source_plan = self.plan_native_sources(&scanned, &fragmented, progress)?;
        let recognized = self.recognize_native_sources(engine, &source_plan, progress)?;
        let mut source_only = source_only_pages(&scanned, &recognized)?;
        for &page in &fragmented {
            source_only.approved_pages.remove(&page);
            source_only.pages.remove(&page);
        }
        let mosaics =
            self.add_reconstructed_mosaics(engine, &scanned, &plans, &mut source_only, progress)?;
        progress(format!(
            "native mosaics: {} fragment pages, {} reconstructed",
            fragmented.len(),
            mosaics
        ));
        progress(format!(
            "source-only fast path approved {} of {} selected pages",
            source_only.approved_pages.len(),
            self.selected.len()
        ));
        // A native-only page needs neither the font mask nor a page render.
        // Retain its word exclusion regions even when it has no OCR output.
        let routes = scanned
            .pages
            .iter()
            .map(|summary| {
                let page = PageNumber::new(summary.page)?;
                let words = self
                    .natives
                    .get(&page)
                    .ok_or_else(|| Error::Invalid("missing native words".into()))?;
                let route = if source_only.approved_pages.contains(&page) {
                    PageRoute::RasterOnly
                } else if !self.policy.keep_native_text
                    && summary.native_text_only()
                    && (summary.native_searchable_shows == 0 || !words.is_empty())
                {
                    PageRoute::NativeOnly
                } else {
                    PageRoute::PageFallback
                };
                Ok((page, route))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        let fallback_selected = self
            .selected
            .iter()
            .copied()
            .filter(|page| routes.get(page) == Some(&PageRoute::PageFallback))
            .collect::<Vec<_>>();
        progress(format!(
            "native-text-only fast path approved {} of {} selected pages",
            routes
                .values()
                .filter(|&&route| route == PageRoute::NativeOnly)
                .count(),
            self.selected.len()
        ));
        Ok(NativePreparation {
            source_only,
            scanned,
            recognized,
            routes,
            fallback_pages: fallback_selected,
        })
    }

    /// Build an ephemeral font-suppressed PDF for page-only OCR rendering.
    fn prepare_raster_pdf(
        &self,
        fallback_pages: &[PageNumber],
        progress: &mut dyn FnMut(String),
    ) -> Result<Option<NamedTempFile>, Error> {
        if self.policy.keep_native_text || fallback_pages.is_empty() {
            return Ok(None);
        }
        let mask = mask_native_fonts(&self.doc, fallback_pages)?;
        progress(format!(
            "native-font render mask: {} searchable shows hidden, {} Type3/unclassified preserved, {} clipping preserved, {} Forms cloned",
            mask.stats.hidden_native_shows,
            mask.stats.preserved_nonsearchable_shows,
            mask.stats.preserved_clipping_shows,
            mask.stats.cloned_forms,
        ));
        mask.bytes
            .map(|bytes| -> Result<NamedTempFile, Error> {
                let mut file = NamedTempFile::new()?;
                file.write_all(&bytes)?;
                file.flush()?;
                Ok(file)
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::NativeSourcePlan;
    use crate::{PageNumber, paint_scan::enumerate_image_paints, pdf_metadata::PdfMetadata};
    use std::collections::BTreeSet;

    #[test]
    fn source_plan_retains_all_requested_repetitions_without_unrelated_pages()
    -> Result<(), crate::Error> {
        let doc = PdfMetadata::from_bytes(
            include_bytes!("../tests/fixtures/repeated-image-six-pages.pdf").to_vec(),
        )?;
        let selected = [6, 3, 1]
            .map(PageNumber::new)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let scan = enumerate_image_paints(&doc, &selected)?;
        let plan = NativeSourcePlan::from_paints(BTreeSet::from(["9:0".into()]), &scan)?;
        assert_eq!(
            plan.pages.iter().map(|p| p.get()).collect::<Vec<_>>(),
            [1, 3, 6]
        );
        assert_eq!(plan.ids, BTreeSet::from(["9:0".to_owned()]));
        let empty = NativeSourcePlan::from_paints(BTreeSet::new(), &scan)?;
        assert_eq!(empty.pages, [] as [PageNumber; 0]);
        Ok(())
    }
}
