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
    catalog_evidence::CatalogEvidence,
    direct_images::{DirectDecodeStats, direct_native_images},
    draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
    geometry::{DisplayMapper, Point, Quad},
    hybrid::{
        MergeInput, MergeProposal, MergeStats, PageRoute, masked_source_conflicts, merge_ocr,
    },
    input_preparation::PreparationAudit,
    mixed_visibility::{MixedVerifier, MixedVisibility, PageVerification},
    mosaic::{
        MosaicBudget, MosaicPlan, mosaic_page_eligible, plan_raster_mosaics,
        requires_composited_recognition,
    },
    mosaic_ocr::reconstruct_native_mosaics,
    native_font_mask::mask_native_fonts,
    native_images::{NativeImageBudget, native_source_images},
    native_words::native_words_for_pages,
    page_selection::parse_page_selection,
    paint_scan::{PaintReport, PdfMatrix, enumerate_image_paints},
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    recognition::{
        RasterCharacterAudit, RasterCharacterMapping, RasterMappingScope, RasterRepairAudit,
        RecognitionBackend, RecognitionCacheBudget, RecognitionSession, RecognitionStats,
    },
    region_ocr::{RegionAudit, RegionPass},
    source_appearances::{PaintAppearances, SourceAppearance},
    source_ocr::{
        SourceAdmissions, SourceLine, SourceObservation, SourceOnlyResult, source_image_mapper,
        source_only_candidates, source_only_pages,
    },
    source_selection::{ImagePlacement, SourcePolicy, choose_mixed_raster_sources},
};

/// Native raster recognition strategy. Safety proofs and resource limits
/// apply identically to every mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum RasterMode {
    /// Select high-detail sources and reconstruct compositor-dependent image pages.
    #[default]
    Auto,
    /// Consider every independently safe source within the configured budget.
    All,
    /// Disable native image and mosaic OCR; retain native-text-only bypass.
    Off,
}

/// Source image extraction backend. The interpreter path is initially
/// explicit so source pixels and OCR quality can be compared independently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum NativeImageDecoder {
    #[default]
    Poppler,
    Hayro,
}

/// Independent source raster, mosaic and fallback policy for an OCR job.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PipelinePolicy {
    pub source_budget: NativeImageBudget,
    pub native_image_decoder: NativeImageDecoder,
    pub raster_mode: RasterMode,
    pub mosaic_budget: MosaicBudget,
    pub render: RenderPolicy,
    pub minimum_confidence: f64,
    pub native_chunk: usize,
    pub keep_native_text: bool,
    /// Reuse verified source-masked rasters instead of ordinary full-page input.
    pub residual_ocr: bool,
    /// Job-local cache limits for unmapped model observations.
    pub recognition_cache: RecognitionCacheBudget,
}

impl Default for PipelinePolicy {
    fn default() -> Self {
        Self {
            source_budget: NativeImageBudget::default(),
            native_image_decoder: NativeImageDecoder::default(),
            raster_mode: RasterMode::Auto,
            mosaic_budget: MosaicBudget::default(),
            render: RenderPolicy::default(),
            minimum_confidence: 0.97,
            native_chunk: 24,
            keep_native_text: false,
            residual_ocr: true,
            recognition_cache: RecognitionCacheBudget::default(),
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

/// Exact native RGB input, independently of OCR model output.
#[derive(Debug, serde::Serialize)]
pub struct SourceRasterIdentity {
    pub width: u32,
    pub height: u32,
    pub rgb_sha256: String,
}

impl SourceRasterIdentity {
    fn measure(rgb: &image::RgbImage) -> Self {
        Self {
            width: rgb.width(),
            height: rgb.height(),
            rgb_sha256: hex::encode(Sha256::digest(rgb.as_raw())),
        }
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
    /// Completed residual alternatives, including rejected inference evidence.
    pub residual_attempts: Vec<ResidualAttempt>,
    /// Model calls and reuse across all raster recognition paths.
    pub recognition: RecognitionStats,
    pub crop_repairs: Vec<RasterRepairAudit>,
    pub character_audits: Vec<RasterCharacterAudit>,
    pub raster_character_mappings: Vec<RasterCharacterMapping>,
    pub catalog_evidence: Option<CatalogEvidence>,
    /// Validated policy used for this complete recognition job.
    pub policy: PipelinePolicy,
    /// Actual source image paint closure and native-only blockers.
    pub paints: PaintReport,
    /// Native image OCR before mapping, visibility proof or merge admission.
    pub source_observations: PaintAppearances<SourceObservation>,
    pub direct_decode: Option<DirectDecodeStats>,
    pub source_rasters: PaintAppearances<SourceRasterIdentity>,
    /// Actual high-detail composited crop inputs and supplementary admission decisions.
    pub regional_ocr: Vec<RegionAudit>,
}

/// The disposition of a fully verified source-suppression alternative.
#[derive(Debug, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ResidualOutcome {
    /// Native font masking changed pixels needed for exact treatment transfer.
    TransferRejected,
    /// At least one required source would fail admission even without page OCR.
    PreflightRejected { conflicts: Vec<ResidualConflict> },
    /// The backend could not recognize the treatment.
    InferenceFailed { error: String },
    /// The treatment produced conflicting labels. Keep the actual observations
    /// for diagnosis, not for automatic importer use.
    MergeRejected {
        conflicts: Vec<ResidualConflict>,
        lines: Vec<OcrLine>,
    },
    /// The treated page passed the same source admission checks as the final merge.
    Accepted,
}

/// A required source observation that cannot survive the attempted page merge.
#[derive(Debug, serde::Serialize)]
pub struct ResidualConflict {
    /// Index in this page's visibility observations (not the global array).
    pub observation_index: usize,
    pub proposal: MergeProposal,
}

/// Evidence for one attempted residual recognition route.
#[derive(Debug, serde::Serialize)]
pub struct ResidualAttempt {
    pub page: PageNumber,
    pub paint_count: usize,
    /// Indices in `visibility.observations` filtered to this page, in original order.
    pub required: BTreeSet<usize>,
    pub outcome: ResidualOutcome,
}

impl ResidualAttempt {
    const fn accepted(&self) -> bool {
        matches!(self.outcome, ResidualOutcome::Accepted)
    }
}

/// The exact raster alternative about to be submitted for recognition.
///
/// Both images are in the same final page display frame, including native
/// font masking. Borrowed data is valid only during the observer callback.
pub struct ResidualInput<'a> {
    pub page: PageNumber,
    pub ordinary: &'a image::RgbImage,
    pub treated: &'a image::RgbImage,
    pub required: &'a BTreeSet<usize>,
}

/// Observe pipeline progress and optionally inspect actual recognition input.
///
/// Implementations can persist explicit diagnostic captures without retaining
/// raster buffers, altering inference inputs, or inserting a second renderer.
pub trait PipelineObserver {
    fn message(&mut self, message: String);

    /// Observe the candidate after source/native preflight and before inference.
    ///
    /// # Errors
    /// Returns an error if the observer cannot complete its requested capture.
    fn residual_input(&mut self, _input: ResidualInput<'_>) -> Result<(), Error> {
        Ok(())
    }
}

impl<F: FnMut(String)> PipelineObserver for F {
    fn message(&mut self, message: String) {
        self(message);
    }
}

struct ResidualWork<'a> {
    ordinary: &'a image::RgbImage,
    treated: image::RgbImage,
    mapper: DisplayMapper,
    native: &'a [Quad],
    evidence: &'a MixedVisibility,
    threshold: f64,
}

impl ResidualAttempt {
    fn recognize(
        &mut self,
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        work: ResidualWork<'_>,
        observer: &mut impl PipelineObserver,
    ) -> Result<Option<Vec<OcrLine>>, Error> {
        let mut candidate = DraftPage {
            page: self.page.get(),
            lines: Vec::new(),
            native_text_regions: work.native.to_vec(),
        };
        let conflicts = |draft: &DraftPage| {
            masked_source_conflicts(
                draft,
                &work.evidence.observations,
                &work.evidence.proofs,
                &self.required,
                work.threshold,
            )
            .map(|conflicts| {
                conflicts
                    .into_iter()
                    .map(|(observation_index, proposal)| ResidualConflict {
                        observation_index,
                        proposal,
                    })
                    .collect::<Vec<_>>()
            })
        };
        let preflight = conflicts(&candidate)?;
        if !preflight.is_empty() {
            self.outcome = ResidualOutcome::PreflightRejected {
                conflicts: preflight,
            };
            return Ok(None);
        }
        observer.residual_input(ResidualInput {
            page: self.page,
            ordinary: work.ordinary,
            treated: &work.treated,
            required: &self.required,
        })?;
        match engine.recognize_document_page(work.treated, work.mapper, self.page, true) {
            Ok(lines) => {
                candidate.lines = lines;
                let rejected = conflicts(&candidate)?;
                if rejected.is_empty() {
                    observer.message(format!(
                        "page {}: verified residual OCR accepted ({} paints, {} source labels)",
                        self.page.get(),
                        self.paint_count,
                        self.required.len()
                    ));
                    self.outcome = ResidualOutcome::Accepted;
                    return Ok(Some(candidate.lines));
                }
                for conflict in &rejected {
                    observer.message(format!(
                        "page {} residual source {:?} rejected: page overlap {:?}, native overlap {:?}",
                        self.page.get(),
                        conflict.proposal.raster_text,
                        conflict.proposal.page_overlap,
                        conflict.proposal.native_overlap,
                    ));
                }
                self.outcome = ResidualOutcome::MergeRejected {
                    conflicts: rejected,
                    lines: candidate.lines,
                };
            }
            Err(error) => {
                self.outcome = ResidualOutcome::InferenceFailed {
                    error: error.to_string(),
                };
            }
        }
        Ok(None)
    }
}

/// Versioned diagnostic evidence separate from the importer draft.
///
/// Geometry in `source_observations` is in unrotated native image pixels
/// with bottom-left origin. Paint geometry and visibility are in PDF user
/// space. Diagnostics cannot authorize an overlay.
#[derive(serde::Serialize)]
pub struct PipelineAudit<'a> {
    schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    preparation: Option<&'a PreparationAudit>,
    pdf_sha256: &'a str,
    model: &'a str,
    policy: PipelinePolicy,
    recognition: RecognitionStats,
    crop_repairs: &'a [RasterRepairAudit],
    character_audits: &'a [RasterCharacterAudit],
    raster_character_mappings: &'a [RasterCharacterMapping],
    catalog_evidence: Option<&'a CatalogEvidence>,
    routes: &'a BTreeMap<PageNumber, PageRoute>,
    residual_pages: &'a BTreeSet<PageNumber>,
    residual_attempts: &'a [ResidualAttempt],
    merge: MergeStats,
    source_observations: &'a PaintAppearances<SourceObservation>,
    direct_decode: Option<&'a DirectDecodeStats>,
    source_rasters: &'a PaintAppearances<SourceRasterIdentity>,
    paints: &'a PaintReport,
    visibility: &'a MixedVisibility,
    proposals: &'a [MergeProposal],
    regional_ocr: &'a [RegionAudit],
}

impl<'a> PipelineAudit<'a> {
    /// Attach verified upstream input preparation provenance.
    ///
    /// # Errors
    /// Rejects a preparation with a different OCR-input digest.
    pub fn with_preparation(mut self, preparation: &'a PreparationAudit) -> Result<Self, Error> {
        if preparation.ocr_input_sha256 != self.pdf_sha256 {
            return Err(Error::Invalid("audit preparation digest mismatch".into()));
        }
        self.preparation = Some(preparation);
        Ok(self)
    }
}

impl PipelineResult {
    /// Associate primary character evidence with exact optimizer templates.
    /// No changes are made to the importer draft.
    ///
    /// # Errors
    /// Rejects mismatched provenance or malformed catalog placements.
    pub fn associate_catalog(&mut self, catalog: &pdf_redox::GlyphCatalog) -> Result<(), Error> {
        self.catalog_evidence = Some(CatalogEvidence::from_audits(
            catalog,
            &self.draft.pdf_sha256,
            &self.character_audits,
            &self.raster_character_mappings,
            &self.residual_pages,
            self.policy.minimum_confidence,
        )?);
        Ok(())
    }

    /// Borrow the complete source-bound diagnostic report without copying evidence.
    #[must_use = "inspect pipeline diagnostics before accepting the processed document"]
    pub const fn audit(&self) -> PipelineAudit<'_> {
        PipelineAudit {
            schema_version: 22,
            preparation: None,
            pdf_sha256: self.draft.pdf_sha256.as_str(),
            model: self.draft.model.as_str(),
            policy: self.policy,
            recognition: self.recognition,
            crop_repairs: self.crop_repairs.as_slice(),
            character_audits: self.character_audits.as_slice(),
            raster_character_mappings: self.raster_character_mappings.as_slice(),
            catalog_evidence: self.catalog_evidence.as_ref(),
            routes: &self.routes,
            residual_pages: &self.residual_pages,
            residual_attempts: self.residual_attempts.as_slice(),
            merge: self.stats,
            source_observations: &self.source_observations,
            direct_decode: self.direct_decode.as_ref(),
            source_rasters: &self.source_rasters,
            paints: &self.paints,
            visibility: &self.visibility,
            proposals: self.proposals.as_slice(),
            regional_ocr: self.regional_ocr.as_slice(),
        }
    }
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
    recognized: PaintAppearances<SourceObservation>,
    direct_decode: Option<DirectDecodeStats>,
    source_rasters: PaintAppearances<SourceRasterIdentity>,
    routes: BTreeMap<PageNumber, PageRoute>,
    fallback_pages: Vec<PageNumber>,
}

fn record_source_mappings(
    engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
    scanned: &PaintReport,
    recognized: &PaintAppearances<SourceObservation>,
    identities: &PaintAppearances<SourceRasterIdentity>,
    admitted: &SourceAdmissions,
) -> Result<(), Error> {
    for paint in &scanned.placements {
        let page = PageNumber::new(paint.page)?;
        let Some(detections) = admitted.get(&(page, paint.paint_index)) else {
            continue;
        };
        if detections.is_empty() {
            continue;
        }
        let (appearance, source) = recognized
            .get(paint)
            .ok_or_else(|| Error::Invalid("admitted source has no recognition".into()))?;
        let (identity_id, identity) = identities
            .get(paint)
            .ok_or_else(|| Error::Invalid("admitted source has no raster identity".into()))?;
        let dimensions = (identity.width, identity.height);
        let source_dimensions = (source.width, source.height);
        if identity_id != appearance
            || dimensions != source_dimensions
            || !detections.iter().all(|index| {
                source
                    .lines
                    .iter()
                    .any(|line| line.detection_index == *index)
            })
        {
            return Err(Error::Invalid(
                "inconsistent admitted source observation".into(),
            ));
        }
        let transform = PdfMatrix(paint.ctm);
        engine.record_character_mapping(RasterCharacterMapping {
            page,
            rgb_sha256: identity.rgb_sha256.clone(),
            dimensions,
            scope: RasterMappingScope::NativeImage {
                paint_index: paint.paint_index,
                admitted_detection_indices: detections.clone(),
            },
            raster_corners: [Point(0., 1.), Point(1., 1.), Point(1., 0.), Point(0., 0.)]
                .map(|point| transform.point(point)),
        });
    }
    Ok(())
}

impl NativePreparation {
    fn record_mixed_source_mappings(
        &self,
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        proposals: &[MergeProposal],
    ) -> Result<(), Error> {
        let mut admitted = SourceAdmissions::new();
        for proposal in proposals
            .iter()
            .filter(|proposal| proposal.included_as_extra)
        {
            admitted
                .entry((PageNumber::new(proposal.page)?, proposal.paint_index))
                .or_default()
                .insert(proposal.detection_index);
        }
        record_source_mappings(
            engine,
            &self.scanned,
            &self.recognized,
            &self.source_rasters,
            &admitted,
        )
    }
}

#[derive(Default)]
struct RecognizedPages {
    raster: Vec<DraftPage>,
    fallback: Vec<DraftPage>,
    visibility: MixedVisibility,
    residual_attempts: Vec<ResidualAttempt>,
}

struct FallbackRecognition {
    lines: Vec<OcrLine>,
    evidence: MixedVisibility,
    attempt: Option<ResidualAttempt>,
}

type CompositedPlans = (BTreeSet<PageNumber>, BTreeMap<PageNumber, MosaicPlan>);

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

fn report_routes(
    observer: &mut impl PipelineObserver,
    stats: &MergeStats,
    regional_ocr: &[RegionAudit],
    residual_page_count: usize,
) {
    observer.message(format!(
        "composited regions: {} attempts, {} additional labels",
        regional_ocr.len(),
        stats.nonoverlapping_region_lines_added
    ));
    observer.message(format!(
            "OCR merge: {} native-only pages, {} source-only pages, {} fallback pages, {} verified native labels added",
            stats.native_only_pages,
            stats.image_source_only_pages,
            stats.page_fallback_pages,
            stats.nonoverlapping_image_lines_added,
        ));
    observer.message(format!(
        "verified residual rasters used on {residual_page_count} pages"
    ));
}

fn report_recognition(recognition: RecognitionStats, observer: &mut impl PipelineObserver) {
    observer.message(format!(
        "recognition: {} model calls, {} exact RGB cache hits, {} retained bytes",
        recognition.inference_calls, recognition.cache_hits, recognition.retained_bytes
    ));
    if recognition.leader_candidates + recognition.boundary_candidates > 0 {
        observer.message(format!(
            "crop repairs: leader {}/{}, boundary {}/{} in {} batches ({} failed)",
            recognition.leader_accepted,
            recognition.leader_candidates,
            recognition.boundary_accepted,
            recognition.boundary_candidates,
            recognition.crop_batches,
            recognition.crop_failures
        ));
    }
    if recognition.orientation_classified > 0 {
        observer.message(format!(
            "line orientation: {}/{} half-turns",
            recognition.orientation_half_turns, recognition.orientation_classified
        ));
    }
    if recognition.confirmation_attempts > 0 {
        observer.message(format!(
            "alternative recognition: {}/{} admitted in {} batches ({} failed)",
            recognition.confirmation_accepted,
            recognition.confirmation_attempts,
            recognition.confirmation_batches,
            recognition.confirmation_failures
        ));
    }
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
    #[must_use = "retain original PDF bytes for non-destructive output construction"]
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
        mut observer: impl PipelineObserver,
    ) -> Result<PipelineResult, Error> {
        self.verify_source()?;
        let mut session = RecognitionSession::new(engine, self.policy.recognition_cache);
        let prepared =
            self.prepare_native_sources(&mut session, &mut |message| observer.message(message))?;
        let temporary_mask = self.prepare_raster_pdf(&prepared.fallback_pages, &mut |message| {
            observer.message(message);
        })?;
        let raster_pdf = temporary_mask
            .as_ref()
            .map_or(self.pdf.as_path(), NamedTempFile::path);
        let RecognizedPages {
            raster,
            fallback,
            visibility,
            residual_attempts,
        } = self.recognize_pages(&mut session, &prepared, raster_pdf, &mut observer)?;
        let residual_pages = residual_attempts
            .iter()
            .filter(|attempt| attempt.accepted())
            .map(|attempt| attempt.page)
            .collect::<BTreeSet<_>>();
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
        let (mut draft, mut stats, proposals) = merge_ocr(MergeInput {
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
        prepared.record_mixed_source_mappings(&mut session, &proposals)?;
        let regional_ocr = RegionPass {
            pdf: &self.pdf,
            document: &self.doc,
            paints: &prepared.scanned,
            routes: &prepared.routes,
            policy: self.policy,
        }
        .recognize(&mut session, &mut draft)?;
        stats.nonoverlapping_region_lines_added =
            regional_ocr.iter().map(RegionAudit::admitted).sum();
        report_routes(&mut observer, &stats, &regional_ocr, residual_pages.len());
        let recognition = session.stats();
        report_recognition(recognition, &mut observer);
        self.verify_source()?;
        Ok(PipelineResult {
            draft,
            stats,
            routes: prepared.routes,
            proposals,
            visibility,
            residual_pages,
            residual_attempts,
            recognition,
            crop_repairs: session.take_repair_audits(),
            character_audits: session.take_character_audits(),
            raster_character_mappings: session.take_raster_character_mappings(),
            catalog_evidence: None,
            policy: self.policy,
            paints: prepared.scanned,
            source_observations: prepared.recognized,
            direct_decode: prepared.direct_decode,
            source_rasters: prepared.source_rasters,
            regional_ocr,
        })
    }

    fn recognize_pages(
        &self,
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        prepared: &NativePreparation,
        raster_pdf: &Path,
        observer: &mut impl PipelineObserver,
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
                        observer.message(format!(
                            "page {} verification unavailable: {error}; using page OCR",
                            page.get()
                        ));
                        PageVerification::default()
                    }
                };
                let FallbackRecognition {
                    lines,
                    evidence,
                    attempt,
                } = self.recognize_fallback(
                    engine,
                    raster_pdf,
                    page,
                    &native_text_regions,
                    page_evidence,
                    observer,
                )?;
                result.residual_attempts.extend(attempt);
                result.visibility.extend(evidence);
                observer.message(format!(
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
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        raster_pdf: &Path,
        page: PageNumber,
        native: &[Quad],
        verified: PageVerification,
        observer: &mut impl PipelineObserver,
    ) -> Result<FallbackRecognition, Error> {
        let PageVerification {
            evidence,
            control,
            residual,
        } = verified;
        let mut attempt = residual.as_ref().map(|r| ResidualAttempt {
            page,
            paint_count: r.paint_count,
            required: r.required.clone(),
            outcome: ResidualOutcome::TransferRejected,
        });
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
        if let (Some(alternative), Some(planned)) = (alternative, attempt.as_mut()) {
            let work = ResidualWork {
                ordinary: &ordinary,
                treated: alternative.image,
                mapper,
                native,
                evidence: &evidence,
                threshold: self.policy.minimum_confidence,
            };
            if let Some(lines) = planned.recognize(engine, work, observer)? {
                return Ok(FallbackRecognition {
                    lines,
                    evidence,
                    attempt,
                });
            }
        }
        if let Some(attempt) = &attempt {
            observer.message(format!(
                "page {}: residual {}; recognizing ordinary page",
                page.get(),
                match &attempt.outcome {
                    ResidualOutcome::TransferRejected => "transfer rejected",
                    ResidualOutcome::PreflightRejected { .. } => "preflight rejected",
                    ResidualOutcome::InferenceFailed { .. } => "inference failed",
                    ResidualOutcome::MergeRejected { .. } => "merge rejected",
                    ResidualOutcome::Accepted => "accepted",
                }
            ));
        }
        Ok(FallbackRecognition {
            lines: engine.recognize_document_page(ordinary, mapper, page, false)?,
            evidence,
            attempt,
        })
    }

    fn plan_composited_pages(&self, scanned: &PaintReport) -> Result<CompositedPlans, Error> {
        let doc = &self.doc;
        let budget = self.policy.mosaic_budget;
        let mut composited = BTreeSet::new();
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
            if !requires_composited_recognition(&paints, box_, mapper) {
                continue;
            }
            composited.insert(page);
            if summary.native_searchable_shows == 0
                && let Some(plan) = plan_raster_mosaics(page, &paints, box_, mapper, budget)?
            {
                plans.insert(page, plan);
            }
        }
        Ok((composited, plans))
    }

    fn plan_native_sources(
        &self,
        scanned: &PaintReport,
        composited: &BTreeSet<PageNumber>,
        progress: &mut dyn FnMut(String),
    ) -> Result<NativeSourcePlan, Error> {
        let source_budget = self.policy.source_budget;
        let mut source_ids = match self.policy.raster_mode {
            RasterMode::Off => BTreeSet::new(),
            RasterMode::All => scanned
                .placements
                .iter()
                .filter(|paint| {
                    paint.unsafe_reasons.is_empty()
                        && paint.width > 0
                        && paint.height > 0
                        && u64::from(paint.width) * u64::from(paint.height)
                            <= source_budget.source_pixels
                })
                .map(|paint| paint.source_id.clone())
                .collect(),
            RasterMode::Auto => self.auto_source_candidates(scanned, progress)?,
        };
        // Prefer the actual composited native-pitch group when standalone
        // rasters would omit page context or join fragmented labels.
        let eligible_elsewhere = scanned
            .placements
            .iter()
            .filter(|paint| {
                paint.unsafe_reasons.is_empty()
                    && !composited.iter().any(|page| page.get() == paint.page)
            })
            .map(|paint| paint.source_id.as_str())
            .collect::<BTreeSet<_>>();
        source_ids.retain(|id| eligible_elsewhere.contains(id.as_str()));
        if self.policy.native_image_decoder == NativeImageDecoder::Poppler {
            // Object-only extraction does not preserve the resolved named
            // color space at each actual paint. Avoid spending its budget
            // on images without an admissible object-level binding.
            for paint in &scanned.placements {
                if paint.sample_layout.contextual {
                    source_ids.remove(&paint.source_id);
                }
            }
        }
        let plan = NativeSourcePlan::from_paints(source_ids, scanned)?;
        progress(format!(
            "native source extraction planned: {} sources across {} candidate pages",
            plan.ids.len(),
            plan.pages.len()
        ));
        Ok(plan)
    }

    fn auto_source_candidates(
        &self,
        scanned: &PaintReport,
        progress: &mut dyn FnMut(String),
    ) -> Result<BTreeSet<String>, Error> {
        let budget = self.policy.source_budget;
        let mut ids = source_only_candidates(scanned, budget.source_pixels);
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
            self.doc.media_boxes(),
            SourcePolicy {
                max_side: self.policy.render.max_side,
                max_inference_pixels: budget.total_pixels,
                max_sources: budget.sources,
                minimum_source_pixels: 20_000,
                min_gain: 2.25,
            },
        )?;
        progress(format!(
            "source candidates: {} high-detail sources, {} native-only sources",
            report.native_resolution_candidates,
            ids.len()
        ));
        ids.extend(mixed);
        Ok(ids)
    }

    fn decode_native_sources(
        &self,
        plan: &NativeSourcePlan,
        scanned: &PaintReport,
        progress: &mut dyn FnMut(String),
    ) -> (PaintAppearances<image::RgbImage>, Option<DirectDecodeStats>) {
        if self.policy.native_image_decoder == NativeImageDecoder::Hayro {
            return match direct_native_images(
                &self.doc,
                &plan.pages,
                &plan.ids,
                &scanned.placements,
                self.policy.source_budget,
            ) {
                Ok(result) => {
                    progress(format!(
                        "direct native decode: {} accepted sources, {} appearances, {} decoded occurrences, {} reused occurrences",
                        result.stats.accepted_sources,
                        result.stats.accepted_appearances,
                        result.stats.decoded_occurrences,
                        result.stats.reused_occurrences
                    ));
                    (result.appearances, Some(result.stats))
                }
                Err(error) => {
                    progress(format!(
                        "Direct native source decoding unavailable: {error}; using page OCR"
                    ));
                    (PaintAppearances::default(), None)
                }
            };
        }
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
        (
            PaintAppearances::from_object_inventory(decoded, &scanned.placements),
            None,
        )
    }

    fn recognize_native_sources(
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        decoded: PaintAppearances<image::RgbImage>,
        progress: &mut dyn FnMut(String),
    ) -> Result<PaintAppearances<SourceObservation>, Error> {
        let mut recognized = PaintAppearances {
            variants: BTreeMap::new(),
            paint_bindings: decoded.paint_bindings,
        };
        let mut variants = decoded.variants.into_iter().collect::<Vec<_>>();
        variants.sort_by(|(a, x), (b, y)| x.source_id.cmp(&y.source_id).then_with(|| a.cmp(b)));
        for (
            id,
            SourceAppearance {
                source_id,
                value: rgb,
            },
        ) in variants
        {
            let (width, height) = (rgb.width(), rgb.height());
            let mapper = source_image_mapper(width, height)?;
            match engine.recognize_pixels(rgb) {
                Ok(pixels) => {
                    let lines = pixels
                        .lines
                        .into_iter()
                        .map(|pixel| SourceLine {
                            detection_index: pixel.detection_index,
                            line: pixel.into_pdf(&|Point(x, y)| {
                                mapper.to_pdf(x / f64::from(width), y / f64::from(height))
                            }),
                        })
                        .collect::<Vec<_>>();
                    progress(format!(
                        "source {source_id}: {} OCR observations",
                        lines.len()
                    ));
                    recognized.variants.insert(
                        id,
                        SourceAppearance {
                            source_id,
                            value: SourceObservation {
                                width,
                                height,
                                lines,
                            },
                        },
                    );
                }
                Err(error) => progress(format!("source {source_id}: {error}; using page OCR")),
            }
        }
        recognized.discard_unavailable_bindings();
        Ok(recognized)
    }

    fn add_reconstructed_mosaics(
        &self,
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
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
            let mut mappings = Vec::new();
            match reconstruct_native_mosaics(
                &self.pdf,
                self.doc.mapper(page)?,
                plan,
                &paints,
                self.policy.mosaic_budget,
                |group_index, rgb, mapper| {
                    let raster = engine.recognize_pixels(rgb)?;
                    let (width, height) = raster.dimensions;
                    let admitted_detection_indices = raster
                        .lines
                        .iter()
                        .map(|line| line.detection_index)
                        .collect();
                    let corners = [
                        Point(0., 0.),
                        Point(f64::from(width), 0.),
                        Point(f64::from(width), f64::from(height)),
                        Point(0., f64::from(height)),
                    ]
                    .map(mapper);
                    mappings.push(RasterCharacterMapping {
                        page,
                        rgb_sha256: raster.rgb_sha256,
                        dimensions: raster.dimensions,
                        scope: RasterMappingScope::Mosaic {
                            group_index,
                            admitted_detection_indices,
                        },
                        raster_corners: corners,
                    });
                    Ok(raster
                        .lines
                        .into_iter()
                        .map(|line| line.into_pdf(mapper))
                        .collect())
                },
            ) {
                Ok(Some(lines)) => {
                    for mapping in mappings {
                        engine.record_character_mapping(mapping);
                    }
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

    /// Decode and recognize original source images, composited native-resolution
    /// mosaics, and verify exact mixed-page source paint visibility.
    fn prepare_native_sources(
        &self,
        engine: &mut RecognitionSession<'_, impl RecognitionBackend>,
        progress: &mut dyn FnMut(String),
    ) -> Result<NativePreparation, Error> {
        let scanned = enumerate_image_paints(&self.doc, &self.selected)?;
        let (composited, plans) = if self.policy.raster_mode == RasterMode::Off {
            (BTreeSet::new(), BTreeMap::new())
        } else {
            self.plan_composited_pages(&scanned)?
        };
        let source_plan = self.plan_native_sources(&scanned, &composited, progress)?;
        let (decoded, direct_decode) = self.decode_native_sources(&source_plan, &scanned, progress);
        let source_rasters = decoded.map_values(SourceRasterIdentity::measure);
        let recognized = Self::recognize_native_sources(engine, decoded, progress)?;
        let mut source_only = source_only_pages(&scanned, &recognized)?;
        for &page in &composited {
            source_only.approved_pages.remove(&page);
            source_only.pages.remove(&page);
        }
        source_only
            .admissions
            .retain(|(page, _), _| !composited.contains(page));
        record_source_mappings(
            engine,
            &scanned,
            &recognized,
            &source_rasters,
            &source_only.admissions,
        )?;
        let mosaics =
            self.add_reconstructed_mosaics(engine, &scanned, &plans, &mut source_only, progress)?;
        progress(format!(
            "native mosaics: {} compositor-required pages, {mosaics} reconstructed",
            composited.len()
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
            direct_decode,
            source_rasters,
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
