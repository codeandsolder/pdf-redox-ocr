//! Recognition of exact subwindows of a padded source raster and atomic
//! research-only reconciliation. No synthetic PDF drafts or re-renders of
//! expanded crops are required.

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    path::Path,
};

use image::imageops;
use serde::Serialize;

use crate::{
    Error, PageNumber,
    candidate_ink::SourceInk,
    draft::{DraftPage, OcrDraft},
    fragment_ink::{InkAccounting, Sample, membership_checks},
    fragment_reconciliation::{conflicts, experimental_draft, propose, superseded},
    geometry::{DisplayMapper, Quad},
    layout_leaders::LeaderEdge,
    line_audit::{
        AuditedLine, BoundaryPeer, LineAuditPolicy, MappedBoundaryExtension, SampledLine,
        audit_line, inside_page,
    },
    mask_reconciliation,
    orthogonal_mask::build_mask,
    pdf_metadata::PdfMetadata,
    pixel_regions::InkBounds,
    refinement::{RefinementPolicy, RefinementScope, recognize_observation, refinement_plan},
    refinement_evidence::{Recognition, RefinementOutcome},
    refinement_recognition::{RefinementModel, RefinementRecognizer},
    stroke_reconciliation,
};

pub use crate::{
    fragment_ink::{RunAccounting, RunInk},
    fragment_reconciliation::{
        Consumption, Expansion, Operation, PeerClaim, Rejection, TokenBinding, Transaction,
    },
    mask_reconciliation::{MaskCorrection, MaskVote, RunPartition},
    orthogonal_mask::MaskEvidence,
    stroke_reconciliation::{Contraction, ContractionVote, PartitionRun},
};

#[derive(Debug)]
pub struct FragmentPolicy {
    pub audit: LineAuditPolicy,
    pub max_page_pixels: u64,
}

impl Default for FragmentPolicy {
    fn default() -> Self {
        Self {
            audit: LineAuditPolicy {
                padding_pixels: None,
                ..LineAuditPolicy::default()
            },
            max_page_pixels: 32_000_000,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum VariantOutcome {
    Skipped {
        reason: Rejection,
    },
    Measured {
        recognition: Box<Recognition>,
        run_accounting: RunAccounting,
        /// Index of the candidate in transactions, even if later conflicted.
        transaction: Option<usize>,
        rejection: Option<Rejection>,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ContractionOutcome {
    Skipped {
        reason: Rejection,
    },
    Measured {
        recognition: Box<Recognition>,
        transaction: Option<usize>,
        rejection: Option<Rejection>,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum OrthogonalOutcome {
    Skipped {
        reason: Rejection,
    },
    Measured {
        evidence: MaskEvidence,
        recognition: Box<Recognition>,
        transaction: Option<usize>,
        rejection: Option<Rejection>,
    },
}

#[derive(Debug, Default, Serialize)]
pub struct StrokeMeasurements {
    pub original: Option<Box<Recognition>>,
    pub contractions: Vec<ContractionOutcome>,
    pub orthogonal: Option<Box<OrthogonalOutcome>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LineOutcome {
    Skipped {
        page: u32,
        line_index: usize,
        reason: String,
    },
    Audited {
        audit: Box<AuditedLine>,
        peer_search: Option<PeerSearch>,
        charged_ink_checks: u64,
        variants: Vec<VariantOutcome>,
        strokes: StrokeMeasurements,
    },
}

/// One-hop extension of the local search to an already ink-supported peer.
#[derive(Debug, Serialize)]
pub struct PeerSearch {
    peers: BTreeSet<usize>,
    initial_sample_quad: Quad,
    initial_normalized_sha256: String,
    padding: [u32; 4],
    horizon: [u32; 2],
    used: bool,
}

#[derive(Debug, Serialize)]
pub struct PageEvidence {
    pub page: u32,
    pub charged_pixels: u64,
    pub lines: Vec<LineOutcome>,
    /// Recognition of original competing observations, at most once per peer.
    pub peers: BTreeMap<usize, RefinementOutcome>,
}

#[derive(Debug, Serialize)]
pub struct FragmentReport {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub draft_model: String,
    pub secondary_model: &'static str,
    pub dpi: u32,
    pub padding_pixels: Option<[u32; 4]>,
    pub segmentation_threshold: u8,
    pub topology_thresholds: Vec<u8>,
    pub polarity: crate::ink_polarity::Polarity,
    pub max_pixels: u64,
    pub max_page_pixels: u64,
    pub max_ink_checks: u64,
    pub pages: Vec<PageEvidence>,
    pub transactions: Vec<Transaction>,
    /// Replaced candidate index and every strictly dominating candidate index.
    pub superseded: BTreeMap<usize, Vec<usize>>,
    pub conflicts: BTreeSet<usize>,
}

struct MaskCrop {
    rgb: image::RgbImage,
    ink: SourceInk,
    evidence: MaskEvidence,
}

struct PageContext<'a> {
    pdf: &'a Path,
    metadata: &'a PdfMetadata,
    number: PageNumber,
    page: &'a DraftPage,
    mapper: DisplayMapper,
    policy: &'a FragmentPolicy,
    peer_policy: RefinementPolicy,
    remaining_pixels: u64,
    peers: BTreeMap<usize, RefinementOutcome>,
}

impl PageContext<'_> {
    fn peer_quads(&self, extension: &MappedBoundaryExtension) -> Vec<Quad> {
        extension
            .overlaps
            .iter()
            .map(|o| match o.peer {
                BoundaryPeer::OcrLine { line_index } => self.page.lines[line_index].quad,
                BoundaryPeer::NativeText { region_index } => {
                    self.page.native_text_regions[region_index]
                }
            })
            .collect()
    }

    fn recognize_peers(
        &mut self,
        extension: &MappedBoundaryExtension,
        predictor: &mut RefinementRecognizer,
    ) -> Result<(), Error> {
        let InkAccounting::Measured { peers, .. } = &extension.ink_accounting else {
            return Ok(());
        };
        for (i, overlap) in extension.overlaps.iter().enumerate() {
            let BoundaryPeer::OcrLine { line_index } = overlap.peer else {
                continue;
            };
            if peers.get(&i).is_some_and(|p| p.axis_added > 0)
                && let std::collections::btree_map::Entry::Vacant(entry) =
                    self.peers.entry(line_index)
            {
                let outcome = recognize_observation(
                    self.pdf,
                    self.number,
                    &self.page.lines[line_index],
                    self.mapper,
                    self.peer_policy,
                    &mut self.remaining_pixels,
                    &mut |rgb| predictor.recognize(rgb),
                )?;
                entry.insert(outcome);
            }
        }
        Ok(())
    }

    fn crop_pixels(
        &mut self,
        sampled: &SampledLine,
        b: InkBounds,
    ) -> Option<(image::RgbImage, image::GrayImage)> {
        let (width, height) = (b.right - b.left, b.bottom - b.top);
        let cost = u64::from(width) * u64::from(height);
        self.remaining_pixels = self.remaining_pixels.checked_sub(cost)?;
        Some((
            imageops::crop_imm(&sampled.rgb, b.left, b.top, width, height).to_image(),
            imageops::crop_imm(&sampled.normalized, b.left, b.top, width, height).to_image(),
        ))
    }

    fn measure_crop(
        &self,
        sampled: &SampledLine,
        rgb: &image::RgbImage,
        normalized: &image::GrayImage,
    ) -> Result<SourceInk, Error> {
        SourceInk::from_shared_raster(
            rgb,
            normalized,
            sampled.audit.polarity.clone(),
            self.policy.audit.segmentation_threshold,
            &self.policy.audit.topology_thresholds,
        )
    }

    fn prepare_crop(
        &mut self,
        sampled: &SampledLine,
        b: InkBounds,
    ) -> Result<Option<(image::RgbImage, SourceInk)>, Error> {
        let Some((rgb, normalized)) = self.crop_pixels(sampled, b) else {
            return Ok(None);
        };
        let shared = self.measure_crop(sampled, &rgb, &normalized)?;
        Ok(Some((rgb, shared)))
    }

    fn prepare_mask(
        &mut self,
        sampled: &SampledLine,
        budget: &mut u64,
    ) -> Result<Option<Result<MaskCrop, Rejection>>, Error> {
        let Some(strokes) = &sampled.audit.foreign_strokes else {
            return Ok(None);
        };
        if strokes.orthogonal.is_empty() {
            return Ok(None);
        }
        let b = sampled.audit.observation_bounds;
        let required = membership_checks(b.right - b.left, b.bottom - b.top, 3);
        let Some(remaining) = budget.checked_sub(required) else {
            return Ok(Some(Err(Rejection::InkBudgetExceeded)));
        };
        let Some((mut rgb, mut normalized)) = self.crop_pixels(sampled, b) else {
            return Ok(Some(Err(Rejection::PixelBudgetExceeded)));
        };
        *budget = remaining;
        let threshold = self.policy.audit.segmentation_threshold;
        let Some(mask) = build_mask(
            &normalized,
            threshold,
            b,
            strokes.orthogonal.iter().map(|s| &s.evidence),
            &sampled.audit.polarity,
        )?
        else {
            return Ok(Some(Err(Rejection::NoMeasuredInk)));
        };
        mask.apply(&mut rgb, &mut normalized, threshold)?;
        let shared = self.measure_crop(sampled, &rgb, &normalized)?;
        if shared
            .groups
            .iter()
            .map(|g| g.run.observation_support.inside_pixels)
            .sum::<u64>()
            != mask.evidence.foreground_after
        {
            return Err(Error::Invalid(
                "masked recognition foreground mismatch".into(),
            ));
        }
        Ok(Some(Ok(MaskCrop {
            rgb,
            ink: shared,
            evidence: mask.evidence,
        })))
    }

    fn recognize_contractions(
        &mut self,
        sampled: &SampledLine,
        predictor: &mut RefinementRecognizer,
    ) -> Result<Vec<ContractionOutcome>, Error> {
        let Some(strokes) = &sampled.audit.foreign_strokes else {
            return Ok(Vec::new());
        };
        let mut variants = Vec::new();
        for c in &strokes.contractions {
            let Some((rgb, shared)) = self.prepare_crop(sampled, c.evidence.retained)? else {
                variants.push(ContractionOutcome::Skipped {
                    reason: Rejection::PixelBudgetExceeded,
                });
                continue;
            };
            let recognition = predictor.recognize_with_ink(&rgb, shared)?;
            variants.push(ContractionOutcome::Measured {
                recognition: Box::new(recognition),
                transaction: None,
                rejection: None,
            });
        }
        Ok(variants)
    }

    fn recognize_strokes(
        &mut self,
        sampled: &SampledLine,
        budget: &mut u64,
        predictor: &mut RefinementRecognizer,
        transactions: &mut Vec<Transaction>,
    ) -> Result<StrokeMeasurements, Error> {
        let Some(strokes) = &sampled.audit.foreign_strokes else {
            return Ok(StrokeMeasurements::default());
        };
        let contractions = &strokes.contractions;
        if contractions.is_empty() && strokes.orthogonal.is_empty() {
            return Ok(StrokeMeasurements::default());
        }
        let Some((rgb, shared)) = self.prepare_crop(sampled, sampled.audit.observation_bounds)?
        else {
            return Ok(StrokeMeasurements {
                original: None,
                contractions: contractions
                    .iter()
                    .map(|_| ContractionOutcome::Skipped {
                        reason: Rejection::PixelBudgetExceeded,
                    })
                    .collect(),
                orthogonal: (!strokes.orthogonal.is_empty()).then(|| {
                    Box::new(OrthogonalOutcome::Skipped {
                        reason: Rejection::PixelBudgetExceeded,
                    })
                }),
            });
        };
        let mut variants = self.recognize_contractions(sampled, predictor)?;
        // Preserve the pixel-budget priority of the contraction crops,
        // then recognize the masked and original crops in one paired batch
        // per model and stretch. Equal dimensions preserve the view contract.
        let prepared = self.prepare_mask(sampled, budget)?;
        let (original, mask_result) = match prepared {
            Some(Ok(mask)) => {
                let [original, masked] = predictor.recognize_pair(
                    (&rgb, shared),
                    (&mask.rgb, mask.ink),
                    &sampled.audit.original_text,
                )?;
                (original, Some(Ok((mask.evidence, masked))))
            }
            Some(Err(reason)) => (
                predictor.recognize_with_ink(&rgb, shared)?,
                Some(Err(reason)),
            ),
            None => (predictor.recognize_with_ink(&rgb, shared)?, None),
        };
        for (i, variant) in variants.iter_mut().enumerate() {
            if let ContractionOutcome::Measured {
                recognition,
                transaction,
                rejection,
            } = variant
            {
                match stroke_reconciliation::propose(&sampled.audit, i, &original, recognition) {
                    Ok(operation) => {
                        *transaction = Some(transactions.len());
                        transactions.push(operation);
                    }
                    Err(reason) => *rejection = Some(reason),
                }
            }
        }
        let orthogonal = mask_result.map(|result| {
            Box::new(match result {
                Err(reason) => OrthogonalOutcome::Skipped { reason },
                Ok((evidence, recognition)) => {
                    let (transaction, rejection) = match mask_reconciliation::propose(
                        &sampled.audit,
                        &original,
                        &recognition,
                        &evidence,
                    ) {
                        Ok(operation) => {
                            let id = transactions.len();
                            transactions.push(operation);
                            (Some(id), None)
                        }
                        Err(reason) => (None, Some(reason)),
                    };
                    OrthogonalOutcome::Measured {
                        evidence,
                        recognition: Box::new(recognition),
                        transaction,
                        rejection,
                    }
                }
            })
        });
        Ok(StrokeMeasurements {
            original: Some(Box::new(original)),
            contractions: variants,
            orthogonal,
        })
    }

    fn recognize_variant(
        &mut self,
        sampled: &SampledLine,
        index: usize,
        budget: &mut u64,
        predictor: &mut RefinementRecognizer,
        transactions: &mut Vec<Transaction>,
    ) -> Result<VariantOutcome, Error> {
        let extension = &sampled.audit.boundary_extensions[index];
        let InkAccounting::Measured { sample, .. } = &extension.ink_accounting else {
            return Ok(VariantOutcome::Skipped {
                reason: Rejection::NoMeasuredInk,
            });
        };
        if sample.axis_added == 0 {
            return Ok(VariantOutcome::Skipped {
                reason: Rejection::NoMeasuredInk,
            });
        }
        let b = extension.evidence.expanded;
        let Some((rgb, shared)) = self.prepare_crop(sampled, b)? else {
            return Ok(VariantOutcome::Skipped {
                reason: Rejection::PixelBudgetExceeded,
            });
        };
        let Some(run_accounting) = (Sample {
            image: &sampled.normalized,
            frame: sampled.frame,
            window: sampled.window,
            threshold: self.policy.audit.segmentation_threshold,
        })
        .account_runs(
            b,
            &shared.groups,
            &self.peer_quads(extension),
            &extension.ink_accounting,
            budget,
        )?
        else {
            return Ok(VariantOutcome::Skipped {
                reason: Rejection::InkBudgetExceeded,
            });
        };
        let recognition = predictor.recognize_with_ink(&rgb, shared)?;
        self.recognize_peers(extension, predictor)?;
        let (transaction, rejection) = match propose(
            self.page,
            sampled.audit.line_index,
            index,
            extension,
            &recognition,
            &run_accounting,
            &self.peers,
        ) {
            Ok(operation) => {
                let id = transactions.len();
                transactions.push(operation);
                (Some(id), None)
            }
            Err(reason) => (None, Some(reason)),
        };
        Ok(VariantOutcome::Measured {
            recognition: Box::new(recognition),
            run_accounting,
            transaction,
            rejection,
        })
    }

    fn peer_search(&self, sampled: &SampledLine) -> Option<PeerSearch> {
        if self.policy.audit.padding_pixels.is_some() {
            return None;
        }
        let mut window = sampled.window;
        let mut plan = PeerSearch {
            peers: BTreeSet::new(),
            initial_sample_quad: sampled.audit.sample_quad,
            initial_normalized_sha256: sampled.audit.normalized_sha256.clone(),
            padding: window.padding,
            horizon: [0; 2],
            used: false,
        };
        let complete = complete_peers(sampled);
        let inside = inside_page(self.mapper);
        for extension in &sampled.audit.boundary_extensions {
            let InkAccounting::Measured { peers, .. } = &extension.ink_accounting else {
                continue;
            };
            for (i, overlap) in extension.overlaps.iter().enumerate() {
                let BoundaryPeer::OcrLine { line_index } = overlap.peer else {
                    continue;
                };
                if complete.contains(&line_index)
                    || peers.get(&i).is_none_or(|p| {
                        p.axis_added == 0 || p.original != 0 || p.orthogonal_added != 0
                    })
                {
                    continue;
                }
                let edge = extension.evidence.edge;
                let Some((next, extent)) = window.cover_peer(
                    sampled.frame,
                    &inside,
                    edge,
                    &self.page.lines[line_index].quad,
                    self.policy.audit.max_pixels,
                ) else {
                    continue;
                };
                if extent <= sampled.frame.height.saturating_mul(2) {
                    continue;
                }
                let side = match edge {
                    LeaderEdge::Left => 0,
                    LeaderEdge::Right => 1,
                };
                plan.horizon[side] = plan.horizon[side].max(extent);
                plan.peers.insert(line_index);
                window = next;
            }
        }
        plan.padding = window.padding;
        (!plan.peers.is_empty()).then_some(plan)
    }

    fn retry_peer_search(
        &mut self,
        index: usize,
        original: &SampledLine,
        budget: &mut u64,
        plan: &mut PeerSearch,
    ) -> Result<Option<SampledLine>, Error> {
        let reserve = variant_checks(original);
        let Some(available) = budget.checked_sub(reserve) else {
            return Ok(None);
        };
        let mut retry_budget = available;
        let mut policy = self.policy.audit.clone();
        policy.padding_pixels = Some(plan.padding);
        let Ok(mut retry) = audit_line(
            self.pdf,
            self.metadata,
            self.number,
            index,
            &self.page.lines[index],
            &policy,
            Some(&mut self.remaining_pixels),
        ) else {
            return Ok(None);
        };
        retry.measure_extensions(self.page, &policy, plan.horizon, &mut retry_budget)?;
        *budget -= available - retry_budget;
        // Retain the ordinary sample unless the larger sample can account
        // for a complete previously unsupported peer. All candidates in a
        // report share the chosen raster and never mix coordinate frames.
        let complete = complete_peers(&retry);
        if plan.peers.is_disjoint(&complete)
            || !complete_peers(original).is_subset(&complete)
            || variant_checks(&retry) > *budget
            || retry
                .audit
                .boundary_extensions
                .iter()
                .any(|e| matches!(e.ink_accounting, InkAccounting::BudgetExceeded { .. }))
        {
            return Ok(None);
        }
        plan.used = true;
        Ok(Some(retry))
    }

    fn line(
        &mut self,
        index: usize,
        predictor: &mut RefinementRecognizer,
        transactions: &mut Vec<Transaction>,
    ) -> Result<LineOutcome, Error> {
        let mut sampled = match audit_line(
            self.pdf,
            self.metadata,
            self.number,
            index,
            &self.page.lines[index],
            &self.policy.audit,
            Some(&mut self.remaining_pixels),
        ) {
            Ok(sampled) => sampled,
            Err(error) => {
                return Ok(LineOutcome::Skipped {
                    page: self.number.get(),
                    line_index: index,
                    reason: error.to_string(),
                });
            }
        };
        let mut budget = self.policy.audit.max_ink_checks;
        sampled.measure_extensions(self.page, &self.policy.audit, [0; 2], &mut budget)?;
        let mut peer_search = self.peer_search(&sampled);
        if let Some(plan) = &mut peer_search
            && let Some(retry) = self.retry_peer_search(index, &sampled, &mut budget, plan)?
        {
            sampled = retry;
        }
        let variants = (0..sampled.audit.boundary_extensions.len())
            .map(|i| self.recognize_variant(&sampled, i, &mut budget, predictor, transactions))
            .collect::<Result<Vec<_>, _>>()?;
        sampled.measure_strokes(&self.policy.audit, &mut budget)?;
        let strokes = self.recognize_strokes(&sampled, &mut budget, predictor, transactions)?;
        Ok(LineOutcome::Audited {
            audit: Box::new(sampled.audit),
            peer_search,
            charged_ink_checks: self.policy.audit.max_ink_checks - budget,
            variants,
            strokes,
        })
    }
}

fn variant_checks(sampled: &SampledLine) -> u64 {
    sampled
        .audit
        .boundary_extensions
        .iter()
        .filter_map(|extension| {
            let InkAccounting::Measured { sample, .. } = &extension.ink_accounting else {
                return None;
            };
            if sample.axis_added == 0 {
                return None;
            }
            let b = extension.evidence.expanded;
            Some(membership_checks(
                b.right - b.left,
                b.bottom - b.top,
                extension.overlaps.len(),
            ))
        })
        .fold(0, u64::saturating_add)
}

fn complete_peers(sampled: &SampledLine) -> BTreeSet<usize> {
    sampled
        .audit
        .boundary_extensions
        .iter()
        .flat_map(|extension| {
            extension
                .overlaps
                .iter()
                .enumerate()
                .filter_map(|(i, overlap)| {
                    let BoundaryPeer::OcrLine { line_index } = overlap.peer else {
                        return None;
                    };
                    let InkAccounting::Measured {
                        peers,
                        peers_fully_sampled,
                        ..
                    } = &extension.ink_accounting
                    else {
                        return None;
                    };
                    (peers_fully_sampled.get(i) == Some(&true)
                        && peers.get(&i).is_some_and(|p| {
                            p.axis_added > 0
                                && p.outside_expansion == 0
                                && p.original == 0
                                && p.orthogonal_added == 0
                        }))
                    .then_some(line_index)
                })
        })
        .collect()
}

/// Refine exact original observations with audited edge expansions.
/// The output draft deliberately uses an unsupported importer schema.
/// # Errors
/// Rejects mismatched PDF provenance, invalid selection/policy, or inference
/// and evidence consistency failures. Individual sampling errors are recorded.
pub fn reconcile_fragments(
    pdf: &Path,
    source: &OcrDraft,
    targets: &BTreeMap<PageNumber, BTreeSet<usize>>,
    policy: &FragmentPolicy,
    threads: Option<NonZeroUsize>,
) -> Result<(FragmentReport, OcrDraft), Error> {
    source.verify_pdf(pdf)?;
    policy.audit.validate()?;
    let metadata = PdfMetadata::open(pdf)?;
    let peer_policy = RefinementPolicy {
        dpi: policy.audit.dpi,
        max_line_pixels: policy.audit.max_pixels,
        max_page_pixels: policy.max_page_pixels,
        ..RefinementPolicy::default()
    }
    .validate()?;
    let plan = refinement_plan(
        source,
        RefinementScope::ExplicitLines(targets),
        metadata.page_count(),
        peer_policy,
    )?;
    let mappers = metadata.select_mappers(&plan.keys().copied().collect::<Vec<_>>())?;
    let mut predictor = RefinementRecognizer::new(RefinementModel::Consensus, threads);
    let mut pages = Vec::new();
    let mut transactions = Vec::new();
    for (number, indices) in plan {
        let page = source
            .pages
            .iter()
            .find(|p| p.page == number.get())
            .ok_or_else(|| Error::Invalid("fragment source page missing".into()))?;
        let mapper = *mappers
            .get(&number)
            .ok_or_else(|| Error::Invalid("fragment page mapper missing".into()))?;
        let mut context = PageContext {
            pdf,
            metadata: &metadata,
            number,
            page,
            mapper,
            policy,
            peer_policy,
            remaining_pixels: policy.max_page_pixels,
            peers: BTreeMap::new(),
        };
        let lines = indices
            .into_iter()
            .map(|i| context.line(i, &mut predictor, &mut transactions))
            .collect::<Result<Vec<_>, _>>()?;
        pages.push(PageEvidence {
            page: number.get(),
            charged_pixels: policy.max_page_pixels - context.remaining_pixels,
            lines,
            peers: context.peers,
        });
    }
    let superseded = superseded(&transactions);
    let conflicts = conflicts(&transactions, &superseded);
    let blocked = conflicts
        .iter()
        .copied()
        .chain(superseded.keys().copied())
        .collect();
    let draft = experimental_draft(source, &transactions, &blocked);
    source.verify_pdf(pdf)?;
    Ok((
        FragmentReport {
            schema_version: 12,
            pdf_sha256: source.pdf_sha256.clone(),
            draft_model: source.model.clone(),
            secondary_model: RefinementModel::Consensus.label(),
            dpi: policy.audit.dpi,
            padding_pixels: policy.audit.padding_pixels,
            segmentation_threshold: policy.audit.segmentation_threshold,
            topology_thresholds: policy.audit.topology_thresholds.clone(),
            polarity: policy.audit.polarity,
            max_pixels: policy.audit.max_pixels,
            max_page_pixels: policy.max_page_pixels,
            max_ink_checks: policy.audit.max_ink_checks,
            pages,
            transactions,
            superseded,
            conflicts,
        },
        draft,
    ))
}
