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
    fragment_ink::{InkAccounting, Sample},
    fragment_reconciliation::{conflicts, experimental_draft, propose, superseded},
    geometry::{DisplayMapper, Quad},
    line_audit::{
        AuditedLine, BoundaryPeer, LineAuditPolicy, MappedBoundaryExtension, SampledLine,
        audit_line,
    },
    pdf_metadata::PdfMetadata,
    refinement::{RefinementPolicy, RefinementScope, recognize_observation, refinement_plan},
    refinement_evidence::{Recognition, RefinementOutcome},
    refinement_recognition::{RefinementModel, RefinementRecognizer},
};

pub use crate::{
    fragment_ink::{RunAccounting, RunInk},
    fragment_reconciliation::{Consumption, PeerClaim, Rejection, TokenBinding, Transaction},
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
pub enum LineOutcome {
    Skipped {
        page: u32,
        line_index: usize,
        reason: String,
    },
    Audited {
        audit: Box<AuditedLine>,
        variants: Vec<VariantOutcome>,
    },
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
    pub padding_pixels: Option<u32>,
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
        let (width, height) = (b.right - b.left, b.bottom - b.top);
        let cost = u64::from(width) * u64::from(height);
        let Some(remaining) = self.remaining_pixels.checked_sub(cost) else {
            return Ok(VariantOutcome::Skipped {
                reason: Rejection::PixelBudgetExceeded,
            });
        };
        self.remaining_pixels = remaining;
        let rgb = imageops::crop_imm(&sampled.rgb, b.left, b.top, width, height).to_image();
        let normalized =
            imageops::crop_imm(&sampled.normalized, b.left, b.top, width, height).to_image();
        let shared = SourceInk::from_shared_raster(
            &rgb,
            &normalized,
            sampled.audit.polarity.clone(),
            self.policy.audit.segmentation_threshold,
            &self.policy.audit.topology_thresholds,
        )?;
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
        sampled.measure_extensions(self.page, &self.policy.audit)?;
        let mut budget = self.policy.audit.max_ink_checks;
        for extension in &sampled.audit.boundary_extensions {
            if let InkAccounting::Measured { charged_checks, .. } = &extension.ink_accounting {
                budget = budget
                    .checked_sub(*charged_checks)
                    .ok_or_else(|| Error::Invalid("inconsistent audit ink budget".into()))?;
            }
        }
        let variants = (0..sampled.audit.boundary_extensions.len())
            .map(|i| self.recognize_variant(&sampled, i, &mut budget, predictor, transactions))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LineOutcome::Audited {
            audit: Box::new(sampled.audit),
            variants,
        })
    }
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
            schema_version: 5,
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
