//! Pixel-space OCR independent of the source's document geometry.

use std::{
    collections::{BTreeSet, VecDeque},
    mem::size_of,
};

use image::RgbImage;
use sha2::{Digest, Sha256};

use crate::{
    Error,
    character_alignment::CharacterAuditReport,
    crop_repair::{CropReason, CropRepair, RepairReport},
    draft::OcrLine,
    geometry::{DisplayMapper, Point, Quad},
};

/// A model observation in the original input raster's pixel-corner frame.
///
/// Coordinates grow rightwards and downwards. Corners are ordered bottom-left,
/// bottom-right, top-right, top-left relative to the observed reading direction
/// for final observations; raw detector audits retain the detector ordering.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PixelLine {
    /// Stable original detector identity within this RGB raster.
    #[serde(skip)]
    pub detection_index: usize,
    pub text: String,
    pub confidence: f64,
    pub quad: Quad,
}

impl PixelLine {
    /// Apply the exact occurrence-specific transform after recognition.
    #[must_use]
    pub fn into_pdf(self, mapper: &dyn Fn(Point) -> Point) -> OcrLine {
        OcrLine {
            text: self.text,
            confidence: self.confidence,
            quad: self.quad.map(mapper),
            reviewed: false,
        }
    }
}

/// Observations and evidence emitted by one complete raster recognition.
#[derive(Default)]
pub struct RasterRecognition {
    pub lines: Vec<PixelLine>,
    pub crop_repair: RepairReport,
    pub character_audit: CharacterAuditReport,
}

impl From<Vec<PixelLine>> for RasterRecognition {
    fn from(lines: Vec<PixelLine>) -> Self {
        Self {
            lines,
            crop_repair: RepairReport::default(),
            character_audit: CharacterAuditReport::default(),
        }
    }
}

/// Original raster identity and all actual crop-repair attempts for that input.
/// Cached occurrences reuse the same pixel-space recognition, not new attempts.
#[derive(Debug, serde::Serialize)]
pub struct RasterRepairAudit {
    pub dimensions: (u32, u32),
    pub rgb_sha256: String,
    pub batches: usize,
    pub repairs: Vec<CropRepair>,
}

/// Position evidence from the actual primary crops of one unique RGB raster.
/// The original detector index may differ from the final draft line index.
#[derive(Debug, serde::Serialize)]
pub struct RasterCharacterAudit {
    pub dimensions: (u32, u32),
    pub rgb_sha256: String,
    pub report: CharacterAuditReport,
}

/// Which accepted raster observations may contribute character evidence.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RasterMappingScope {
    WholePage {
        residual: bool,
    },
    Region {
        seed_paint_index: u32,
        admitted_detection_indices: BTreeSet<usize>,
    },
}

/// Identifies an actual page-space raster occurrence, including cached results.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RasterCharacterMapping {
    pub page: crate::PageNumber,
    pub rgb_sha256: String,
    pub dimensions: (u32, u32),
    pub scope: RasterMappingScope,
    /// PDF corners of the actual raster extent in TL,TR,BR,BL order.
    pub raster_corners: Quad,
}

impl RasterCharacterMapping {
    /// Whether this mapping belongs to the final accepted page route.
    #[cfg(feature = "pdf-integration")]
    pub(crate) fn selected(&self, residual_pages: &BTreeSet<crate::PageNumber>) -> bool {
        match &self.scope {
            RasterMappingScope::WholePage { residual } => {
                *residual == residual_pages.contains(&self.page)
            }
            RasterMappingScope::Region { .. } => true,
        }
    }

    /// Restrict supplemental raster evidence to explicitly admitted detections.
    #[cfg(feature = "pdf-integration")]
    pub(crate) fn includes_detection(&self, detection_index: usize) -> bool {
        match &self.scope {
            RasterMappingScope::WholePage { .. } => true,
            RasterMappingScope::Region {
                admitted_detection_indices,
                ..
            } => admitted_detection_indices.contains(&detection_index),
        }
    }

    /// Reject ambiguous repeated executions of the same raster before matching.
    #[cfg(feature = "pdf-integration")]
    pub(crate) fn unique_audit<'a>(
        &self,
        audits: &'a [RasterCharacterAudit],
    ) -> Option<&'a RasterCharacterAudit> {
        let mut matching = audits
            .iter()
            .filter(|a| a.rgb_sha256 == self.rgb_sha256 && a.dimensions == self.dimensions);
        let first = matching.next()?;
        matching.next().is_none().then_some(first)
    }

    /// Map actual raster pixel corners to PDF space with the occurrence affine transform.
    #[must_use]
    pub fn to_pdf(&self, quad: Quad) -> Quad {
        let [top_left, top_right, _, bottom_left] = self.raster_corners;
        quad.map(|Point(x, y)| {
            let (u, v) = (
                x / f64::from(self.dimensions.0),
                y / f64::from(self.dimensions.1),
            );
            Point(
                (top_right.0 - top_left.0)
                    .mul_add(u, (bottom_left.0 - top_left.0).mul_add(v, top_left.0)),
                (top_right.1 - top_left.1)
                    .mul_add(u, (bottom_left.1 - top_left.1).mul_add(v, top_left.1)),
            )
        })
    }
}

/// A reusable recognizer whose result depends only on the input RGB raster.
///
/// A backend must use the same inference policy throughout a recognition
/// session. Mapping into PDF space belongs to the caller, never the model.
pub trait RecognitionBackend {
    /// Stable model and preprocessing provenance for the generated draft.
    fn model_profile(&self) -> &str;

    /// Recognize text in the original raster's pixel coordinates.
    ///
    /// # Errors
    /// Returns an error when recognition fails.
    fn recognize_pixels(&self, rgb: RgbImage) -> Result<RasterRecognition, Error>;
}

/// Bound retained recognition results, not the input raster allocation.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RecognitionCacheBudget {
    /// Maximum number of distinct RGB inputs remembered.
    pub max_entries: usize,
    /// Maximum retained observation payload, including string allocations.
    pub max_bytes: usize,
}

impl Default for RecognitionCacheBudget {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_bytes: 4 * 1024 * 1024,
        }
    }
}

impl RecognitionCacheBudget {
    /// Disable result caching without changing recognition or mapping.
    pub const DISABLED: Self = Self {
        max_entries: 0,
        max_bytes: 0,
    };

    const fn enabled(self) -> bool {
        self.max_entries > 0 && self.max_bytes > 0
    }
}

/// Job-local cache accounting, independent of page and source routing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct RecognitionStats {
    pub inference_calls: usize,
    pub cache_hits: usize,
    pub evictions: usize,
    pub retained_bytes: usize,
    pub crop_batches: usize,
    pub leader_candidates: usize,
    pub leader_accepted: usize,
    pub boundary_candidates: usize,
    pub boundary_accepted: usize,
    pub crop_failures: usize,
    pub orientation_classified: usize,
    pub orientation_half_turns: usize,
    pub confirmation_batches: usize,
    pub confirmation_attempts: usize,
    pub confirmation_accepted: usize,
    pub confirmation_failures: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RasterKey {
    dimensions: (u32, u32),
    digest: [u8; 32],
}

impl RasterKey {
    fn new(rgb: &RgbImage) -> Self {
        Self {
            dimensions: rgb.dimensions(),
            digest: Sha256::digest(rgb.as_raw()).into(),
        }
    }
}

struct CacheEntry {
    key: RasterKey,
    lines: Vec<PixelLine>,
    bytes: usize,
}

/// One inference session and its bounded, content-addressed LRU result cache.
///
/// Cache identity includes every RGB byte and both dimensions. No perceptual
/// hashes, PDF object numbers, transforms or page numbers participate.
/// Failures are not cached, so subsequent occurrences can retry inference.
pub struct RecognitionSession<'a, B> {
    backend: &'a B,
    budget: RecognitionCacheBudget,
    entries: VecDeque<CacheEntry>,
    stats: RecognitionStats,
    repair_audits: Vec<RasterRepairAudit>,
    character_audits: Vec<RasterCharacterAudit>,
    raster_character_mappings: Vec<RasterCharacterMapping>,
}

impl<'a, B: RecognitionBackend> RecognitionSession<'a, B> {
    #[must_use]
    pub const fn new(backend: &'a B, budget: RecognitionCacheBudget) -> Self {
        Self {
            backend,
            budget,
            entries: VecDeque::new(),
            repair_audits: Vec::new(),
            character_audits: Vec::new(),
            raster_character_mappings: Vec::new(),
            stats: RecognitionStats {
                inference_calls: 0,
                cache_hits: 0,
                evictions: 0,
                retained_bytes: 0,
                crop_batches: 0,
                leader_candidates: 0,
                leader_accepted: 0,
                boundary_candidates: 0,
                boundary_accepted: 0,
                crop_failures: 0,
                orientation_classified: 0,
                orientation_half_turns: 0,
                confirmation_batches: 0,
                confirmation_attempts: 0,
                confirmation_accepted: 0,
                confirmation_failures: 0,
            },
        }
    }

    #[must_use]
    pub const fn stats(&self) -> RecognitionStats {
        self.stats
    }

    /// Finish collecting evidence without retaining input raster buffers.
    pub fn take_repair_audits(&mut self) -> Vec<RasterRepairAudit> {
        std::mem::take(&mut self.repair_audits)
    }

    pub fn take_character_audits(&mut self) -> Vec<RasterCharacterAudit> {
        std::mem::take(&mut self.character_audits)
    }

    pub fn take_raster_character_mappings(&mut self) -> Vec<RasterCharacterMapping> {
        std::mem::take(&mut self.raster_character_mappings)
    }

    /// Retain the exact occurrence transform for page-raster character audits.
    /// Source-image and mosaic rasters deliberately use their own mapping paths.
    ///
    /// # Errors
    /// Returns model/raster errors from page recognition.
    pub fn recognize_document_page(
        &mut self,
        rgb: RgbImage,
        mapper: DisplayMapper,
        page: crate::PageNumber,
        residual: bool,
    ) -> Result<Vec<OcrLine>, Error> {
        let key = RasterKey::new(&rgb);
        let width = f64::from(rgb.width());
        let height = f64::from(rgb.height());
        let lines = self
            .recognize_keyed(rgb, key)?
            .into_iter()
            .map(|line| line.into_pdf(&|Point(x, y)| mapper.to_pdf(x / width, y / height)))
            .collect();
        self.record_character_mapping(RasterCharacterMapping {
            page,
            rgb_sha256: hex::encode(key.digest),
            dimensions: key.dimensions,
            scope: RasterMappingScope::WholePage { residual },
            raster_corners: [Point(0., 0.), Point(1., 0.), Point(1., 1.), Point(0., 1.)]
                .map(|Point(u, v)| mapper.to_pdf(u, v)),
        });
        Ok(lines)
    }

    /// Record an occurrence only when actual character evidence exists.
    pub(crate) fn record_character_mapping(&mut self, mapping: RasterCharacterMapping) {
        if self.character_audits.iter().any(|audit| {
            audit.rgb_sha256 == mapping.rgb_sha256 && audit.dimensions == mapping.dimensions
        }) {
            self.raster_character_mappings.push(mapping);
        }
    }

    fn record_characters(&mut self, key: RasterKey, report: CharacterAuditReport) -> bool {
        self.stats.orientation_classified += report.orientation.len();
        self.stats.orientation_half_turns +=
            report.orientation.iter().filter(|e| e.applied).count();
        self.stats.confirmation_batches += report.confirmation_batches;
        self.stats.confirmation_attempts += report.confirmations.len();
        self.stats.confirmation_accepted += report
            .confirmations
            .iter()
            .filter(|c| c.outcome.accepted())
            .count();
        let failures = report
            .confirmations
            .iter()
            .filter(|c| c.outcome.failed())
            .count();
        self.stats.confirmation_failures += failures;
        if !report.orientation.is_empty()
            || !report.lines.is_empty()
            || report.skipped_budget > 0
            || report.invalid_metadata > 0
        {
            self.character_audits.push(RasterCharacterAudit {
                dimensions: key.dimensions,
                rgb_sha256: hex::encode(key.digest),
                report,
            });
        }
        failures == 0
    }

    fn record_repairs(&mut self, key: RasterKey, report: RepairReport) -> bool {
        self.stats.crop_batches += report.batches;
        for repair in &report.repairs {
            let accepted = usize::from(repair.outcome.accepted());
            match repair.reason {
                CropReason::PeriodicLeader { .. } => {
                    self.stats.leader_candidates += 1;
                    self.stats.leader_accepted += accepted;
                }
                CropReason::BoundaryInk { .. } => {
                    self.stats.boundary_candidates += 1;
                    self.stats.boundary_accepted += accepted;
                }
            }
        }
        let failures = report.repairs.iter().filter(|r| r.outcome.failed()).count();
        self.stats.crop_failures += failures;
        if !report.repairs.is_empty() {
            self.repair_audits.push(RasterRepairAudit {
                dimensions: key.dimensions,
                rgb_sha256: hex::encode(key.digest),
                batches: report.batches,
                repairs: report.repairs,
            });
        }
        failures == 0
    }

    fn remember(&mut self, key: RasterKey, lines: &[PixelLine]) {
        let bytes = lines.iter().fold(
            size_of::<CacheEntry>()
                .saturating_add(lines.len().saturating_mul(size_of::<PixelLine>())),
            |total, line| total.saturating_add(line.text.len()),
        );
        if bytes > self.budget.max_bytes {
            return;
        }
        while self.entries.len() >= self.budget.max_entries
            || bytes > self.budget.max_bytes - self.stats.retained_bytes
        {
            let Some(evicted) = self.entries.pop_front() else {
                return;
            };
            self.stats.retained_bytes -= evicted.bytes;
            self.stats.evictions += 1;
        }
        self.entries.push_back(CacheEntry {
            key,
            lines: lines.to_vec(),
            bytes,
        });
        self.stats.retained_bytes += bytes;
    }

    /// Recognize a raster, reusing successful results for identical inputs.
    ///
    /// # Errors
    /// Returns an error for an empty raster or failed model inference.
    pub fn recognize_pixels(&mut self, rgb: RgbImage) -> Result<Vec<PixelLine>, Error> {
        let key = RasterKey::new(&rgb);
        self.recognize_keyed(rgb, key)
    }

    fn recognize_keyed(&mut self, rgb: RgbImage, key: RasterKey) -> Result<Vec<PixelLine>, Error> {
        if rgb.width() == 0 || rgb.height() == 0 {
            return Err(Error::Invalid("empty OCR raster".into()));
        }
        if self.budget.enabled()
            && let Some(index) = self.entries.iter().position(|entry| entry.key == key)
            && let Some(entry) = self.entries.remove(index)
        {
            self.stats.cache_hits += 1;
            let result = entry.lines.clone();
            self.entries.push_back(entry);
            return Ok(result);
        }
        self.stats.inference_calls += 1;
        let RasterRecognition {
            lines,
            crop_repair,
            character_audit,
        } = self.backend.recognize_pixels(rgb)?;
        let mut detections = BTreeSet::new();
        if lines
            .iter()
            .any(|line| !detections.insert(line.detection_index))
        {
            return Err(Error::Invalid("duplicate raster detection identity".into()));
        }
        let complete_characters = self.record_characters(key, character_audit);
        let complete = self.record_repairs(key, crop_repair) && complete_characters;
        if self.budget.enabled() && complete {
            self.remember(key, &lines);
        }
        Ok(lines)
    }

    /// Recognize in pixel space and project into the current occurrence.
    ///
    /// # Errors
    /// Returns an error for an empty raster or failed model inference.
    pub fn recognize_mapped(
        &mut self,
        rgb: RgbImage,
        mapper: &dyn Fn(Point) -> Point,
    ) -> Result<Vec<OcrLine>, Error> {
        Ok(self
            .recognize_pixels(rgb)?
            .into_iter()
            .map(|line| line.into_pdf(mapper))
            .collect())
    }

    /// Recognize a complete raster using its normalized display mapping.
    ///
    /// # Errors
    /// Returns an error for an empty raster or failed model inference.
    pub fn recognize_page(
        &mut self,
        rgb: RgbImage,
        mapper: DisplayMapper,
    ) -> Result<Vec<OcrLine>, Error> {
        let width = f64::from(rgb.width());
        let height = f64::from(rgb.height());
        self.recognize_mapped(rgb, &|point| {
            mapper.to_pdf(point.0 / width, point.1 / height)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PixelLine, RasterRecognition, RecognitionBackend, RecognitionCacheBudget,
        RecognitionSession,
    };
    use crate::{Error, geometry::Point};
    use image::{Rgb, RgbImage};
    use std::cell::Cell;

    #[derive(Default)]
    struct Backend {
        calls: Cell<usize>,
        fail_next: Cell<bool>,
        empty: bool,
    }

    impl RecognitionBackend for Backend {
        fn model_profile(&self) -> &'static str {
            "pixel cache fixture"
        }

        fn recognize_pixels(&self, _rgb: RgbImage) -> Result<RasterRecognition, Error> {
            self.calls.set(self.calls.get() + 1);
            if self.fail_next.replace(false) {
                return Err(Error::Inference("transient fixture failure".into()));
            }
            Ok(if self.empty {
                vec![]
            } else {
                vec![PixelLine {
                    detection_index: 0,
                    text: "source text".into(),
                    confidence: 0.99,
                    quad: [
                        Point(1.0, 3.0),
                        Point(5.0, 3.0),
                        Point(5.0, 1.0),
                        Point(1.0, 1.0),
                    ],
                }]
            }
            .into())
        }
    }

    #[test]
    fn pixel_identity_is_separate_from_occurrence_geometry() -> Result<(), Error> {
        let backend = Backend::default();
        let mut session = RecognitionSession::new(&backend, RecognitionCacheBudget::default());
        let rgb = RgbImage::from_pixel(8, 4, Rgb([12, 34, 56]));
        let first = session.recognize_mapped(rgb.clone(), &|p| p)?;
        let second = session.recognize_mapped(rgb.clone(), &|Point(x, y)| {
            Point(y.mul_add(-2.0, 100.0), x.mul_add(3.0, 20.0))
        })?;
        assert_eq!(backend.calls.get(), 1);
        assert_eq!(first[0].text, second[0].text);
        assert_eq!(
            second[0].quad,
            [
                Point(94.0, 23.0),
                Point(94.0, 35.0),
                Point(98.0, 35.0),
                Point(98.0, 23.0)
            ]
        );
        let mut changed = rgb.clone();
        changed.put_pixel(7, 3, Rgb([12, 34, 57]));
        session.recognize_pixels(changed)?;
        let reshaped = RgbImage::from_raw(4, 8, rgb.into_raw())
            .ok_or_else(|| Error::Invalid("invalid fixture raster".into()))?;
        session.recognize_pixels(reshaped)?;
        assert_eq!(session.stats().inference_calls, 3);
        assert_eq!(session.stats().cache_hits, 1);
        Ok(())
    }

    #[test]
    fn bounded_lru_caches_empty_successes_but_not_errors() -> Result<(), Error> {
        let backend = Backend {
            empty: true,
            ..Backend::default()
        };
        let budget = RecognitionCacheBudget {
            max_entries: 2,
            max_bytes: 4096,
        };
        let mut session = RecognitionSession::new(&backend, budget);
        let image = |v| RgbImage::from_pixel(8, 4, Rgb([v; 3]));
        backend.fail_next.set(true);
        assert!(session.recognize_pixels(image(0)).is_err());
        for value in [0, 1, 0, 2, 0, 1] {
            assert_eq!(
                session.recognize_pixels(image(value))?,
                [] as [PixelLine; 0]
            );
        }
        assert_eq!(session.stats().inference_calls, 5);
        assert_eq!(session.stats().cache_hits, 2);
        assert_eq!(session.stats().evictions, 2);
        assert!(session.stats().retained_bytes <= budget.max_bytes);
        for cache in [
            RecognitionCacheBudget::DISABLED,
            RecognitionCacheBudget {
                max_entries: 8,
                max_bytes: 1,
            },
        ] {
            let mut other = RecognitionSession::new(&backend, cache);
            other.recognize_pixels(image(0))?;
            other.recognize_pixels(image(0))?;
            assert_eq!(other.stats().inference_calls, 2);
            assert_eq!(other.stats().retained_bytes, 0);
        }
        Ok(())
    }
}
