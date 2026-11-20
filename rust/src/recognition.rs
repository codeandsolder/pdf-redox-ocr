//! Pixel-space OCR independent of the source's document geometry.

use std::{collections::VecDeque, mem::size_of};

use image::RgbImage;
use sha2::{Digest, Sha256};

use crate::{
    Error,
    draft::OcrLine,
    geometry::{DisplayMapper, Point, Quad},
};

/// A model observation in the original input raster's pixel-corner frame.
///
/// Coordinates grow rightwards and downwards. Corners are ordered bottom-left,
/// bottom-right, top-right, top-left relative to the detected text orientation.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelLine {
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
    fn recognize_pixels(&self, rgb: RgbImage) -> Result<Vec<PixelLine>, Error>;
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
}

impl<'a, B: RecognitionBackend> RecognitionSession<'a, B> {
    #[must_use]
    pub const fn new(backend: &'a B, budget: RecognitionCacheBudget) -> Self {
        Self {
            backend,
            budget,
            entries: VecDeque::new(),
            stats: RecognitionStats {
                inference_calls: 0,
                cache_hits: 0,
                evictions: 0,
                retained_bytes: 0,
            },
        }
    }

    #[must_use]
    pub const fn stats(&self) -> RecognitionStats {
        self.stats
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
        if rgb.width() == 0 || rgb.height() == 0 {
            return Err(Error::Invalid("empty OCR raster".into()));
        }
        let key = self.budget.enabled().then(|| RasterKey::new(&rgb));
        if let Some(key) = key
            && let Some(index) = self.entries.iter().position(|entry| entry.key == key)
            && let Some(entry) = self.entries.remove(index)
        {
            self.stats.cache_hits += 1;
            let result = entry.lines.clone();
            self.entries.push_back(entry);
            return Ok(result);
        }
        self.stats.inference_calls += 1;
        let result = self.backend.recognize_pixels(rgb)?;
        if let Some(key) = key {
            self.remember(key, &result);
        }
        Ok(result)
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
    use super::{PixelLine, RecognitionBackend, RecognitionCacheBudget, RecognitionSession};
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

        fn recognize_pixels(&self, _rgb: RgbImage) -> Result<Vec<PixelLine>, Error> {
            self.calls.set(self.calls.get() + 1);
            if self.fail_next.replace(false) {
                return Err(Error::Inference("transient fixture failure".into()));
            }
            Ok(if self.empty {
                vec![]
            } else {
                vec![PixelLine {
                    text: "source text".into(),
                    confidence: 0.99,
                    quad: [
                        Point(1.0, 3.0),
                        Point(5.0, 3.0),
                        Point(5.0, 1.0),
                        Point(1.0, 1.0),
                    ],
                }]
            })
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
