//! Select Image `XObjects` worth native-resolution OCR before page fallback.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Error, PageNumber, geometry::PageBox, rasterize::DEFAULT_PAGE_MAX_SIDE};

/// One source image occurrence within a PDF paint stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePlacement {
    /// One-based page number.
    pub page: u32,
    /// Stable indirect object identifier.
    pub source_id: String,
    /// Native width in pixels.
    pub width: u32,
    /// Native height in pixels.
    pub height: u32,
    /// Current PDF transformation matrix.
    pub ctm: [f64; 6],
    /// Reasons forbidding source-level OCR.
    pub unsafe_reasons: Vec<String>,
}

/// Explicit resource limits for extra inference.
#[derive(Debug, Clone, Copy)]
pub struct SourcePolicy {
    /// Longest page-render side.
    pub max_side: u32,
    /// Maximum additional native-source pixels.
    pub max_inference_pixels: u64,
    /// Maximum independent source `XObjects`.
    pub max_sources: usize,
    /// Minimum source pixel area.
    pub minimum_source_pixels: u64,
    /// Minimum native-to-rendered pixel-area gain.
    pub min_gain: f64,
}

impl Default for SourcePolicy {
    fn default() -> Self {
        Self {
            max_side: DEFAULT_PAGE_MAX_SIDE,
            max_inference_pixels: 8_000_000,
            max_sources: 12,
            minimum_source_pixels: 20_000,
            min_gain: 2.25,
        }
    }
}

impl SourcePolicy {
    fn validate(self) -> Result<(), Error> {
        if self.max_side < 100
            || self.max_inference_pixels == 0
            || self.max_sources == 0
            || self.minimum_source_pixels == 0
            || !self.min_gain.is_finite()
            || self.min_gain <= 1.0
        {
            return Err(Error::Invalid(
                "invalid source-native OCR selection policy".into(),
            ));
        }
        Ok(())
    }
}

/// Per-source diagnostic for auditing native image decisions.
#[derive(Debug, Clone, Serialize)]
pub struct SourceDiagnostic {
    /// Source object identifier.
    pub source_id: String,
    /// Native pixels charged for one recognition.
    pub native_source_pixels: u64,
    /// Highest resolution gain over page rendering.
    pub native_to_page_render_area_gain: f64,
    /// Occurrences with lost page-render detail.
    pub eligible_placements: usize,
    /// Recoverable pixel area per one source inference cost.
    pub score: f64,
    /// Whether source was admitted under the policy.
    pub selected: bool,
}

/// Deterministic report consumed by the full hybrid coordinator.
#[derive(Debug, Clone, Serialize)]
pub struct SelectionReport {
    /// Source candidates before budget enforcement.
    pub native_resolution_candidates: usize,
    /// Sources admitted under the policy.
    pub selected_sources: usize,
    /// Total native pixels admitted.
    pub pixel_budget_used: u64,
    /// Detailed source decisions.
    pub sources: Vec<SourceDiagnostic>,
}

#[derive(Clone, Copy)]
struct Appearance {
    pixels: u64,
    pixel_area: f64,
    rendered: f64,
    gain: f64,
}

struct Candidate {
    source_id: String,
    native_pixels: u64,
    gain: f64,
    score: f64,
    uses: usize,
}

/// Select sources whose original pixels materially exceed page-render detail.
///
/// Reports remain stable regardless of the input placement order. A missing
/// `MediaBox` fails closed rather than using an incorrect crop or page default.
///
/// # Errors
/// Returns an error for invalid budgets or missing page `MediaBox` geometry.
pub fn choose_mixed_raster_sources(
    placements: &[ImagePlacement],
    pages: &BTreeMap<PageNumber, PageBox>,
    policy: SourcePolicy,
) -> Result<(BTreeSet<String>, SelectionReport), Error> {
    policy.validate()?;
    let scale = pages
        .iter()
        .map(|(&page, &box_)| {
            (
                page.get(),
                f64::from(policy.max_side) / box_.width().max(box_.height()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut sources: BTreeMap<&str, Vec<Appearance>> = BTreeMap::new();

    for placement in placements {
        let page_scale = *scale.get(&placement.page).ok_or_else(|| {
            Error::Invalid(format!("missing MediaBox for page {}", placement.page))
        })?;
        if !placement.unsafe_reasons.is_empty() {
            continue;
        }
        let pixels = u64::from(placement.width) * u64::from(placement.height);
        let pixel_area = f64::from(placement.width) * f64::from(placement.height);
        if pixels < policy.minimum_source_pixels || placement.width.min(placement.height) < 24 {
            continue;
        }
        let [scale_x, shear_y, shear_x, scale_y, _, _] = placement.ctm;
        let rendered = shear_y.mul_add(-shear_x, scale_x * scale_y).abs() * page_scale.powi(2);
        if !rendered.is_finite() || rendered <= 0.0 {
            continue;
        }
        let gain = pixel_area / rendered;
        if gain < policy.min_gain {
            continue;
        }
        sources
            .entry(&placement.source_id)
            .or_default()
            .push(Appearance {
                pixels,
                pixel_area,
                rendered,
                gain,
            });
    }

    let mut ranked = Vec::new();
    for (source_id, appearances) in sources {
        let Some(best) = appearances.iter().max_by(|a, b| a.gain.total_cmp(&b.gain)) else {
            continue;
        };
        let saved = appearances
            .iter()
            .map(|appearance| (appearance.pixel_area - appearance.rendered).max(0.0))
            .sum::<f64>();
        ranked.push(Candidate {
            source_id: source_id.to_owned(),
            native_pixels: best.pixels,
            gain: best.gain,
            score: saved / best.pixel_area,
            uses: appearances.len(),
        });
    }
    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.gain.total_cmp(&a.gain))
            .then_with(|| b.uses.cmp(&a.uses))
            .then_with(|| a.source_id.cmp(&b.source_id))
    });

    let mut selected = BTreeSet::new();
    let mut budget = 0_u64;
    let mut diagnostics = Vec::with_capacity(ranked.len());
    for item in &ranked {
        let eligible = selected.len() < policy.max_sources
            && item.native_pixels <= policy.max_inference_pixels.saturating_sub(budget);
        diagnostics.push(SourceDiagnostic {
            source_id: item.source_id.clone(),
            native_source_pixels: item.native_pixels,
            native_to_page_render_area_gain: round3(item.gain),
            eligible_placements: item.uses,
            score: round3(item.score),
            selected: eligible,
        });
        if eligible {
            budget += item.native_pixels;
            selected.insert(item.source_id.clone());
        }
    }

    Ok((
        selected.clone(),
        SelectionReport {
            native_resolution_candidates: ranked.len(),
            selected_sources: selected.len(),
            pixel_budget_used: budget,
            sources: diagnostics,
        },
    ))
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round_ties_even() / 1000.0
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{ImagePlacement, SourcePolicy, choose_mixed_raster_sources};
    use crate::{PageNumber, geometry::PageBox};

    fn image(page: u32) -> ImagePlacement {
        ImagePlacement {
            page,
            source_id: "15:0".into(),
            width: 1200,
            height: 300,
            ctm: [240.0, 0.0, 0.0, 60.0, 80.0, 400.0],
            unsafe_reasons: Vec::new(),
        }
    }

    #[test]
    fn sampling_matches_mediabox_reference() -> Result<(), crate::Error> {
        let pages =
            BTreeMap::from([(PageNumber::new(1)?, PageBox::new([0.0, 0.0, 800.0, 600.0])?)]);
        let (selected, report) = choose_mixed_raster_sources(
            &[image(1)],
            &pages,
            SourcePolicy {
                max_side: 1600,
                ..SourcePolicy::default()
            },
        )?;
        assert!(selected.contains("15:0"));
        assert_eq!(report.sources[0].native_to_page_render_area_gain, 6.25);
        Ok(())
    }

    #[test]
    fn missing_geometry_is_rejected_and_budget_is_enforced() -> Result<(), crate::Error> {
        assert!(
            choose_mixed_raster_sources(&[image(2)], &BTreeMap::new(), SourcePolicy::default())
                .is_err()
        );
        let pages =
            BTreeMap::from([(PageNumber::new(2)?, PageBox::new([0.0, 0.0, 800.0, 600.0])?)]);
        let (selected, report) = choose_mixed_raster_sources(
            &[image(2)],
            &pages,
            SourcePolicy {
                max_inference_pixels: 100_000,
                ..SourcePolicy::default()
            },
        )?;
        assert!(selected.is_empty());
        assert_eq!(report.native_resolution_candidates, 1);
        Ok(())
    }
}
