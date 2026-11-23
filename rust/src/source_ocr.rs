//! Source-native recognition for PDF pages with image-only visible content.
//!
//! A single source `XObject` may be painted repeatedly. Recognition is cached
//! per appearance and geometrically projected separately per paint invocation.
//! Admission always uses the conservative actual-paint inspection evidence.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Error, PageNumber,
    draft::OcrLine,
    geometry::{DisplayMapper, PageBox, Point, Rotation},
    hybrid::{ImagePaintEvidence, RasterPageEvidence, approve_raster_only_pages},
    paint_scan::{ImagePaint, PaintReport, PdfMatrix},
    polygon::area,
    source_appearances::PaintAppearances,
};

/// One actual detector identity and its source-local OCR line.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceLine {
    pub detection_index: usize,
    #[serde(flatten)]
    pub line: OcrLine,
}

/// One native image's OCR observations in the image's own user-space frame.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceObservation {
    /// Original native image width.
    pub width: u32,
    /// Original native image height.
    pub height: u32,
    /// OCR observation on an unrotated image with bottom-left origin.
    pub lines: Vec<SourceLine>,
}

impl SourceObservation {
    /// Borrow the source-local polygons in the original detector output order.
    #[must_use = "use source-local recognized lines when attributing PDF text"]
    pub fn local_lines(&self) -> Vec<&OcrLine> {
        self.lines.iter().map(|entry| &entry.line).collect()
    }
}

/// Project PDF image-source space into each original PDF paint occurrence.
///
/// # Errors
/// Rejects mismatched dimensions, invalid confidence, degenerate/out-of-bounds
/// source quads, or non-finite projections.
pub fn map_observations_to_paint(
    source: &SourceObservation,
    paint: &ImagePaint,
) -> Result<Vec<OcrLine>, Error> {
    if source.width == 0
        || source.height == 0
        || source.width != paint.width
        || source.height != paint.height
    {
        return Err(Error::Invalid(
            "source OCR pixel dimensions differ from PDF `XObject`".into(),
        ));
    }
    let width = f64::from(source.width);
    let height = f64::from(source.height);
    let matrix = PdfMatrix(paint.ctm);
    let mapped = source
        .lines
        .iter()
        .map(|source| {
            let line = &source.line;
            if !line.confidence.is_finite()
                || !(0.0..=1.0).contains(&line.confidence)
                || area(&line.quad) <= 1e-9
                || line.quad.iter().any(|point| {
                    !point.0.is_finite()
                        || !point.1.is_finite()
                        || !(0.0..=width).contains(&point.0)
                        || !(0.0..=height).contains(&point.1)
                })
            {
                return Err(Error::Invalid(
                    "invalid native source OCR confidence or quad".into(),
                ));
            }
            let projected = line
                .quad
                .map(|point| matrix.point(Point(point.0 / width, point.1 / height)));
            if projected
                .iter()
                .any(|point| !point.0.is_finite() || !point.1.is_finite())
            {
                return Err(Error::Invalid("source OCR projection was nonfinite".into()));
            }
            let mut projected_line = line.clone();
            projected_line.quad = projected;
            Ok(projected_line)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(mapped)
}

/// Eligible full-source `XObjects` for source-native OCR, based on actual paints.
///
/// Pages with visible vector content or unsupported graphics state still use
/// page fallback. Only image-only pages are considered for the fast path.
#[must_use = "use eligible source-only candidates when selecting OCR regions"]
pub fn source_only_candidates(report: &PaintReport, max_pixels: u64) -> BTreeSet<String> {
    let mut eligible = report
        .pages
        .iter()
        .map(|page| (page.page, page.image_only_after_native))
        .collect::<BTreeMap<_, _>>();
    for paint in &report.placements {
        let safe = paint.unsafe_reasons.is_empty()
            && paint.width > 0
            && paint.height > 0
            && u64::from(paint.width) * u64::from(paint.height) <= max_pixels;
        if !safe {
            eligible.insert(paint.page, false);
        }
    }
    report
        .placements
        .iter()
        .filter(|paint| eligible.get(&paint.page) == Some(&true))
        .map(|paint| paint.source_id.clone())
        .collect()
}

/// Exact detector IDs accepted at each expanded source paint occurrence.
pub(crate) type SourceAdmissions = BTreeMap<(PageNumber, u32), BTreeSet<usize>>;

/// The subset of selected pages proven to be safe source-only OCR targets.
#[derive(Debug, Default)]
pub struct SourceOnlyResult {
    /// Recognitions mapped into final PDF page coordinates.
    pub pages: BTreeMap<PageNumber, Vec<OcrLine>>,
    /// Fully source-only pages, eligible to skip regular page OCR.
    pub approved_pages: BTreeSet<PageNumber>,
    pub(crate) admissions: SourceAdmissions,
}

/// Decide whether source OCR can replace ordinary page OCR.
///
/// A recognized image is not enough: all actual painted images must be
/// decoded/recognized, all source quads inside the native raster and known clips, there must be
/// at least one positive text observation, and every image layer must be
/// mutually non-overlapping. Named `XObject` color-space aliases require a
/// separate rendered visibility proof, even if every paint decoded natively.
///
/// # Errors
/// Returns an error for invalid page numbering; unusable source observations trigger page fallback.
pub fn source_only_pages(
    report: &PaintReport,
    recognized: &PaintAppearances<SourceObservation>,
) -> Result<SourceOnlyResult, Error> {
    let mut result = SourceOnlyResult::default();
    for summary in &report.pages {
        let page = PageNumber::new(summary.page)?;
        if !summary.image_only_after_native {
            continue;
        }
        let paints = report
            .placements
            .iter()
            .filter(|paint| paint.page == summary.page)
            .collect::<Vec<_>>();
        let mut lines = Vec::new();
        let mut admissions = SourceAdmissions::new();
        let mut evidence = Vec::with_capacity(paints.len());
        for paint in paints {
            let known_source = recognized.get(paint);
            // Named XObject color spaces can be decoded by Hayro but omitted
            // by the actual page renderer. They require a differential proof.
            let mut safe = paint.unsafe_reasons.is_empty()
                && !paint.sample_layout.contextual
                && known_source.is_some();
            if safe && let Some((_, observation)) = known_source {
                match map_observations_to_paint(observation, paint) {
                    Ok(mapped) => {
                        safe &= mapped
                            .iter()
                            .all(|line| paint.wholly_visible_in_clip(&line.quad));
                        if safe {
                            let admitted = admissions.entry((page, paint.paint_index)).or_default();
                            for (projected, source) in mapped.into_iter().zip(&observation.lines) {
                                if !projected.text.trim().is_empty() && projected.confidence > 0.0 {
                                    admitted.insert(source.detection_index);
                                    lines.push(projected);
                                }
                            }
                        }
                    }
                    Err(Error::Invalid(_)) => safe = false,
                    Err(error) => return Err(error),
                }
            }
            evidence.push(ImagePaintEvidence {
                page_quad: paint.page_quad,
                recognized: safe,
            });
        }
        let decisions = approve_raster_only_pages(&[RasterPageEvidence {
            page: summary.page,
            image_only_after_native: summary.image_only_after_native,
            paints: evidence,
            recognized_lines: lines.len(),
        }]);
        if decisions.first().is_some_and(|decision| decision.approved) {
            result.approved_pages.insert(page);
            result.pages.insert(page, lines);
            result.admissions.extend(admissions);
        }
    }
    Ok(result)
}

/// Standard source-local mapper for an unrotated native decoded bitmap.
///
/// # Errors
/// Rejects invalid native raster dimensions.
pub fn source_image_mapper(width: u32, height: u32) -> Result<DisplayMapper, Error> {
    let bounds = PageBox::new([0.0, 0.0, f64::from(width), f64::from(height)])?;
    Ok(DisplayMapper::new(bounds, Rotation::Deg0))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{SourceLine, SourceObservation, source_only_pages};
    use crate::source_appearances::PaintAppearances;
    use crate::{
        draft::OcrLine,
        geometry::Point,
        paint_scan::{ImagePaint, PagePaintSummary, PaintReport},
    };

    fn quad(x: f64, y: f64) -> [Point; 4] {
        [
            Point(x, y),
            Point(x + 80.0, y),
            Point(x + 80.0, y + 20.0),
            Point(x, y + 20.0),
        ]
    }

    #[test]
    fn repeated_image_has_distinct_pdf_placements_with_single_source_result()
    -> Result<(), crate::Error> {
        let source = SourceObservation {
            width: 200,
            height: 100,
            lines: vec![SourceLine {
                detection_index: 0,
                line: OcrLine {
                    text: "Capacitor 47uF".into(),
                    confidence: 0.99,
                    quad: quad(10.0, 10.0),
                    reviewed: false,
                },
            }],
        };
        let paint = |page, tx| ImagePaint {
            page,
            paint_index: 0,
            paint_path: vec![0],
            source_id: "9:0".into(),
            object_number: 9,
            generation: 0,
            width: 200,
            height: 100,
            sample_layout: crate::sample_layout::ImageSampleLayout::default(),
            ctm: [200.0, 0.0, 0.0, 100.0, tx, 20.0],
            page_quad: [
                Point(tx, 20.0),
                Point(tx + 200.0, 20.0),
                Point(tx + 200.0, 120.0),
                Point(tx, 120.0),
            ],
            clips: vec![[
                Point(0.0, 0.0),
                Point(600.0, 0.0),
                Point(600.0, 800.0),
                Point(0.0, 800.0),
            ]],
            unsafe_reasons: Vec::new(),
        };
        let report = PaintReport {
            pages: vec![
                PagePaintSummary {
                    page: 1,
                    image_paints: 1,
                    image_only_after_native: true,
                    ..PagePaintSummary::default()
                },
                PagePaintSummary {
                    page: 2,
                    image_paints: 1,
                    image_only_after_native: true,
                    ..PagePaintSummary::default()
                },
            ],
            placements: vec![paint(1, 20.0), paint(2, 120.0)],
        };
        let result = source_only_pages(
            &report,
            &PaintAppearances::from_object_inventory(
                BTreeMap::from([("9:0".into(), source)]),
                &report.placements,
            ),
        )?;
        assert_eq!(result.approved_pages.len(), 2);
        let first = &result.pages[&crate::PageNumber::new(1)?][0];
        let second = &result.pages[&crate::PageNumber::new(2)?][0];
        assert_eq!(first.quad[0], Point(30.0, 30.0));
        assert_eq!(second.quad[0], Point(130.0, 30.0));
        assert_eq!(first.text, second.text);
        Ok(())
    }

    fn single_paint_report() -> PaintReport {
        PaintReport {
            pages: vec![PagePaintSummary {
                page: 1,
                image_only_after_native: true,
                image_paints: 1,
                ..PagePaintSummary::default()
            }],
            placements: vec![ImagePaint {
                page: 1,
                paint_index: 0,
                paint_path: vec![0],
                source_id: "9:0".into(),
                object_number: 9,
                generation: 0,
                width: 200,
                height: 100,
                sample_layout: crate::sample_layout::ImageSampleLayout::default(),
                ctm: [200.0, 0.0, 0.0, 100.0, 20.0, 20.0],
                page_quad: [
                    Point(20.0, 20.0),
                    Point(220.0, 20.0),
                    Point(220.0, 120.0),
                    Point(20.0, 120.0),
                ],
                clips: vec![],
                unsafe_reasons: vec![],
            }],
        }
    }

    #[test]
    fn missing_or_invalid_source_observation_never_approves_page() -> Result<(), crate::Error> {
        let report = single_paint_report();
        let result = source_only_pages(&report, &PaintAppearances::default())?;
        assert!(result.approved_pages.is_empty());

        let invalid_source = SourceObservation {
            width: 200,
            height: 100,
            lines: vec![SourceLine {
                detection_index: 0,
                line: OcrLine {
                    text: "Outside image".into(),
                    confidence: 0.99,
                    quad: quad(150.0, 90.0),
                    reviewed: false,
                },
            }],
        };
        assert!(super::map_observations_to_paint(&invalid_source, &report.placements[0]).is_err());
        let result = source_only_pages(
            &report,
            &PaintAppearances::from_object_inventory(
                BTreeMap::from([("9:0".into(), invalid_source)]),
                &report.placements,
            ),
        )?;
        assert!(result.approved_pages.is_empty());
        assert!(result.pages.is_empty());

        let wrong_size = SourceObservation {
            width: 199,
            height: 100,
            lines: Vec::new(),
        };
        let result = source_only_pages(
            &report,
            &PaintAppearances::from_object_inventory(
                BTreeMap::from([("9:0".into(), wrong_size)]),
                &report.placements,
            ),
        )?;
        assert!(result.approved_pages.is_empty());

        let mut uninformative = OcrLine {
            text: "   ".into(),
            confidence: 0.99,
            quad: quad(10.0, 10.0),
            reviewed: false,
        };
        for (text, confidence) in [
            ("   ", 0.99),
            ("Unreliable", 0.0),
            ("Nonfinite", f64::NAN),
            ("Out of range", 1.01),
        ] {
            uninformative.text = text.into();
            uninformative.confidence = confidence;
            let source = SourceObservation {
                width: 200,
                height: 100,
                lines: vec![SourceLine {
                    detection_index: 0,
                    line: uninformative.clone(),
                }],
            };
            let result = source_only_pages(
                &report,
                &PaintAppearances::from_object_inventory(
                    BTreeMap::from([("9:0".into(), source)]),
                    &report.placements,
                ),
            )?;
            assert!(
                result.approved_pages.is_empty(),
                "uninformative line {text:?} with confidence {confidence} approved source-only OCR"
            );
        }
        let degenerate = SourceObservation {
            width: 200,
            height: 100,
            lines: vec![SourceLine {
                detection_index: 0,
                line: OcrLine {
                    text: "Degenerate geometry".into(),
                    confidence: 0.99,
                    quad: [Point(10.0, 20.0); 4],
                    reviewed: false,
                },
            }],
        };
        assert!(super::map_observations_to_paint(&degenerate, &report.placements[0]).is_err());
        let result = source_only_pages(
            &report,
            &PaintAppearances::from_object_inventory(
                BTreeMap::from([("9:0".into(), degenerate)]),
                &report.placements,
            ),
        )?;
        assert!(result.approved_pages.is_empty());

        Ok(())
    }

    #[test]
    fn blank_line_does_not_prevent_valid_source_admission() -> Result<(), crate::Error> {
        let report = single_paint_report();
        let meaningful = OcrLine {
            text: "CAPACITOR".into(),
            confidence: 0.99,
            quad: quad(10.0, 10.0),
            reviewed: false,
        };
        let blank = OcrLine {
            text: "  ".into(),
            confidence: 0.99,
            quad: quad(10.0, 40.0),
            reviewed: false,
        };
        let source = SourceObservation {
            width: 200,
            height: 100,
            lines: vec![
                SourceLine {
                    detection_index: 0,
                    line: blank,
                },
                SourceLine {
                    detection_index: 1,
                    line: meaningful,
                },
            ],
        };
        let result = source_only_pages(
            &report,
            &PaintAppearances::from_object_inventory(
                BTreeMap::from([("9:0".into(), source)]),
                &report.placements,
            ),
        )?;
        let admitted = result
            .pages
            .get(&crate::PageNumber::new(1)?)
            .ok_or_else(|| crate::Error::Invalid("textful raster-only page rejected".into()))?;
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].text, "CAPACITOR");
        Ok(())
    }
}
