//! Complete native OCR orchestration with deterministic mapped recognition.

#![cfg(feature = "pdf-integration")]

use std::{cell::RefCell, fs};

use image::RgbImage;
use pdf_redox_ocr::{
    Error, PageNumber,
    draft::OcrLine,
    geometry::{Point, Quad},
    native_images::NativeImageBudget,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    pipeline::{PipelinePolicy, PreparedOcr},
    rasterize::RenderPolicy,
    rasterize::render_rgb_page,
    recognition::RecognitionBackend,
};
use sha2::{Digest, Sha256};

#[derive(Default)]
struct Recognizer {
    calls: RefCell<Vec<(u32, u32)>>,
    fail_native: bool,
    recognize_fallback: bool,
    fallback_digests: RefCell<Vec<[u8; 32]>>,
    conflict_once: Option<Quad>,
    fail_residual_once: bool,
}

impl RecognitionBackend for Recognizer {
    fn model_profile(&self) -> &'static str {
        "fixture recognizer"
    }

    fn recognize_mapped(
        &self,
        rgb: RgbImage,
        mapper: &dyn Fn(Point) -> Point,
    ) -> Result<Vec<OcrLine>, Error> {
        let (w, h) = rgb.dimensions();
        self.calls.borrow_mut().push((w, h));
        if (w, h) != (1200, 300) {
            let mut digests = self.fallback_digests.borrow_mut();
            digests.push(Sha256::digest(rgb.as_raw()).into());
            if digests.len() == 1 {
                if self.fail_residual_once {
                    return Err(Error::Inference("fixture residual failure".into()));
                }
                if let Some(quad) = self.conflict_once {
                    return Ok(vec![OcrLine {
                        text: "Conflicting residual".into(),
                        confidence: 0.99,
                        quad,
                        reviewed: false,
                    }]);
                }
            }
        }
        let quad = if (w, h) == (1200, 300) {
            if self.fail_native {
                return Err(Error::Inference(
                    "fixture native recognition failure".into(),
                ));
            }
            [
                Point(75.0, 180.0),
                Point(1060.0, 180.0),
                Point(1060.0, 100.0),
                Point(75.0, 100.0),
            ]
        } else if self.recognize_fallback {
            let (w, h) = (f64::from(w), f64::from(h));
            [
                Point(w * 0.25, h * 0.75),
                Point(w * 0.75, h * 0.75),
                Point(w * 0.75, h * 0.25),
                Point(w * 0.25, h * 0.25),
            ]
        } else {
            return Ok(vec![]);
        };
        Ok(vec![OcrLine {
            text: "CAPACITOR 47uF 16V".into(),
            confidence: 0.99,
            quad: quad.map(mapper),
            reviewed: false,
        }])
    }
}

fn policy() -> PipelinePolicy {
    PipelinePolicy {
        render: RenderPolicy {
            max_side: 1600,
            dpi: 72,
        },
        ..PipelinePolicy::default()
    }
}

#[test]
fn repeated_visible_source_is_recognized_once_and_occluded_paints_never_admitted()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("mixed.pdf");
    let original = include_bytes!("fixtures/repeated-nested-paint-occlusion.pdf");
    fs::write(&pdf, original)?;
    let job = PreparedOcr::open(&pdf, None, policy())?;
    let recognizer = Recognizer::default();
    let mut progress = Vec::new();
    let output = job.recognize(&recognizer, |message| progress.push(message))?;
    assert_eq!(output.stats.page_fallback_pages, 3, "{progress:#?}");
    assert_eq!(output.stats.image_source_only_pages, 0);
    assert_eq!(
        output
            .residual_pages
            .iter()
            .map(|p| p.get())
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert_eq!(
        output.stats.nonoverlapping_image_lines_added, 3,
        "{progress:#?} {:#?}",
        output.proposals
    );
    assert_eq!(
        recognizer.calls.borrow().as_slice(),
        &[(1200, 300), (1600, 1200), (1600, 1200), (1600, 1200)]
    );
    assert_eq!(output.proposals.len(), 3);
    assert_eq!(output.visibility.audited.len(), 6);
    assert_eq!(output.visibility.proofs.len(), 3);
    for page in &output.draft.pages {
        assert_eq!(page.lines.len(), 1);
        assert_eq!(page.lines[0].text, "CAPACITOR 47uF 16V");
        assert_eq!(
            page.native_text_regions,
            [] as [pdf_redox_ocr::geometry::Quad; 0]
        );
        let candidates = output
            .proposals
            .iter()
            .filter(|p| p.page == page.page)
            .collect::<Vec<_>>();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].verified_visible_source && candidates[0].included_as_extra);
        let audited = output
            .visibility
            .audited
            .iter()
            .filter(|a| a.page.get() == page.page)
            .collect::<Vec<_>>();
        assert_eq!(audited.len(), 2);
        assert_eq!(audited[0].visible_lines, 1);
        assert_eq!(audited[1].visible_lines, 0);
        assert_eq!(candidates[0].paint_index, audited[0].paint_index);
    }
    assert!(
        output
            .draft
            .model
            .contains("fixture recognizer native-source")
    );
    assert!(
        output
            .draft
            .model
            .contains("page-raster-max-side=1600 page-raster-dpi=72")
    );
    output.draft.verify_pdf(&pdf)?;
    assert_eq!(fs::read(&pdf)?, original);
    assert_eq!(job.original_bytes(), original);
    Ok(())
}

#[test]
fn diagnostic_can_disable_only_residual_recognition() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("mixed.pdf");
    fs::write(
        &pdf,
        include_bytes!("fixtures/repeated-nested-paint-occlusion.pdf"),
    )?;
    let job = PreparedOcr::open(
        &pdf,
        Some("1"),
        PipelinePolicy {
            residual_ocr: false,
            ..policy()
        },
    )?;
    let recognizer = Recognizer::default();
    let result = job.recognize(&recognizer, |_| {})?;
    assert_eq!(result.residual_pages, std::collections::BTreeSet::new());
    assert_eq!(result.visibility.proofs.len(), 1);
    assert_eq!(result.stats.nonoverlapping_image_lines_added, 1);
    let ordinary: [u8; 32] =
        Sha256::digest(render_rgb_page(&pdf, PageNumber::new(1)?, policy().render)?.as_raw())
            .into();
    assert_eq!(recognizer.fallback_digests.borrow().as_slice(), &[ordinary]);
    Ok(())
}

#[test]
fn residual_conflict_or_inference_failure_retries_exact_ordinary_raster()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("mixed.pdf");
    fs::write(
        &pdf,
        include_bytes!("fixtures/repeated-nested-paint-occlusion.pdf"),
    )?;
    let page = PageNumber::new(1)?;
    let doc = PdfMetadata::open(&pdf)?;
    let scan = enumerate_image_paints(&doc, &[page])?;
    let source = &scan.placements[0];
    let observation = pdf_redox_ocr::source_ocr::SourceObservation {
        width: 1200,
        height: 300,
        lines: vec![OcrLine {
            text: "CAPACITOR 47uF 16V".into(),
            confidence: 0.99,
            quad: [
                Point(75.0, 120.0),
                Point(1060.0, 120.0),
                Point(1060.0, 200.0),
                Point(75.0, 200.0),
            ],
            reviewed: false,
        }],
    };
    let conflict =
        pdf_redox_ocr::source_ocr::map_observations_to_paint(&observation, source)?[0].quad;
    let ordinary: [u8; 32] =
        Sha256::digest(render_rgb_page(&pdf, page, policy().render)?.as_raw()).into();
    let job = PreparedOcr::open(&pdf, Some("1"), policy())?;
    for recognizer in [
        Recognizer {
            conflict_once: Some(conflict),
            ..Recognizer::default()
        },
        Recognizer {
            fail_residual_once: true,
            ..Recognizer::default()
        },
    ] {
        let result = job.recognize(&recognizer, |_| {})?;
        assert_eq!(result.residual_pages, std::collections::BTreeSet::new());
        let digests = recognizer.fallback_digests.borrow();
        assert_eq!(digests.len(), 2);
        assert_ne!(digests[0], ordinary);
        assert_eq!(digests[1], ordinary);
        assert_eq!(result.draft.pages[0].lines.len(), 1);
        assert_eq!(result.draft.pages[0].lines[0].text, "CAPACITOR 47uF 16V");
        assert_eq!(result.stats.nonoverlapping_image_lines_added, 1);
    }
    Ok(())
}

#[test]
fn invisible_native_text_still_excludes_duplicate_proven_source_lines()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("mixed.pdf");
    fs::write(
        &pdf,
        include_bytes!("fixtures/mixed-paint-visible-occluded.pdf"),
    )?;
    let job = PreparedOcr::open(&pdf, None, policy())?;
    let output = job.recognize(&Recognizer::default(), |_| {})?;
    assert_eq!(output.stats.nonoverlapping_image_lines_added, 0);
    assert_eq!(output.residual_pages, std::collections::BTreeSet::new());
    assert!(
        output
            .draft
            .pages
            .iter()
            .all(|p| p.lines.is_empty() && p.native_text_regions.len() == 1)
    );
    let visible = output
        .proposals
        .iter()
        .find(|p| p.page == 1)
        .ok_or("missing visible proposal")?;
    assert!(visible.verified_visible_source);
    assert!(visible.native_overlap >= 0.20);
    assert!(!visible.included_as_extra);
    assert!(
        output
            .proposals
            .iter()
            .filter(|p| p.page == 2)
            .all(|p| !p.verified_visible_source && !p.included_as_extra)
    );
    Ok(())
}

#[test]
fn native_budget_and_recognizer_failures_preserve_full_page_fallback()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("mixed.pdf");
    fs::write(
        &pdf,
        include_bytes!("fixtures/mixed-paint-visible-occluded.pdf"),
    )?;
    for budget in [
        NativeImageBudget::default(),
        NativeImageBudget {
            max_source_pixels: 100,
            ..NativeImageBudget::default()
        },
    ] {
        let job = PreparedOcr::open(
            &pdf,
            Some("2"),
            PipelinePolicy {
                source_budget: budget,
                ..policy()
            },
        )?;
        let recognizer = Recognizer {
            fail_native: true,
            recognize_fallback: true,
            ..Recognizer::default()
        };
        let output = job.recognize(&recognizer, |_| {})?;
        assert_eq!(output.stats.page_fallback_pages, 1);
        assert_eq!(output.stats.nonoverlapping_image_lines_added, 0);
        assert_eq!(output.draft.pages.len(), 1);
        assert_eq!(output.draft.pages[0].page, 2);
        assert_eq!(output.draft.pages[0].lines.len(), 1);
        let calls = recognizer.calls.borrow();
        let expected = if budget.max_source_pixels == 100 {
            1
        } else {
            2
        };
        assert_eq!(calls.len(), expected, "{calls:?}");
        assert_eq!(calls.last(), Some(&(1600, 1200)));
    }
    Ok(())
}

#[test]
fn fragment_mosaics_use_same_backend_without_full_page_inference()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("mosaic.pdf");
    fs::write(
        &pdf,
        include_bytes!("fixtures/fragmented-native-mosaics.pdf"),
    )?;
    let job = PreparedOcr::open(&pdf, Some("1,3-5"), policy())?;
    let recognizer = Recognizer {
        recognize_fallback: true,
        ..Recognizer::default()
    };
    let mut progress = Vec::new();
    let output = job.recognize(&recognizer, |message| progress.push(message))?;
    assert_eq!(output.stats.image_source_only_pages, 4, "{progress:#?}");
    assert_eq!(output.stats.page_fallback_pages, 0);
    assert_eq!(recognizer.calls.borrow().len(), 4);
    for page in output.draft.pages {
        assert_eq!(page.lines.len(), 1);
        for point in page.lines[0].quad {
            assert!((75.0..=230.0).contains(&point.0));
            assert!((125.0..=175.0).contains(&point.1));
        }
    }
    Ok(())
}

#[test]
fn invalid_policy_selection_and_modified_prepared_source_are_rejected()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("source.pdf");
    fs::write(
        &pdf,
        include_bytes!("fixtures/mixed-paint-visible-occluded.pdf"),
    )?;
    assert!(PreparedOcr::open(&pdf, Some("3"), policy()).is_err());
    assert!(
        PreparedOcr::open(
            &pdf,
            None,
            PipelinePolicy {
                minimum_confidence: f64::NAN,
                ..policy()
            }
        )
        .is_err()
    );
    assert!(
        PreparedOcr::open(
            &pdf,
            None,
            PipelinePolicy {
                native_chunk: 0,
                ..policy()
            }
        )
        .is_err()
    );
    let job = PreparedOcr::open(&pdf, Some("1"), policy())?;
    fs::write(&pdf, b"changed")?;
    let recognizer = Recognizer::default();
    assert!(job.recognize(&recognizer, |_| {}).is_err());
    assert!(recognizer.calls.borrow().is_empty());
    Ok(())
}
