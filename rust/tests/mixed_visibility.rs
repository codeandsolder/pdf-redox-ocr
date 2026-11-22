//! Real PDF source-paint visibility: admit the visible source label but reject
//! the same reused image hidden under a later opaque vector fill.

#![cfg(feature = "pdf-integration")]

use std::{collections::BTreeMap, fs};

use pdf_redox_ocr::source_appearances::PaintAppearances;
use pdf_redox_ocr::{
    PageNumber,
    draft::OcrLine,
    geometry::Point,
    mixed_visibility::prove_mixed_source_text,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    rasterize::RenderPolicy,
    source_ocr::SourceObservation,
    source_selection::{ImagePlacement, SourcePolicy, choose_mixed_raster_sources},
};

#[test]
fn source_isolation_accepts_visible_and_rejects_opaque_occlusion()
-> Result<(), Box<dyn std::error::Error>> {
    let pdf_bytes = include_bytes!("fixtures/mixed-paint-visible-occluded.pdf");
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("mixed.pdf");
    fs::write(&path, pdf_bytes)?;
    let document = PdfMetadata::open(&path)?;
    assert_eq!(document.page_count(), 2);
    let pages = [PageNumber::new(1)?, PageNumber::new(2)?];
    let painted = enumerate_image_paints(&document, &pages)?;
    assert_eq!(painted.placements.len(), 2, "{painted:#?}");
    assert!(painted.pages.iter().all(|p| !p.image_only_after_native));
    let first = &painted.placements[0];
    let second = &painted.placements[1];
    assert_eq!(first.source_id, second.source_id);
    let image = SourceObservation {
        width: 1200,
        height: 300,
        lines: vec![pdf_redox_ocr::source_ocr::SourceLine {
            detection_index: 0,
            line: OcrLine {
                text: "CAPACITOR 47uF 16V".to_owned(),
                confidence: 0.99,
                quad: [
                    Point(75.0, 120.0),
                    Point(1060.0, 120.0),
                    Point(1060.0, 200.0),
                    Point(75.0, 200.0),
                ],
                reviewed: false,
            },
        }],
    };
    let recognized = BTreeMap::from([(first.source_id.clone(), image)]);
    let evidence = prove_mixed_source_text(
        &path,
        &document,
        &painted,
        &PaintAppearances::from_object_inventory(recognized, &painted.placements),
        &pages,
        RenderPolicy {
            max_side: 1600,
            dpi: 72,
        },
        0.97,
    )?;
    assert_eq!(evidence.audited.len(), 2);
    assert_eq!(evidence.proofs.len(), 1);
    assert_eq!(evidence.proofs[0].page, 1);
    assert_eq!(evidence.proofs[0].source_id, first.source_id);
    assert_eq!(evidence.proofs[0].paint_index, first.paint_index);
    assert_eq!(evidence.proofs[0].text, "CAPACITOR 47uF 16V");
    assert!(evidence.proven_pages.contains(&pages[0]));
    assert!(!evidence.proven_pages.contains(&pages[1]));
    assert_eq!(evidence.audited[0].visible_lines, 1);
    assert_eq!(evidence.audited[1].visible_lines, 0);

    let candidates = painted
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
    let (selected, _report) =
        choose_mixed_raster_sources(&candidates, document.media_boxes(), SourcePolicy::default())?;
    assert!(selected.contains(&first.source_id));
    Ok(())
}

#[test]
fn invalid_native_recognition_retains_page_fallback() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("source-geometry.pdf");
    fs::write(
        &path,
        include_bytes!("fixtures/mixed-paint-visible-occluded.pdf"),
    )?;
    let document = PdfMetadata::open(&path)?;
    let pages = [PageNumber::new(1)?, PageNumber::new(2)?];
    let painted = enumerate_image_paints(&document, &pages)?;
    let id = painted.placements[0].source_id.clone();
    let policy = RenderPolicy {
        max_side: 1600,
        dpi: 72,
    };
    let invalid_line = OcrLine {
        text: "OUTSIDE IMAGE".into(),
        confidence: 0.99,
        quad: [
            Point(1150.0, 120.0),
            Point(1250.0, 120.0),
            Point(1250.0, 200.0),
            Point(1150.0, 200.0),
        ],
        reviewed: false,
    };
    for source in [
        SourceObservation {
            width: 1200,
            height: 300,
            lines: vec![pdf_redox_ocr::source_ocr::SourceLine {
                detection_index: 0,
                line: invalid_line.clone(),
            }],
        },
        SourceObservation {
            width: 1199,
            height: 300,
            lines: vec![pdf_redox_ocr::source_ocr::SourceLine {
                detection_index: 0,
                line: invalid_line,
            }],
        },
    ] {
        let result = prove_mixed_source_text(
            &path,
            &document,
            &painted,
            &PaintAppearances::from_object_inventory(
                BTreeMap::from([(id.clone(), source)]),
                &painted.placements,
            ),
            &pages,
            policy,
            0.97,
        )?;
        assert!(result.proofs.is_empty());
        assert!(result.observations.is_empty());
        assert!(result.proven_pages.is_empty());
    }
    Ok(())
}
