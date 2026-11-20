//! Noncanonical Form rectangle coordinates are common in real supplier PDFs.

#![cfg(feature = "pdf-integration")]

use std::fs;

use pdf_redox_ocr::{
    PageNumber,
    geometry::Point,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
};

const PDF: &[u8] = include_bytes!("fixtures/noncanonical-form-bboxes.pdf");

#[test]
fn inverted_form_bbox_matches_poppler_and_empty_bbox_is_not_source_safe()
-> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let pdf = tmp.path().join("form-bboxes.pdf");
    fs::write(&pdf, PDF)?;
    let meta = PdfMetadata::open(&pdf)?;
    let page = PageNumber::new(1)?;
    let scanned = enumerate_image_paints(&meta, &[page])?;
    assert_eq!(scanned.placements.len(), 3);
    assert_eq!(scanned.pages.len(), 1);
    assert_eq!(scanned.pages[0].image_paints, 3);
    assert_eq!(scanned.pages[0].unsafe_image_paints, 1);

    for (index, x0) in [(0, 30.0), (1, 150.0)] {
        let paint = &scanned.placements[index];
        assert_eq!(paint.page_quad[0], Point(x0, 55.0));
        assert_eq!(paint.page_quad[2], Point(x0 + 50.0, 95.0));
        assert!(paint.unsafe_reasons.is_empty(), "{paint:?}");
        let clip = paint.clips.last().ok_or("missing normalized Form clip")?;
        assert_eq!(clip[0], Point(x0, 55.0));
        assert_eq!(clip[2], Point(x0 + 50.0, 95.0));
    }
    let degenerate = &scanned.placements[2];
    assert!(
        degenerate
            .unsafe_reasons
            .contains(&"nonrectangular_or_untracked_clip")
    );
    assert!(!scanned.pages[0].image_only_after_native);

    let rendered = render_rgb_page(
        &pdf,
        page,
        RenderPolicy {
            dpi: 72,
            max_side: 420,
        },
    )?;
    assert_eq!(rendered.dimensions(), (420, 150));
    for (x, should_be_blue) in [(55, true), (175, true), (295, false)] {
        let pixel = rendered.get_pixel(x, 75).0;
        assert_eq!(pixel[2] > 150 && pixel[0] < 50, should_be_blue);
    }
    assert_eq!(fs::read(&pdf)?, PDF);
    Ok(())
}
