//! Counterfactual source-image masking on the same genuine six-page PDF that
//! exercises extraction and resource reuse.

#![cfg(feature = "pdf-integration")]

use std::fs;

use pdf_redox::{ObjectId, SourcePdf};
use pdf_redox_ocr::{
    PageNumber,
    draft::OcrLine,
    geometry::Point,
    rasterize::{RenderPolicy, render_rgb_page},
    source_mask::mask_source_text,
};

#[test]
fn only_masked_source_pixels_change_in_rendered_pdf() -> Result<(), Box<dyn std::error::Error>> {
    let original = include_bytes!("fixtures/repeated-image-six-pages.pdf");
    let mask = OcrLine {
        text: "Source bitmap technical label".into(),
        confidence: 0.99,
        quad: [
            Point(75.0, 120.0),
            Point(1060.0, 120.0),
            Point(1060.0, 200.0),
            Point(75.0, 200.0),
        ],
        reviewed: false,
    };
    let masked = mask_source_text(original, ObjectId::new(9, 0), &[mask], 0.97)?
        .ok_or("8-bit RGB XObject was unexpectedly rejected")?;
    assert_eq!(masked.metrics.len(), 1);
    assert!(masked.metrics[0].changed_source_pixels > 1000);

    let doc = SourcePdf::from_bytes(masked.bytes.clone())?;
    assert_eq!(doc.page_count(), 6);

    let dir = tempfile::tempdir()?;
    let before_path = dir.path().join("before.pdf");
    let after_path = dir.path().join("after.pdf");
    fs::write(&before_path, original)?;
    fs::write(&after_path, &masked.bytes)?;
    let page = PageNumber::new(2)?;
    let before = render_rgb_page(&before_path, page, RenderPolicy::default())?;
    let after = render_rgb_page(&after_path, page, RenderPolicy::default())?;
    assert_eq!(before.dimensions(), after.dimensions());
    let difference = before
        .pixels()
        .zip(after.pixels())
        .filter(|(a, b)| {
            a.0.iter()
                .zip(b.0)
                .any(|(left, right)| left.abs_diff(right) > 8)
        })
        .count();
    assert!(
        difference > 100,
        "expected a visible counterfactual image text change"
    );
    Ok(())
}

#[test]
fn cannot_mask_outside_original_source_bounds() {
    let original = include_bytes!("fixtures/repeated-image-six-pages.pdf");
    let invalid = OcrLine {
        text: "offsource".into(),
        confidence: 0.99,
        quad: [
            Point(-2.0, 100.0),
            Point(40.0, 100.0),
            Point(40.0, 130.0),
            Point(-2.0, 130.0),
        ],
        reviewed: false,
    };
    assert!(mask_source_text(original, ObjectId::new(9, 0), &[invalid], 0.97).is_err());
}
