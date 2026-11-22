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

mod support;

#[test]
fn masks_preserve_exact_samples_of_neighboring_excluded_labels()
-> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox::{OwnedObject, StreamData};
    use pdf_redox_ocr::source_mask::{MaskSource, mask_source_object, mask_source_subset};

    let mut samples = vec![255_u8; 96 * 32];
    for row in samples.as_chunks_mut::<96>().0.iter_mut().skip(8).take(16) {
        row[14..44].fill(0);
        row[50..78].fill(0);
    }
    // Native color components are preserved, not merely visually similar.
    for row in samples.as_chunks_mut::<96>().0.iter_mut().skip(8).take(16) {
        row[46..50].fill(123);
    }
    let source = SourcePdf::from_bytes(support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 96 32] /Resources << /XObject << /Im 5 0 R >> >> >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>".to_vec(),
        support::stream("",b"q 96 0 0 32 0 0 cm /Im Do Q"),
        support::stream("/Type /XObject /Subtype /Image /Width 96 /Height 32 /BitsPerComponent 8 /ColorSpace /DeviceGray",&samples),
    ]))?;
    let line = |text: &str, confidence, left, right| OcrLine {
        text: text.into(),
        confidence,
        quad: [
            Point(left, 8.0),
            Point(right, 8.0),
            Point(right, 24.0),
            Point(left, 24.0),
        ],
        reviewed: false,
    };
    let id = MaskSource::Object(ObjectId::new(5, 0));
    let selected = line("Selected", 0.99, 10.0, 50.0);
    let low_confidence = line("Neighbor", 0.5, 49.0, 80.0);
    let thresholded = mask_source_object(&source, id, &[&selected, &low_confidence], 0.97)?
        .ok_or("no thresholded mask")?;
    let explicit = mask_source_subset(
        &source,
        id,
        &[&selected, &line("Neighbor", 0.99, 49.0, 80.0)],
        &[0],
    )?
    .ok_or("no explicit mask")?;
    let short = mask_source_object(
        &source,
        id,
        &[&selected, &line("N", 0.999, 49.0, 80.0)],
        0.97,
    )?
    .ok_or("no short-protected mask")?;
    let unprotected =
        mask_source_object(&source, id, &[&selected], 0.97)?.ok_or("no unprotected control")?;
    let data = |mask: pdf_redox_ocr::source_mask::MaskedSourceObject| {
        let OwnedObject::Stream {
            data: StreamData::Owned(bytes),
            ..
        } = mask.object
        else {
            return Err("fixture replacement must be raw packed samples");
        };
        assert_eq!(mask.metrics.len(), 1);
        assert!(mask.metrics[0].changed_source_pixels > 100);
        Ok(bytes)
    };
    let expected = data(thresholded)?;
    assert_eq!(data(explicit)?, expected);
    assert_eq!(data(short)?, expected);
    let ordinary = data(unprotected)?;
    assert_ne!(
        ordinary[16 * 96 + 50],
        samples[16 * 96 + 50],
        "mask must overlap excluded label in the unprotected control"
    );
    for y in 5..27 {
        assert_eq!(
            &expected[y * 96 + 46..y * 96 + 83],
            &samples[y * 96 + 46..y * 96 + 83]
        );
    }
    assert_eq!(expected[16 * 96 + 30], 255);
    assert_eq!(expected[16 * 96 + 50], 0);
    assert!(
        mask_source_subset(
            &source,
            id,
            &[&line("Duplicate", 0.99, 10.0, 50.0)],
            &[0, 0]
        )
        .is_err()
    );
    assert!(mask_source_subset(&source, id, &[], &[0]).is_err());
    Ok(())
}
