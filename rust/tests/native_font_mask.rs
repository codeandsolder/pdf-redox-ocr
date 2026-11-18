//! Render-only native font suppression must not alter extraction or Type3 ink.

#![cfg(feature = "pdf-integration")]

use std::{fs, process::Command};

use pdf_redox_ocr::{
    PageNumber,
    native_font_mask::mask_native_fonts,
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
};

const ORIGINAL: &[u8] = include_bytes!("fixtures/native-font-form-masking.pdf");

fn font_fixture()
-> Result<(tempfile::TempDir, std::path::PathBuf, PdfMetadata), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("original.pdf");
    fs::write(&input, ORIGINAL)?;
    let metadata = PdfMetadata::open(&input)?;
    assert_eq!(metadata.page_count(), 5);
    Ok((dir, input, metadata))
}

fn words(pdf: &std::path::Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let process = Command::new("pdftotext").arg(pdf).arg("-").output()?;
    assert!(process.status.success());
    Ok(process.stdout)
}

fn changed_pixels(
    before: &image::RgbImage,
    after: &image::RgbImage,
    allowed: Option<[u32; 4]>,
) -> (u64, u64) {
    assert_eq!(before.dimensions(), after.dimensions());
    let mut changed = 0_u64;
    let mut outside = 0_u64;
    for (x, y, pixel) in before.enumerate_pixels() {
        if pixel == after.get_pixel(x, y) {
            continue;
        }
        changed += 1;
        if !allowed.is_some_and(|[x0, y0, x1, y1]| (x0..=x1).contains(&x) && (y0..=y1).contains(&y))
        {
            outside += 1;
        }
    }
    (changed, outside)
}

#[test]
fn native_only_mask_hides_searchable_glyphs_but_preserves_type3_and_clipping()
-> Result<(), Box<dyn std::error::Error>> {
    let (_dir, input, document) = font_fixture()?;
    let pages = [1, 2, 3, 4]
        .map(PageNumber::new)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mask = mask_native_fonts(&document, &pages)?;
    assert_eq!(mask.stats.hidden_native_shows, 3, "{:#?}", mask.stats);
    assert_eq!(mask.stats.preserved_clipping_shows, 1);
    assert_eq!(mask.stats.already_invisible_shows, 1);
    assert!(mask.stats.preserved_nonsearchable_shows >= 3);
    assert_eq!(mask.stats.rewritten_pages, 3);
    assert_eq!(mask.stats.cloned_forms, 3);
    let dir = tempfile::tempdir()?;
    let filtered = dir.path().join("masked.pdf");
    fs::write(&filtered, mask.bytes.ok_or("no masked PDF produced")?)?;
    assert_eq!(
        words(&input)?,
        words(&filtered)?,
        "native text extraction changed"
    );
    let policy = RenderPolicy {
        dpi: 72,
        max_side: 500,
    };
    let selected = [
        (1, Some([40, 45, 330, 135]), true),
        (2, Some([65, 100, 285, 220]), true),
        (3, None, false),
        (4, Some([65, 100, 285, 220]), true),
        (5, None, false),
    ];
    for (number, bounds, should_change) in selected {
        let page = PageNumber::new(number)?;
        let before = render_rgb_page(&input, page, policy)?;
        let after = render_rgb_page(&filtered, page, policy)?;
        let (changed, outside) = changed_pixels(&before, &after, bounds);
        assert_eq!(
            outside, 0,
            "page {number}: {outside} pixels changed outside native text"
        );
        assert_eq!(
            changed > 20,
            should_change,
            "page {number}: changed {changed} pixels"
        );
    }
    assert_eq!(fs::read(&input)?, ORIGINAL, "original PDF was modified");
    Ok(())
}

#[test]
fn nonsearchable_type3_form_only_returns_no_mask_and_does_not_touch_siblings()
-> Result<(), Box<dyn std::error::Error>> {
    let (_dir, _input, doc) = font_fixture()?;
    let mask = mask_native_fonts(&doc, &[PageNumber::new(3)?])?;
    assert!(mask.bytes.is_none());
    assert_eq!(mask.stats.hidden_native_shows, 0);
    assert_eq!(mask.stats.rewritten_pages, 0);
    assert_eq!(mask.stats.cloned_forms, 0);
    let mask = mask_native_fonts(&doc, &[PageNumber::new(2)?])?;
    assert_eq!(mask.stats.hidden_native_shows, 1);
    assert_eq!(mask.stats.rewritten_pages, 1);
    assert_eq!(mask.stats.cloned_forms, 1);
    let result = PdfMetadata::from_bytes(mask.bytes.ok_or("missing Form rewrite")?)?;
    assert_eq!(result.page_count(), 5);
    assert!(mask_native_fonts(&doc, &[]).is_err());
    assert!(mask_native_fonts(&doc, &[PageNumber::new(6)?]).is_err());
    assert!(mask_native_fonts(&doc, &[PageNumber::new(1)?, PageNumber::new(1)?]).is_err());
    Ok(())
}
