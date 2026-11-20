//! Distinct invocations of the same PDF Form inherit independent graphics state.
//!
//! A render-only native-font mask must not change an already invisible Form,
//! a text-clipping Form, Type3 ink, the unselected sibling page or glyph advances.

#![cfg(feature = "pdf-integration")]

use std::{fs, path::Path, process::Command};

use image::RgbImage;
use pdf_redox_ocr::{
    PageNumber,
    native_font_mask::mask_native_fonts,
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
};

const PDF: &[u8] = include_bytes!("fixtures/native-font-invocation-context.pdf");
const CALLS: &[(u32, u32)] = &[
    (35, 370),
    (145, 370),
    (255, 370),
    (35, 280),
    (145, 280),
    (255, 280),
    (35, 190),
    (145, 190),
    (255, 100), // Form with its own /F resource, different from the caller's selected /F.
];

fn extracted(pdf: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let out = Command::new("pdftotext").arg(pdf).arg("-").output()?;
    assert!(out.status.success());
    Ok(out.stdout)
}

fn region(x: u32, y: u32) -> [u32; 4] {
    // Raster origin is top-left; PDF's origin is bottom-left.
    [x - 2, 450 - y - 44, x + 48, 450 - y + 10]
}

fn inside(x: u32, y: u32, [x0, y0, x1, y1]: [u32; 4]) -> bool {
    (x0..=x1).contains(&x) && (y0..=y1).contains(&y)
}

fn check_page(original: &RgbImage, modified: &RgbImage, expected: &[usize], page: u32) {
    assert_eq!(original.dimensions(), (400, 450));
    assert_eq!(original.dimensions(), modified.dimensions());
    let mut changed_per_call = [0_u64; 9];
    for (x, y, pixel) in original.enumerate_pixels() {
        if pixel == modified.get_pixel(x, y) {
            continue;
        }
        let matched = CALLS
            .iter()
            .enumerate()
            .find(|&(_, &(px, py))| inside(x, y, region(px, py)));
        assert!(
            matched.is_some(),
            "page {page} changed unrelated pixel at ({x}, {y})"
        );
        if let Some((index, _)) = matched {
            changed_per_call[index] += 1;
        }
    }
    for (index, count) in changed_per_call.into_iter().enumerate() {
        if expected.contains(&index) {
            assert!(
                count >= 20,
                "page {page} missing native mask for call {index}: {count} pixels"
            );
        } else {
            assert_eq!(
                count, 0,
                "page {page} modified unmasked Form invocation {index}"
            );
        }
    }
}

#[test]
fn same_form_invocations_keep_distinct_font_and_text_rendering_modes()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("original.pdf");
    let output = dir.path().join("render-only.pdf");
    fs::write(&input, PDF)?;
    let doc = PdfMetadata::open(&input)?;
    let mask = mask_native_fonts(&doc, &[PageNumber::new(1)?, PageNumber::new(2)?])?;
    assert_eq!(mask.stats.hidden_native_shows, 7);
    assert_eq!(mask.stats.preserved_clipping_shows, 2);
    assert_eq!(mask.stats.already_invisible_shows, 4);
    assert_eq!(mask.stats.cloned_forms, 9);
    assert_eq!(mask.stats.rewritten_pages, 2);
    fs::write(&output, mask.bytes.ok_or("no masked PDF produced")?)?;
    assert_eq!(extracted(&input)?, extracted(&output)?);
    let policy = RenderPolicy {
        dpi: 72,
        max_side: 450,
    };
    for (page_number, expected) in [(1, vec![0, 4, 6, 7, 8]), (2, vec![4, 7]), (3, vec![])] {
        let page = PageNumber::new(page_number)?;
        let original = render_rgb_page(&input, page, policy)?;
        let filtered = render_rgb_page(&output, page, policy)?;
        check_page(&original, &filtered, &expected, page_number);
    }
    assert_eq!(fs::read(&input)?, PDF);
    Ok(())
}

#[test]
fn selected_pages_only_rewrite_their_own_native_form_invocations()
-> Result<(), Box<dyn std::error::Error>> {
    let doc = PdfMetadata::from_bytes(PDF.to_vec())?;
    let no_visible_source = mask_native_fonts(&doc, &[PageNumber::new(2)?])?;
    assert_eq!(no_visible_source.stats.hidden_native_shows, 2);
    assert_eq!(no_visible_source.stats.rewritten_pages, 1);
    assert_eq!(no_visible_source.stats.cloned_forms, 3);
    assert_eq!(no_visible_source.stats.preserved_clipping_shows, 1);
    let skipped = mask_native_fonts(&doc, &[PageNumber::new(3)?])?;
    assert_eq!(skipped.stats.hidden_native_shows, 4);
    Ok(())
}
