//! Extended graphics-state font selection inherits real font objects through Forms.

#![cfg(feature = "pdf-integration")]

use std::{fs, path::Path, process::Command};

use image::RgbImage;
use pdf_redox_ocr::{
    PageNumber,
    native_font_mask::mask_native_fonts,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
};

const PDF: &[u8] = include_bytes!("fixtures/extgstate-font-invocations.pdf");
const POSITIONS: [(u32, u32); 6] = [
    (25, 190),
    (125, 190),
    (225, 190),
    (325, 190),
    (25, 90),
    (125, 90),
];

fn extracted(pdf: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let output = Command::new("pdftotext").arg(pdf).arg("-").output()?;
    assert!(output.status.success());
    Ok(output.stdout)
}

fn verify_render_diff(original: &RgbImage, masked: &RgbImage, expected: &[usize]) {
    assert_eq!(original.dimensions(), (430, 260));
    assert_eq!(original.dimensions(), masked.dimensions());
    let mut changed = [0_u64; POSITIONS.len()];
    for (x, y, pixel) in original.enumerate_pixels() {
        if pixel == masked.get_pixel(x, y) {
            continue;
        }
        let call = POSITIONS.iter().position(|&(px, py)| {
            (px.saturating_sub(3)..=px + 48).contains(&x)
                && (260 - py - 45..=260 - py + 6).contains(&y)
        });
        assert!(call.is_some(), "mask changed unrelated pixel at ({x}, {y})");
        if let Some(index) = call {
            changed[index] += 1;
        }
    }
    for (i, count) in changed.into_iter().enumerate() {
        if expected.contains(&i) {
            assert!(count > 20, "did not hide searchable invocation {i}");
        } else {
            assert_eq!(count, 0, "changed Type3 or unselected invocation {i}");
        }
    }
}

#[test]
fn extended_graphics_state_changes_native_font_admission_and_render_only_mask()
-> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let original = tmp.path().join("source.pdf");
    let output = tmp.path().join("render-only.pdf");
    fs::write(&original, PDF)?;
    let doc = PdfMetadata::open(&original)?;
    let page1 = PageNumber::new(1)?;
    let page2 = PageNumber::new(2)?;
    let scanned = enumerate_image_paints(&doc, &[page1, page2])?;
    assert!(scanned.placements.is_empty());
    for (page, native, graphical) in [(1, 3, 3), (2, 2, 4)] {
        let report = scanned
            .pages
            .iter()
            .find(|entry| entry.page == page)
            .ok_or("missing page in paint scan")?;
        assert_eq!(report.native_searchable_shows, native, "page {page}");
        assert_eq!(report.non_image_visual_paint, graphical, "page {page}");
        assert_eq!(report.unknown_paint, 0);
        assert_eq!(report.unsupported_extgs, 0);
    }

    let mask = mask_native_fonts(&doc, &[page1])?;
    assert_eq!(mask.stats.hidden_native_shows, 3);
    assert_eq!(mask.stats.preserved_nonsearchable_shows, 3);
    assert_eq!(mask.stats.cloned_forms, 3);
    assert_eq!(mask.stats.rewritten_pages, 1);
    fs::write(&output, mask.bytes.ok_or("missing masked PDF")?)?;
    assert_eq!(extracted(&original)?, extracted(&output)?);

    let render = RenderPolicy {
        dpi: 72,
        max_side: 430,
    };
    let before1 = render_rgb_page(&original, page1, render)?;
    let after1 = render_rgb_page(&output, page1, render)?;
    verify_render_diff(&before1, &after1, &[0, 2, 4]);
    let before2 = render_rgb_page(&original, page2, render)?;
    let after2 = render_rgb_page(&output, page2, render)?;
    verify_render_diff(&before2, &after2, &[]);
    assert_eq!(fs::read(&original)?, PDF);
    Ok(())
}
