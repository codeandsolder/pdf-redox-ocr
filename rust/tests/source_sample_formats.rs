//! Real PDF mask round trips across packed and wide source sample encodings.

#![cfg(feature = "pdf-integration")]

use std::{fs, path::Path};

use pdf_redox::{ObjectId, SourcePdf};
use pdf_redox_ocr::{
    PageNumber,
    draft::OcrLine,
    geometry::Point,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    source_mask::mask_source_text,
};

fn original_line() -> OcrLine {
    OcrLine {
        text: "Source local OCR candidate".to_owned(),
        confidence: 0.99,
        quad: [
            Point(60.0, 45.0),
            Point(210.0, 45.0),
            Point(210.0, 100.0),
            Point(60.0, 100.0),
        ],
        reviewed: false,
    }
}

fn count_rendered_changes(
    before: &Path,
    after: &Path,
    page: PageNumber,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let policy = RenderPolicy {
        dpi: 72,
        max_side: 450,
    };
    let baseline = render_rgb_page(before, page, policy)?;
    let masked = render_rgb_page(after, page, policy)?;
    assert_eq!(baseline.dimensions(), masked.dimensions());
    let (mut changed_inside, mut changed_outside) = (0, 0);
    let width = usize::try_from(baseline.width())?;
    for (idx, (a, b)) in baseline.pixels().zip(masked.pixels()).enumerate() {
        if a.0 == b.0 {
            continue;
        }
        let (x, y) = (idx % width, idx / width);
        // OCR source quad x=[60,210], y=[45,100], with PDF image placed
        // at (50,90) on a 450×320 page. 4px margin allows AA boundaries.
        if (106..=264).contains(&x) && (126..=189).contains(&y) {
            changed_inside += 1;
        } else {
            changed_outside += 1;
        }
    }
    Ok((changed_inside, changed_outside))
}

#[test]
fn source_mask_preserves_pdf_color_semantics_across_ten_sample_formats()
-> Result<(), Box<dyn std::error::Error>> {
    let original = include_bytes!("fixtures/source-sample-formats.pdf");
    let dir = tempfile::tempdir()?;
    let before = dir.path().join("original.pdf");
    let after = dir.path().join("masked.pdf");
    fs::write(&before, original)?;
    let metadata = PdfMetadata::open(&before)?;
    assert_eq!(metadata.page_count(), 10);
    let pages = (1..=10)
        .map(PageNumber::new)
        .collect::<Result<Vec<_>, _>>()?;
    let report = enumerate_image_paints(&metadata, &pages)?;
    assert_eq!(report.placements.len(), 10);
    let formats = [
        "Gray1",
        "Gray2",
        "Gray4",
        "Gray8",
        "RGB16",
        "CMYK8",
        "Indexed4",
        "RGB8+Decode",
        "CalGray8",
        "ICCBased RGB8",
    ];

    for (&page, &name) in pages.iter().zip(&formats) {
        let paint = report
            .placements
            .iter()
            .find(|paint| paint.page == page.get())
            .ok_or("missing native image paint")?;
        assert!(paint.unsafe_reasons.is_empty(), "{name}: {paint:#?}");
        let id = ObjectId::new(
            i32::try_from(paint.object_number)?,
            i32::from(paint.generation),
        );
        let changed = mask_source_text(original, id, &[original_line()], 0.97)?
            .ok_or_else(|| format!("{name}: source mask rejected supported image"))?;
        assert_eq!(changed.metrics.len(), 1, "{name}");
        assert!(
            changed.metrics[0].changed_source_pixels > 1_000,
            "{name}: no native image pixels changed"
        );
        fs::write(&after, &changed.bytes)?;
        let roundtrip = SourcePdf::from_bytes(changed.bytes)?;
        assert_eq!(roundtrip.page_count(), 10, "{name}");
        let (inside, outside) = count_rendered_changes(&before, &after, page)?;
        assert!(inside > 500, "{name}: only {inside} visible changed pixels");
        assert_eq!(
            outside, 0,
            "{name}: rewriting changed {outside} pixels outside the OCR source region"
        );
    }
    Ok(())
}
