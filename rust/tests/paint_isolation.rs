//! Isolated source masking when one `XObject` appears twice on a page,
//! twice through one reused Form, and twice through nested Forms.

#![cfg(feature = "pdf-integration")]

use std::{collections::BTreeMap, fs};

use pdf_redox_ocr::{
    PageNumber, draft::OcrLine, geometry::Point, mixed_visibility::prove_mixed_source_text,
    paint_scan::enumerate_image_paints, pdf_metadata::PdfMetadata, rasterize::RenderPolicy,
    source_ocr::SourceObservation,
};

#[test]
fn attributes_repeated_direct_form_and_nested_form_paints() -> Result<(), Box<dyn std::error::Error>>
{
    let original = include_bytes!("fixtures/repeated-nested-paint-occlusion.pdf");
    let tmp = tempfile::tempdir()?;
    let pdf = tmp.path().join("repeated-nested.pdf");
    fs::write(&pdf, original)?;
    let document = PdfMetadata::open(&pdf)?;
    let pages = [
        PageNumber::new(1)?,
        PageNumber::new(2)?,
        PageNumber::new(3)?,
    ];
    let painted = enumerate_image_paints(&document, &pages)?;
    assert_eq!(painted.placements.len(), 6, "{painted:#?}");
    let source = painted.placements[0].source_id.clone();
    assert!(painted.placements.iter().all(|p| p.source_id == source));
    let paths: Vec<Vec<u32>> = painted
        .placements
        .iter()
        .map(|paint| paint.paint_path.clone())
        .collect();
    assert_eq!(
        paths,
        [
            vec![0],
            vec![1],
            vec![0, 0],
            vec![1, 0],
            vec![0, 0, 0],
            vec![1, 0, 0],
        ]
    );
    let recognized = BTreeMap::from([(
        source,
        SourceObservation {
            width: 1200,
            height: 300,
            lines: vec![OcrLine {
                text: "CAPACITOR 47uF 16V".to_owned(),
                confidence: 0.99,
                quad: [
                    Point(75.0, 120.0),
                    Point(1060.0, 120.0),
                    Point(1060.0, 200.0),
                    Point(75.0, 200.0),
                ],
                reviewed: false,
            }],
        },
    )]);
    let result = prove_mixed_source_text(
        &pdf,
        &document,
        &painted,
        &recognized,
        &pages,
        RenderPolicy {
            max_side: 1600,
            dpi: 72,
        },
        0.97,
    )?;
    assert_eq!(result.audited.len(), 6, "{:#?}", result.audited);
    assert_eq!(result.proofs.len(), 3, "{:#?}", result.audited);
    for page in 1..=3 {
        let first = painted
            .placements
            .iter()
            .find(|paint| paint.page == page)
            .ok_or("missing first occurrence")?;
        let proofs = result
            .proofs
            .iter()
            .filter(|proof| proof.page == page)
            .collect::<Vec<_>>();
        assert_eq!(proofs.len(), 1, "page {page}");
        assert_eq!(proofs[0].paint_index, first.paint_index);
        assert_eq!(proofs[0].text, "CAPACITOR 47uF 16V");
        assert!(result.proven_pages.contains(&PageNumber::new(page)?));
        let audits = result
            .audited
            .iter()
            .filter(|audit| audit.page.get() == page)
            .collect::<Vec<_>>();
        assert_eq!(audits.len(), 2);
        assert_eq!(audits[0].visible_lines, 1);
        assert_eq!(audits[1].visible_lines, 0);
    }
    Ok(())
}

#[test]
fn isolated_paint_does_not_modify_other_visible_uses_or_reserved_aliases()
-> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox::ObjectId;
    use pdf_redox_ocr::{
        paint_isolation::isolate_source_paint, rasterize::render_rgb_page,
        source_mask::mask_source_object,
    };

    let original = include_bytes!("fixtures/repeated-nested-paint-both-visible.pdf");
    let temp = tempfile::tempdir()?;
    let before_path = temp.path().join("before.pdf");
    let after_path = temp.path().join("after.pdf");
    fs::write(&before_path, original)?;
    let metadata = PdfMetadata::open(&before_path)?;
    let pages = [
        PageNumber::new(1)?,
        PageNumber::new(2)?,
        PageNumber::new(3)?,
    ];
    let painted = enumerate_image_paints(&metadata, &pages)?;
    assert_eq!(painted.placements.len(), 6);
    let original_line = OcrLine {
        text: "CAPACITOR 47uF 16V".to_owned(),
        confidence: 0.99,
        quad: [
            Point(75.0, 120.0),
            Point(1060.0, 120.0),
            Point(1060.0, 200.0),
            Point(75.0, 200.0),
        ],
        reviewed: false,
    };
    for &page in &pages {
        let first = painted
            .placements
            .iter()
            .find(|paint| paint.page == page.get())
            .ok_or("missing first image")?;
        let masked = mask_source_object(
            metadata.source(),
            ObjectId::new(
                i32::try_from(first.object_number)?,
                i32::from(first.generation),
            ),
            std::slice::from_ref(&original_line),
            0.97,
        )?
        .ok_or("source image could not be masked")?;
        let mut invalid = first.clone();
        invalid.paint_path = vec![50];
        assert!(isolate_source_paint(&metadata, &invalid, masked.object.clone()).is_err());
        let isolated = isolate_source_paint(&metadata, first, masked.object)?;
        fs::write(&after_path, isolated)?;
        let policy = RenderPolicy {
            max_side: 1600,
            dpi: 72,
        };
        let before = render_rgb_page(&before_path, page, policy)?;
        let after = render_rgb_page(&after_path, page, policy)?;
        assert_eq!(before.dimensions(), after.dimensions());
        let mut left_changed = 0;
        let mut right_changed = 0;
        for (index, (a, b)) in before.pixels().zip(after.pixels()).enumerate() {
            let x = index % usize::try_from(before.width())?;
            if a.0 != b.0 {
                if x < usize::try_from(before.width())? / 2 {
                    left_changed += 1;
                } else {
                    right_changed += 1;
                }
            }
        }
        assert!(
            left_changed > 100,
            "page {} had no paint-local change",
            page.get()
        );
        assert_eq!(
            right_changed,
            0,
            "page {} changed a different, still-visible occurrence",
            page.get()
        );
    }
    Ok(())
}
