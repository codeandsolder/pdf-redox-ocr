//! Isolated source masking when one `XObject` appears twice on a page,
//! twice through one reused Form, and twice through nested Forms.

#![cfg(feature = "pdf-integration")]

use std::{collections::BTreeMap, fs};

use pdf_redox_ocr::source_appearances::PaintAppearances;
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
        },
    )]);
    let result = prove_mixed_source_text(
        &pdf,
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
    use pdf_redox_ocr::{
        paint_isolation::{PaintReplacement, isolate_source_paints},
        rasterize::render_rgb_page,
        source_mask::{MaskSource, mask_source_object},
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
            MaskSource::Paint(first),
            &[&original_line],
            0.97,
        )?
        .ok_or("source image could not be masked")?;
        let mut invalid = first.clone();
        invalid.paint_path = vec![50];
        assert!(
            isolate_source_paints(
                &metadata,
                [PaintReplacement {
                    image: masked.object.clone(),
                    paints: vec![&invalid]
                }]
            )
            .is_err()
        );
        let isolated = isolate_source_paints(
            &metadata,
            [PaintReplacement {
                image: masked.object,
                paints: vec![first],
            }],
        )?;
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

mod support;

fn shared_form_fixture() -> Vec<u8> {
    use support::stream;
    support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /MediaBox [0 0 200 100] /Resources << /XObject << /Fm 5 0 R /OcrIsolatedPaint0 5 0 R >> >> >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /Contents 7 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /Contents 7 0 R >>".to_vec(),
        stream(
            "/Type /XObject /Subtype /Form /BBox [0 0 50 20] /Resources << /XObject << /Im 6 0 R /OcrIsolatedPaint0 6 0 R >> >>",
            b"q 20 0 0 20 0 0 cm /Im Do Q q 20 0 0 20 25 0 cm /OcrIsolatedPaint0 Do Q",
        ),
        stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray", &[0]),
        stream("", b"q 1 0 0 1 10 20 cm /Fm Do Q q 1 0 0 1 100 20 cm /OcrIsolatedPaint0 Do Q"),
    ])
}

fn gray_image(value: u8) -> pdf_redox::OwnedObject {
    use pdf_redox::{OwnedObject, StreamData};
    let dictionary = [
        (b"Type".to_vec(), OwnedObject::Name(b"XObject".to_vec())),
        (b"Subtype".to_vec(), OwnedObject::Name(b"Image".to_vec())),
        (b"Width".to_vec(), OwnedObject::Integer(1)),
        (b"Height".to_vec(), OwnedObject::Integer(1)),
        (b"BitsPerComponent".to_vec(), OwnedObject::Integer(8)),
        (
            b"ColorSpace".to_vec(),
            OwnedObject::Name(b"DeviceGray".to_vec()),
        ),
        (b"Length".to_vec(), OwnedObject::Integer(1)),
    ]
    .into_iter()
    .collect();
    OwnedObject::Stream {
        dictionary,
        data: StreamData::Owned(vec![value]),
    }
}

#[test]
fn combined_isolation_shares_form_prefixes_and_replacement_images()
-> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox_ocr::{
        paint_isolation::{PaintReplacement, isolate_source_paints},
        rasterize::render_rgb_page,
    };
    let original = shared_form_fixture();
    let doc = PdfMetadata::from_bytes(original)?;
    let pages = [PageNumber::new(1)?, PageNumber::new(2)?];
    let scan = enumerate_image_paints(&doc, &pages)?;
    let p = &scan.placements;
    assert_eq!(p.len(), 8);
    assert_eq!(p[0].paint_path, [0, 0]);
    assert_eq!(p[1].paint_path, [0, 1]);
    assert_eq!(p[6].paint_path, [1, 0]);
    let result = isolate_source_paints(
        &doc,
        [
            PaintReplacement {
                image: gray_image(255),
                paints: vec![&p[0], &p[6]],
            },
            PaintReplacement {
                image: gray_image(128),
                paints: vec![&p[1]],
            },
        ],
    )?;
    let rewritten = PdfMetadata::from_bytes(result.clone())?;
    let after = enumerate_image_paints(&rewritten, &pages)?;
    let a = &after.placements;
    assert_eq!(a.len(), p.len());
    assert_eq!(a[0].source_id, a[6].source_id);
    assert_ne!(a[0].source_id, a[1].source_id);
    for index in [2, 3, 4, 5, 7] {
        assert_eq!(a[index].source_id, a[2].source_id);
        assert_ne!(a[index].source_id, a[0].source_id);
        assert_ne!(a[index].source_id, a[1].source_id);
    }
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("combined.pdf");
    fs::write(&path, result)?;
    let policy = RenderPolicy {
        max_side: 200,
        dpi: 72,
    };
    let first = render_rgb_page(&path, pages[0], policy)?;
    let second = render_rgb_page(&path, pages[1], policy)?;
    assert_eq!(first.dimensions(), (200, 100));
    assert_eq!(second.dimensions(), (200, 100));
    assert_eq!(first.get_pixel(20, 70).0, [255; 3]);
    assert_eq!(first.get_pixel(45, 70).0, [128; 3]);
    for x in [110, 135] {
        assert_eq!(first.get_pixel(x, 70).0, [0; 3]);
    }
    for x in [20, 45, 135] {
        assert_eq!(second.get_pixel(x, 70).0, [0; 3]);
    }
    assert_eq!(second.get_pixel(110, 70).0, [255; 3]);
    Ok(())
}

#[test]
fn combined_isolation_rejects_conflicting_or_invalid_paths()
-> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox_ocr::paint_isolation::{PaintReplacement, isolate_source_paints};
    let original = shared_form_fixture();
    let doc = PdfMetadata::from_bytes(original.clone())?;
    let scan = enumerate_image_paints(&doc, &[PageNumber::new(1)?])?;
    let first = &scan.placements[0];
    let replace = |paints| {
        isolate_source_paints(
            &doc,
            [PaintReplacement {
                image: gray_image(255),
                paints,
            }],
        )
    };
    assert!(replace(vec![first, first]).is_err());
    let mut prefix = first.clone();
    prefix.paint_path.pop();
    assert!(replace(vec![first, &prefix]).is_err());
    assert!(replace(vec![&prefix, first]).is_err());
    let mut bad = first.clone();
    bad.object_number += 1;
    assert!(replace(vec![&bad]).is_err());
    let mut bad = first.clone();
    bad.page = 0;
    assert!(replace(vec![&bad]).is_err());
    let mut bad = first.clone();
    bad.paint_path = vec![0, 99];
    assert!(replace(vec![&bad]).is_err());
    assert_eq!(isolate_source_paints(&doc, [])?, original);
    Ok(())
}
