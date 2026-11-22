//! Real PDF mask round trips across packed and wide source sample encodings.

#![cfg(feature = "pdf-integration")]

use std::{collections::BTreeMap, fs, path::Path};

use pdf_redox::{ObjectId, SourcePdf};
use pdf_redox_ocr::source_appearances::PaintAppearances;
use pdf_redox_ocr::{
    PageNumber,
    draft::OcrLine,
    geometry::Point,
    mixed_visibility::prove_mixed_source_text,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    source_mask::mask_source_text,
    source_ocr::SourceObservation,
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
        let id = ObjectId::new(paint.object_number, i32::from(paint.generation));
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

#[test]
fn native_formats_admit_exclusively_pixel_verified_source_labels()
-> Result<(), Box<dyn std::error::Error>> {
    let original = include_bytes!("fixtures/source-sample-formats.pdf");
    let temp = tempfile::tempdir()?;
    let pdf = temp.path().join("ten-formats.pdf");
    fs::write(&pdf, original)?;
    let document = PdfMetadata::open(&pdf)?;
    let pages = (1..=10)
        .map(PageNumber::new)
        .collect::<Result<Vec<_>, _>>()?;
    let painted = enumerate_image_paints(&document, &pages)?;
    assert_eq!(painted.placements.len(), 10);
    let recognized = painted
        .placements
        .iter()
        .map(|paint| {
            (
                paint.source_id.clone(),
                SourceObservation {
                    width: paint.width,
                    height: paint.height,
                    lines: vec![original_line()],
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(recognized.len(), 10);
    let verified = prove_mixed_source_text(
        &pdf,
        &document,
        &painted,
        &PaintAppearances::from_object_inventory(recognized, &painted.placements),
        &pages,
        RenderPolicy {
            dpi: 72,
            max_side: 450,
        },
        0.97,
    )?;
    assert_eq!(verified.audited.len(), 10, "{:#?}", verified.audited);
    assert_eq!(verified.proofs.len(), 10, "{:#?}", verified.audited);
    assert_eq!(verified.observations.len(), 10);
    assert_eq!(verified.proven_pages.len(), 10);
    assert!(
        verified
            .audited
            .iter()
            .all(|audit| audit.visible_lines == 1)
    );
    Ok(())
}

mod support;

fn contextual_palette_fixture() -> Vec<u8> {
    let mut samples = vec![0u8; 12 * 32];
    for row in samples.as_chunks_mut::<12>().0.iter_mut().skip(8).take(16) {
        row[2..9].fill(255);
    }
    support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 64] /Resources << /XObject << /A 5 0 R /B 6 0 R >> >> /Contents 4 0 R >>".to_vec(),
        support::stream("", b"0.5 g 0 0 240 2 re f q 1 0 0 1 12 12 cm /A Do Q q 1 0 0 1 132 12 cm /B Do Q"),
        support::stream("/Type /XObject /Subtype /Form /BBox [0 0 96 32] /Resources << /ColorSpace << /Local [/Indexed /DeviceRGB 1 <ffffff000000>] >> /XObject << /Im 7 0 R >> >>", b"q 96 0 0 32 0 0 cm /Im Do Q"),
        support::stream("/Type /XObject /Subtype /Form /BBox [0 0 96 32] /Resources << /ColorSpace << /Local [/Indexed /DeviceRGB 1 <ffff000000ff>] >> /XObject << /Im 7 0 R >> >>", b"q 96 0 0 32 0 0 cm /Im Do Q"),
        support::stream("/Type /XObject /Subtype /Image /Width 96 /Height 32 /BitsPerComponent 1 /ColorSpace /Local", &samples),
    ])
}

#[test]
fn contextual_palette_masks_preserve_form_appearances_without_false_visibility()
-> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox_ocr::{direct_images::direct_native_images, native_images::NativeImageBudget};
    use std::collections::BTreeSet;

    let bytes = contextual_palette_fixture();
    let tmp = tempfile::tempdir()?;
    let file = tmp.path().join("context.pdf");
    fs::write(&file, &bytes)?;
    let doc = PdfMetadata::from_bytes(bytes)?;
    let pages = [PageNumber::new(1)?];
    let report = enumerate_image_paints(&doc, &pages)?;
    assert_eq!(report.placements.len(), 2);
    assert!(
        report
            .placements
            .iter()
            .all(|p| p.sample_layout.contextual && p.sample_layout.components == Some(1))
    );
    let decoded = direct_native_images(
        &doc,
        &pages,
        &BTreeSet::from(["7:0".into()]),
        &report.placements,
        NativeImageBudget::default(),
    )?;
    assert_eq!(
        decoded.stats.accepted_appearances, 2,
        "{:#?}",
        decoded.stats
    );
    let recognized = contextual_observations(&decoded.appearances);
    let verified = prove_mixed_source_text(
        &file,
        &doc,
        &report,
        &recognized,
        &pages,
        RenderPolicy {
            dpi: 72,
            max_side: 240,
        },
        0.97,
    )?;
    // Poppler rejects nonstandard named XObject color spaces; their native
    // decoding alone cannot prove that any page pixels are actually visible.
    assert!(verified.proofs.is_empty(), "{:#?}", verified.audited);
    assert_eq!(verified.audited.len(), 2);
    assert_eq!(verified.audited[0].paint_index, 0);
    assert_eq!(verified.audited[1].paint_index, 1);
    assert_ne!(
        verified.audited[0].appearance_id,
        verified.audited[1].appearance_id
    );
    assert_isolated_contextual_masks(&doc, &report, &decoded.appearances, &recognized, &pages)?;
    let unbound = PaintAppearances::from_object_inventory(
        BTreeMap::from([(
            "7:0".into(),
            recognized
                .variants
                .values()
                .next()
                .ok_or("missing recognition")?
                .value
                .clone(),
        )]),
        &report.placements,
    );
    assert!(
        unbound.is_empty(),
        "object-only decoder cannot prove contextual aliases"
    );
    Ok(())
}

fn contextual_observations(
    decoded: &PaintAppearances<image::RgbImage>,
) -> PaintAppearances<SourceObservation> {
    use pdf_redox_ocr::source_appearances::SourceAppearance;
    PaintAppearances {
        variants: decoded
            .variants
            .iter()
            .map(|(&id, appearance)| {
                let text = if appearance.value.get_pixel(0, 0).0 == [255, 255, 255] {
                    "BLACK LABEL"
                } else {
                    "BLUE LABEL"
                };
                (
                    id,
                    SourceAppearance {
                        source_id: appearance.source_id.clone(),
                        value: SourceObservation {
                            width: 96,
                            height: 32,
                            lines: vec![OcrLine {
                                text: text.into(),
                                confidence: 0.99,
                                reviewed: false,
                                quad: [
                                    Point(12., 6.),
                                    Point(76., 6.),
                                    Point(76., 26.),
                                    Point(12., 26.),
                                ],
                            }],
                        },
                    },
                )
            })
            .collect(),
        paint_bindings: decoded.paint_bindings.clone(),
    }
}

fn assert_isolated_contextual_masks(
    doc: &PdfMetadata,
    report: &pdf_redox_ocr::paint_scan::PaintReport,
    decoded: &PaintAppearances<image::RgbImage>,
    recognized: &PaintAppearances<SourceObservation>,
    pages: &[PageNumber],
) -> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox_ocr::{
        direct_images::direct_native_images,
        native_images::NativeImageBudget,
        paint_isolation::{PaintReplacement, isolate_source_paints},
        source_mask::{MaskSource, mask_source_object},
    };
    for (index, paint) in report.placements.iter().enumerate() {
        let (_, observation) = recognized
            .get(paint)
            .ok_or("missing contextual observation")?;
        let masked = mask_source_object(
            doc.source(),
            MaskSource::Paint(paint),
            &observation.lines,
            0.97,
        )?
        .ok_or("contextual sample mask unavailable")?;
        assert!(masked.metrics[0].changed_source_pixels > 500);
        let treatment = isolate_source_paints(
            doc,
            [PaintReplacement {
                image: masked.object,
                paints: vec![paint],
            }],
        )?;
        let treatment = PdfMetadata::from_bytes(treatment)?;
        let after = enumerate_image_paints(&treatment, pages)?;
        let wanted = after
            .placements
            .iter()
            .map(|p| p.source_id.clone())
            .collect();
        let images = direct_native_images(
            &treatment,
            pages,
            &wanted,
            &after.placements,
            NativeImageBudget::default(),
        )?;
        for (i, (original, replaced)) in report.placements.iter().zip(&after.placements).enumerate()
        {
            let before_rgb = decoded
                .get(original)
                .ok_or("original appearance missing")?
                .1;
            let after_rgb = images
                .appearances
                .get(replaced)
                .ok_or("treatment appearance missing")?
                .1;
            if i == index {
                assert!(
                    after_rgb
                        .pixels()
                        .all(|pixel| pixel == before_rgb.get_pixel(0, 0))
                );
            } else {
                assert_eq!(before_rgb, after_rgb);
            }
        }
    }
    Ok(())
}

/// Compare truly high-contrast adjacent indices with distant codes that have
/// nearly equal displayed colors. Raw tuple counts must remain independent.
#[test]
fn palette_visibility_uses_rgb_contrast_not_index_distance()
-> Result<(), Box<dyn std::error::Error>> {
    for noisy_background in [false, true] {
        check_palette_contrast(noisy_background)?;
    }
    Ok(())
}

fn check_palette_contrast(noisy_background: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = vec![0u8; 96 * 32];
    if noisy_background {
        for row in samples.as_chunks_mut::<96>().0.iter_mut().skip(6).take(20) {
            row[12..76].fill(255);
        }
    }
    for row in samples.as_chunks_mut::<96>().0.iter_mut().skip(8).take(16) {
        row[16..72].fill(1);
    }
    let mut palette = vec![255_u8; 256 * 3];
    palette[3..6].fill(0);
    palette[255 * 3..].fill(247);
    let colors = hex::encode(palette);
    let dictionary = format!(
        "/Type /XObject /Subtype /Image /Width 96 /Height 32 /BitsPerComponent 8 /ColorSpace [/Indexed /DeviceRGB 255 <{colors}>]"
    );
    let bytes = support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 64] /Resources << /XObject << /Im 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        support::stream("", b"0.5 g 0 0 120 2 re f q 96 0 0 32 12 12 cm /Im Do Q"),
        support::stream(&dictionary, &samples),
    ]);
    let tmp = tempfile::tempdir()?;
    let file = tmp.path().join("indexed.pdf");
    fs::write(&file, &bytes)?;
    let doc = PdfMetadata::from_bytes(bytes.clone())?;
    let pages = [PageNumber::new(1)?];
    let report = enumerate_image_paints(&doc, &pages)?;
    let line = OcrLine {
        text: "PALETTE INDEX".into(),
        confidence: 0.99,
        reviewed: false,
        quad: [
            Point(12., 6.),
            Point(76., 6.),
            Point(76., 26.),
            Point(12., 26.),
        ],
    };
    let changed_samples = if noisy_background { 64 * 20 } else { 56 * 16 };
    let masked = mask_source_text(
        &bytes,
        ObjectId::new(5, 0),
        std::slice::from_ref(&line),
        0.97,
    )?
    .ok_or("one-step palette index change was overlooked")?;
    assert_eq!(masked.metrics[0].changed_source_pixels, changed_samples);
    let recognized = PaintAppearances::from_object_inventory(
        BTreeMap::from([(
            "5:0".into(),
            SourceObservation {
                width: 96,
                height: 32,
                lines: vec![line],
            },
        )]),
        &report.placements,
    );
    let verified = prove_mixed_source_text(
        &file,
        &doc,
        &report,
        &recognized,
        &pages,
        RenderPolicy {
            dpi: 72,
            max_side: 240,
        },
        0.97,
    )?;
    assert_eq!(verified.proofs.len(), 1, "{:#?}", verified.audited);
    assert_eq!(verified.proofs[0].text, "PALETTE INDEX");
    assert_eq!(verified.audited[0].contrast[0].native_rgb_pixels, 896);
    assert_eq!(
        verified.audited[0].contrast[0].changed_samples,
        changed_samples
    );
    Ok(())
}
