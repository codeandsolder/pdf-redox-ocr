//! Exercise actual native decoding through the shared parsed source and
//! interpreter, including per-occurrence context and targeted resource limits.
#![cfg(feature = "pdf-integration")]

use pdf_redox_ocr::{
    PageNumber,
    direct_images::{DirectImages, direct_native_images},
    native_images::{NativeImageBudget, native_source_images},
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
};
use std::{collections::BTreeSet, fs};

mod support;

fn image<'a>(result: &'a DirectImages, source: &str) -> Option<&'a image::RgbImage> {
    result
        .appearances
        .variants
        .values()
        .find(|v| v.source_id == source)
        .map(|v| &v.value)
}

fn extract(
    bytes: &[u8],
    pages: &[PageNumber],
    budget: NativeImageBudget,
) -> Result<DirectImages, pdf_redox_ocr::Error> {
    let doc = PdfMetadata::from_bytes(bytes.to_vec())?;
    let paints = enumerate_image_paints(&doc, pages)?;
    let wanted = paints
        .placements
        .iter()
        .map(|paint| paint.source_id.clone())
        .collect();
    direct_native_images(&doc, pages, &wanted, &paints.placements, budget)
}

#[test]
fn repeated_source_matches_poppler_and_validates_every_context()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = include_bytes!("fixtures/repeated-image-six-pages.pdf");
    let pages = (1..=6)
        .map(PageNumber::new)
        .collect::<Result<Vec<_>, _>>()?;
    let actual = extract(bytes, &pages, NativeImageBudget::default())?;
    assert_eq!(actual.stats.matched_occurrences, 6, "{:?}", actual.stats);
    assert_eq!(actual.stats.decoded_occurrences, 1);
    assert_eq!(actual.stats.reused_occurrences, 5);
    assert_eq!(actual.stats.reused_pixels, 1_800_000);
    assert_eq!(actual.stats.decoded_pixels, 360_000);
    assert_eq!(actual.stats.accepted_sources, 1);
    assert!(actual.stats.rejected_paints.is_empty());
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("fixture.pdf");
    fs::write(&file, bytes)?;
    let previous = native_source_images(
        &file,
        &pages,
        &BTreeSet::from(["9:0".into()]),
        NativeImageBudget::default(),
    )?;
    assert_eq!(image(&actual, "9:0"), previous.get("9:0"));
    Ok(())
}

#[test]
fn unrelated_inline_image_does_not_consume_targeted_decode_budget()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = include_bytes!("fixtures/inline-plus-source-budget.pdf");
    let pages = [PageNumber::new(1)?];
    for (budget, expected, reason) in [
        (
            NativeImageBudget {
                extraction_pixels: 360_000,
                ..NativeImageBudget::default()
            },
            1,
            None,
        ),
        (
            NativeImageBudget {
                extraction_pixels: 359_999,
                ..NativeImageBudget::default()
            },
            0,
            Some("page_decode_budget"),
        ),
        (
            NativeImageBudget {
                source_pixels: 359_999,
                ..NativeImageBudget::default()
            },
            0,
            Some("source_pixel_budget"),
        ),
        (
            NativeImageBudget {
                total_pixels: 359_999,
                ..NativeImageBudget::default()
            },
            0,
            Some("resident_appearance_budget"),
        ),
    ] {
        let actual = extract(bytes, &pages, budget)?;
        assert_eq!(actual.appearances.len(), expected, "{:?}", actual.stats);
        assert_eq!(
            actual.stats.decoded_pixels,
            if expected == 1 { 360_000 } else { 0 }
        );
        assert_eq!(
            actual
                .stats
                .rejected_paints
                .values()
                .flat_map(|p| p.values())
                .next()
                .copied(),
            reason
        );
    }
    Ok(())
}

fn color_fixture() -> Vec<u8> {
    support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /XObject << /Gray 5 0 R /RGB 6 0 R /Indexed 7 0 R /CMYK 8 0 R /Alpha 9 0 R /Stencil 10 0 R >> >> /Contents 4 0 R >>".to_vec(),
        support::stream("",b"q 20 0 0 10 0 0 cm /Gray Do Q q 20 0 0 10 20 0 cm /RGB Do Q q 20 0 0 10 40 0 cm /Indexed Do Q q 20 0 0 10 60 0 cm /CMYK Do Q q 20 0 0 10 80 0 cm /Alpha Do Q q 20 0 0 10 100 0 cm /Stencil Do Q"),
        support::stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0]",&[0,255]),
        support::stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",&[255,0,0,0,255,0]),
        support::stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace [/Indexed /DeviceRGB 1 <0000ffffffff>] /BitsPerComponent 8",&[0,1]),
        support::stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceCMYK /BitsPerComponent 8",&[0,0,0,0,0,0,0,255]),
        support::stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask [255 255 0 0 0 0]",&[255,0,0,0,255,0]),
        support::stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ImageMask true /BitsPerComponent 1",&[0b1000_0000]),
    ])
}

#[test]
fn native_color_decode_and_compositing_rejection() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = color_fixture();
    let pages = [PageNumber::new(1)?];
    let actual = extract(&bytes, &pages, NativeImageBudget::default())?;
    assert_eq!(actual.appearances.len(), 4, "{:?}", actual.stats);
    for (id, expected) in [
        ("5:0", [255, 255, 255, 0, 0, 0]),
        ("6:0", [255, 0, 0, 0, 255, 0]),
        ("7:0", [0, 0, 255, 255, 255, 255]),
    ] {
        assert_eq!(
            image(&actual, id).ok_or("missing decoded image")?.as_raw(),
            &expected,
            "{id}"
        );
    }
    assert!(image(&actual, "8:0").is_some());
    assert!(image(&actual, "9:0").is_none());
    assert!(image(&actual, "10:0").is_none());
    Ok(())
}

/// The same image stream painted via two named color-space resources.
fn context_fixture(
    same_page: bool,
    second_palette: &str,
    samples: [u8; 2],
    repeats: usize,
) -> Vec<u8> {
    let first=b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /XObject << /A 7 0 R /B 8 0 R >> >> /Contents 5 0 R >>".to_vec();
    let second=b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /XObject << /B 8 0 R >> >> /Contents 6 0 R >>".to_vec();
    let pages = if same_page {
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec()
    } else {
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec()
    };
    let second_content = "q 1 0 0 1 10 10 cm /B Do Q ".repeat(repeats);
    let mut content = "q 1 0 0 1 10 10 cm /A Do Q ".repeat(repeats);
    if same_page {
        content.push_str(&second_content);
    }
    support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        pages,
        first,
        second,
        support::stream("", content.as_bytes()),
        support::stream("", second_content.as_bytes()),
        support::stream(
            "/Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /ColorSpace << /Local [/Indexed /DeviceRGB 1 <ff000000ff00>] >> /XObject << /Im 9 0 R >> >>",
            b"q 40 0 0 20 10 10 cm /Im Do Q",
        ),
        support::stream(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /ColorSpace << /Local [/Indexed /DeviceRGB 1 <{second_palette}>] >> /XObject << /Im 9 0 R >> >>"
            ),
            b"q 40 0 0 20 10 10 cm /Im Do Q",
        ),
        support::stream(
            "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /Local /BitsPerComponent 8",
            &samples,
        ),
    ])
}

#[test]
fn different_resolved_resource_contexts_do_not_share_pixels()
-> Result<(), Box<dyn std::error::Error>> {
    let source = context_fixture(false, "0000ffffffff", [0, 1], 1);
    for (page, expected) in [(1, [255, 0, 0, 0, 255, 0]), (2, [0, 0, 255, 255, 255, 255])] {
        let single = extract(
            &source,
            &[PageNumber::new(page)?],
            NativeImageBudget::default(),
        )?;
        assert_eq!(
            image(&single, "9:0")
                .ok_or("context source missing")?
                .as_raw(),
            &expected
        );
    }
    for same_page in [true, false] {
        let pages = (1..=if same_page { 1 } else { 2 })
            .map(PageNumber::new)
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = context_fixture(same_page, "0000ffffffff", [0, 1], 2);
        let actual = extract(&bytes, &pages, NativeImageBudget::default())?;
        let doc = PdfMetadata::from_bytes(bytes)?;
        let paints = enumerate_image_paints(&doc, &pages)?;
        assert_eq!(actual.stats.decoded_occurrences, 2);
        assert_eq!(actual.stats.reused_occurrences, 2);
        assert_eq!(actual.stats.accepted_appearances, 2);
        assert_eq!(actual.stats.accepted_occurrences, 4);
        assert!(actual.stats.rejected_paints.is_empty());
        for (index, paint) in paints.placements.iter().enumerate() {
            let (_, rgb) = actual.appearances.get(paint).ok_or("binding missing")?;
            let expected = if index < 2 {
                [255, 0, 0, 0, 255, 0]
            } else {
                [0, 0, 255, 255, 255, 255]
            };
            assert_eq!(rgb.as_raw(), &expected);
        }
        assert_eq!(paints.placements[0].ctm, paints.placements[2].ctm);
        let limited = extract(
            &context_fixture(same_page, "0000ffffffff", [0, 1], 2),
            &pages,
            NativeImageBudget {
                sources: 1,
                ..NativeImageBudget::default()
            },
        )?;
        assert_eq!(limited.stats.accepted_occurrences, 2);
        assert_eq!(limited.stats.accepted_appearances, 1);
    }
    Ok(())
}

#[test]
fn each_distinct_equal_pixel_context_is_decoded_only_once() -> Result<(), Box<dyn std::error::Error>>
{
    for same_page in [true, false] {
        let pages = (1..=if same_page { 1 } else { 2 })
            .map(PageNumber::new)
            .collect::<Result<Vec<_>, _>>()?;
        for (palette, decoded) in [("ff000000ff00", 1), ("ff00000000ff", 2)] {
            let result = extract(
                &context_fixture(same_page, palette, [0, 0], 2),
                &pages,
                NativeImageBudget {
                    source_pixels: 2,
                    total_pixels: 2,
                    sources: 1,
                    extraction_pixels: 2 * decoded,
                },
            )?;
            assert_eq!(result.stats.matched_occurrences, 4);
            assert_eq!(result.stats.decoded_pixels, 2 * decoded);
            assert_eq!(result.stats.reused_pixels, 8 - 2 * decoded);
            assert_eq!(
                result.stats.decoded_occurrences + result.stats.reused_occurrences,
                4
            );
            assert!(result.stats.rejected_paints.is_empty());
            assert_eq!(
                image(&result, "9:0").ok_or("source missing")?.as_raw(),
                &[255, 0, 0, 255, 0, 0]
            );
        }
    }
    Ok(())
}

#[test]
fn unsupported_root_color_contexts_cannot_publish_independent_rgb()
-> Result<(), Box<dyn std::error::Error>> {
    for (resources, page_extra, reason) in [
        (
            "/ColorSpace << /DefaultRGB [/CalRGB << /WhitePoint [0.95 1 1.09] /Gamma [2.2 2.2 2.2] >>] >>",
            "",
            "unsafe_scanned_occurrence",
        ),
        (
            "",
            "/Group << /S /Transparency /CS /DeviceGray >>",
            "page_transparency_group",
        ),
    ] {
        let bytes=support::pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << {resources} /XObject << /Im 5 0 R >> >> /Contents 4 0 R {page_extra} >>").into_bytes(),
            support::stream("",b"q 100 0 0 100 0 0 cm /Im Do Q"),
            support::stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",&[128,64,32]),
        ]);
        let pages = [PageNumber::new(1)?];
        let doc = PdfMetadata::from_bytes(bytes)?;
        let paints = enumerate_image_paints(&doc, &pages)?;
        if !resources.is_empty() {
            assert_eq!(
                paints.placements[0].unsafe_reasons,
                vec!["resource_default_color_space_requires_resolution"]
            );
        }
        let result = direct_native_images(
            &doc,
            &pages,
            &BTreeSet::from(["5:0".into()]),
            &paints.placements,
            NativeImageBudget::default(),
        )?;
        assert_eq!(result.stats.decoded_occurrences, 0);
        assert_eq!(
            result.stats.rejected_paints.get(&1).and_then(|p| p.get(&0)),
            Some(&reason)
        );
    }
    Ok(())
}

#[test]
fn unmatched_geometry_cannot_publish_a_source() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = include_bytes!("fixtures/repeated-image-six-pages.pdf");
    let doc = PdfMetadata::from_bytes(bytes.to_vec())?;
    let pages = [PageNumber::new(1)?];
    let mut paints = enumerate_image_paints(&doc, &pages)?.placements;
    paints[0].ctm[4] += 1.;
    let actual = direct_native_images(
        &doc,
        &pages,
        &BTreeSet::from(["9:0".into()]),
        &paints,
        NativeImageBudget::default(),
    )?;
    assert!(actual.appearances.is_empty());
    assert_eq!(actual.stats.decoded_occurrences, 0);
    assert_eq!(
        actual.stats.rejected_pages.get(&1),
        Some(&"unmatched_interpreter_occurrence")
    );
    Ok(())
}

#[test]
fn later_incompatible_page_does_not_starve_independent_source()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /XObject << /A 7 0 R /B 8 0 R >> >> /Contents 5 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Group << /S /Transparency /CS /DeviceGray >> /Resources << /XObject << /A 7 0 R >> >> /Contents 6 0 R >>".to_vec(),
        support::stream("", b"q 50 0 0 50 0 0 cm /A Do Q q 50 0 0 50 50 0 cm /B Do Q"),
        support::stream("", b"q 50 0 0 50 0 0 cm /A Do Q"),
        support::stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8", &[255, 0, 0]),
        support::stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8", &[0, 255, 0]),
    ]);
    let pages = [PageNumber::new(1)?, PageNumber::new(2)?];
    let result = extract(
        &bytes,
        &pages,
        NativeImageBudget {
            sources: 2,
            total_pixels: 2,
            extraction_pixels: 2,
            ..NativeImageBudget::default()
        },
    )?;
    assert_eq!(
        result.stats.rejected_pages.get(&2),
        Some(&"page_transparency_group")
    );
    assert_eq!(result.stats.decoded_pixels, 2);
    assert_eq!(result.stats.decoded_occurrences, 2);
    assert_eq!(result.stats.accepted_sources, 2);
    assert_eq!(result.stats.accepted_occurrences, 2);
    assert_eq!(
        image(&result, "8:0")
            .ok_or("independent source missing")?
            .as_raw(),
        &[0, 255, 0]
    );
    Ok(())
}

fn split_program_fixture(first: &[u8], second: &[u8]) -> Vec<u8> {
    support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /XObject << /Im 6 0 R >> >> /Contents [4 0 R 5 0 R] >>".to_vec(),
        support::stream("", first),
        support::stream("", second),
        support::stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8", &[255, 0, 0]),
    ])
}

#[test]
fn exact_do_paths_disambiguate_equal_transforms_after_skipped_operations()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = split_program_fixture(b"Do /Missing Do /Im Do", b"/Im Do");
    let doc = PdfMetadata::from_bytes(bytes)?;
    let pages = [PageNumber::new(1)?];
    let mut paints = enumerate_image_paints(&doc, &pages)?.placements;
    let wanted = BTreeSet::from(["6:0".into()]);
    assert_eq!(paints.len(), 2);
    assert_eq!(paints[0].paint_path, [2]);
    assert_eq!(paints[1].paint_path, [3]);
    assert_eq!(paints[0].ctm, paints[1].ctm);
    let actual =
        direct_native_images(&doc, &pages, &wanted, &paints, NativeImageBudget::default())?;
    assert_eq!(actual.stats.matched_occurrences, 2);
    assert_eq!(actual.stats.decoded_occurrences, 1);
    assert_eq!(actual.stats.reused_occurrences, 1);
    assert!(image(&actual, "6:0").is_some());

    paints[0].paint_path = vec![1];
    let missing =
        direct_native_images(&doc, &pages, &wanted, &paints, NativeImageBudget::default())?;
    assert_eq!(
        missing.stats.rejected_pages.get(&1),
        Some(&"unmatched_interpreter_occurrence")
    );
    assert_eq!(missing.stats.decoded_occurrences, 0);

    paints[0].paint_path = paints[1].paint_path.clone();
    let duplicate =
        direct_native_images(&doc, &pages, &wanted, &paints, NativeImageBudget::default())?;
    assert_eq!(
        duplicate.stats.rejected_pages.get(&1),
        Some(&"duplicate_scanned_occurrence")
    );
    assert_eq!(duplicate.stats.decoded_occurrences, 0);
    Ok(())
}

#[test]
fn divergent_content_array_programs_are_rejected_before_decode()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = split_program_fixture(b"/Im Do % comment", b"/Im Do");
    let pages = [PageNumber::new(1)?];
    let doc = PdfMetadata::from_bytes(bytes.clone())?;
    assert_eq!(enumerate_image_paints(&doc, &pages)?.placements.len(), 2);
    let actual = extract(&bytes, &pages, NativeImageBudget::default())?;
    assert_eq!(
        actual.stats.rejected_pages.get(&1),
        Some(&"page_content_path_mismatch")
    );
    assert_eq!(actual.stats.decoded_occurrences, 0);
    assert!(actual.appearances.is_empty());
    Ok(())
}

#[test]
fn late_page_mismatch_discards_only_its_appearance_bindings()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = context_fixture(false, "0000ffffffff", [0, 1], 2);
    let doc = PdfMetadata::from_bytes(bytes)?;
    let pages = [PageNumber::new(1)?, PageNumber::new(2)?];
    let mut paints = enumerate_image_paints(&doc, &pages)?.placements;
    assert_eq!(paints.len(), 4);
    paints[3].ctm[4] += 1.;
    let result = direct_native_images(
        &doc,
        &pages,
        &BTreeSet::from(["9:0".into()]),
        &paints,
        NativeImageBudget::default(),
    )?;
    assert_eq!(result.stats.decoded_occurrences, 2);
    assert_eq!(result.stats.accepted_appearances, 1);
    assert_eq!(result.stats.accepted_occurrences, 2);
    assert_eq!(
        result.stats.rejected_pages.get(&2),
        Some(&"unmatched_interpreter_occurrence")
    );
    assert!(result.appearances.get(&paints[2]).is_none());
    assert!(result.appearances.get(&paints[3]).is_none());
    for paint in &paints[0..2] {
        assert_eq!(
            result
                .appearances
                .get(paint)
                .ok_or("verified first page lost")?
                .1
                .as_raw(),
            &[255, 0, 0, 0, 255, 0]
        );
    }
    Ok(())
}
