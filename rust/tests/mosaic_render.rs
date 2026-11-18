//! Real PDF native-resolution crops agree exactly with complete page painting,
//! including clipping, alpha masks, rotated pages and nonzero page origins.

#![cfg(feature = "pdf-integration")]

use std::{fs, process::Command};

use image::ImageFormat;
use pdf_redox_ocr::{
    PageNumber,
    draft::OcrLine,
    geometry::{PageBox, Point},
    mosaic::{MosaicBudget, mosaic_page_eligible, plan_raster_mosaics},
    mosaic_ocr::reconstruct_native_mosaics,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    rasterize::render_rgb_region,
};

#[test]
fn native_mosaic_crop_matches_poppler_full_page_across_real_pdf_effects()
-> Result<(), Box<dyn std::error::Error>> {
    let input = include_bytes!("fixtures/fragmented-native-mosaics.pdf");
    let temp = tempfile::tempdir()?;
    let pdf = temp.path().join("fragmented.pdf");
    fs::write(&pdf, input)?;
    let doc = PdfMetadata::open(&pdf)?;
    let pages = (1..=5)
        .map(PageNumber::new)
        .collect::<Result<Vec<_>, _>>()?;
    let report = enumerate_image_paints(&doc, &pages)?;
    assert_eq!(report.placements.len(), 81);
    let media = PageBox::new([30.0, 40.0, 430.0, 340.0])?;

    for page in pages {
        let mapper = doc.mapper(page)?;
        let paints = report
            .placements
            .iter()
            .filter(|p| p.page == page.get())
            .cloned()
            .collect::<Vec<_>>();
        let expected = if page.get() == 2 { 1 } else { 20 };
        assert_eq!(paints.len(), expected, "page {}", page.get());
        let summary = report
            .pages
            .iter()
            .find(|s| s.page == page.get())
            .ok_or("missing paint summary")?;
        assert!(
            mosaic_page_eligible(summary, &paints),
            "page {} mosaic eligibility",
            page.get()
        );
        if page.get() == 4 {
            assert!(
                !summary.image_only_after_native,
                "alpha source should remain unsafe for direct OCR"
            );
        }
        let region = if page.get() == 2 {
            PageBox::new([80.0, 130.0, 224.0, 168.4])?
        } else {
            let plan = plan_raster_mosaics(page, &paints, media, mapper, MosaicBudget::default())?
                .ok_or_else(|| {
                    format!("missing source-resolution mosaic on page {}", page.get())
                })?;
            assert_eq!(plan.groups.len(), 1);
            assert_eq!(plan.groups[0].paint_indices.len(), 20);
            assert!(plan.estimated_pixels <= 12_000_000);
            plan.groups[0].region
        };
        let crop = render_rgb_region(&pdf, page, mapper, region, 300, 12_000_000)?;
        let full = Command::new("pdftoppm")
            .args([
                "-f",
                &page.get().to_string(),
                "-l",
                &page.get().to_string(),
                "-r",
                "300",
                "-singlefile",
            ])
            .arg(&pdf)
            .output()?;
        assert!(full.status.success());
        let reference =
            image::load_from_memory_with_format(&full.stdout, ImageFormat::Pnm)?.into_rgb8();
        let width_points = (mapper.to_pdf(1.0, 0.0).0 - mapper.to_pdf(0.0, 0.0).0)
            .hypot(mapper.to_pdf(1.0, 0.0).1 - mapper.to_pdf(0.0, 0.0).1);
        let height_points = (mapper.to_pdf(0.0, 1.0).0 - mapper.to_pdf(0.0, 0.0).0)
            .hypot(mapper.to_pdf(0.0, 1.0).1 - mapper.to_pdf(0.0, 0.0).1);
        for (x, y, pixel) in crop.rgb.enumerate_pixels() {
            let source = crop.pixel_to_pdf(Point(f64::from(x) + 0.5, f64::from(y) + 0.5));
            let normalized = mapper.to_raster(source);
            let full_x = num_traits::ToPrimitive::to_u32(
                &(normalized.0 * width_points * 300.0 / 72.0).floor(),
            )
            .ok_or("full page coordinate X out of bounds")?;
            let full_y = num_traits::ToPrimitive::to_u32(
                &(normalized.1 * height_points * 300.0 / 72.0).floor(),
            )
            .ok_or("full page coordinate Y out of bounds")?;
            assert_eq!(
                pixel,
                reference.get_pixel(full_x, full_y),
                "page {} crop ({x},{y}) maps to full ({full_x},{full_y})",
                page.get()
            );
        }
    }
    Ok(())
}

#[test]
fn group_estimate_or_source_orientation_does_not_bypass_resource_limits()
-> Result<(), Box<dyn std::error::Error>> {
    let doc =
        PdfMetadata::from_bytes(include_bytes!("fixtures/fragmented-native-mosaics.pdf").to_vec())?;
    let pages = [PageNumber::new(1)?, PageNumber::new(3)?];
    let report = enumerate_image_paints(&doc, &pages)?;
    for page in pages {
        let group = report
            .placements
            .iter()
            .filter(|p| p.page == page.get())
            .cloned()
            .collect::<Vec<_>>();
        let box_ = *doc.media_boxes().get(&page).ok_or("missing media")?;
        let mapper = doc.mapper(page)?;
        let plan = plan_raster_mosaics(page, &group, box_, mapper, MosaicBudget::default())?
            .ok_or("valid fragment group not found")?;
        assert_eq!(plan.groups[0].paint_indices.len(), 20);
        assert_eq!(
            plan.groups[0].clockwise_turns,
            if page.get() == 3 { 3 } else { 0 }
        );
        assert!(
            plan_raster_mosaics(
                page,
                &group,
                box_,
                mapper,
                MosaicBudget {
                    max_group_pixels: 100,
                    ..MosaicBudget::default()
                }
            )?
            .is_none()
        );
    }
    Ok(())
}

#[test]
fn complete_fragment_group_reconstructs_text_geometry_without_model_download()
-> Result<(), Box<dyn std::error::Error>> {
    let original = include_bytes!("fixtures/fragmented-native-mosaics.pdf");
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("mosaics.pdf");
    fs::write(&file, original)?;
    let doc = PdfMetadata::open(&file)?;
    for number in [1, 3, 4, 5] {
        let page = PageNumber::new(number)?;
        let report = enumerate_image_paints(&doc, &[page])?;
        assert_eq!(report.placements.len(), 20);
        let mapper = doc.mapper(page)?;
        let bounds = *doc.media_boxes().get(&page).ok_or("missing MediaBox")?;
        let plan = plan_raster_mosaics(
            page,
            &report.placements,
            bounds,
            mapper,
            MosaicBudget::default(),
        )?
        .ok_or("fragment plan missing")?;
        let mut calls = 0_usize;
        let found = reconstruct_native_mosaics(
            &file,
            mapper,
            &plan,
            &report.placements,
            MosaicBudget::default(),
            |rgb, to_pdf| {
                calls += 1;
                // The caller receives only one source-native crop for all 20
                // strips, including the true PDF paint stack alpha and clip.
                let dark = rgb.pixels().filter(|p| p.0[0] < 205).count();
                assert!(
                    dark > 50,
                    "page {number} has no recovered source ink ({dark})"
                );
                let w = f64::from(rgb.width());
                let h = f64::from(rgb.height());
                let corners = [
                    to_pdf(Point(w * 0.25, h * 0.75)),
                    to_pdf(Point(w * 0.75, h * 0.75)),
                    to_pdf(Point(w * 0.75, h * 0.25)),
                    to_pdf(Point(w * 0.25, h * 0.25)),
                ];
                Ok(vec![OcrLine {
                    text: "FRAGMENTED OCR".into(),
                    confidence: 0.99,
                    quad: corners,
                    reviewed: false,
                }])
            },
        )?
        .ok_or("fragment recognition admission failed")?;
        assert_eq!(calls, 1, "expected one model call for twenty source strips");
        assert_eq!(found.len(), 1);
        for point in found[0].quad {
            assert!(
                (75.0..=230.0).contains(&point.0),
                "page {number} PDF x {}",
                point.0
            );
            assert!(
                (125.0..=175.0).contains(&point.1),
                "page {number} PDF y {}",
                point.1
            );
        }
    }
    Ok(())
}
