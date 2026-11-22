//! Preserve independent labels while increasing actual source sampling detail.
#![cfg(feature = "pdf-integration")]

use std::{collections::BTreeSet, fs};

use pdf_redox_ocr::{
    PageNumber,
    character_alignment::{BatchGeometry, CharacterAuditReport},
    crop_geometry::CropGeometry,
    draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
    geometry::{PageBox, Point, Quad},
    pdf_metadata::PdfMetadata,
    quality_eval::{GroundTruth, ReferenceSample},
    rasterize::render_rgb_region,
    recognition::{PixelLine, RasterCharacterAudit, RasterCharacterMapping, RasterMappingScope},
    reviewed_glyphs::ReviewedGlyphBank,
    source_glyphs::resample,
};
use sha2::{Digest, Sha256};

#[test]
fn resampling_uses_actual_pdf_ink_for_independently_labeled_runs()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("source.pdf");
    fs::write(&pdf, include_bytes!("fixtures/line-audit-rotations.pdf"))?;
    let metadata = PdfMetadata::open(&pdf)?;
    let page = PageNumber::new(1)?;
    let crop = render_rgb_region(
        &pdf,
        page,
        metadata.mapper(page)?,
        PageBox::new([50., 65., 125., 91.])?,
        150,
        2_000_000,
    )?;
    let dimensions = crop.rgb.dimensions();
    let (w, h) = dimensions;
    let w64 = f64::from(w);
    let h64 = f64::from(h);
    let corners: Quad = [
        Point(0., 0.),
        Point(w64, 0.),
        Point(w64, h64),
        Point(0., h64),
    ]
    .map(|p| crop.pixel_to_pdf(p));
    let line = PixelLine {
        text: "08i".into(),
        confidence: 0.99,
        detection_index: 17,
        quad: [
            Point(0., h64),
            Point(w64, h64),
            Point(w64, 0.),
            Point(0., 0.),
        ],
    };
    let batch = BatchGeometry::new([dimensions])?;
    let resized_width = usize::try_from((w * 48).div_ceil(h))?;
    let columns = [12, 33, 59].map(|x| x * resized_width / 75);
    let mut report = CharacterAuditReport::default();
    let audit = report
        .observe(&line, &crop.rgb, batch, &columns, batch.tensor_width)
        .ok_or("fixture audit")?;
    audit.geometry = CropGeometry::from_quad(line.quad, dimensions, dimensions);
    assert!(audit.geometry.is_some());
    let digest = hex::encode(Sha256::digest(crop.rgb.as_raw()));
    let mapping = RasterCharacterMapping {
        page,
        rgb_sha256: digest.clone(),
        dimensions,
        scope: RasterMappingScope::WholePage { residual: false },
        raster_corners: corners,
    };
    let draft = OcrDraft::new(
        sha256_file(&pdf)?,
        "fixture".into(),
        vec![DraftPage {
            page: 1,
            lines: vec![OcrLine {
                text: line.text,
                confidence: line.confidence,
                quad: mapping.to_pdf(line.quad),
                reviewed: false,
            }],
            native_text_regions: vec![],
        }],
    );
    let reference = GroundTruth {
        schema_version: 1,
        source_pdf_sha256: draft.pdf_sha256.clone(),
        coordinate_system: "PDF user space".into(),
        provenance: "independently identified fixture shapes".into(),
        samples: vec![ReferenceSample {
            page: 1,
            center: Point(85., 78.),
            text: "08i".into(),
            kind: "fixture".into(),
        }],
    };
    let audits = [RasterCharacterAudit {
        dimensions,
        rgb_sha256: digest,
        report,
    }];
    let mappings = [mapping];
    let residual = BTreeSet::new();
    let mut bank =
        ReviewedGlyphBank::from_audits(&reference, &draft, &audits, &mappings, &residual)?;
    assert_eq!(bank.glyphs.len(), 3, "{:?}", bank.stats);
    resample(&mut bank, &pdf, &audits, &mappings, &residual, 600)?;
    assert_sampled_shapes(&bank, dimensions)?;
    Ok(())
}

fn assert_sampled_shapes(
    bank: &ReviewedGlyphBank,
    dimensions: (u32, u32),
) -> Result<(), Box<dyn std::error::Error>> {
    let stats = bank
        .source_sampling
        .as_ref()
        .ok_or("missing source sampling")?;
    assert_eq!(stats.sampled_lines, 1, "{:?}", stats.failures);
    assert_eq!(stats.matched_glyphs, 3, "{stats:?}");
    for (glyph, label) in bank.glyphs.iter().zip(['0', '8', 'i']) {
        assert_eq!(glyph.label, label);
        assert_eq!(glyph.prediction, label);
        let source = glyph
            .source_appearance
            .as_ref()
            .ok_or("missing appearance")?;
        assert!(source.line_dimensions.0 > dimensions.0 * 3);
        assert_eq!(source.normalized_line_sha256.len(), 64);
    }
    Ok(())
}
