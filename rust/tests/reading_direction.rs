//! Verify reading-frame geometry through the actual PDF overlay importer.

#![cfg(feature = "pdf-integration")]

use std::{fs, process::Command};

use pdf_redox::{
    ObjectHandle, OcrDocument, OcrLine, OcrOverlayOptions, OcrPage, OwnedObject, SourcePdf,
};
use pdf_redox_ocr::{
    PageNumber,
    crop_geometry::CropGeometry,
    geometry::{Point, Quad},
    rasterize::{RenderPolicy, render_rgb_page},
};
use sha2::{Digest, Sha256};

mod support;

fn fixture() -> Vec<u8> {
    support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>"
            .to_vec(),
        support::stream("", b"0.5 G 10 10 80 80 re S"),
    ])
}

fn text_matrix(pdf: Vec<u8>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let source = SourcePdf::from_bytes(pdf)?;
    let page = source.materialize(source.page_ids()[0])?;
    let Some(OwnedObject::Reference(ObjectHandle::Existing(id))) = page
        .as_dictionary()
        .and_then(|dict| dict.get(b"Contents".as_slice()))
    else {
        return Err("expected indirect overlay content".into());
    };
    let decoded = source.decoded_stream_data(*id)?;
    let text = std::str::from_utf8(&decoded)?;
    let before_matrix = text
        .split_once(" Tm ")
        .ok_or("missing actual text matrix")?
        .0;
    let mut values = before_matrix
        .split_whitespace()
        .rev()
        .take(6)
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()?;
    values.reverse();
    Ok(values)
}

#[test]
fn all_reading_directions_reach_the_pdf_baseline_and_remain_searchable()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let original = fixture();
    let input = dir.path().join("source.pdf");
    let output = dir.path().join("overlay.pdf");
    fs::write(&input, &original)?;
    let page = PageNumber::new(1)?;
    let render = RenderPolicy {
        max_side: 128,
        dpi: 72,
    };
    let pixels = render_rgb_page(&input, page, render)?;
    let digest = hex::encode(Sha256::digest(&original));
    for (width, height, half_turn, expected) in [
        (40., 10., false, [40., 0.]),
        (40., 10., true, [-40., 0.]),
        (10., 40., false, [0., -40.]),
        (10., 40., true, [0., 40.]),
    ] {
        let detector: Quad = [
            Point(20., 30. + height),
            Point(20. + width, 30. + height),
            Point(20. + width, 30.),
            Point(20., 30.),
        ];
        let order = CropGeometry::from_quad(detector, (100, 100), (40, 10))
            .and_then(|geometry| geometry.with_half_turn(half_turn).reading_order(detector))
            .ok_or("missing reading-frame geometry")?;
        let quad = order.map(|i| [detector[i].0, 100. - detector[i].1]);
        let draft = OcrDocument {
            schema_version: 3,
            pdf_sha256: digest.clone(),
            model: "direction fixture".into(),
            pages: vec![OcrPage {
                page: 1,
                native_text_regions: vec![],
                lines: vec![OcrLine {
                    text: "Searchable".into(),
                    confidence: 0.99,
                    quad,
                    reviewed: false,
                }],
            }],
        };
        let (annotated, report) = pdf_redox::apply_ocr_overlay(
            &original,
            &draft,
            OcrOverlayOptions {
                apply_unreviewed: true,
                min_confidence: 0.97,
            },
        )?;
        assert_eq!(report.accepted_lines, 1);
        fs::write(&output, &annotated)?;
        let matrix = text_matrix(annotated)?;
        assert_eq!(matrix.len(), 6);
        for (value, delta) in matrix.iter().zip(expected) {
            assert!((value * 0.667 - delta).abs() < 1e-6);
        }
        assert!((matrix[4] - quad[0][0]).abs() < 1e-6);
        assert!((matrix[5] - quad[0][1]).abs() < 1e-6);
        let extracted = Command::new("pdftotext")
            .arg("-raw")
            .arg(&output)
            .arg("-")
            .output()?;
        assert!(extracted.status.success());
        assert!(String::from_utf8(extracted.stdout)?.contains("Searchable"));
        assert_eq!(pixels, render_rgb_page(&output, page, render)?);
    }
    Ok(())
}
