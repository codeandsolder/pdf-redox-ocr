//! Native PDF line evidence must retain its geometry across page rotations.

#![cfg(feature = "pdf-integration")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

use pdf_redox_ocr::{
    PageNumber,
    draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
    geometry::Point,
    line_audit::{LineAuditPolicy, audit_lines},
};

#[test]
fn source_ink_topology_and_pdf_mapping_survive_all_page_rotations()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let pdf = directory.path().join("source.pdf");
    fs::write(&pdf, include_bytes!("fixtures/line-audit-rotations.pdf"))?;
    let quad = [
        Point(50.0, 65.0),
        Point(125.0, 65.0),
        Point(125.0, 91.0),
        Point(50.0, 91.0),
    ];
    let pages = (1..=4)
        .map(|page| DraftPage {
            page,
            lines: vec![OcrLine {
                text: "08i".into(),
                confidence: 0.999,
                quad,
                reviewed: false,
            }],
            native_text_regions: vec![],
        })
        .collect();
    let mut draft = OcrDraft::new(sha256_file(&pdf)?, "synthetic vector ink".into(), pages);
    let targets = (1..=4)
        .map(|number| Ok((PageNumber::new(number)?, BTreeSet::from([0]))))
        .collect::<Result<BTreeMap<_, _>, pdf_redox_ocr::Error>>()?;
    let report = audit_lines(&pdf, &draft, &targets, &LineAuditPolicy::default())?;
    assert_eq!(report.lines.len(), 4);
    for line in report.lines {
        assert_eq!(line.original_quad, quad);
        assert_eq!(line.groups.len(), 3, "page {}", line.page);
        for ((group, expected_holes), expected_left) in
            line.groups.iter().zip([1, 2, 0]).zip([55.0, 75.0, 100.0])
        {
            assert!(!group.group.touches_line_edge);
            assert!(
                (group.ink_quad[0].0 - expected_left).abs() < 0.3,
                "page {}: {:?}",
                line.page,
                group.ink_quad
            );
            assert!((group.ink_quad[0].1 - 70.0).abs() < 0.3);
            for topology in &group.group.topology {
                assert_eq!(topology.enclosed_background_regions.len(), expected_holes);
            }
        }
    }
    assert!(
        audit_lines(
            &pdf,
            &draft,
            &targets,
            &LineAuditPolicy {
                max_pixels: 9,
                ..LineAuditPolicy::default()
            }
        )
        .is_err()
    );
    draft.pdf_sha256 = "00".repeat(32);
    assert!(audit_lines(&pdf, &draft, &targets, &LineAuditPolicy::default()).is_err());
    Ok(())
}
