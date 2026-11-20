//! Native PDF line evidence must retain its geometry across page rotations.

#![cfg(feature = "pdf-integration")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use pdf_redox_ocr::{
    PageNumber,
    draft::{DraftPage, OcrDraft, OcrLine, sha256_file},
    geometry::{Point, Quad},
    line_audit::{LineAuditPolicy, audit_lines},
};

type Targets = BTreeMap<PageNumber, BTreeSet<usize>>;

fn fixture_draft(
    pdf: &Path,
    quad: Quad,
    text: &str,
) -> Result<(OcrDraft, Targets), Box<dyn std::error::Error>> {
    fs::write(pdf, include_bytes!("fixtures/line-audit-rotations.pdf"))?;
    let pages = (1..=4)
        .map(|page| DraftPage {
            page,
            lines: vec![OcrLine {
                text: text.into(),
                confidence: 0.999,
                quad,
                reviewed: false,
            }],
            native_text_regions: vec![],
        })
        .collect();
    let draft = OcrDraft::new(sha256_file(pdf)?, "synthetic vector ink".into(), pages);
    let targets = (1..=4)
        .map(|number| Ok((PageNumber::new(number)?, BTreeSet::from([0]))))
        .collect::<Result<Targets, pdf_redox_ocr::Error>>()?;
    Ok((draft, targets))
}

#[test]
fn source_ink_topology_and_pdf_mapping_survive_all_page_rotations()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let pdf = directory.path().join("source.pdf");
    let quad = [
        Point(50.0, 65.0),
        Point(125.0, 65.0),
        Point(125.0, 91.0),
        Point(50.0, 91.0),
    ];
    let (draft, targets) = fixture_draft(&pdf, quad, "08i")?;
    let report = audit_lines(&pdf, &draft, &targets, &LineAuditPolicy::default())?;
    assert_eq!(report.lines.len(), 4);
    for line in report.lines {
        assert_eq!(line.original_quad, quad);
        assert_eq!(line.groups.len(), 3, "page {}", line.page);
        assert_eq!(line.components.len(), 4);
        assert_eq!(line.supported_runs.len(), 3);
        let mut band_components = line
            .supported_bands
            .iter()
            .flat_map(|band| band.column_runs.iter())
            .flat_map(|run| run.run.component_indices.iter().copied())
            .collect::<Vec<_>>();
        band_components.sort_unstable();
        assert_eq!(band_components, vec![0, 1, 2, 3]);
        for band in &line.supported_bands {
            let mut expected = band.band.run.component_indices.clone();
            expected.sort_unstable();
            let mut actual = band
                .column_runs
                .iter()
                .flat_map(|run| run.run.component_indices.iter().copied())
                .collect::<Vec<_>>();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
        assert_eq!(line.groups[2].group.run.component_indices.len(), 2);
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
    Ok(())
}

#[test]
fn source_padding_recovers_clipped_stroke_across_rotations()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let pdf = directory.path().join("source.pdf");
    // Crop the first ring through its top border. A sufficiently padded
    // audit restores the real stroke, rather than drawing a closing edge.
    let clipped_quad = [
        Point(50.0, 65.0),
        Point(70.0, 65.0),
        Point(70.0, 82.5),
        Point(50.0, 82.5),
    ];
    let (mut draft, targets) = fixture_draft(&pdf, clipped_quad, "0")?;
    let unpadded = audit_lines(&pdf, &draft, &targets, &LineAuditPolicy::default())?;
    let padded = audit_lines(
        &pdf,
        &draft,
        &targets,
        &LineAuditPolicy {
            padding_pixels: 36,
            ..LineAuditPolicy::default()
        },
    )?;
    for (before, after) in unpadded.lines.iter().zip(&padded.lines) {
        assert_eq!(before.groups.len(), 1);
        assert!(
            before.groups[0]
                .group
                .topology
                .iter()
                .all(|topology| topology.enclosed_background_regions.is_empty())
        );
        assert_eq!(after.groups.len(), 1);
        let group = &after.groups[0];
        assert!(
            group
                .group
                .topology
                .iter()
                .all(|topology| topology.enclosed_background_regions.len() == 1)
        );
        assert!(group.group.run.observation_support.inside_pixels > 0);
        assert!(group.group.run.observation_support.outside_pixels > 0);
        assert!(group.group.run.observation_support.touches_observation_edge);
        assert!(!group.group.touches_line_edge);
        assert_eq!(before.polarity.foreground, after.polarity.foreground);
        assert_eq!(before.polarity.class_medians, after.polarity.class_medians);
        assert_eq!(before.original_quad, after.original_quad);
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
