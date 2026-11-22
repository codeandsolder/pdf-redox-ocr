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
            assert!(group.appearance.is_some());
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
        assert!(
            group.appearance.is_none(),
            "outside-observation ink cannot supply a matched appearance"
        );
        assert!(before.groups[0].appearance.is_none());
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

mod support;

#[test]
fn periodic_source_leader_mapping_preserves_body_ink_across_rotations()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let pdf = dir.path().join("leaders.pdf");
    // The body has three text-like strokes and a separate internal dot.
    // Ten compact vector dots form a leader extending beyond the original
    // recognition's right edge. All four pages paint the same source geometry.
    let content = b"0 g 42 54 7 12 re f 54 54 7 12 re f 64 54 2 2 re f 68 54 6 12 re f 84 54 2 2 re f 88 54 2 2 re f 92 54 2 2 re f 96 54 2 2 re f 100 54 2 2 re f 104 54 2 2 re f 108 54 2 2 re f 112 54 2 2 re f 116 54 2 2 re f 120 54 2 2 re f";
    fs::write(
        &pdf,
        support::pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R 6 0 R] /Count 4 /MediaBox [0 0 180 120] >>"
                .to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Contents 7 0 R >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Contents 7 0 R /Rotate 90 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Contents 7 0 R /Rotate 180 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Contents 7 0 R /Rotate 270 >>".to_vec(),
            support::stream("", content),
        ]),
    )?;
    let quad = [
        Point(36.0, 50.0),
        Point(95.0, 50.0),
        Point(95.0, 70.0),
        Point(36.0, 70.0),
    ];
    let draft = OcrDraft::new(
        sha256_file(&pdf)?,
        "leader fixture".into(),
        (1..=4)
            .map(|page| DraftPage {
                page,
                native_text_regions: vec![],
                lines: vec![OcrLine {
                    text: "A.B...".into(),
                    confidence: 0.99,
                    quad,
                    reviewed: false,
                }],
            })
            .collect(),
    );
    let targets = (1..=4)
        .map(|page| Ok((PageNumber::new(page)?, BTreeSet::from([0]))))
        .collect::<Result<Targets, pdf_redox_ocr::Error>>()?;
    let report = audit_lines(
        &pdf,
        &draft,
        &targets,
        &LineAuditPolicy {
            padding_pixels: 160,
            ..LineAuditPolicy::default()
        },
    )?;
    for line in report.lines {
        assert_eq!(
            line.leaders.len(),
            1,
            "page {} {:#?}",
            line.page,
            line.leaders
        );
        let leader = &line.leaders[0];
        assert_eq!(
            leader.evidence.edge,
            pdf_redox_ocr::layout_leaders::LeaderEdge::Right
        );
        assert_eq!(leader.evidence.inside_dots, 3);
        assert!(leader.evidence.outside_dots >= 2);
        assert!((leader.body_quad[1].0 - 79.0).abs() < 0.5);
        assert!((leader.body_quad[2].0 - 79.0).abs() < 0.5);
        assert_eq!(line.original_quad, quad);
        assert_eq!(line.original_text, "A.B...");
    }
    Ok(())
}

#[test]
fn source_edge_extensions_report_competing_observations_across_rotations()
-> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox_ocr::{layout_leaders::LeaderEdge, line_audit::BoundaryPeer};

    let directory = tempfile::tempdir()?;
    let pdf = directory.path().join("source.pdf");
    let quad = [
        Point(50.0, 65.0),
        Point(70.0, 65.0),
        Point(70.0, 91.0),
        Point(50.0, 91.0),
    ];
    let peer_quad = [
        Point(73.0, 65.0),
        Point(94.0, 65.0),
        Point(94.0, 91.0),
        Point(73.0, 91.0),
    ];
    let (mut draft, targets) = fixture_draft(&pdf, quad, "0")?;
    for page in &mut draft.pages {
        page.lines.push(OcrLine {
            text: "8".into(),
            confidence: 0.99,
            quad: peer_quad,
            reviewed: false,
        });
        page.native_text_regions.extend([peer_quad, quad]);
    }
    let report = audit_lines(
        &pdf,
        &draft,
        &targets,
        &LineAuditPolicy {
            padding_pixels: 240,
            ..LineAuditPolicy::default()
        },
    )?;
    assert_eq!(report.schema_version, 8);
    for line in report.lines {
        let extension = line
            .boundary_extensions
            .iter()
            .find(|e| {
                e.evidence.edge == LeaderEdge::Right
                    && e.overlaps
                        .iter()
                        .any(|o| matches!(o.peer, BoundaryPeer::OcrLine { line_index: 1 }))
            })
            .ok_or("missing adjacent glyph proposal")?;
        assert_eq!(extension.overlaps.len(), 3);
        let pdf_redox_ocr::line_audit::InkAccounting::Measured {
            peers_fully_sampled,
            peers,
            peer_union,
            sample,
            ..
        } = &extension.ink_accounting
        else {
            return Err("expected measured ink accounting".into());
        };
        assert_eq!(peers_fully_sampled, &[true, true, true]);
        assert_eq!(peers[&0], peers[&1]);
        assert_eq!(peer_union.original, sample.original);
        assert_eq!(peer_union.added, peers[&0].added);
        assert_eq!(peer_union.outside_expansion, peers[&0].outside_expansion);
        assert_eq!(peers[&2].original, sample.original);
        assert_eq!(peers[&2].added, 0);
        assert!(peer_union.added > 0);

        assert!(
            extension
                .overlaps
                .iter()
                .any(|o| matches!(o.peer, BoundaryPeer::NativeText { region_index: 0 }))
        );
        for overlap in &extension.overlaps[..2] {
            assert!(overlap.strip_fraction > 0.0 && overlap.strip_fraction <= 1.0);
            assert!(overlap.peer_fraction > 0.0 && overlap.peer_fraction <= 1.0);
            assert_eq!(
                overlap.component_indices,
                extension.evidence.component_indices
            );
        }
        let original_only = &extension.overlaps[2];
        assert!(matches!(
            original_only.peer,
            BoundaryPeer::NativeText { region_index: 1 }
        ));
        assert!(original_only.strip_fraction.abs() < 1e-9);
        assert!((original_only.expanded_peer_fraction - 1.0).abs() < 1e-9);
        assert!(original_only.component_indices.is_empty());
        assert!((extension.added_strip[0].0 - 70.0).abs() < 1e-8);
        assert!(extension.proposed_quad[1].0 > 85.0);
        assert_eq!(line.original_quad, quad);
    }
    Ok(())
}
