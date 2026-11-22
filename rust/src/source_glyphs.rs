//! Independently sample the original visible PDF at higher resolution.
//!
//! Preserve the association to actual primary detector ink without assigning
//! characters from position alone or splitting ambiguous joined components.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use crate::{
    Error, PageNumber,
    character_alignment::{CharacterLineAudit, RunAlignment},
    draft::{OcrLine, sha256_file},
    geometry::{Quad, QuadBounds},
    line_audit::{AuditedLine, LineAuditPolicy, audit_line},
    pdf_metadata::PdfMetadata,
    polygon::{area, clip_convex},
    recognition::{RasterCharacterAudit, RasterCharacterMapping, RasterLocation},
    reviewed_glyphs::{
        ReviewedGlyphBank, SourceGlyphAppearance, SourceSampling, SourceSamplingFailure,
    },
};

type LineKey = (u32, RasterLocation, usize);

/// Find mutually unique, substantially overlapping ink groups across two
/// independently segmented rasters in the same source coordinate space.
/// Partial, split and merged groups cannot transfer a glyph association.
fn corresponding_groups(primary: &[Option<Quad>], source: &[Quad]) -> BTreeMap<usize, usize> {
    let mut forward = vec![Vec::new(); primary.len()];
    let mut reverse = vec![0_usize; source.len()];
    for (i, ink) in primary.iter().enumerate() {
        let Some(ink) = ink else {
            continue;
        };
        let primary_area = area(ink);
        if !primary_area.is_finite() || primary_area <= 1e-9 {
            continue;
        }
        let primary_bounds = QuadBounds::new(ink);
        for (j, group) in source.iter().enumerate() {
            if !primary_bounds.intersects(QuadBounds::new(group)) {
                continue;
            }
            let source_area = area(group);
            if !source_area.is_finite() || source_area <= 1e-9 {
                continue;
            }
            let intersection = area(&clip_convex(ink, group));
            let (a, b) = (intersection / primary_area, intersection / source_area);
            // A small isolated mark may belong to a much larger primary
            // run. Count substantial coverage of EITHER group as a
            // competing association, even when the areas differ greatly.
            if a >= 0.12 || b >= 0.12 {
                forward[i].push((j, a >= 0.55 && b >= 0.70));
                reverse[j] += 1;
            }
        }
    }
    forward
        .into_iter()
        .enumerate()
        .filter_map(|(i, candidates)| {
            if let [(j, true)] = candidates.as_slice() {
                (reverse[*j] == 1).then_some((i, *j))
            } else {
                None
            }
        })
        .collect()
}

struct SampleContext<'a> {
    pdf: &'a Path,
    metadata: &'a PdfMetadata,
    policy: LineAuditPolicy,
}

impl SampleContext<'_> {
    fn source_line(
        &self,
        mapping: &RasterCharacterMapping,
        line: &CharacterLineAudit,
    ) -> Result<AuditedLine, Error> {
        let geometry = line
            .geometry
            .ok_or_else(|| Error::Invalid("primary crop geometry unavailable".into()))?;
        let order = geometry
            .reading_order(line.original.quad)
            .ok_or_else(|| Error::Invalid("primary reading frame unavailable".into()))?;
        let original = OcrLine {
            text: line.original.text.clone(),
            confidence: line.original.confidence,
            quad: mapping.to_pdf(order.map(|idx| line.original.quad[idx])),
            reviewed: false,
        };
        let sampled = audit_line(
            self.pdf,
            self.metadata,
            mapping.page,
            line.detection_index,
            &original,
            &self.policy,
        )?
        .audit;
        Ok(sampled)
    }
}

fn match_line(
    context: &SampleContext<'_>,
    bank: &mut ReviewedGlyphBank,
    indices: &[usize],
    mapping: &RasterCharacterMapping,
    line: &CharacterLineAudit,
) -> Result<usize, Error> {
    let source = context.source_line(mapping, line)?;
    let primary = line
        .runs
        .iter()
        .map(|run| {
            line.geometry
                .and_then(|g| g.ink_quad(run.run.ink))
                .map(|quad| mapping.to_pdf(quad))
        })
        .collect::<Vec<_>>();
    let quads = source.groups.iter().map(|g| g.ink_quad).collect::<Vec<_>>();
    let correspondence = corresponding_groups(&primary, &quads);
    let mut updates = Vec::new();
    for &index in indices {
        let glyph = &bank.glyphs[index];
        let anchor = line
            .anchors
            .get(glyph.character_index)
            .ok_or_else(|| Error::Invalid("reviewed character not in primary detection".into()))?;
        let RunAlignment::Unique { run } = anchor.alignment else {
            return Err(Error::Invalid(
                "reviewed character lacks unique primary ink".into(),
            ));
        };
        if (
            glyph.crop_sha256.as_str(),
            glyph.prediction,
            glyph.raster_sha256.as_str(),
        ) != (
            line.crop_sha256.as_str(),
            anchor.character,
            mapping.rgb_sha256.as_str(),
        ) || primary.get(run).copied().flatten() != Some(glyph.ink_quad)
        {
            return Err(Error::Invalid(
                "reviewed glyph does not belong to its primary detection".into(),
            ));
        }
        let Some(&group_index) = correspondence.get(&run) else {
            continue;
        };
        let group = &source.groups[group_index];
        let Some(shape) = group.appearance.clone() else {
            continue;
        };
        updates.push((
            index,
            SourceGlyphAppearance {
                normalized_line_sha256: source.normalized_sha256.clone(),
                line_dimensions: (source.width, source.height),
                source_group_index: group_index,
                ink_quad: group.ink_quad,
                shape,
            },
        ));
    }
    let count = updates.len();
    for (index, appearance) in updates {
        bank.glyphs[index].source_appearance = Some(appearance);
    }
    Ok(count)
}

/// Attach measured high-detail source ink to an already reviewed glyph bank.
///
/// Every accepted match has a unique primary and source run; labels and
/// primary predictions stay unchanged. Source sampling failures are recorded
/// per line rather than silently substituting the recognition crop.
/// # Errors
/// Rejects PDF provenance mismatch or invalid source sampling policy.
pub fn resample(
    bank: &mut ReviewedGlyphBank,
    pdf: &Path,
    audits: &[RasterCharacterAudit],
    mappings: &[RasterCharacterMapping],
    residual: &BTreeSet<PageNumber>,
    dpi: u32,
) -> Result<(), Error> {
    let policy = LineAuditPolicy {
        dpi,
        ..LineAuditPolicy::default()
    };
    policy.validate()?;
    if sha256_file(pdf)? != bank.pdf_sha256 {
        return Err(Error::Invalid("source glyph PDF digest mismatch".into()));
    }
    let metadata = PdfMetadata::open(pdf)?;
    let context = SampleContext {
        pdf,
        metadata: &metadata,
        policy,
    };
    let mut lines = BTreeMap::<LineKey, Vec<usize>>::new();
    for (i, glyph) in bank.glyphs.iter_mut().enumerate() {
        glyph.source_appearance = None;
        lines
            .entry((glyph.page, glyph.raster_location, glyph.detection_index))
            .or_default()
            .push(i);
    }
    let mut sampling = SourceSampling {
        dpi,
        attempted_lines: lines.len(),
        sampled_lines: 0,
        matched_glyphs: 0,
        failures: Vec::new(),
    };
    for ((page, location, detection), indices) in lines {
        let raster_sha256 = bank.glyphs[indices[0]].raster_sha256.clone();
        let result = (|| {
            let mut candidates = mappings.iter().filter(|mapping| {
                mapping.page.get() == page
                    && mapping.scope.location() == location
                    && mapping.rgb_sha256 == raster_sha256
                    && mapping.selected(residual)
                    && mapping.includes_detection(detection)
            });
            let mapping = candidates
                .next()
                .filter(|_| candidates.next().is_none())
                .ok_or_else(|| Error::Invalid("nonunique source raster occurrence".into()))?;
            let audit = mapping
                .unique_audit(audits)
                .ok_or_else(|| Error::Invalid("nonunique character audit".into()))?;
            let line = audit
                .report
                .lines
                .iter()
                .find(|line| line.detection_index == detection)
                .ok_or_else(|| Error::Invalid("primary detection not found".into()))?;
            match_line(&context, bank, &indices, mapping, line)
        })();
        match result {
            Ok(count) => {
                sampling.sampled_lines += 1;
                sampling.matched_glyphs += count;
            }
            Err(error) => sampling.failures.push(SourceSamplingFailure {
                page,
                raster_location: location,
                detection_index: detection,
                reason: error.to_string(),
            }),
        }
    }
    if sha256_file(pdf)? != bank.pdf_sha256 {
        return Err(Error::Invalid(
            "PDF changed during source glyph sampling".into(),
        ));
    }
    bank.source_sampling = Some(sampling);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::corresponding_groups;
    use crate::geometry::{Point, Quad};
    use std::collections::BTreeMap;

    fn rect(x: f64, w: f64) -> Quad {
        [
            Point(x, 0.),
            Point(x + w, 0.),
            Point(x + w, 10.),
            Point(x, 10.),
        ]
    }

    #[test]
    fn independent_segmentation_rejects_split_merge_and_partial_matches() {
        let primary = [
            Some(rect(0., 10.)),
            Some(rect(20., 10.)),
            Some(rect(40., 10.)),
            Some(rect(52., 10.)),
            Some(rect(80., 10.)),
            Some(rect(100., 10.)),
        ];
        let source = [
            rect(0.5, 9.),
            rect(20., 4.),
            rect(25., 5.),
            rect(40., 22.),
            rect(87., 10.),
            rect(100., 10.),
        ];
        assert_eq!(
            corresponding_groups(&primary, &source),
            BTreeMap::from([(0, 0), (5, 5)])
        );
        // Even a thin detached stroke can invalidate the one-to-one transfer.
        assert!(
            corresponding_groups(&[Some(rect(0., 10.))], &[rect(0., 9.), rect(9.5, 0.5)])
                .is_empty()
        );
    }
}
