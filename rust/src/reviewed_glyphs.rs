//! Independently labeled appearance samples in exact source PDF coordinates.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Error, PageNumber,
    character_alignment::{CharacterLineAudit, RunAlignment},
    draft::OcrDraft,
    geometry::Quad,
    glyph_appearance::GlyphShape,
    quality_eval::{GroundTruth, contains, validate},
    recognition::{RasterCharacterAudit, RasterCharacterMapping, RasterMappingScope},
};

/// One independently transcribed character and the primary model's actual
/// source-ink association. The reviewed Unicode is not an OCR vote.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewedGlyph {
    pub page: u32,
    pub crop_sha256: String,
    pub detection_index: usize,
    pub character_index: usize,
    pub label: char,
    pub prediction: char,
    pub line_score: f64,
    pub ink_quad: Quad,
    pub shape: GlyphShape,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ReviewStats {
    pub references: usize,
    pub matched_lines: usize,
    pub missing_or_ambiguous_lines: usize,
    pub unaligned_characters: usize,
    pub missing_or_clipped_ink: usize,
    pub labeled_glyphs: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReviewedGlyphBank {
    pub schema_version: u32,
    pub pdf_sha256: String,
    pub model: String,
    pub reference_provenance: String,
    pub stats: ReviewStats,
    pub glyphs: Vec<ReviewedGlyph>,
}

/// Label a prediction only when every minimum-cost edit path pairs that
/// prediction with the same reviewed scalar (and never with an insertion).
/// Whitespace handling is explicit at the caller; there is no case folding
/// or compatibility normalization. Bound quadratic storage independently of
/// the line length so malformed metadata cannot allocate without limit.
fn stable_labels(reference: &[char], prediction: &[char]) -> Option<Vec<Option<char>>> {
    let width = prediction.len().checked_add(1)?;
    let height = reference.len().checked_add(1)?;
    let cells = width.checked_mul(height)?;
    if cells > 1_000_000 {
        return None;
    }
    let mut forward = vec![0; cells];
    let mut backward = vec![0; cells];
    let at = |row: usize, col: usize| row * width + col;
    for row in 0..height {
        for col in 0..width {
            forward[at(row, col)] = match (row, col) {
                (0, _) => col,
                (_, 0) => row,
                _ => (forward[at(row - 1, col)] + 1)
                    .min(forward[at(row, col - 1)] + 1)
                    .min(
                        forward[at(row - 1, col - 1)]
                            + usize::from(reference[row - 1] != prediction[col - 1]),
                    ),
            };
        }
    }
    for row in (0..height).rev() {
        for col in (0..width).rev() {
            backward[at(row, col)] = if row == height - 1 {
                width - 1 - col
            } else if col == width - 1 {
                height - 1 - row
            } else {
                (backward[at(row + 1, col)] + 1)
                    .min(backward[at(row, col + 1)] + 1)
                    .min(
                        backward[at(row + 1, col + 1)]
                            + usize::from(reference[row] != prediction[col]),
                    )
            };
        }
    }
    let optimal = forward[cells - 1];
    Some(
        prediction
            .iter()
            .enumerate()
            .map(|(col, &observed)| {
                let mut label = None;
                for row in 0..height {
                    let prefix = forward[at(row, col)];
                    if prefix + 1 + backward[at(row, col + 1)] == optimal {
                        return None;
                    }
                    if row < reference.len()
                        && prefix
                            + usize::from(reference[row] != observed)
                            + backward[at(row + 1, col + 1)]
                            == optimal
                    {
                        let expected = reference[row];
                        if label.is_some_and(|previous| previous != expected) {
                            return None;
                        }
                        label = Some(expected);
                    }
                }
                label
            })
            .collect(),
    )
}

struct MappedLine<'a> {
    mapping: &'a RasterCharacterMapping,
    line: &'a CharacterLineAudit,
    quad: Quad,
}

fn collect_lines<'a>(
    audits: &'a [RasterCharacterAudit],
    mappings: &'a [RasterCharacterMapping],
    residual_pages: &BTreeSet<PageNumber>,
) -> BTreeMap<u32, Vec<MappedLine<'a>>> {
    let mut pages = BTreeMap::<u32, Vec<MappedLine>>::new();
    for mapping in mappings {
        if !mapping.selected(residual_pages) {
            continue;
        }
        let Some(audit) = mapping.unique_audit(audits) else {
            continue;
        };
        let page = pages.entry(mapping.page.get()).or_default();
        for line in &audit.report.lines {
            if !mapping.includes_detection(line.detection_index) {
                continue;
            }
            if page.iter().any(|entry| {
                entry.mapping.rgb_sha256 == mapping.rgb_sha256
                    && entry.mapping.scope == mapping.scope
                    && entry.line.detection_index == line.detection_index
            }) {
                continue;
            }
            page.push(MappedLine {
                mapping,
                line,
                quad: mapping.to_pdf(line.original.quad),
            });
        }
    }
    pages
}

fn collect_glyphs(
    mapped: &MappedLine<'_>,
    reference: &str,
    stats: &mut ReviewStats,
    glyphs: &mut Vec<ReviewedGlyph>,
) {
    let line = mapped.line;
    let expected = reference
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<Vec<_>>();
    let anchors = line
        .anchors
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.character.is_whitespace())
        .collect::<Vec<_>>();
    let predicted = anchors.iter().map(|(_, a)| a.character).collect::<Vec<_>>();
    let Some(labels) = stable_labels(&expected, &predicted) else {
        stats.unaligned_characters += anchors.len();
        return;
    };
    for ((character_index, anchor), label) in anchors.into_iter().zip(labels) {
        let Some(label) = label else {
            stats.unaligned_characters += 1;
            continue;
        };
        let RunAlignment::Unique { run } = anchor.alignment else {
            stats.missing_or_clipped_ink += 1;
            continue;
        };
        let Some(shape) = line.run_appearances.get(run).and_then(Option::as_ref) else {
            stats.missing_or_clipped_ink += 1;
            continue;
        };
        let Some(ink_quad) = line
            .geometry
            .and_then(|g| g.ink_quad(line.runs[run].run.ink))
        else {
            stats.missing_or_clipped_ink += 1;
            continue;
        };
        glyphs.push(ReviewedGlyph {
            page: mapped.mapping.page.get(),
            crop_sha256: line.crop_sha256.clone(),
            detection_index: line.detection_index,
            character_index,
            label,
            prediction: anchor.character,
            line_score: line.original.confidence,
            ink_quad: mapped.mapping.to_pdf(ink_quad),
            shape: shape.clone(),
        });
    }
}

impl ReviewedGlyphBank {
    /// Collect glyphs from unique raw primary line occurrences. A reviewed
    /// anchor falling in multiple observations is deliberately excluded.
    ///
    /// # Errors
    /// Rejects reference/draft source mismatch or invalid reference geometry.
    pub fn from_audits(
        reference: &GroundTruth,
        draft: &OcrDraft,
        audits: &[RasterCharacterAudit],
        mappings: &[RasterCharacterMapping],
        residual_pages: &BTreeSet<PageNumber>,
    ) -> Result<Self, Error> {
        validate(reference, draft)?;
        let pages = collect_lines(audits, mappings, residual_pages);
        let mut stats = ReviewStats {
            references: reference.samples.len(),
            ..Default::default()
        };
        let mut glyphs = Vec::new();
        let mut matched =
            BTreeMap::<(u32, &str, &RasterMappingScope, usize), Vec<(&MappedLine<'_>, &str)>>::new(
            );
        for sample in &reference.samples {
            let mut candidates = pages
                .get(&sample.page)
                .into_iter()
                .flatten()
                .filter(|line| contains(&line.quad, sample.center));
            let Some(line) = candidates.next() else {
                stats.missing_or_ambiguous_lines += 1;
                continue;
            };
            if candidates.next().is_some() {
                stats.missing_or_ambiguous_lines += 1;
                continue;
            }
            matched
                .entry((
                    sample.page,
                    &line.mapping.rgb_sha256,
                    &line.mapping.scope,
                    line.line.detection_index,
                ))
                .or_default()
                .push((line, &sample.text));
        }
        for group in matched.values() {
            if let [(line, text)] = group.as_slice() {
                stats.matched_lines += 1;
                collect_glyphs(line, text, &mut stats, &mut glyphs);
            } else {
                stats.missing_or_ambiguous_lines += group.len();
            }
        }
        stats.labeled_glyphs = glyphs.len();
        Ok(Self {
            schema_version: 1,
            pdf_sha256: draft.pdf_sha256.clone(),
            model: draft.model.clone(),
            reference_provenance: reference.provenance.clone(),
            stats,
            glyphs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn labels(reference: &str, prediction: &str) -> Option<Vec<Option<char>>> {
        stable_labels(
            &reference.chars().collect::<Vec<_>>(),
            &prediction.chars().collect::<Vec<_>>(),
        )
    }
    #[test]
    fn unique_substitutions_and_ambiguous_insertions_are_distinct() {
        assert_eq!(
            labels("Ø20.0", "020.0"),
            Some(vec![Some('Ø'), Some('2'), Some('0'), Some('.'), Some('0')])
        );
        assert_eq!(labels("AB", "AAB"), Some(vec![None, None, Some('B')]));
        assert_eq!(labels("ABA", "BA"), Some(vec![Some('B'), Some('A')]));
        assert_eq!(labels("AB", "BA"), Some(vec![None, None]));
        assert_eq!(labels("", "X"), Some(vec![None]));
    }
}
