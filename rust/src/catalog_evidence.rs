//! Match actual recognition-crop ink to exact reusable PDF outline components.
//!
//! Template identity is appearance evidence, not a Unicode label. Multiple
//! detections of one occurrence cannot inflate vote counts.

use pdf_redox::GlyphCatalog;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Error, PageNumber,
    character_alignment::RunAlignment,
    geometry::{Point, Quad, QuadBounds},
    polygon::{area, clip_convex},
    recognition::{RasterCharacterAudit, RasterCharacterMapping, RasterLocation},
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct OccurrenceId {
    pub template: String,
    pub page: PageNumber,
    pub ordinal: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CharacterVote {
    pub occurrence: OccurrenceId,
    pub raster_location: RasterLocation,
    pub raster_sha256: String,
    pub detection_index: usize,
    pub character_index: usize,
    pub character: char,
    /// Line-level prediction score: not a character probability.
    pub line_score: f64,
    pub crop_sha256: String,
    pub ink_quad: Quad,
    pub vector_quad: Quad,
    pub ink_coverage: f64,
    pub vector_coverage: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MatchStats {
    pub unique_anchors: usize,
    pub unmappable: usize,
    pub no_component: usize,
    pub partial_or_multiple_components: usize,
    pub duplicate_occurrences: usize,
    pub conflicting_occurrences: usize,
    pub associated_occurrences: usize,
}

#[derive(Debug, Serialize)]
pub struct TemplateEvidence {
    pub template: String,
    pub observations: usize,
    pub pages: usize,
    pub labels: BTreeMap<char, usize>,
    /// At least three agreeing observations on more than one page. Still
    /// only repeated predictions from a single source model.
    pub repeated_label: Option<char>,
}

#[derive(Debug, Serialize)]
pub struct CatalogEvidence {
    pub pdf_sha256: String,
    pub catalog_templates: usize,
    pub catalog_occurrences: usize,
    pub stats: MatchStats,
    pub votes: Vec<CharacterVote>,
    pub templates: Vec<TemplateEvidence>,
}

struct Placement<'a> {
    template: &'a str,
    ordinal: usize,
    quad: Quad,
    bounds: QuadBounds,
    area: f64,
}

struct CatalogIndex<'a> {
    pages: BTreeMap<PageNumber, Vec<Placement<'a>>>,
    occurrences: usize,
}

impl<'a> CatalogIndex<'a> {
    fn new(catalog: &'a GlyphCatalog, digest: &str) -> Result<Self, Error> {
        if catalog.schema_version != 2 || catalog.optimized_pdf_sha256 != digest {
            return Err(Error::Invalid(
                "glyph catalog does not describe the exact OCR input PDF".into(),
            ));
        }
        let mut pages = BTreeMap::<PageNumber, Vec<Placement>>::new();
        let mut template_ids = BTreeSet::new();
        let mut occurrences = 0;
        for template in &catalog.glyphs {
            if !template_ids.insert(template.id.as_str()) {
                return Err(Error::Invalid("duplicate catalog template identity".into()));
            }
            let mut identities = BTreeSet::new();
            for placement in &template.occurrences {
                let page = PageNumber::new(
                    u32::try_from(placement.page)
                        .map_err(|_| Error::Invalid("catalog page overflow".into()))?,
                )?;
                if !identities.insert((page, placement.index_on_page)) {
                    return Err(Error::Invalid("duplicate catalog occurrence".into()));
                }
                let quad = placement.page_quad.map(|[x, y]| Point(x, y));
                let measure = area(&quad);
                if !measure.is_finite()
                    || measure <= 1e-9
                    || quad.iter().any(|p| !p.0.is_finite() || !p.1.is_finite())
                {
                    return Err(Error::Invalid("invalid vector glyph geometry".into()));
                }
                let bounds = QuadBounds::new(&quad);
                pages.entry(page).or_default().push(Placement {
                    template: &template.id,
                    ordinal: placement.index_on_page,
                    quad,
                    bounds,
                    area: measure,
                });
                occurrences += 1;
            }
        }
        for page in pages.values_mut() {
            page.sort_by(|a, b| a.bounds.min_x.total_cmp(&b.bounds.min_x));
        }
        Ok(Self { pages, occurrences })
    }
}

struct Match<'a> {
    placement: &'a Placement<'a>,
    ink_coverage: f64,
    vector_coverage: f64,
}

enum Association<'a> {
    None,
    PartialOrMultiple,
    Unique(Match<'a>),
}

fn associate<'a>(ink: &Quad, placements: &'a [Placement<'a>]) -> Association<'a> {
    let bounds = QuadBounds::new(ink);
    let ink_area = area(ink);
    if ink_area <= 1e-9 || !ink_area.is_finite() {
        return Association::None;
    }
    let end = placements.partition_point(|p| p.bounds.min_x < bounds.max_x);
    let mut match_ = None;
    let mut significant = 0;
    for p in &placements[..end] {
        if !bounds.intersects(p.bounds) {
            continue;
        }
        let intersection = area(&clip_convex(ink, &p.quad));
        let ink_coverage = intersection / ink_area;
        let vector_coverage = intersection / p.area;
        // A large unrelated rule may cross the glyph, or a tiny detached
        // mark may be just a fragment. Neither gives full-component evidence.
        if ink_coverage < 0.12 || vector_coverage < 0.12 {
            continue;
        }
        significant += 1;
        if ink_coverage >= 0.55 && vector_coverage >= 0.70 {
            match_ = Some(Match {
                placement: p,
                ink_coverage,
                vector_coverage,
            });
        }
    }
    match (significant, match_) {
        (0, _) => Association::None,
        (1, Some(m)) => Association::Unique(m),
        _ => Association::PartialOrMultiple,
    }
}

fn collect_votes(
    mapping: &RasterCharacterMapping,
    audit: &RasterCharacterAudit,
    placements: &[Placement<'_>],
    stats: &mut MatchStats,
    by_occurrence: &mut BTreeMap<OccurrenceId, Vec<CharacterVote>>,
) {
    for line in &audit.report.lines {
        if !mapping.includes_detection(line.detection_index) {
            continue;
        }
        for (character_index, anchor) in line.anchors.iter().enumerate() {
            let RunAlignment::Unique { run } = anchor.alignment else {
                continue;
            };
            stats.unique_anchors += 1;
            let Some(ink) = line
                .geometry
                .and_then(|geometry| geometry.ink_quad(line.runs[run].run.ink))
            else {
                stats.unmappable += 1;
                continue;
            };
            let ink = mapping.to_pdf(ink);
            let matched = match associate(&ink, placements) {
                Association::None => {
                    stats.no_component += 1;
                    continue;
                }
                Association::PartialOrMultiple => {
                    stats.partial_or_multiple_components += 1;
                    continue;
                }
                Association::Unique(matched) => matched,
            };
            let occurrence = OccurrenceId {
                template: matched.placement.template.to_owned(),
                page: mapping.page,
                ordinal: matched.placement.ordinal,
            };
            let vote = CharacterVote {
                occurrence: occurrence.clone(),
                raster_location: mapping.scope.location(),
                raster_sha256: mapping.rgb_sha256.clone(),
                detection_index: line.detection_index,
                character_index,
                character: anchor.character,
                line_score: line.original.confidence,
                crop_sha256: line.crop_sha256.clone(),
                ink_quad: ink,
                vector_quad: matched.placement.quad,
                ink_coverage: matched.ink_coverage,
                vector_coverage: matched.vector_coverage,
            };
            by_occurrence.entry(occurrence).or_default().push(vote);
        }
    }
}

fn summarize(votes: &[CharacterVote], min_score: f64) -> Vec<TemplateEvidence> {
    let mut by_template = BTreeMap::<&str, Vec<&CharacterVote>>::new();
    for vote in votes {
        if vote.line_score >= min_score {
            by_template
                .entry(&vote.occurrence.template)
                .or_default()
                .push(vote);
        }
    }
    by_template
        .into_iter()
        .map(|(template, group)| {
            let mut labels = BTreeMap::new();
            let mut pages = BTreeSet::new();
            for vote in &group {
                *labels.entry(vote.character).or_insert(0) += 1;
                pages.insert(vote.occurrence.page);
            }
            let repeated_label = (labels.len() == 1 && group.len() >= 3 && pages.len() >= 2)
                .then(|| labels.keys().next().copied())
                .flatten();
            TemplateEvidence {
                template: template.into(),
                observations: group.len(),
                pages: pages.len(),
                labels,
                repeated_label,
            }
        })
        .collect()
}

impl CatalogEvidence {
    /// Build source-bound evidence from the accepted page and regional routes.
    /// Cached rasters may map to multiple pages; each physical vector
    /// occurrence contributes at most one vote. Rejected residual attempts
    /// and non-page source rasters do not participate. Regional mappings
    /// contribute only admitted detector identities.
    ///
    /// # Errors
    /// Rejects mismatched catalog provenance or malformed placements.
    pub fn from_audits(
        catalog: &GlyphCatalog,
        pdf_sha256: &str,
        audits: &[RasterCharacterAudit],
        mappings: &[RasterCharacterMapping],
        residual_pages: &BTreeSet<PageNumber>,
        minimum_score: f64,
    ) -> Result<Self, Error> {
        let index = CatalogIndex::new(catalog, pdf_sha256)?;
        let mut stats = MatchStats::default();
        let mut by_occurrence = BTreeMap::<OccurrenceId, Vec<CharacterVote>>::new();
        for mapping in mappings {
            // Native-image ink is not evidence of an overlapping vector outline.
            // Mosaic routes are certified image-only.
            if !matches!(
                mapping.scope,
                crate::recognition::RasterMappingScope::WholePage { .. }
                    | crate::recognition::RasterMappingScope::Region { .. }
            ) || !mapping.selected(residual_pages)
            {
                continue;
            }
            let Some(placements) = index.pages.get(&mapping.page) else {
                continue;
            };
            let Some(audit) = mapping.unique_audit(audits) else {
                continue;
            };
            collect_votes(mapping, audit, placements, &mut stats, &mut by_occurrence);
        }
        let mut votes = Vec::new();
        for (_, mut group) in by_occurrence {
            if group.len() > 1 {
                stats.duplicate_occurrences += 1;
                if group.iter().any(|v| v.character != group[0].character) {
                    stats.conflicting_occurrences += 1;
                    continue;
                }
            }
            // Retain one actual observation, not a synthetic confidence.
            group.sort_by(|a, b| b.line_score.total_cmp(&a.line_score));
            if let Some(vote) = group.into_iter().next() {
                votes.push(vote);
            }
        }
        stats.associated_occurrences = votes.len();
        let templates = summarize(&votes, minimum_score);
        Ok(Self {
            pdf_sha256: pdf_sha256.into(),
            catalog_templates: catalog.glyphs.len(),
            catalog_occurrences: index.occurrences,
            stats,
            votes,
            templates,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rectangle(x: f64, y: f64, w: f64, h: f64) -> Quad {
        [
            Point(x, y),
            Point(x + w, y),
            Point(x + w, y + h),
            Point(x, y + h),
        ]
    }
    fn placement(quad: Quad) -> Placement<'static> {
        Placement {
            template: "a",
            ordinal: 0,
            quad,
            bounds: QuadBounds::new(&quad),
            area: area(&quad),
        }
    }
    #[test]
    fn reciprocal_coverage_rejects_parts_and_multiple_components() {
        let ink = rectangle(10., 10., 10., 12.);
        assert!(matches!(
            associate(&ink, &[placement(rectangle(10.5, 10.5, 9., 11.))]),
            Association::Unique(_)
        ));
        assert!(matches!(
            associate(&ink, &[placement(rectangle(0., 0., 100., 100.))]),
            Association::None
        ));
        assert!(matches!(
            associate(
                &ink,
                &[
                    placement(rectangle(10., 10., 5., 12.)),
                    placement(rectangle(15., 10., 5., 12.))
                ]
            ),
            Association::PartialOrMultiple
        ));
    }
    #[test]
    fn catalog_must_be_bound_to_actual_input() {
        let catalog = GlyphCatalog {
            schema_version: 2,
            optimized_pdf_sha256: "actual".into(),
            glyphs: vec![],
        };
        assert!(CatalogIndex::new(&catalog, "wrong").is_err());
        assert!(CatalogIndex::new(&catalog, "actual").is_ok());
    }
}
