//! PDF-structure-aware Rust OCR with original vector page preservation.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use clap::Parser;
use pdf_redox::{OcrDocument, OcrLine as PdfOcrLine, OcrOverlayOptions, OcrPage};
use pdf_redox_ocr::{
    PageNumber,
    draft::{DraftPage, OcrDraft},
    hybrid::{MergeInput, merge_ocr},
    inference::OcrInference,
    mixed_visibility::{MixedVisibility, prove_mixed_source_text},
    mosaic::{MosaicBudget, MosaicPlan, contains_native_fragments, plan_raster_mosaics},
    mosaic_ocr::recognize_native_mosaics,
    native_images::{NativeImageBudget, native_source_images},
    native_words::native_words_for_pages,
    paint_scan::{PaintReport, enumerate_image_paints},
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    source_ocr::{
        SourceObservation, SourceOnlyResult, source_image_mapper, source_only_candidates,
        source_only_pages,
    },
    source_selection::{ImagePlacement, SourcePolicy, choose_mixed_raster_sources},
};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

/// Generate SHA-bound PDF OCR geometry and optionally apply invisible semantic text.
#[derive(Debug, Parser)]
#[command(name = "pdf-redox-ocr", version)]
struct Args {
    /// Exact source PDF, which is never overwritten.
    #[arg(long)]
    pdf: PathBuf,
    /// Write schema-version 3 OCR geometry as JSON.
    #[arg(long)]
    output: PathBuf,
    /// Write an optional PDF annotated through the Rust pdf-redox importer.
    #[arg(long)]
    output_pdf: Option<PathBuf>,
    /// One-based pages separated by commas or inclusive ranges, e.g. 1,3-5.
    #[arg(long)]
    pages: Option<String>,
    /// Longest rendered dimension in pixels.
    #[arg(long, default_value_t = 1600)]
    max_side: u32,
    /// Base rendering DPI before page size limiting.
    #[arg(long, default_value_t = 72)]
    dpi: u32,
    /// Minimum confidence for automatic invisible semantic text.
    #[arg(long, default_value_t = 0.97)]
    confidence: f64,
    /// Poppler page-extraction batch size.
    #[arg(long, default_value_t = 24)]
    native_chunk: usize,
    /// Maximum pixels allowed in a single decoded native-image OCR source.
    #[arg(long, default_value_t = 8_000_000)]
    max_native_source_pixels: u64,
    /// Maximum aggregate pixels held by distinct native source images.
    #[arg(long, default_value_t = 16_000_000)]
    max_native_total_pixels: u64,
    /// Maximum number of unique image sources recognized.
    #[arg(long, default_value_t = 12)]
    max_native_sources: usize,
    /// Largest native-resolution mosaic crop, in pixels.
    #[arg(long, default_value_t = 12_000_000)]
    max_mosaic_group_pixels: u64,
    /// Per-page aggregate native-resolution mosaic pixels.
    #[arg(long, default_value_t = 24_000_000)]
    max_mosaic_page_pixels: u64,
    /// Largest number of independent mosaic groups rendered on one page.
    #[arg(long, default_value_t = 24)]
    max_mosaic_groups: usize,
    /// Upper Poppler renderer DPI for native-resolution mosaic crops.
    #[arg(long, default_value_t = 2400)]
    max_mosaic_dpi: u32,
    /// Maximum aggregate pixels that one Poppler extraction may produce.
    #[arg(long, default_value_t = 64_000_000)]
    max_native_extraction_pixels: u64,
}

fn parse_page_selection(
    specification: Option<&str>,
    count: usize,
) -> Result<Vec<PageNumber>, Box<dyn Error>> {
    let count = u32::try_from(count)?;
    let Some(specification) = specification else {
        return (1..=count)
            .map(|index| PageNumber::new(index).map_err(Into::into))
            .collect();
    };
    let mut pages = BTreeSet::new();
    for part in specification.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("empty page in selection".into());
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start.trim().parse::<u32>()?;
            let end = end.trim().parse::<u32>()?;
            if start == 0 || start > end || end > count {
                return Err("invalid page range".into());
            }
            for number in start..=end {
                if !pages.insert(number) {
                    return Err("duplicate page selection".into());
                }
            }
        } else {
            let number = part.parse::<u32>()?;
            if number == 0 || number > count || !pages.insert(number) {
                return Err("invalid or duplicate selected page".into());
            }
        }
    }
    if pages.is_empty() {
        return Err("empty page selection".into());
    }
    pages
        .into_iter()
        .map(|number| PageNumber::new(number).map_err(Into::into))
        .collect()
}

fn normalized_existing(path: &Path) -> Result<Option<PathBuf>, std::io::Error> {
    match fs::canonicalize(path) {
        Ok(normalized) => Ok(Some(normalized)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn check_paths(input: &Path, paths: &[&Path]) -> Result<(), Box<dyn Error>> {
    let input = fs::canonicalize(input)?;
    let mut targets = BTreeSet::new();
    for output in paths {
        let parent = output
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let file_name = output.file_name().ok_or("output must name a file")?;
        let normalized = match normalized_existing(output)? {
            Some(existing) => existing,
            None => fs::canonicalize(parent)?.join(file_name),
        };
        if normalized == input {
            return Err("output path aliases the original source PDF".into());
        }
        if !targets.insert(normalized) {
            return Err("two output paths resolve to the same file".into());
        }
    }
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = NamedTempFile::new_in(parent)?;
    {
        let mut writer = BufWriter::new(&mut temporary);
        writer.write_all(bytes)?;
        writer.flush()?;
    }
    temporary.persist(path)?;
    Ok(())
}

fn importer_document(draft: &OcrDraft) -> Result<OcrDocument, Box<dyn Error>> {
    let mut pages = Vec::with_capacity(draft.pages.len());
    for page in &draft.pages {
        let lines = page
            .lines
            .iter()
            .map(|line| PdfOcrLine {
                text: line.text.clone(),
                confidence: line.confidence,
                quad: line.quad.map(|point| [point.0, point.1]),
                reviewed: line.reviewed,
            })
            .collect();
        pages.push(OcrPage {
            page: usize::try_from(page.page)?,
            lines,
            native_text_regions: page
                .native_text_regions
                .iter()
                .map(|quad| quad.map(|point| [point.0, point.1]))
                .collect(),
        });
    }
    Ok(OcrDocument {
        schema_version: draft.schema_version,
        pdf_sha256: draft.pdf_sha256.clone(),
        model: draft.model.clone(),
        pages,
    })
}

/// Distinct source paints proved independently before the final OCR merge.
struct NativePreparation {
    source_only: SourceOnlyResult,
    mixed: MixedVisibility,
}

#[derive(Clone, Copy)]
struct NativePolicies {
    source_budget: NativeImageBudget,
    mosaic_budget: MosaicBudget,
    render: RenderPolicy,
    minimum_confidence: f64,
}

fn plan_native_fragment_pages(
    scanned: &PaintReport,
    doc: &PdfMetadata,
    budget: MosaicBudget,
) -> Result<(BTreeSet<PageNumber>, BTreeMap<PageNumber, MosaicPlan>), Box<dyn Error>> {
    let mut fragmented = BTreeSet::new();
    let mut plans = BTreeMap::new();
    for summary in &scanned.pages {
        if !summary.image_only_after_native {
            continue;
        }
        let page = PageNumber::new(summary.page)?;
        let paints = scanned
            .placements
            .iter()
            .filter(|p| p.page == summary.page)
            .cloned()
            .collect::<Vec<_>>();
        let box_ = *doc
            .media_boxes()
            .get(&page)
            .ok_or("selected page missing MediaBox")?;
        let mapper = doc.mapper(page)?;
        if !contains_native_fragments(&paints, box_, mapper) {
            continue;
        }
        fragmented.insert(page);
        if summary.native_searchable_shows == 0
            && let Some(plan) = plan_raster_mosaics(page, &paints, box_, mapper, budget)?
        {
            plans.insert(page, plan);
        }
    }
    Ok((fragmented, plans))
}

fn native_source_ids(
    scanned: &PaintReport,
    doc: &PdfMetadata,
    fragmented: &BTreeSet<PageNumber>,
    source_budget: NativeImageBudget,
    render: RenderPolicy,
) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let mut source_ids = source_only_candidates(scanned, source_budget.max_source_pixels);
    let sampling = scanned
        .placements
        .iter()
        .map(|paint| ImagePlacement {
            page: paint.page,
            source_id: paint.source_id.clone(),
            width: paint.width,
            height: paint.height,
            ctm: paint.ctm,
            unsafe_reasons: paint
                .unsafe_reasons
                .iter()
                .map(ToString::to_string)
                .collect(),
        })
        .collect::<Vec<_>>();
    let (mixed, report) = choose_mixed_raster_sources(
        &sampling,
        doc.media_boxes(),
        SourcePolicy {
            max_side: render.max_side,
            max_inference_pixels: source_budget.max_total_pixels,
            max_sources: source_budget.max_sources,
            minimum_source_pixels: 20_000,
            min_gain: 2.25,
        },
    )?;
    eprintln!(
        "source candidates: {} high-detail sources, {} native-only sources",
        report.native_resolution_candidates,
        source_ids.len()
    );
    source_ids.extend(mixed);
    // Native source OCR is insufficient on a page where a text line can cross
    // neighboring fragments; recognize the complete native-pitch group instead.
    let eligible_elsewhere = scanned
        .placements
        .iter()
        .filter(|paint| !fragmented.iter().any(|page| page.get() == paint.page))
        .map(|paint| paint.source_id.as_str())
        .collect::<BTreeSet<_>>();
    source_ids.retain(|id| eligible_elsewhere.contains(id.as_str()));
    Ok(source_ids)
}

fn recognize_native_sources(
    engine: &OcrInference,
    pdf: &Path,
    selected: &[PageNumber],
    source_ids: &BTreeSet<String>,
    budget: NativeImageBudget,
) -> Result<BTreeMap<String, SourceObservation>, Box<dyn Error>> {
    let decoded = match native_source_images(pdf, selected, source_ids, budget) {
        Ok(images) => images,
        Err(error) => {
            eprintln!("Native source extraction unavailable: {error}; using page OCR");
            BTreeMap::new()
        }
    };
    let mut recognized = BTreeMap::new();
    for (source_id, rgb) in decoded {
        let (width, height) = (rgb.width(), rgb.height());
        let mapper = source_image_mapper(width, height)?;
        match engine.recognize_page(rgb, mapper) {
            Ok(lines) => {
                eprintln!("source {source_id}: {} OCR observations", lines.len());
                recognized.insert(
                    source_id,
                    SourceObservation {
                        width,
                        height,
                        lines,
                    },
                );
            }
            Err(error) => eprintln!("source {source_id}: {error}; using page OCR"),
        }
    }
    Ok(recognized)
}

fn add_reconstructed_mosaics(
    engine: &OcrInference,
    pdf: &Path,
    doc: &PdfMetadata,
    scanned: &PaintReport,
    plans: &BTreeMap<PageNumber, MosaicPlan>,
    budget: MosaicBudget,
    source_only: &mut SourceOnlyResult,
) -> Result<usize, Box<dyn Error>> {
    let mut count = 0;
    for (&page, plan) in plans {
        let paints = scanned
            .placements
            .iter()
            .filter(|paint| paint.page == page.get())
            .cloned()
            .collect::<Vec<_>>();
        match recognize_native_mosaics(engine, pdf, doc.mapper(page)?, plan, &paints, budget) {
            Ok(Some(lines)) => {
                eprintln!(
                    "page {} native mosaic: {} OCR lines from {} groups",
                    page.get(),
                    lines.len(),
                    plan.groups.len()
                );
                source_only.approved_pages.insert(page);
                source_only.pages.insert(page, lines);
                count += 1;
            }
            Ok(None) => eprintln!(
                "page {} native mosaic incomplete; using page OCR",
                page.get()
            ),
            Err(error) => eprintln!(
                "page {} native mosaic rejected: {error}; using page OCR",
                page.get()
            ),
        }
    }
    Ok(count)
}

/// Decode and recognize original source images, source-resolution fragment
/// mosaics, and verify exact mixed-page source paint visibility.
fn prepare_native_sources(
    engine: &OcrInference,
    pdf: &Path,
    doc: &PdfMetadata,
    selected: &[PageNumber],
    policies: NativePolicies,
) -> Result<NativePreparation, Box<dyn Error>> {
    let NativePolicies {
        source_budget,
        mosaic_budget,
        render,
        minimum_confidence,
    } = policies;
    mosaic_budget.validate()?;
    let scanned = enumerate_image_paints(doc, selected)?;
    let (fragmented, plans) = plan_native_fragment_pages(&scanned, doc, mosaic_budget)?;
    let source_ids = native_source_ids(&scanned, doc, &fragmented, source_budget, render)?;
    let recognized = recognize_native_sources(engine, pdf, selected, &source_ids, source_budget)?;
    let mut source_only = source_only_pages(&scanned, &recognized)?;
    for &page in &fragmented {
        source_only.approved_pages.remove(&page);
        source_only.pages.remove(&page);
    }
    let mosaics = add_reconstructed_mosaics(
        engine,
        pdf,
        doc,
        &scanned,
        &plans,
        mosaic_budget,
        &mut source_only,
    )?;
    eprintln!(
        "native mosaics: {} fragment pages, {} reconstructed",
        fragmented.len(),
        mosaics
    );
    eprintln!(
        "source-only fast path approved {} of {} selected pages",
        source_only.approved_pages.len(),
        selected.len()
    );
    let fallback_selected = selected
        .iter()
        .copied()
        .filter(|page| !source_only.approved_pages.contains(page))
        .collect::<Vec<_>>();
    let mixed = match prove_mixed_source_text(
        pdf,
        doc,
        &scanned,
        &recognized,
        &fallback_selected,
        render,
        minimum_confidence,
    ) {
        Ok(mixed) => mixed,
        Err(error) => {
            eprintln!("Differential proof unavailable: {error}; using page OCR");
            MixedVisibility::default()
        }
    };
    eprintln!(
        "mixed source visibility: {} proven labels across {} source paint audits",
        mixed.proofs.len(),
        mixed.audited.len()
    );
    Ok(NativePreparation { source_only, mixed })
}

/// Bind OCR observations to their original PDF and atomically publish outputs.
fn write_draft_and_overlay(
    args: &Args,
    doc: &PdfMetadata,
    draft: &OcrDraft,
) -> Result<(), Box<dyn Error>> {
    // Catch a source replacement or mutation while page images were rendered.
    draft.verify_pdf(&args.pdf)?;

    let mut serialized = serde_json::to_vec_pretty(draft)?;
    serialized.push(b'\n');
    let annotated = if args.output_pdf.is_some() {
        let (pdf, report) = pdf_redox::apply_ocr_overlay(
            doc.original_bytes(),
            &importer_document(draft)?,
            OcrOverlayOptions {
                apply_unreviewed: true,
                min_confidence: args.confidence,
            },
        )?;
        eprintln!(
            "OCR overlay: {} accepted, {} native duplicates suppressed",
            report.accepted_lines, report.skipped_native_text_overlap
        );
        Some(pdf)
    } else {
        None
    };
    write_atomic(&args.output, &serialized)?;
    if let (Some(output_path), Some(bytes)) = (args.output_pdf.as_ref(), annotated) {
        write_atomic(output_path, &bytes)?;
    }
    Ok(())
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    let mut paths = vec![args.output.as_path()];
    if let Some(pdf) = &args.output_pdf {
        paths.push(pdf);
    }
    check_paths(&args.pdf, &paths)?;
    if !args.confidence.is_finite() || !(0.0..=1.0).contains(&args.confidence) {
        return Err("confidence threshold outside 0..=1".into());
    }
    let policy = RenderPolicy {
        max_side: args.max_side,
        dpi: args.dpi,
    }
    .validate()?;
    let doc = PdfMetadata::open(&args.pdf)?;
    let selection = parse_page_selection(args.pages.as_deref(), doc.page_count())?;
    let transforms = doc.select_mappers(&selection)?;
    let natives = native_words_for_pages(&args.pdf, &transforms, args.native_chunk)?;
    let digest = hex::encode(Sha256::digest(doc.original_bytes()));

    let engine = OcrInference::pp_ocrv6_small()?;
    let prepared = prepare_native_sources(
        &engine,
        &args.pdf,
        &doc,
        &selection,
        NativePolicies {
            source_budget: NativeImageBudget {
                max_source_pixels: args.max_native_source_pixels,
                max_total_pixels: args.max_native_total_pixels,
                max_sources: args.max_native_sources,
                max_extraction_pixels: args.max_native_extraction_pixels,
            },
            mosaic_budget: MosaicBudget {
                max_group_pixels: args.max_mosaic_group_pixels,
                max_page_pixels: args.max_mosaic_page_pixels,
                max_groups: args.max_mosaic_groups,
                max_dpi: args.max_mosaic_dpi,
            },
            render: policy,
            minimum_confidence: args.confidence,
        },
    )?;
    let mut raster_pages = Vec::with_capacity(selection.len());
    let mut fallback_pages = Vec::new();
    for &page in &selection {
        let native_text_regions = natives
            .get(&page)
            .cloned()
            .ok_or("Poppler native text omitted selected page")?;
        let native_lines = prepared
            .source_only
            .pages
            .get(&page)
            .cloned()
            .unwrap_or_default();
        raster_pages.push(DraftPage {
            page: page.get(),
            lines: native_lines,
            native_text_regions: native_text_regions.clone(),
        });
        if !prepared.source_only.approved_pages.contains(&page) {
            let image = render_rgb_page(&args.pdf, page, policy)?;
            let mapper = doc.mapper(page)?;
            let lines = engine.recognize_page(image, mapper)?;
            eprintln!(
                "page {} (fallback): {} OCR lines, {} native words",
                page.get(),
                lines.len(),
                native_text_regions.len()
            );
            fallback_pages.push(DraftPage {
                page: page.get(),
                lines,
                native_text_regions,
            });
        }
    }
    let raster_draft = OcrDraft::new(
        digest.clone(),
        "oar-ocr PP-OCRv6 Small native-source".into(),
        raster_pages,
    );
    let fallback_draft = OcrDraft::new(
        digest,
        "oar-ocr PP-OCRv6 Small page-fallback".into(),
        fallback_pages,
    );
    let (draft, stats, _proposals) = merge_ocr(MergeInput {
        pdf: &args.pdf,
        selected: &selection,
        raster: &raster_draft,
        image_observations: &prepared.mixed.observations,
        baseline: &fallback_draft,
        raster_only: &prepared.source_only.approved_pages,
        proven_pages: &prepared.mixed.proven_pages,
        visibility_proofs: &prepared.mixed.proofs,
        threshold: args.confidence,
    })?;
    eprintln!(
        "OCR merge: {} source-only pages, {} fallback pages, {} verified native labels added",
        stats.image_source_only_pages,
        stats.page_fallback_pages,
        stats.nonoverlapping_image_lines_added,
    );
    write_draft_and_overlay(args, &doc, &draft)
}

fn main() -> Result<(), Box<dyn Error>> {
    run(&Args::parse())
}

#[cfg(test)]
mod tests {
    use super::{check_paths, parse_page_selection};
    use std::fs;

    #[test]
    fn refuses_output_overwrites_and_aliases() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("original.pdf");
        let json = dir.path().join("result.json");
        let output_pdf = dir.path().join("searchable.pdf");
        fs::write(&input, b"untouched source")?;
        assert!(check_paths(&input, &[&json, &output_pdf]).is_ok());
        assert!(check_paths(&input, &[&input]).is_err());
        assert!(check_paths(&input, &[&json, &json]).is_err());
        #[cfg(unix)]
        {
            let link = dir.path().join("alias.pdf");
            std::os::unix::fs::symlink(&input, &link)?;
            assert!(check_paths(&input, &[&link]).is_err());
        }
        Ok(())
    }

    #[test]
    fn parses_ranges_and_rejects_out_of_range_or_duplicate_pages()
    -> Result<(), Box<dyn std::error::Error>> {
        let pages = parse_page_selection(Some("1, 3-5, 7"), 10)?;
        assert_eq!(
            pages.iter().map(|number| number.get()).collect::<Vec<_>>(),
            vec![1, 3, 4, 5, 7]
        );
        assert!(parse_page_selection(Some("1,2,2"), 10).is_err());
        assert!(parse_page_selection(Some("1-12"), 10).is_err());
        assert!(parse_page_selection(Some("0"), 10).is_err());
        assert!(parse_page_selection(Some("10-5"), 10).is_err());
        Ok(())
    }
}
