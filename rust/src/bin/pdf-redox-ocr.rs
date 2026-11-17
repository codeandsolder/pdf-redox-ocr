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
    inference::OcrInference,
    native_images::{NativeImageBudget, native_source_images},
    native_words::native_words_for_pages,
    paint_scan::enumerate_image_paints,
    pdf_metadata::PdfMetadata,
    rasterize::{RenderPolicy, render_rgb_page},
    source_ocr::{
        SourceObservation, source_image_mapper, source_only_candidates, source_only_pages,
    },
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

/// Decode and recognize independent native-image sources before page fallback.
fn recognize_native_source_pages(
    engine: &OcrInference,
    pdf: &Path,
    doc: &PdfMetadata,
    selection: &[PageNumber],
    budget: NativeImageBudget,
) -> Result<pdf_redox_ocr::source_ocr::SourceOnlyResult, Box<dyn Error>> {
    let scanned = enumerate_image_paints(doc, selection)?;
    let selected_source_ids = source_only_candidates(&scanned, budget.max_source_pixels);
    let decoded_sources = match native_source_images(pdf, selection, &selected_source_ids, budget) {
        Ok(decoded) => decoded,
        Err(error) => {
            eprintln!("Native source extraction unavailable: {error}; using page OCR");
            BTreeMap::new()
        }
    };
    let mut source_observations = BTreeMap::new();
    for (source_id, rgb) in decoded_sources {
        let width = rgb.width();
        let height = rgb.height();
        let mapper = source_image_mapper(width, height)?;
        match engine.recognize_page(rgb, mapper) {
            Ok(lines) => {
                eprintln!("source {source_id}: {} OCR observations", lines.len());
                source_observations.insert(
                    source_id,
                    SourceObservation {
                        width,
                        height,
                        lines,
                    },
                );
            }
            Err(error) => {
                eprintln!("source {source_id}: {error}; source-only OCR disabled for its pages");
            }
        }
    }
    let source_only = source_only_pages(&scanned, &source_observations)?;
    eprintln!(
        "source raster fast path approved {} of {} selected pages",
        source_only.approved_pages.len(),
        selection.len()
    );
    Ok(source_only)
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
    let source_only = recognize_native_source_pages(
        &engine,
        &args.pdf,
        &doc,
        &selection,
        NativeImageBudget {
            max_source_pixels: args.max_native_source_pixels,
            max_total_pixels: args.max_native_total_pixels,
            max_sources: args.max_native_sources,
            max_extraction_pixels: args.max_native_extraction_pixels,
        },
    )?;
    let mut pages = Vec::with_capacity(selection.len());
    for page in selection {
        let (lines, path) = if let Some(lines) = source_only.pages.get(&page) {
            (lines.clone(), "native-source")
        } else {
            let image = render_rgb_page(&args.pdf, page, policy)?;
            let mapper = doc.mapper(page)?;
            (engine.recognize_page(image, mapper)?, "page-fallback")
        };
        let native_text_regions = natives
            .get(&page)
            .cloned()
            .ok_or("Poppler native text omitted selected page")?;
        eprintln!(
            "page {} ({path}): {} OCR lines, {} native words",
            page.get(),
            lines.len(),
            native_text_regions.len()
        );
        pages.push(DraftPage {
            page: page.get(),
            lines,
            native_text_regions,
        });
    }
    let draft = OcrDraft::new(
        digest,
        "oar-ocr PP-OCRv6 Small (Rust), native source and page fallback".into(),
        pages,
    );
    // Catch a source replacement or mutation while page images were rendered.
    draft.verify_pdf(&args.pdf)?;

    let mut serialized = serde_json::to_vec_pretty(&draft)?;
    serialized.push(b'\n');
    let annotated = if args.output_pdf.is_some() {
        let (pdf, report) = pdf_redox::apply_ocr_overlay(
            doc.original_bytes(),
            &importer_document(&draft)?,
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
