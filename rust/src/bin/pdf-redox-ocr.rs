//! PDF-structure-aware Rust OCR with original vector page preservation.

use std::{error::Error, path::PathBuf};

use clap::Parser;
use pdf_redox::{OcrDocument, OcrLine as PdfOcrLine, OcrOverlayOptions, OcrPage};
use pdf_redox_ocr::{
    artifacts::{check_output_paths as check_paths, write_atomic},
    draft::OcrDraft,
    inference::OcrInference,
    mosaic::MosaicBudget,
    native_images::NativeImageBudget,
    pipeline::{PipelinePolicy, PreparedOcr},
    rasterize::{DEFAULT_PAGE_MAX_SIDE, RenderPolicy},
};

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
    #[arg(long, default_value_t = DEFAULT_PAGE_MAX_SIDE)]
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
    /// Diagnostic: render searchable native text rather than temporarily hiding it.
    #[arg(long)]
    keep_native_text: bool,
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

const fn configured_policy(args: &Args) -> PipelinePolicy {
    PipelinePolicy {
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
        render: RenderPolicy {
            max_side: args.max_side,
            dpi: args.dpi,
        },
        native_chunk: args.native_chunk,
        keep_native_text: args.keep_native_text,
        minimum_confidence: args.confidence,
    }
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

/// Bind OCR observations to their original PDF and atomically publish outputs.
fn write_draft_and_overlay(
    args: &Args,
    original_bytes: &[u8],
    draft: &OcrDraft,
) -> Result<(), Box<dyn Error>> {
    // Catch a source replacement or mutation while page images were rendered.
    draft.verify_pdf(&args.pdf)?;

    let mut serialized = serde_json::to_vec_pretty(draft)?;
    serialized.push(b'\n');
    let annotated = if args.output_pdf.is_some() {
        let (pdf, report) = pdf_redox::apply_ocr_overlay(
            original_bytes,
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
    let job = PreparedOcr::open(&args.pdf, args.pages.as_deref(), configured_policy(args))?;
    let engine = OcrInference::pp_ocrv6_small()?;
    let result = job.recognize(&engine, |message| eprintln!("{message}"))?;
    write_draft_and_overlay(args, job.original_bytes(), &result.draft)
}

fn main() -> Result<(), Box<dyn Error>> {
    run(&Args::parse())
}

#[cfg(test)]
mod tests {
    use super::check_paths;
    use pdf_redox_ocr::page_selection::parse_page_selection;
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
