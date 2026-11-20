//! PDF-structure-aware Rust OCR with original vector page preservation.

use std::{
    error::Error,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use clap::Parser;
use image::{ImageEncoder, RgbImage, codecs::png::PngEncoder};
use pdf_redox::{OcrDocument, OcrLine as PdfOcrLine, OcrOverlayOptions, OcrPage};
use pdf_redox_ocr::{
    artifacts::{check_output_paths as check_paths, write_atomic},
    draft::OcrDraft,
    inference::OcrInference,
    input_preparation::PreparedInput,
    mosaic::MosaicBudget,
    native_images::NativeImageBudget,
    pipeline::{
        PipelineObserver, PipelinePolicy, PipelineResult, PreparedOcr, RasterMode, ResidualInput,
    },
    rasterize::{DEFAULT_PAGE_MAX_SIDE, RenderPolicy},
    recognition::RecognitionCacheBudget,
};

/// Generate SHA-bound PDF OCR geometry and optionally apply invisible semantic text.
#[derive(Debug, Parser)]
#[command(name = "pdf-redox-ocr", version)]
struct Args {
    /// Exact source PDF, which is never overwritten.
    #[arg(long)]
    pdf: PathBuf,
    /// Retain the exact schema-version 3 importer draft. With optimization,
    /// also retain its corresponding PDF using --optimized-pdf.
    #[arg(long)]
    output: Option<PathBuf>,
    #[command(flatten)]
    preparation: PreparationArgs,
    #[command(flatten)]
    recognition: RecognitionArgs,
    /// Write an optional PDF annotated through the Rust pdf-redox importer.
    #[arg(long)]
    output_pdf: Option<PathBuf>,
    /// Optional SHA-bound JSON audit of source recognition, paint proofs and routing.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Explicit new directory for lossless ordinary/treated residual raster captures.
    #[arg(long)]
    capture_residual_dir: Option<PathBuf>,
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
    /// Diagnostic: recognize the complete fallback raster, even with source proofs.
    #[arg(long)]
    no_residual_ocr: bool,
    /// Automatic, exhaustive bounded, or disabled native-raster recognition.
    #[arg(long, value_enum, default_value = "auto")]
    raster_mode: RasterMode,
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

#[derive(Debug, clap::Args)]
struct PreparationArgs {
    /// Optimize using pdf-redox's standard processing objective before OCR.
    #[arg(long)]
    optimize: bool,
    /// Retain the exact optimized PDF to which the OCR draft is bound.
    #[arg(long, requires = "optimize")]
    optimized_pdf: Option<PathBuf>,
    /// Directory for the scoped intermediate optimized PDF.
    #[arg(long, requires = "optimize")]
    work_dir: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
struct RecognitionArgs {
    /// Diagnostic: disable job-local exact RGB recognition reuse.
    #[arg(long)]
    no_recognition_cache: bool,
    /// Disable geometry-confirmed re-recognition of dotted layout leaders.
    #[arg(long)]
    no_leader_repair: bool,
    /// ONNX intra-operation CPU thread count (default: backend policy).
    #[arg(long)]
    ocr_threads: Option<NonZeroUsize>,
}

fn configured_policy(args: &Args) -> PipelinePolicy {
    PipelinePolicy {
        raster_mode: args.raster_mode,
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
        residual_ocr: !args.no_residual_ocr,
        recognition_cache: if args.recognition.no_recognition_cache {
            RecognitionCacheBudget::DISABLED
        } else {
            RecognitionCacheBudget::default()
        },
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
fn write_outputs(
    args: &Args,
    input: &PreparedInput,
    prepared_bytes: &[u8],
    result: &PipelineResult,
) -> Result<(), Box<dyn Error>> {
    let draft = &result.draft;
    let serialized = args
        .output
        .as_ref()
        .map(|_| json_bytes(draft))
        .transpose()?;
    let annotated = if args.output_pdf.is_some() {
        let (pdf, report) = pdf_redox::apply_ocr_overlay(
            prepared_bytes,
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
    let audit = args
        .report
        .as_ref()
        .map(|_| -> Result<Vec<u8>, Box<dyn Error>> {
            Ok(json_bytes(
                &result.audit().with_preparation(input.audit())?,
            )?)
        })
        .transpose()?;
    // Catch a change to either source immediately before publishing.
    input.verify_recognized(draft)?;
    if let Some(path) = &args.preparation.optimized_pdf {
        write_atomic(path, prepared_bytes)?;
    }
    if let (Some(path), Some(bytes)) = (&args.output, serialized) {
        write_atomic(path, &bytes)?;
    }
    if let (Some(output_path), Some(bytes)) = (args.output_pdf.as_ref(), annotated) {
        write_atomic(output_path, &bytes)?;
    }
    if let (Some(path), Some(bytes)) = (&args.report, audit) {
        write_atomic(path, &bytes)?;
    }
    Ok(())
}

fn json_bytes(value: &impl serde::Serialize) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

struct CliObserver {
    capture: Option<PathBuf>,
}

fn write_capture(path: &Path, rgb: &RgbImage) -> Result<(), pdf_redox_ocr::Error> {
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes).write_image(
        rgb.as_raw(),
        rgb.width(),
        rgb.height(),
        image::ExtendedColorType::Rgb8,
    )?;
    write_atomic(path, &bytes)
}

impl PipelineObserver for CliObserver {
    fn message(&mut self, message: String) {
        eprintln!("{message}");
    }

    fn residual_input(&mut self, input: ResidualInput<'_>) -> Result<(), pdf_redox_ocr::Error> {
        if let Some(dir) = &self.capture {
            for (name, image) in [("ordinary", input.ordinary), ("treated", input.treated)] {
                write_capture(
                    &dir.join(format!("page-{:06}-{name}.png", input.page.get())),
                    image,
                )?;
            }
        }
        Ok(())
    }
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.output.is_none() && args.output_pdf.is_none() && args.report.is_none() {
        return Err("request at least one draft, searchable PDF or audit output".into());
    }
    if args.preparation.optimize
        && args.output.is_some()
        && args.preparation.optimized_pdf.is_none()
    {
        return Err("retaining an optimized-input draft requires --optimized-pdf".into());
    }
    let paths = [
        args.output.as_deref(),
        args.output_pdf.as_deref(),
        args.report.as_deref(),
        args.preparation.optimized_pdf.as_deref(),
        args.capture_residual_dir.as_deref(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    check_paths(&args.pdf, &paths)?;
    let input = PreparedInput::open(
        &args.pdf,
        args.preparation.optimize,
        args.preparation.work_dir.as_deref(),
    )?;
    if let Some(evidence) = &input.audit().processing {
        eprintln!(
            "PDF processing optimization: {} -> {} bytes",
            evidence.report.before.input_bytes, evidence.report.after_bytes
        );
    }
    let job = PreparedOcr::open(input.path(), args.pages.as_deref(), configured_policy(args))?;
    if let Some(directory) = &args.capture_residual_dir {
        fs::create_dir(directory)?;
    }
    let engine = OcrInference::pp_ocrv6_small(
        args.recognition.ocr_threads,
        (!args.recognition.no_leader_repair).then_some(args.confidence.max(0.97)),
    );
    let result = job.recognize(
        &engine,
        CliObserver {
            capture: args.capture_residual_dir.clone(),
        },
    )?;
    write_outputs(args, &input, job.original_bytes(), &result)
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
