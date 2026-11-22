//! Research-only source-resolution line refinement in exact draft reading order.

use std::{collections::BTreeSet, error::Error, fs, num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use pdf_redox_ocr::{
    PageNumber,
    artifacts::{check_output_paths, write_atomic},
    draft::OcrDraft,
    page_selection::{parse_exact_lines, parse_page_selection},
    pdf_metadata::PdfMetadata,
    refinement::{RefinementPolicy, RefinementRequest, RefinementScope, refine_draft},
    refinement_recognition::{RefinementModel, RefinementRecognizer},
};

/// Compare original visible PDF line crops with actual recognition hypotheses.
#[derive(Parser, Debug)]
#[command(name = "pdf-redox-ocr-refine", version)]
struct Args {
    /// Exact original PDF.
    #[arg(long)]
    pdf: PathBuf,
    /// Primary version-3 draft in original PDF reading-order geometry.
    #[arg(long)]
    primary_ocr: PathBuf,
    /// Actual recognizer, or two-model four-view consensus.
    #[arg(long, value_enum, default_value = "medium")]
    recognizer: RefinementModel,
    /// Output per-line source and model evidence.
    #[arg(long)]
    report: PathBuf,
    /// Distinct experimental draft, entirely zero-confidence and unreviewed.
    #[arg(long)]
    experimental_draft: PathBuf,
    /// Optional page numbers and ranges.
    #[arg(long, conflicts_with = "lines")]
    pages: Option<String>,
    /// Exact original draft observation PAGE:ZERO_BASED_INDEX; repeatable.
    #[arg(long = "line", value_name = "PAGE:INDEX")]
    lines: Vec<String>,
    /// Maximum primary score when selecting by confidence.
    #[arg(long, default_value_t = 0.97)]
    max_primary_score: f64,
    #[arg(long, default_value_t = 2)]
    min_chars: usize,
    /// Source sampling DPI, not full-page resolution.
    #[arg(long, default_value_t = 600)]
    dpi: u32,
    /// Combined rendered and rectified source pixels per line.
    #[arg(long, default_value_t = 2_000_000)]
    max_line_pixels: u64,
    /// Combined rendered and rectified source pixels per page.
    #[arg(long, default_value_t = 32_000_000)]
    max_page_pixels: u64,
    /// ONNX intra-operation CPU threads.
    #[arg(long)]
    ocr_threads: Option<NonZeroUsize>,
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    check_output_paths(&args.pdf, &[&args.report, &args.experimental_draft])?;
    check_output_paths(&args.primary_ocr, &[&args.report, &args.experimental_draft])?;
    let source: OcrDraft = serde_json::from_slice(&fs::read(&args.primary_ocr)?)?;
    source.verify_pdf(&args.pdf)?;
    let metadata = PdfMetadata::open(&args.pdf)?;
    let selection = args
        .pages
        .as_deref()
        .map(|pages| parse_page_selection(Some(pages), metadata.page_count()))
        .transpose()?
        .map(|pages| pages.into_iter().collect::<BTreeSet<PageNumber>>());
    let exact_lines = parse_exact_lines(&args.lines)?;
    let scope = if exact_lines.is_empty() {
        RefinementScope::ByConfidence {
            pages: selection.as_ref(),
        }
    } else {
        RefinementScope::ExplicitLines(&exact_lines)
    };
    let policy = RefinementPolicy {
        max_primary_score: args.max_primary_score,
        min_chars: args.min_chars,
        dpi: args.dpi,
        max_line_pixels: args.max_line_pixels,
        max_page_pixels: args.max_page_pixels,
    }
    .validate()?;
    let mut predictor = RefinementRecognizer::new(args.recognizer, args.ocr_threads);
    let (report, experimental) = refine_draft(
        RefinementRequest {
            pdf: &args.pdf,
            metadata: &metadata,
            source: &source,
            scope,
            secondary_model: args.recognizer.label(),
            policy,
        },
        |image| predictor.recognize(image),
    )?;
    let mut report_bytes = serde_json::to_vec_pretty(&report)?;
    report_bytes.push(b'\n');
    let mut draft_bytes = serde_json::to_vec_pretty(&experimental)?;
    draft_bytes.push(b'\n');
    write_atomic(&args.report, &report_bytes)?;
    write_atomic(&args.experimental_draft, &draft_bytes)?;
    eprintln!(
        "source refinement: {}/{} recognized, {} skipped, {} differing hypotheses, {} source pixels; experimental confidences all zero",
        report.summary.recognized_lines,
        report.summary.selected_lines,
        report.summary.skipped_lines,
        report.summary.revised_predictions,
        report.summary.charged_pixels
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run(&Args::parse())
}

#[cfg(test)]
mod tests {
    use super::parse_exact_lines;
    use pdf_redox_ocr::PageNumber;

    #[test]
    fn exact_line_syntax_and_duplicates_are_unambiguous() -> Result<(), pdf_redox_ocr::Error> {
        let targets = parse_exact_lines(&["28:11".into(), "28:2".into(), "8:0".into()])?;
        let page = PageNumber::new(28)?;
        assert_eq!(
            targets[&page].iter().copied().collect::<Vec<_>>(),
            vec![2, 11],
        );
        assert_eq!(targets[&PageNumber::new(8)?].len(), 1);
        for invalid in ["0:1", "28", ":2", "x:2", "28:-1", "28:1:0", "28:x"] {
            assert!(parse_exact_lines(&[invalid.into()]).is_err(), "{invalid}");
        }
        assert!(parse_exact_lines(&["28:2".into(), "28:2".into()]).is_err());
        Ok(())
    }
}
