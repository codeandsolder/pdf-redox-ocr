//! Native source-fragment reconciliation diagnostics.
use clap::Parser;
use pdf_redox_ocr::{
    artifacts::{check_output_paths, write_atomic},
    draft::{OcrDraft, sha256_file},
    fragment_refinement::{FragmentPolicy, reconcile_fragments},
    page_selection::parse_exact_lines,
};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, num::NonZeroUsize, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    pdf: PathBuf,
    #[arg(long)]
    primary_ocr: PathBuf,
    #[arg(long = "line", required = true, value_name = "PAGE:INDEX")]
    lines: Vec<String>,
    #[arg(long)]
    report: PathBuf,
    #[arg(long)]
    experimental_draft: PathBuf,
    #[arg(long, default_value_t = 600)]
    dpi: u32,
    /// Override geometry-scaled padding with a uniform pixel margin.
    #[arg(long)]
    padding_pixels: Option<u32>,
    #[arg(long, default_value_t = 2_000_000)]
    max_line_pixels: u64,
    #[arg(long, default_value_t = 32_000_000)]
    max_page_pixels: u64,
    #[arg(long, default_value_t = 32_000_000)]
    max_ink_checks: u64,
    #[arg(long)]
    ocr_threads: Option<NonZeroUsize>,
}

#[derive(serde::Serialize)]
struct Provenance<'a> {
    #[serde(flatten)]
    report: &'a pdf_redox_ocr::fragment_refinement::FragmentReport,
    primary_ocr_sha256: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let outputs = [args.report.as_path(), args.experimental_draft.as_path()];
    for input in [&args.pdf, &args.primary_ocr] {
        check_output_paths(input, &outputs)?;
    }
    let source_bytes = fs::read(&args.primary_ocr)?;
    let primary_sha256 = hex::encode(Sha256::digest(&source_bytes));
    let source: OcrDraft = serde_json::from_slice(&source_bytes)?;
    let targets = parse_exact_lines(&args.lines)?;
    let mut policy = FragmentPolicy::default();
    policy.audit.dpi = args.dpi;
    policy.audit.padding_pixels = args.padding_pixels;
    policy.audit.max_pixels = args.max_line_pixels;
    policy.audit.max_ink_checks = args.max_ink_checks;
    policy.max_page_pixels = args.max_page_pixels;
    let (report, draft) =
        reconcile_fragments(&args.pdf, &source, &targets, &policy, args.ocr_threads)?;
    if sha256_file(&args.primary_ocr)? != primary_sha256 {
        return Err("primary OCR changed during fragment reconciliation".into());
    }
    let provenance = Provenance {
        report: &report,
        primary_ocr_sha256: primary_sha256,
    };
    for (path, mut bytes) in [
        (&args.report, serde_json::to_vec_pretty(&provenance)?),
        (&args.experimental_draft, serde_json::to_vec_pretty(&draft)?),
    ] {
        bytes.push(b'\n');
        write_atomic(path, &bytes)?;
    }
    eprintln!(
        "{} proposed fragment transactions, {} superseded, {} conflicts, {} accepted (non-importable research)",
        report.transactions.len(),
        report.superseded.len(),
        report.conflicts.len(),
        report.transactions.len() - report.conflicts.len() - report.superseded.len()
    );
    Ok(())
}
