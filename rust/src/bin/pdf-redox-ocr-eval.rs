//! Source-bound human-reference evaluation for OCR drafts (no model required).

use std::{error::Error, fs, path::PathBuf};

use clap::Parser;
use pdf_redox_ocr::{
    artifacts::{check_output_paths, write_atomic},
    draft::OcrDraft,
    quality_eval::{GroundTruth, evaluate},
};

/// Compare OCR predictions with manually transcribed PDF-space anchors.
#[derive(Debug, Parser)]
#[command(name = "pdf-redox-ocr-eval", version)]
struct Args {
    /// Exact original PDF whose SHA-256 is stored in the draft and reference.
    #[arg(long)]
    pdf: PathBuf,
    /// JSON OCR draft bound to the original PDF.
    #[arg(long)]
    draft: PathBuf,
    /// Manually reviewed ground truth JSON with source hash and anchor centers.
    #[arg(long)]
    reference: PathBuf,
    /// New or replaced structured evaluation report JSON.
    #[arg(long)]
    report: PathBuf,
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    for source in [&args.pdf, &args.draft, &args.reference] {
        check_output_paths(source, &[&args.report])?;
    }
    let draft: OcrDraft = serde_json::from_slice(&fs::read(&args.draft)?)?;
    draft.verify_pdf(&args.pdf)?;
    let reference: GroundTruth = serde_json::from_slice(&fs::read(&args.reference)?)?;
    let report = evaluate(&reference, &draft)?;
    let mut json = serde_json::to_vec_pretty(&report)?;
    json.push(b'\n');
    write_atomic(&args.report, &json)?;
    eprintln!(
        "manual OCR evaluation: {}/{} matched, {} exact, {} normalized exact, CER {:.4}",
        report.overall.matched,
        report.overall.samples,
        report.overall.exact,
        report.overall.normalized_exact,
        report.overall.character_error_rate,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run(&Args::parse())
}
