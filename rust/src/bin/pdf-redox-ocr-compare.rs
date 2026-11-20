//! Source-bound comparison of alternate OCR drafts, independent of models.
use std::{error::Error, fs, path::PathBuf};

use clap::Parser;
use pdf_redox_ocr::{
    artifacts::{check_output_paths, write_atomic},
    draft::OcrDraft,
    draft_compare::compare,
};

#[derive(Debug, Parser)]
#[command(name = "pdf-redox-ocr-compare", version)]
struct Args {
    #[arg(long)]
    pdf: PathBuf,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    report: PathBuf,
    #[arg(long, default_value_t = 0.5)]
    minimum_overlap: f64,
    #[arg(long, default_value_t = 0.97)]
    confidence: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    for input in [&args.pdf, &args.baseline, &args.candidate] {
        check_output_paths(input, &[&args.report])?;
    }
    let baseline: OcrDraft = serde_json::from_slice(&fs::read(&args.baseline)?)?;
    let candidate: OcrDraft = serde_json::from_slice(&fs::read(&args.candidate)?)?;
    baseline.verify_pdf(&args.pdf)?;
    candidate.verify_pdf(&args.pdf)?;
    let report = compare(&baseline, &candidate, args.minimum_overlap, args.confidence)?;
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    write_atomic(&args.report, &bytes)?;
    eprintln!("spatial OCR comparison: {:?}", report.counts);
    Ok(())
}
