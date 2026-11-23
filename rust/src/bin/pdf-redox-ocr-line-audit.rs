//! Geometry-only high-resolution source PDF line diagnostics.

use std::{error::Error, fs, path::PathBuf};

use clap::Parser;
use pdf_redox_ocr::{
    artifacts::{check_output_paths, write_atomic},
    draft::OcrDraft,
    ink_polarity::Polarity,
    line_audit::{LineAuditPolicy, audit_lines},
    page_selection::parse_exact_lines,
};

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ForegroundMode {
    Auto,
    Dark,
    Light,
}

impl From<ForegroundMode> for Polarity {
    fn from(value: ForegroundMode) -> Self {
        match value {
            ForegroundMode::Auto => Self::Auto,
            ForegroundMode::Dark => Self::Dark,
            ForegroundMode::Light => Self::Light,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "pdf-redox-ocr-line-audit", version)]
struct Args {
    /// Exact original PDF.
    #[arg(long)]
    pdf: PathBuf,
    /// Unchanged, PDF-bound original OCR draft.
    #[arg(long)]
    primary_ocr: PathBuf,
    /// Original one-based page and zero-based line index; repeatable.
    #[arg(long = "line", required = true, value_name = "PAGE:INDEX")]
    lines: Vec<String>,
    /// Native region render resolution.
    #[arg(long, default_value_t = 600)]
    dpi: u32,
    /// Maximum source or oriented crop size, in pixels, per line.
    #[arg(long, default_value_t = 2_000_000)]
    max_pixels: u64,
    /// Maximum total raster-pixel/peer membership checks per line.
    #[arg(long, default_value_t = 32_000_000)]
    max_ink_checks: u64,
    /// Foreground polarity; auto uses measured perimeter luminance classes.
    #[arg(long, value_enum, default_value = "auto")]
    polarity: ForegroundMode,
    /// Extra source pixels around the observation, clipped to page bounds.
    #[arg(long, default_value_t = 0)]
    padding_pixels: u32,
    /// Contrast-normalized foreground cutoff for column segmentation.
    #[arg(long, default_value_t = 192)]
    segmentation_threshold: u8,
    /// Topology luminance cutoffs; repeatable.
    #[arg(long)]
    threshold: Vec<u8>,
    /// Destination JSON containing mapped ink groups.
    #[arg(long)]
    report: PathBuf,
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    check_output_paths(&args.pdf, &[&args.report])?;
    check_output_paths(&args.primary_ocr, &[&args.report])?;
    let draft: OcrDraft = serde_json::from_slice(&fs::read(&args.primary_ocr)?)?;
    let targets = parse_exact_lines(&args.lines)?;
    let mut policy = LineAuditPolicy {
        dpi: args.dpi,
        max_pixels: args.max_pixels,
        max_ink_checks: args.max_ink_checks,
        segmentation_threshold: args.segmentation_threshold,
        polarity: args.polarity.into(),
        padding_pixels: Some([args.padding_pixels; 4]),
        ..LineAuditPolicy::default()
    };
    if !args.threshold.is_empty() {
        policy.topology_thresholds.clone_from(&args.threshold);
        policy.topology_thresholds.sort_unstable();
        policy.topology_thresholds.dedup();
    }
    let report = audit_lines(&args.pdf, &draft, &targets, &policy)?;
    let mut json = serde_json::to_vec_pretty(&report)?;
    json.push(b'\n');
    write_atomic(&args.report, &json)?;
    eprintln!(
        "audited {} original OCR lines, {} visible ink groups at {} DPI",
        report.lines.len(),
        report
            .lines
            .iter()
            .map(|line| line.groups.len())
            .sum::<usize>(),
        report.dpi,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run(&Args::parse())
}
