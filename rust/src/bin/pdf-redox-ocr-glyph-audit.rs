//! Structural glyph diagnostics independent of OCR character probabilities.

use std::{error::Error, fs, path::PathBuf};

use clap::Parser;
use pdf_redox_ocr::{
    artifacts::{check_output_paths, write_atomic},
    glyph_topology::{GlyphTopology, analyze},
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(name = "pdf-redox-ocr-glyph-audit", version)]
struct Args {
    /// A tightly selected grayscale/RGB crop with a margin around its glyph.
    #[arg(long)]
    image: PathBuf,
    /// Output JSON containing thresholded enclosed-background topology.
    #[arg(long)]
    report: PathBuf,
    /// Luminance cutoffs, repeatable; defaults to 64,128,192,224.
    #[arg(long)]
    threshold: Vec<u8>,
}

#[derive(Serialize)]
struct AuditReport {
    schema_version: u32,
    image_sha256: String,
    thresholds: Vec<GlyphTopology>,
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    check_output_paths(&args.image, &[&args.report])?;
    let bytes = fs::read(&args.image)?;
    let image_sha256 = hex::encode(Sha256::digest(&bytes));
    let image = image::load_from_memory(&bytes)?.to_luma8();
    let mut thresholds = if args.threshold.is_empty() {
        vec![64, 128, 192, 224]
    } else {
        args.threshold.clone()
    };
    thresholds.sort_unstable();
    thresholds.dedup();
    let diagnostics = thresholds
        .into_iter()
        .map(|threshold| analyze(&image, threshold))
        .collect::<Result<Vec<_>, _>>()?;
    let report = AuditReport {
        schema_version: 1,
        image_sha256,
        thresholds: diagnostics,
    };
    let mut json = serde_json::to_vec_pretty(&report)?;
    json.push(b'\n');
    write_atomic(&args.report, &json)?;
    for entry in &report.thresholds {
        eprintln!(
            "glyph ink cutoff {}: {} enclosed background regions, {} dark pixels",
            entry.threshold,
            entry.enclosed_background_regions.len(),
            entry.foreground_pixels
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run(&Args::parse())
}
