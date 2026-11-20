//! Compare reviewed glyph appearance on held-out source pages.

use clap::Parser;
use pdf_redox_ocr::{
    artifacts::{check_output_paths, write_atomic},
    glyph_matching::{MatchPolicy, evaluate},
    reviewed_glyphs::ReviewedGlyphBank,
};
use std::{
    collections::BTreeSet,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(version)]
struct Args {
    /// Ground-truth-backed query glyph bank.
    #[arg(long)]
    query: PathBuf,
    /// Donor glyph bank, repeatable. Same-page and identical-crop examples are excluded.
    #[arg(long, required = true)]
    training: Vec<PathBuf>,
    /// Optional one-based query page, repeatable.
    #[arg(long)]
    page: Vec<u32>,
    #[arg(long)]
    report: PathBuf,
    /// Maximum normalized appearance distance (not confidence).
    #[arg(long)]
    max_distance: f64,
    /// Required difference between first and second distinct label distances.
    #[arg(long)]
    min_margin: f64,
    /// Minimum number of independently cropped matching examples for the winning label.
    #[arg(long, default_value_t = 2)]
    min_independent_crops: usize,
}

fn read(path: &Path) -> Result<ReviewedGlyphBank, Box<dyn Error>> {
    Ok(serde_json::from_reader(std::io::BufReader::new(
        fs::File::open(path)?,
    ))?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    for path in std::iter::once(&args.query).chain(&args.training) {
        check_output_paths(path, &[&args.report])?;
    }
    let query = read(&args.query)?;
    let banks = args
        .training
        .iter()
        .map(|p| read(p))
        .collect::<Result<Vec<_>, _>>()?;
    let report = evaluate(
        &query,
        &banks.iter().collect::<Vec<_>>(),
        &args.page.into_iter().collect::<BTreeSet<_>>(),
        MatchPolicy {
            max_distance: args.max_distance,
            min_margin: args.min_margin,
            min_independent_crops: args.min_independent_crops,
        },
    )?;
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    write_atomic(&args.report, &bytes)?;
    eprintln!(
        "{} glyphs, {} candidates, {} correct, {} corrected errors, {} changed correct primaries",
        report.summary.queries,
        report.summary.candidates,
        report.summary.correct_candidates,
        report.summary.corrected_primary_errors,
        report.summary.changed_correct_primary
    );
    Ok(())
}
