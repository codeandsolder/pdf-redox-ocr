//! Offline, research-only ranking of saved CTC hypotheses.

use clap::Parser;
use pdf_redox_ocr::{
    artifacts::{check_output_paths, write_atomic},
    candidate_ranking::{RankingReport, rank},
    draft::OcrDraft,
    refinement_evidence::RefinementReport,
    reviewed_glyphs::ReviewedGlyphBank,
};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    pdf: PathBuf,
    #[arg(long)]
    primary_ocr: PathBuf,
    #[arg(long)]
    refinement_report: PathBuf,
    #[arg(long, required = true)]
    donor: Vec<PathBuf>,
    #[arg(long)]
    report: PathBuf,
    #[arg(long)]
    experimental_draft: PathBuf,
}

#[derive(serde::Serialize)]
struct Provenance<'a> {
    #[serde(flatten)]
    ranking: RankingReport<'a>,
    refinement_report_sha256: String,
    primary_ocr_sha256: String,
    donor_bank_sha256: Vec<String>,
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<(T, String), Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(path)?;
    let digest = hex::encode(Sha256::digest(&bytes));
    Ok((serde_json::from_slice(&bytes)?, digest))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let outputs = [args.report.as_path(), args.experimental_draft.as_path()];
    for input in [&args.pdf, &args.primary_ocr, &args.refinement_report]
        .into_iter()
        .chain(&args.donor)
    {
        check_output_paths(input, &outputs)?;
    }
    let (source, primary_ocr_sha256): (OcrDraft, _) = read(&args.primary_ocr)?;
    source.verify_pdf(&args.pdf)?;
    let (refinement, refinement_report_sha256): (RefinementReport, _) =
        read(&args.refinement_report)?;
    let (banks, donor_bank_sha256): (Vec<ReviewedGlyphBank>, Vec<String>) = args
        .donor
        .iter()
        .map(|p| read(p))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    let (ranking, draft) = rank(&source, &refinement, &banks.iter().collect::<Vec<_>>())?;
    let evidence = Provenance {
        ranking,
        refinement_report_sha256,
        primary_ocr_sha256,
        donor_bank_sha256,
    };
    source.verify_pdf(&args.pdf)?;
    for (path, bytes) in [
        (args.report.as_path(), serde_json::to_vec_pretty(&evidence)?),
        (
            args.experimental_draft.as_path(),
            serde_json::to_vec_pretty(&draft)?,
        ),
    ] {
        write_json(path, bytes)?;
    }
    eprintln!(
        "candidate topology research changes: {}",
        evidence.ranking.selected_changes
    );
    Ok(())
}
fn write_json(path: &Path, mut bytes: Vec<u8>) -> Result<(), pdf_redox_ocr::Error> {
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}
