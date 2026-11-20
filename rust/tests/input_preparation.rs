//! Exact processing-source provenance and end-to-end native-only CLI coverage.

#![cfg(feature = "pdf-integration")]

use std::fs;

use pdf_redox::{Config, optimize_pdf};
use pdf_redox_ocr::{
    Error,
    draft::{OcrDraft, sha256_file},
    input_preparation::{PreparationMode, PreparedInput},
};

mod support;

fn fixture() -> Vec<u8> {
    support::pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F 4 0 R >> >> /Contents 5 0 R >>".to_vec(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        support::stream("",b"BT /F 12 Tf 20 150 Td (Native searchable text) Tj ET"),
    ])
}

#[test]
fn processing_preparation_exactly_matches_library_and_cleans_up() -> Result<(), Error> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("source.pdf");
    let original = fixture();
    fs::write(&input, &original)?;
    let raw = PreparedInput::open(&input, PreparationMode::Original { catalog: None })?;
    assert_eq!(raw.path(), input.canonicalize()?);
    assert_eq!(raw.audit().original_sha256, raw.audit().ocr_input_sha256);
    assert!(raw.audit().processing.is_none());
    let work = dir.path().join("work");
    fs::create_dir(&work)?;
    let prepared = PreparedInput::open(
        &input,
        PreparationMode::Processing {
            work_dir: Some(&work),
            collect_catalog: true,
        },
    )?;
    let path = prepared.path().to_path_buf();
    let (expected, _) = optimize_pdf(&original, &Config::optimize_only().for_processing())?;
    assert_eq!(fs::read(&path)?, expected);
    let audit = prepared.audit();
    assert_eq!(
        prepared.catalog().map(|c| c.optimized_pdf_sha256.as_str()),
        Some(audit.ocr_input_sha256.as_str())
    );
    assert_eq!(audit.original_sha256, sha256_file(&input)?);
    assert_eq!(audit.ocr_input_sha256, sha256_file(&path)?);
    assert_ne!(audit.original_sha256, audit.ocr_input_sha256);
    assert!(audit.processing.is_some());
    let draft = OcrDraft::new(audit.ocr_input_sha256.clone(), "fixture".into(), vec![]);
    let wrong = OcrDraft::new(audit.original_sha256.clone(), "fixture".into(), vec![]);
    prepared.verify_recognized(&draft)?;
    assert!(prepared.verify_recognized(&wrong).is_err());
    fs::write(&path, b"changed prepared PDF")?;
    assert!(prepared.verify_recognized(&draft).is_err());
    fs::write(&path, &expected)?;
    assert_eq!(fs::read(&input)?, original);
    fs::write(&input, b"changed")?;
    assert!(prepared.verify_recognized(&draft).is_err());
    drop(prepared);
    assert!(!path.exists());
    assert_eq!(fs::read_dir(work)?.count(), 0);
    Ok(())
}

#[cfg(all(feature = "cli", feature = "native-inference"))]
#[test]
fn processing_cli_retains_exact_draft_input_without_model_or_optimizer_binary()
-> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("source.pdf");
    let output = dir.path().join("searchable.pdf");
    let draft_path = dir.path().join("draft.json");
    let prepared_path = dir.path().join("optimized.pdf");
    let report_path = dir.path().join("audit.json");
    let catalog_path = dir.path().join("glyphs.json");
    let work = dir.path().join("work");
    let cache = dir.path().join("invalid-model-cache");
    fs::create_dir(&work)?;
    fs::write(&cache, b"not a model directory")?;
    fs::write(&input, fixture())?;
    let cli = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_pdf-redox-ocr"));
        command
            .current_dir(dir.path())
            .env("OAR_HOME", &cache)
            .arg("--pdf")
            .arg(&input)
            .arg("--optimize")
            .arg("--work-dir")
            .arg(&work);
        command
    };
    let rejected = cli().arg("--output").arg(&draft_path).output()?;
    assert!(!rejected.status.success());
    assert!(!draft_path.exists());
    assert_eq!(fs::read_dir(&work)?.count(), 0);
    let rejected = cli().arg("--output-pdf").arg(&input).output()?;
    assert!(!rejected.status.success());
    assert_eq!(fs::read(&input)?, fixture());
    let processed = cli()
        .arg("--output-pdf")
        .arg(&output)
        .arg("--output")
        .arg(&draft_path)
        .arg("--optimized-pdf")
        .arg(&prepared_path)
        .arg("--report")
        .arg(&report_path)
        .arg("--output-glyph-catalog")
        .arg(&catalog_path)
        .args(["--confirm-with", "medium"])
        .output()?;
    assert!(
        processed.status.success(),
        "{}",
        String::from_utf8_lossy(&processed.stderr)
    );
    let draft: OcrDraft = serde_json::from_slice(&fs::read(&draft_path)?)?;
    draft.verify_pdf(&prepared_path)?;
    assert!(draft.verify_pdf(&input).is_err());
    assert_eq!(draft.pages.len(), 1);
    assert!(draft.pages[0].lines.is_empty());
    let audit: serde_json::Value = serde_json::from_slice(&fs::read(&report_path)?)?;
    assert_eq!(audit["schema_version"], 9);
    assert_eq!(
        audit["preparation"]["original_sha256"],
        sha256_file(&input)?
    );
    assert_eq!(audit["preparation"]["ocr_input_sha256"], draft.pdf_sha256);
    assert_eq!(audit["pdf_sha256"], draft.pdf_sha256);
    assert_eq!(audit["recognition"]["inference_calls"], 0);
    assert_eq!(audit["recognition"]["confirmation_attempts"], 0);
    let catalog: serde_json::Value = serde_json::from_slice(&fs::read(&catalog_path)?)?;
    assert_eq!(catalog["optimized_pdf_sha256"], draft.pdf_sha256);
    assert_eq!(
        audit["preparation"]["processing"]["config"]["optimization_goal"],
        "processing"
    );
    assert_eq!(fs::read(&prepared_path)?, fs::read(&output)?);
    assert_eq!(fs::read_dir(work)?.count(), 0);
    assert_eq!(fs::read(cache)?, b"not a model directory");
    assert_eq!(fs::read(input)?, fixture());
    Ok(())
}
