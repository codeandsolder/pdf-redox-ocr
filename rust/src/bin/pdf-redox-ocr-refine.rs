//! Research-only PP-OCRv6 medium recognition on uncertain PDF text quads.

use std::{collections::BTreeSet, error::Error, fs, io::Write, path::PathBuf};

use clap::Parser;
use image::{RgbImage, imageops};
use oar_ocr::{
    download::resolve_path,
    predictors::{TextLineOrientationPredictor, TextRecognitionPredictor},
};
use pdf_redox_ocr::{
    PageNumber,
    artifacts::{check_output_paths, write_atomic},
    draft::OcrDraft,
    native_font_mask::mask_native_fonts,
    page_selection::{parse_exact_lines, parse_page_selection},
    pdf_metadata::PdfMetadata,
    rasterize::{DEFAULT_PAGE_MAX_SIDE, RenderPolicy},
    refinement::{
        Recognition, RefinementPolicy, RefinementRequest, RefinementScope, refine_draft,
        refinement_plan,
    },
};
use tempfile::NamedTempFile;

/// Compare a recognition-only second model with a SHA-bound primary draft.
#[derive(Parser, Debug)]
#[command(name = "pdf-redox-ocr-refine", version)]
struct Args {
    /// Original PDF used to generate the primary OCR draft.
    #[arg(long)]
    pdf: PathBuf,
    /// Primary version-3 draft in original PDF geometry.
    #[arg(long)]
    primary_ocr: PathBuf,
    /// Recognition-only ONNX model; the default is auto-resolved from OAR.
    #[arg(long, default_value = "pp-ocrv6_medium_rec.onnx")]
    secondary_model: PathBuf,
    /// OAR/PaddleOCR full-vocabulary text dictionary.
    #[arg(long, default_value = "ppocrv6_dict.txt")]
    character_dict: PathBuf,
    /// Where to write all primary/secondary text and confidence comparisons.
    #[arg(long)]
    report: PathBuf,
    /// Separate hypothesis draft with every line unreviewed and confidence zero.
    #[arg(long)]
    experimental_draft: PathBuf,
    /// Optional one-based page numbers and inclusive ranges.
    #[arg(long, conflicts_with = "lines")]
    pages: Option<String>,
    /// Exact original draft observation `PAGE:ZERO_BASED_LINE_INDEX`; repeatable.
    #[arg(long = "line", value_name = "PAGE:INDEX")]
    lines: Vec<String>,
    /// Maximum primary OCR confidence to reconsider.
    #[arg(long, default_value_t = 0.97)]
    max_primary_score: f64,
    #[arg(long, default_value_t = 2)]
    min_chars: usize,
    /// Longest side of the Poppler page render.
    #[arg(long, default_value_t = DEFAULT_PAGE_MAX_SIDE)]
    max_side: u32,
    #[arg(long, default_value_t = 72)]
    dpi: u32,
    /// Maximum number of crops per recognizer batch.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    /// OAR text-line orientation model used to correct upside-down crop images.
    #[arg(long, default_value = "pp-lcnet_x0_25_textline_ori.onnx")]
    orientation_model: PathBuf,
    /// Disable optional text-line orientation classification.
    #[arg(long)]
    no_cls: bool,
    /// Preserve searchable native text in the diagnostic raster.
    #[arg(long)]
    keep_native_text: bool,
}

/// Lazily loaded recognition and orientation models; never detect text again.
struct OnnxRecognizer {
    recognition: TextRecognitionPredictor,
    orientation: Option<TextLineOrientationPredictor>,
}

impl OnnxRecognizer {
    fn load(args: &Args) -> Result<Self, pdf_redox_ocr::Error> {
        let resolve = |path: &PathBuf| {
            resolve_path(path).map_err(|e| pdf_redox_ocr::Error::Inference(e.to_string()))
        };
        let recognition = TextRecognitionPredictor::builder()
            .dict_path(resolve(&args.character_dict)?)
            .build(resolve(&args.secondary_model)?)
            .map_err(|e| pdf_redox_ocr::Error::Inference(e.to_string()))?;
        let orientation = if args.no_cls {
            None
        } else {
            Some(
                TextLineOrientationPredictor::builder()
                    .input_shape((80, 160))
                    .build(resolve(&args.orientation_model)?)
                    .map_err(|e| pdf_redox_ocr::Error::Inference(e.to_string()))?,
            )
        };
        Ok(Self {
            recognition,
            orientation,
        })
    }

    fn predict(&self, images: Vec<RgbImage>) -> Result<Vec<Recognition>, pdf_redox_ocr::Error> {
        let images = if let Some(orientation) = &self.orientation {
            let results = orientation
                .predict(images.clone())
                .map_err(|e| pdf_redox_ocr::Error::Inference(e.to_string()))?;
            if results.orientations.len() != images.len() {
                return Err(pdf_redox_ocr::Error::Inference(
                    "line orientation classifier returned wrong batch size".into(),
                ));
            }
            images
                .into_iter()
                .zip(results.orientations)
                .map(
                    |(image, predictions)| match predictions.first().map(|p| p.class_id) {
                        Some(1) => Ok(imageops::rotate180(&image)),
                        None | Some(0) => Ok(image),
                        Some(other) => Err(pdf_redox_ocr::Error::Inference(format!(
                            "invalid line orientation class {other}"
                        ))),
                    },
                )
                .collect::<Result<Vec<_>, _>>()?
        } else {
            images
        };
        let results = self
            .recognition
            .predict(images)
            .map_err(|e| pdf_redox_ocr::Error::Inference(e.to_string()))?;
        if results.texts.len() != results.scores.len() {
            return Err(pdf_redox_ocr::Error::Inference(
                "secondary recognizer returned inconsistent texts and confidences".into(),
            ));
        }
        Ok(results
            .texts
            .into_iter()
            .zip(results.scores)
            .map(|(text, confidence)| Recognition {
                text,
                confidence: f64::from(confidence),
            })
            .collect())
    }
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
        batch_size: args.batch_size,
        render: RenderPolicy {
            max_side: args.max_side,
            dpi: args.dpi,
        },
    }
    .validate()?;
    let plan = refinement_plan(&source, scope, metadata.page_count(), policy)?;
    let candidate_pages = plan.keys().copied().collect::<Vec<_>>();
    let mut masked_pdf = None;
    if !candidate_pages.is_empty()
        && !args.keep_native_text
        && source.pages.iter().any(|page| {
            plan.keys().any(|candidate| candidate.get() == page.page)
                && !page.native_text_regions.is_empty()
        })
    {
        let masking = mask_native_fonts(&metadata, &candidate_pages)?;
        if let Some(bytes) = masking.bytes {
            let mut temporary = NamedTempFile::new()?;
            temporary.write_all(&bytes)?;
            temporary.flush()?;
            masked_pdf = Some(temporary);
        }
    }
    let render_pdf = masked_pdf
        .as_ref()
        .map_or(args.pdf.as_path(), NamedTempFile::path);
    let label = format!(
        "OAR PP-OCRv6 recognition-only {}, line orientation {}",
        args.secondary_model.display(),
        if args.no_cls {
            "disabled".to_owned()
        } else {
            args.orientation_model.display().to_string()
        }
    );
    let mut predictor: Option<OnnxRecognizer> = None;
    let (report, experimental) = refine_draft(
        RefinementRequest {
            pdf: &args.pdf,
            render_pdf,
            metadata: &metadata,
            source: &source,
            scope,
            secondary_model: &label,
            policy,
        },
        |images| {
            if predictor.is_none() {
                predictor = Some(OnnxRecognizer::load(args)?);
            }
            predictor
                .as_ref()
                .ok_or_else(|| {
                    pdf_redox_ocr::Error::Inference("missing secondary recognizer".into())
                })?
                .predict(images)
        },
    )?;
    let mut report_bytes = serde_json::to_vec_pretty(&report)?;
    report_bytes.push(b'\n');
    let mut draft_bytes = serde_json::to_vec_pretty(&experimental)?;
    draft_bytes.push(b'\n');
    write_atomic(&args.report, &report_bytes)?;
    write_atomic(&args.experimental_draft, &draft_bytes)?;
    eprintln!(
        "recognition-only refinement: {} crops, {} differing predictions across {} pages; experimental confidences all zero",
        report.summary.selected_lines,
        report.summary.revised_predictions,
        report.summary.pages_refined
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
