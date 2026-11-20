//! Exercise native-only paint closure, the three-way page partition and
//! native-word preservation using actual PDFs and Poppler without a model.

#![cfg(feature = "pdf-integration")]

use std::{cell::RefCell, fs};

use image::RgbImage;
use pdf_redox_ocr::{
    Error, PageNumber,
    draft::OcrLine,
    geometry::Point,
    hybrid::PageRoute,
    paint_scan::{NativeOnlyBlocker, enumerate_image_paints},
    pdf_metadata::PdfMetadata,
    pipeline::{PipelinePolicy, PreparedOcr},
    rasterize::RenderPolicy,
    recognition::RecognitionBackend,
};

fn stream(dictionary: &str, bytes: &[u8]) -> Vec<u8> {
    [
        format!("<< {dictionary} /Length {} >>\nstream\n", bytes.len()).as_bytes(),
        bytes,
        b"\nendstream",
    ]
    .concat()
}

/// A page per distinct paint-closure condition, all sharing actual resources.
fn fixture() -> Vec<u8> {
    const CONTENTS: &[&[u8]] = &[
        b"q /RGB cs 0 0 0 sc /Span BMC BT /F 12 Tf 20 150 Td (Native words) Tj ET EMC Q",
        b"q /Outer Do Q",
        b"",
        b"BT /F 12 Tf 20 150 Td (Vector label) Tj ET 20 30 m 60 30 l S",
        b"/PatCS cs /P scn BT /F 30 Tf 20 150 Td (Pattern text) Tj ET",
        b"q 10 10 180 180 re W n BT /F 12 Tf 20 150 Td (Clipped native) Tj ET Q",
        b"/GS gs BT /F 12 Tf 20 150 Td (Extended state) Tj ET",
        b"BT /T3 16 Tf 20 150 Td (A) Tj ET",
        b"BT /F 12 Tf 20 150 Td (Annotated) Tj ET",
        b"BT /F 12 Tf 20 150 Td ( ) Tj ET",
        b"q 20 0 0 20 10 30 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x00 EI Q",
        b"EMC BT /F 12 Tf 20 150 Td (Unbalanced) Tj ET",
        b"BT /F 12 Tf 3 Tr 20 150 Td (Invisible text) Tj ET",
        b"q 80 0 0 20 10 30 cm /Im Do Q",
        b"q /Inner Do Q",
    ];
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        Vec::new(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        stream(
            "/Type /XObject /Subtype /Form /BBox [0 0 200 200]",
            b"q /Inner Do Q",
        ),
        stream(
            "/Type /XObject /Subtype /Form /BBox [0 0 200 200]",
            b"BT /F 12 Tf 20 150 Td (Native form) Tj ET",
        ),
        b"<< /Type /ExtGState /ca 1 /CA 1 >>".to_vec(),
        stream(
            "/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 4 4] /XStep 4 /YStep 4 /Resources << >>",
            b"0 g 0 0 2 2 re f",
        ),
        b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 600 600] /FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << /A 9 0 R >> /Encoding << /Type /Encoding /Differences [65 /A] >> /FirstChar 65 /LastChar 65 /Widths [600] /Resources << >> >>".to_vec(),
        stream("", b"600 0 0 0 600 600 d1 0 0 600 600 re f"),
        b"<< /Type /Annot /Subtype /Square /Rect [30 30 60 60] /C [1 0 0] >>".to_vec(),
        stream(
            "/Type /XObject /Subtype /Image /Width 80 /Height 20 /BitsPerComponent 8 /ColorSpace /DeviceGray",
            &[0; 1600],
        ),
    ];
    let mut kids = String::new();
    for (index, contents) in CONTENTS.iter().enumerate() {
        use std::fmt::Write as _;
        let page_id = objects.len() + 1;
        let content_id = page_id + 1;
        let annotation = if index == 8 { "/Annots [10 0 R]" } else { "" };
        let group = if index == 14 {
            "/Group << /S /Transparency /CS /DeviceRGB >>"
        } else {
            ""
        };
        objects.push(
            format!(
                "<< /Type /Page /Parent 2 0 R /Contents {content_id} 0 R {annotation} {group} >>"
            )
            .into_bytes(),
        );
        objects.push(stream("", contents));
        let _ = write!(kids, "{page_id} 0 R ");
    }
    objects[1] = format!(
        "<< /Type /Pages /Kids [{kids}] /Count {} /MediaBox [0 0 200 200] /Resources << /Font << /F 3 0 R /T3 8 0 R >> /XObject << /Outer 4 0 R /Inner 5 0 R /Im 11 0 R >> /ColorSpace << /RGB /DeviceRGB /PatCS /Pattern >> /ExtGState << /GS 6 0 R >> /Pattern << /P 7 0 R >> >> >>",
        CONTENTS.len()
    ).into_bytes();
    let mut bytes = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (index, object) in objects.iter().enumerate() {
        offsets.push(bytes.len());
        bytes.extend(format!("{} 0 obj\n", index + 1).as_bytes());
        bytes.extend(object);
        bytes.extend(b"\nendobj\n");
    }
    let xref = bytes.len();
    let size = objects.len() + 1;
    bytes.extend(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for offset in offsets {
        bytes.extend(format!("{offset:010} 00000 n \n").as_bytes());
    }
    bytes.extend(
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    bytes
}

#[derive(Default)]
struct Recognizer(RefCell<Vec<(u32, u32)>>);

impl RecognitionBackend for Recognizer {
    fn model_profile(&self) -> &'static str {
        "native-only fixture"
    }

    fn recognize_mapped(
        &self,
        image: RgbImage,
        to_pdf: &dyn Fn(Point) -> Point,
    ) -> Result<Vec<OcrLine>, Error> {
        let dimensions = image.dimensions();
        self.0.borrow_mut().push(dimensions);
        if dimensions != (80, 20) {
            return Ok(Vec::new());
        }
        Ok(vec![OcrLine {
            text: "Raster source".into(),
            confidence: 0.99,
            quad: [
                Point(10.0, 15.0),
                Point(70.0, 15.0),
                Point(70.0, 5.0),
                Point(10.0, 5.0),
            ]
            .map(to_pdf),
            reviewed: false,
        }])
    }
}

#[test]
fn native_only_certificate_rejects_indirect_paint_and_invalid_state()
-> Result<(), Box<dyn std::error::Error>> {
    let document = PdfMetadata::from_bytes(fixture())?;
    let pages = (1..=15)
        .map(PageNumber::new)
        .collect::<Result<Vec<_>, _>>()?;
    let scan = enumerate_image_paints(&document, &pages)?;
    let certified = scan
        .pages
        .iter()
        .filter(|p| p.native_text_only())
        .map(|p| p.page)
        .collect::<Vec<_>>();
    // An empty show is structurally native-only, but lacks extraction evidence.
    assert_eq!(certified, [1, 2, 3, 10, 13], "{:#?}", scan.pages);
    assert_eq!(scan.pages[1].forms_visited, 2);
    assert!(
        scan.pages[4]
            .native_only_blockers
            .contains(&NativeOnlyBlocker::ColorPattern)
    );
    assert!(
        scan.pages[5]
            .native_only_blockers
            .contains(&NativeOnlyBlocker::Clipping)
    );
    assert!(
        scan.pages[6]
            .native_only_blockers
            .contains(&NativeOnlyBlocker::ExtendedGraphicsState)
    );
    assert!(
        scan.pages[11]
            .native_only_blockers
            .contains(&NativeOnlyBlocker::StreamState)
    );
    assert!(
        scan.pages[14]
            .native_only_blockers
            .contains(&NativeOnlyBlocker::PageContext)
    );
    Ok(())
}

#[test]
fn all_three_routes_preserve_selected_order_and_native_words()
-> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("native-only.pdf");
    fs::write(&path, fixture())?;
    let doc = PreparedOcr::open(
        &path,
        Some("1-15"),
        PipelinePolicy {
            render: RenderPolicy {
                max_side: 300,
                dpi: 72,
            },
            ..PipelinePolicy::default()
        },
    )?;
    let model = Recognizer::default();
    let result = doc.recognize(&model, |_| {})?;
    assert_eq!(result.stats.native_only_pages, 4, "{:?}", result.routes);
    assert_eq!(result.stats.image_source_only_pages, 1);
    assert_eq!(result.stats.page_fallback_pages, 10);
    assert_eq!(result.draft.pages.len(), 15);
    assert_eq!(model.0.borrow().len(), 11, "{:?}", model.0.borrow());
    assert_eq!(
        model
            .0
            .borrow()
            .iter()
            .filter(|&&size| size == (80, 20))
            .count(),
        1
    );
    for (index, page) in result.draft.pages.iter().enumerate() {
        let number = PageNumber::new(u32::try_from(index + 1)?)?;
        assert_eq!(page.page, number.get());
        let route = match page.page {
            1 | 2 | 3 | 13 => PageRoute::NativeOnly,
            14 => PageRoute::RasterOnly,
            _ => PageRoute::PageFallback,
        };
        assert_eq!(result.routes[&number], route, "page {}", page.page);
        if page.page == 14 {
            assert_eq!(page.lines.len(), 1);
            assert_eq!(page.lines[0].text, "Raster source");
        } else {
            assert_eq!(page.lines.len(), 0);
        }
        if matches!(page.page, 1 | 2 | 13) {
            assert!(!page.native_text_regions.is_empty());
        }
    }
    result.draft.verify_pdf(&path)?;
    assert_eq!(fs::read(&path)?, fixture());
    Ok(())
}

#[test]
fn native_only_selection_and_keep_native_text_control_recognition()
-> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("native-only.pdf");
    fs::write(&path, fixture())?;
    let policy = PipelinePolicy {
        render: RenderPolicy {
            max_side: 300,
            dpi: 72,
        },
        ..PipelinePolicy::default()
    };
    let doc = PreparedOcr::open(&path, Some("1-3,13"), policy)?;
    let recognizer = Recognizer::default();
    let result = doc.recognize(&recognizer, |_| {})?;
    assert_eq!(recognizer.0.borrow().len(), 0);
    assert_eq!(result.stats.native_only_pages, 4);
    let forced = PreparedOcr::open(
        &path,
        Some("1-3,13"),
        PipelinePolicy {
            keep_native_text: true,
            ..policy
        },
    )?;
    let result = forced.recognize(&recognizer, |_| {})?;
    assert_eq!(result.stats.page_fallback_pages, 4);
    assert_eq!(recognizer.0.borrow().len(), 4);
    assert!(!result.draft.pages[0].native_text_regions.is_empty());
    Ok(())
}

#[cfg(all(feature = "cli", feature = "native-inference"))]
#[test]
fn native_cli_needs_no_model_cache_or_onnx_session() -> Result<(), Box<dyn std::error::Error>> {
    use pdf_redox_ocr::draft::OcrDraft;
    use std::process::Command;

    let temp = tempfile::tempdir()?;
    let pdf = temp.path().join("original.pdf");
    let json = temp.path().join("draft.json");
    let cache = temp.path().join("unusable-model-cache");
    fs::write(&pdf, fixture())?;
    fs::write(&cache, b"a file is not a model-cache directory")?;
    // OAR would fail before any download if model initialization ran.
    let process = Command::new(env!("CARGO_BIN_EXE_pdf-redox-ocr"))
        .current_dir(temp.path())
        .env("OAR_HOME", &cache)
        .arg("--pdf")
        .arg(&pdf)
        .arg("--output")
        .arg(&json)
        .args(["--pages", "1-3,13"])
        .output()?;
    let stderr = String::from_utf8_lossy(&process.stderr);
    assert!(process.status.success(), "{stderr}");
    assert!(stderr.contains("4 native-only pages, 0 source-only pages, 0 fallback pages"));
    assert!(!stderr.contains("native-font render mask"));
    let draft: OcrDraft = serde_json::from_slice(&fs::read(json)?)?;
    assert_eq!(
        draft.pages.iter().map(|p| p.page).collect::<Vec<_>>(),
        [1, 2, 3, 13]
    );
    assert!(draft.pages.iter().all(|p| p.lines.is_empty()));
    draft.verify_pdf(&pdf)?;
    assert_eq!(fs::read(cache)?, b"a file is not a model-cache directory");
    Ok(())
}
