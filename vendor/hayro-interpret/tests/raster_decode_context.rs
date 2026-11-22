//! Regression tests for resolved raster decoding identities.

use hayro_interpret::{
    BlendMode, ClipPath, Context, Device, DrawMode, DrawProps, Image, ImageData, ImageDrawProps,
    InterpreterCache, InterpreterSettings, RasterDecodeContext, SoftMask,
    font::GlyphRun,
    interpret, interpret_page,
    kurbo::{Affine, BezPath, Rect},
};
use hayro_syntax::Pdf;
use std::io::Write;

fn stream(dictionary: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = format!("<< {dictionary} /Length {} >>\nstream\n", data.len()).into_bytes();
    bytes.extend(data);
    bytes.extend(b"\nendstream");
    bytes
}

fn pdf(objects: &[Vec<u8>]) -> Pdf {
    let mut bytes = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(bytes.len());
        writeln!(bytes, "{} 0 obj", index + 1).unwrap();
        bytes.extend(object);
        bytes.extend(b"\nendobj\n");
    }
    let xref = bytes.len();
    let size = objects.len() + 1;
    write!(bytes, "xref\n0 {size}\n0000000000 65535 f \n").unwrap();
    for offset in offsets {
        writeln!(bytes, "{offset:010} 00000 n ").unwrap();
    }
    write!(
        bytes,
        "trailer\n<< /Root 1 0 R /Size {size} >>\nstartxref\n{xref}\n%%EOF\n"
    )
    .unwrap();
    Pdf::new(bytes).unwrap()
}

struct Observation {
    context: Option<RasterDecodeContext>,
    rgb: Vec<u8>,
    path: Option<Vec<u32>>,
}

#[derive(Default)]
struct Capture(Vec<Observation>);

impl Device<'_> for Capture {
    fn draw_path(&mut self, _: &BezPath, _: DrawProps<'_>, _: &DrawMode) {}
    fn push_clip_path(&mut self, _: &ClipPath) {}
    fn pop_clip(&mut self) {}
    fn push_transparency_group(
        &mut self,
        _: f32,
        _: Option<SoftMask<'_>>,
        _: BlendMode,
        _: bool,
        _: bool,
    ) {
    }
    fn pop_transparency_group(&mut self) {}
    fn draw_glyph_run(&mut self, _: &GlyphRun<'_, '_>, _: DrawProps<'_>, _: &DrawMode) {}

    fn draw_image(&mut self, image: Image<'_, '_>, props: ImageDrawProps<'_>) {
        if let Image::Raster(raster) = image {
            let context = raster.decode_context(None);
            if context.is_some() {
                assert_ne!(context, raster.decode_context(Some((1, 1))));
            }
            raster.with_rgba(
                |image, alpha| {
                    assert!(alpha.is_none());
                    let rgb = match image {
                        ImageData::Rgb(rgb) => rgb.data,
                        ImageData::Luma(luma) => {
                            luma.data.into_iter().flat_map(|v| [v; 3]).collect()
                        }
                    };
                    self.0.push(Observation {
                        context,
                        rgb,
                        path: props.page_xobject_path.map(|path| path.into_vec()),
                    });
                },
                None,
            );
        }
    }
}

#[test]
fn decoding_context_tracks_resolved_color_and_transfer_across_pages() {
    let doc = pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /XObject << /A 7 0 R /B 8 0 R /C 9 0 R /Other 11 0 R >> /ExtGState << /Invert << /TR2 13 0 R >> /Four << /TR2 [13 0 R 13 0 R 13 0 R 13 0 R] >> >> >> /Contents 5 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /XObject << /A 7 0 R /B 8 0 R >> >> /Contents 6 0 R >>".to_vec(),
        stream("", b"/A Do /B Do /C Do q /Invert gs /A Do Q q /Four gs /A Do Q /A Do /Other Do BI /W 1 /H 1 /CS /RGB /BPC 8 ID abc EI"),
        stream("", b"/A Do /B Do"),
        stream("/Type /XObject /Subtype /Form /BBox [0 0 2 2] /Resources << /ColorSpace << /Local 12 0 R >> /XObject << /Im 10 0 R >> >>", b"/Im Do"),
        stream("/Type /XObject /Subtype /Form /BBox [0 0 2 2] /Resources << /ColorSpace << /Local [/Indexed /DeviceRGB 1 <0000ffffffff>] >> /XObject << /Im 10 0 R >> >>", b"/Im Do"),
        stream("/Type /XObject /Subtype /Form /BBox [0 0 2 2] /Resources << /ColorSpace << /Local [/Indexed /DeviceRGB 1 <ff000000ff00>] >> /XObject << /Im 10 0 R >> >>", b"/Im Do"),
        stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /Local /BitsPerComponent 8", &[0, 1]),
        stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace [/Indexed /DeviceRGB 1 <ff000000ff00>] /BitsPerComponent 8", &[0, 1]),
        b"[/Indexed /DeviceRGB 1 <ff000000ff00>]".to_vec(),
        b"<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>".to_vec(),
    ]);
    let mut capture = Capture::default();
    for page in doc.pages().iter() {
        let cache = InterpreterCache::new();
        let mut context = Context::new(
            Affine::IDENTITY,
            Rect::new(0.0, 0.0, 100.0, 100.0),
            &cache,
            doc.xref(),
            InterpreterSettings::default(),
        );
        interpret_page(page, &mut context, &mut capture);
    }
    let rows = &capture.0;
    assert_eq!(rows.len(), 10);
    let expected: &[Option<&[u32]>] = &[
        Some(&[0, 0]),
        Some(&[1, 0]),
        Some(&[2, 0]),
        Some(&[3, 0]),
        Some(&[4, 0]),
        Some(&[5, 0]),
        Some(&[6]),
        None,
        Some(&[0, 0]),
        Some(&[1, 0]),
    ];
    assert_eq!(
        rows.iter()
            .map(|row| row.path.as_deref())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(rows[0].rgb, [255, 0, 0, 0, 255, 0]);
    assert_eq!(rows[1].rgb, [0, 0, 255, 255, 255, 255]);
    assert_eq!(rows[3].rgb, [0, 255, 255, 255, 0, 255]);
    assert_eq!(rows[0].context, rows[2].context);
    assert_eq!(rows[0].context, rows[5].context);
    assert_eq!(rows[0].context, rows[8].context);
    assert_eq!(rows[1].context, rows[9].context);
    assert_ne!(rows[0].context, rows[1].context);
    assert_ne!(rows[0].context, rows[3].context);
    assert_ne!(rows[3].context, rows[4].context);
    assert_eq!(rows[3].rgb, rows[4].rgb);
    assert_ne!(rows[0].context, rows[6].context);
    assert_eq!(rows[0].rgb, rows[6].rgb);
    assert!(rows[7].context.is_none());
    assert!(
        rows.iter()
            .enumerate()
            .all(|(i, row)| i == 7 || row.context.is_some())
    );
}

#[test]
fn page_xobject_paths_count_skipped_invocations_and_exclude_other_programs() {
    let doc = pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R /OCProperties << /OCGs [11 0 R] /D << /OFF [11 0 R] >> >> >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << /XObject << /F 6 0 R /Hidden 8 0 R /Im 7 0 R >> >> /Contents [4 0 R 5 0 R] /Annots [9 0 R] >>".to_vec(),
        stream("", b"Do /Missing Do /Hidden Do /F Do"),
        stream("", b"/F Do /Im Do"),
        stream("/Type /XObject /Subtype /Form /BBox [0 0 2 2] /Resources << /XObject << /Im 7 0 R >> >>", b"123 Do /Missing Do BI /W 1 /H 1 /CS /RGB /BPC 8 ID abc EI /Im Do"),
        stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8", &[255,0,0]),
        stream("/Type /XObject /Subtype /Form /BBox [0 0 2 2] /OC 11 0 R /Resources << /XObject << /Im 7 0 R >> >>", b"/Im Do"),
        b"<< /Type /Annot /Subtype /Stamp /Rect [0 0 2 2] /AP << /N 10 0 R >> >>".to_vec(),
        stream("/Type /XObject /Subtype /Form /BBox [0 0 2 2] /Resources << /XObject << /Im 7 0 R >> >>", b"/Im Do"),
        b"<< /Type /OCG /Name (Hidden) >>".to_vec(),
    ]);
    let cache = InterpreterCache::new();
    let mut context = Context::new(
        Affine::IDENTITY,
        Rect::new(0.0, 0.0, 100.0, 100.0),
        &cache,
        doc.xref(),
        InterpreterSettings::default(),
    );
    let page = &doc.pages()[0];
    let mut capture = Capture::default();
    interpret_page(page, &mut context, &mut capture);
    let paths: Vec<_> = capture.0.iter().map(|row| row.path.as_deref()).collect();
    let expected: &[Option<&[u32]>] = &[None, Some(&[3, 2]), None, Some(&[4, 2]), Some(&[5]), None];
    assert_eq!(paths, expected);
    assert_eq!(capture.0[1].rgb, [255, 0, 0]);
    assert_eq!(capture.0[5].rgb, [255, 0, 0]);
    capture.0.clear();
    interpret(
        page.typed_operations(),
        page.resources(),
        &mut context,
        &mut capture,
    );
    assert_eq!(capture.0.len(), 5);
    assert!(capture.0.iter().all(|row| row.path.is_none()));
}
