//! Exact image paint occurrences using Hayro's native content instruction parser.
//!
//! Resource-table `XObjects` are not paint: they count only when referenced by an
//! actual Do operation. Page and nested Form graphics state remain distinct.
//! Unsupported PDF visual instructions never certify a raster-only page.

use std::collections::BTreeSet;

use hayro_syntax::{content::UntypedIter, object::Object as HayroObject};
use pdf_redox::{ObjectHandle, ObjectId, OwnedDictionary, OwnedObject, SourcePdf};
use serde::Serialize;

use crate::{
    Error, PageNumber,
    geometry::{PageBox, Point, Quad},
    pdf_metadata::{PdfMetadata, resolve_owned},
    polygon::area,
};

const MAX_FORM_DEPTH: usize = 32;
const MAX_CONTENT_STREAM_NEST: usize = 64;

/// A linear affine PDF transformation (a,b,c,d,e,f).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PdfMatrix(pub [f64; 6]);

impl PdfMatrix {
    /// Identity current-transform matrix.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    /// Compose the outer state transform with a new inner transformation.
    #[must_use]
    pub fn concat(self, inner: Self) -> Self {
        let outer = self.0;
        let inner = inner.0;
        Self([
            outer[2].mul_add(inner[1], outer[0] * inner[0]),
            outer[3].mul_add(inner[1], outer[1] * inner[0]),
            outer[2].mul_add(inner[3], outer[0] * inner[2]),
            outer[3].mul_add(inner[3], outer[1] * inner[2]),
            outer[2].mul_add(inner[5], outer[0].mul_add(inner[4], outer[4])),
            outer[3].mul_add(inner[5], outer[1].mul_add(inner[4], outer[5])),
        ])
    }

    /// Transform a point in the unit square into PDF user space.
    #[must_use]
    pub const fn point(self, Point(point_x, point_y): Point) -> Point {
        let matrix = self.0;
        Point(
            matrix[2].mul_add(point_y, matrix[0].mul_add(point_x, matrix[4])),
            matrix[3].mul_add(point_y, matrix[1].mul_add(point_x, matrix[5])),
        )
    }

    /// Transform an axis-aligned rectangle into its original PDF-space quad.
    #[must_use]
    pub const fn rect_quad(self, bounds: PageBox) -> Quad {
        [
            self.point(Point(bounds.x0, bounds.y0)),
            self.point(Point(bounds.x1, bounds.y0)),
            self.point(Point(bounds.x1, bounds.y1)),
            self.point(Point(bounds.x0, bounds.y1)),
        ]
    }

    /// Return the absolute determinant of this CTM.
    #[must_use]
    pub fn area_scale(self) -> f64 {
        let [a, b, c, d, _, _] = self.0;
        b.mul_add(-c, a * d).abs()
    }
}

impl TryFrom<&[f64]> for PdfMatrix {
    type Error = Error;

    fn try_from(values: &[f64]) -> Result<Self, Self::Error> {
        let values: [f64; 6] = values
            .try_into()
            .map_err(|_| Error::Invalid("PDF CTM needs six numbers".into()))?;
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Error::Invalid("nonfinite PDF CTM".into()));
        }
        Ok(Self(values))
    }
}

/// A native Image `XObject` invoked at a particular paint position.
#[derive(Debug, Clone, Serialize)]
pub struct ImagePaint {
    /// Original one-based page index.
    pub page: u32,
    /// Paint invocation index expanded through nested Forms.
    pub paint_index: u32,
    /// Zero-based Do ordinal at each invocation from the page through nested Forms.
    pub paint_path: Vec<u32>,
    /// Original image `XObject` identifier.
    pub source_id: String,
    /// Resolved source object number.
    pub object_number: u32,
    /// Resolved source generation.
    pub generation: u16,
    /// Image native pixel width.
    pub width: u32,
    /// Image native pixel height.
    pub height: u32,
    /// PDF user-space CTM at paint invocation.
    pub ctm: [f64; 6],
    /// Image unit square in PDF user coordinates.
    pub page_quad: Quad,
    /// Ordered convex graphics clip regions.
    pub clips: Vec<Quad>,
    /// All reasons forbidding independent source-image OCR.
    pub unsafe_reasons: Vec<&'static str>,
}

impl ImagePaint {
    /// PDF page-space quad for an OCR box on the source's native pixel grid.
    ///
    /// # Errors
    /// Rejects invalid or nonfinite coordinates, including outside the source raster.
    pub fn source_box_to_pdf(&self, pixels: &Quad) -> Result<Quad, Error> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::Invalid("invalid source dimensions".into()));
        }
        let matrix = PdfMatrix(self.ctm);
        let to_pdf = |point: Point| -> Result<Point, Error> {
            if !point.0.is_finite()
                || !point.1.is_finite()
                || point.0 < 0.0
                || point.1 < 0.0
                || point.0 > f64::from(self.width)
                || point.1 > f64::from(self.height)
            {
                return Err(Error::Invalid(
                    "image OCR point outside native raster".into(),
                ));
            }
            Ok(matrix.point(Point(
                point.0 / f64::from(self.width),
                1.0 - point.1 / f64::from(self.height),
            )))
        };
        Ok([
            to_pdf(pixels[3])?,
            to_pdf(pixels[2])?,
            to_pdf(pixels[1])?,
            to_pdf(pixels[0])?,
        ])
    }

    /// Every vertex must be within every known clipping polygon.
    #[must_use]
    pub fn wholly_visible_in_clip(&self, quad: &Quad) -> bool {
        self.unsafe_reasons.is_empty()
            && self
                .clips
                .iter()
                .all(|clip| quad.iter().all(|point| inside_convex(*point, clip, 1e-3)))
    }
}

/// Source paint totals and page purity evidence.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PagePaintSummary {
    /// One-based page number.
    pub page: u32,
    /// Total actual Image `XObject` paints.
    pub image_paints: usize,
    /// Image paints requiring full page fallback.
    pub unsafe_image_paints: usize,
    /// Nested Form `XObjects` visited.
    pub forms_visited: usize,
    /// Visible searchable text paint operations.
    pub native_searchable_shows: usize,
    /// Visual paints that cannot be independently decoded as source raster.
    pub non_image_visual_paint: usize,
    /// Unsupported or invalid PDF operators.
    pub unknown_paint: usize,
    /// Unsupported alpha/blend/overprint graphics state.
    pub unsupported_extgs: usize,
    /// Inline images whose occurrence has no standalone `XObject`.
    pub unsupported_inline_images: usize,
    /// Recursive or excessively nested Form paint.
    pub unsafe_form_recursion: usize,
    /// PDF graphics-state Q stack underflow.
    pub unbalanced_q: usize,
    /// Unterminated PDF clipping or state stacks.
    pub unbalanced_stream: usize,
    /// Annotations or other separately rendered page content.
    pub annotations: usize,
    /// Source paints alone account for all visible page ink except native text.
    pub image_only_after_native: bool,
}

impl PagePaintSummary {
    const fn certify(&mut self) {
        self.image_only_after_native = self.image_paints > 0
            && self.unsafe_image_paints == 0
            && self.non_image_visual_paint == 0
            && self.unknown_paint == 0
            && self.unsupported_extgs == 0
            && self.unsupported_inline_images == 0
            && self.unsafe_form_recursion == 0
            && self.unbalanced_q == 0
            && self.unbalanced_stream == 0
            && self.annotations == 0;
    }
}

/// All source image appearances and per-page safety diagnostics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PaintReport {
    /// Image `XObject` actual paint occurrences.
    pub placements: Vec<ImagePaint>,
    /// One summary per requested PDF page.
    pub pages: Vec<PagePaintSummary>,
}

#[derive(Clone, Copy, Debug, Default)]
enum SelectedFont {
    #[default]
    Unsearchable,
    Searchable,
}

#[derive(Clone, Debug)]
struct State {
    ctm: PdfMatrix,
    clips: Vec<Quad>,
    clip_unknown: bool,
    gs_unsafe: bool,
    // The PDF graphics state holds an already selected *font object*.
    // Once Tf executes, nested Forms must inherit its classification even
    // when their own Resources reuse the same font name for another font.
    font: SelectedFont,
    render_mode: u8,
    text_open: bool,
}

impl State {
    fn for_page(box_: PageBox) -> Self {
        Self {
            ctm: PdfMatrix::IDENTITY,
            clips: vec![PdfMatrix::IDENTITY.rect_quad(box_)],
            clip_unknown: false,
            gs_unsafe: false,
            font: SelectedFont::default(),
            render_mode: 0,
            text_open: false,
        }
    }
}

#[derive(Default)]
struct PathState {
    rects: Vec<Quad>,
    subpaths: Vec<Vec<Point>>,
    curved_or_unknown: bool,
    pending_clip: bool,
}

impl PathState {
    fn reset(&mut self) {
        self.rects.clear();
        self.subpaths.clear();
        self.curved_or_unknown = false;
        self.pending_clip = false;
    }

    fn apply_clip(&self, state: &mut State) {
        if !self.pending_clip {
            return;
        }
        if !self.curved_or_unknown && self.rects.len() == 1 && self.subpaths.is_empty() {
            state.clips.push(self.rects[0]);
        } else if !self.curved_or_unknown && self.rects.is_empty() && self.subpaths.len() == 1 {
            if let Some(quad) = rectangle_path_quad(&self.subpaths[0]) {
                state.clips.push(quad);
            } else {
                state.clip_unknown = true;
            }
        } else {
            state.clip_unknown = true;
        }
    }
    /// Parse path construction, clipping and paint without losing graphics-state scope.
    fn consume(&mut self, op: &[u8], arguments: Option<&[f64]>, state: &mut State) -> Option<bool> {
        match op {
            b"re" => {
                if let Some([x, y, width, height]) = arguments {
                    let x1 = x + width;
                    let y1 = y + height;
                    if let Ok(bounds) = PageBox::new([x.min(x1), y.min(y1), x.max(x1), y.max(y1)]) {
                        self.rects.push(state.ctm.rect_quad(bounds));
                    } else {
                        self.curved_or_unknown = true;
                    }
                } else {
                    self.curved_or_unknown = true;
                }
            }
            b"m" => {
                if let Some([x, y]) = arguments {
                    self.subpaths.push(vec![state.ctm.point(Point(*x, *y))]);
                } else {
                    self.curved_or_unknown = true;
                }
            }
            b"l" => {
                if let Some([x, y]) = arguments {
                    if let Some(points) = self.subpaths.last_mut() {
                        points.push(state.ctm.point(Point(*x, *y)));
                    } else {
                        self.curved_or_unknown = true;
                    }
                } else {
                    self.curved_or_unknown = true;
                }
            }
            b"h" => {
                if let Some(points) = self.subpaths.last_mut() {
                    if let Some(&first) = points.first() {
                        points.push(first);
                    } else {
                        self.curved_or_unknown = true;
                    }
                } else {
                    self.curved_or_unknown = true;
                }
            }
            b"c" | b"v" | b"y" => self.curved_or_unknown = true,
            b"W" | b"W*" => self.pending_clip = true,
            b"n" | b"S" | b"s" | b"f" | b"F" | b"f*" | b"B" | b"B*" | b"b" | b"b*" => {
                self.apply_clip(state);
                self.reset();
                return Some(op != b"n");
            }
            _ => return None,
        }
        Some(false)
    }
}

fn rectangle_path_quad(points: &[Point]) -> Option<Quad> {
    let mut cleaned = Vec::with_capacity(points.len());
    for &point in points {
        if cleaned.last().is_none_or(|previous: &Point| {
            (point.0 - previous.0).hypot(point.1 - previous.1) > 1e-5
        }) {
            cleaned.push(point);
        }
    }
    if cleaned.len() == 5 && (cleaned[0].0 - cleaned[4].0).hypot(cleaned[0].1 - cleaned[4].1) < 1e-4
    {
        cleaned.pop();
    }
    let quad: Quad = cleaned.try_into().ok()?;
    let lengths = (0..4)
        .map(|i| (quad[(i + 1) % 4].0 - quad[i].0).hypot(quad[(i + 1) % 4].1 - quad[i].1))
        .collect::<Vec<_>>();
    if lengths.iter().any(|&length| length < 1e-4) || area(&quad) < 1e-8 {
        return None;
    }
    for index in 0..4 {
        let begin = quad[index];
        let middle = quad[(index + 1) % 4];
        let end = quad[(index + 2) % 4];
        let edge_a = Point(middle.0 - begin.0, middle.1 - begin.1);
        let edge_b = Point(end.0 - middle.0, end.1 - middle.1);
        let dot = edge_a.0.mul_add(edge_b.0, edge_a.1 * edge_b.1);
        if dot.abs() > 0.025 * lengths[index] * lengths[(index + 1) % 4] {
            return None;
        }
    }
    let expected_fourth = Point(
        (quad[1].0 + quad[3].0) - quad[2].0,
        (quad[1].1 + quad[3].1) - quad[2].1,
    );
    if (expected_fourth.0 - quad[0].0).hypot(expected_fourth.1 - quad[0].1)
        > 0.02 * lengths.iter().copied().fold(0.0, f64::max)
    {
        return None;
    }
    Some(quad)
}

fn inside_convex(point: Point, polygon: &Quad, epsilon: f64) -> bool {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for index in 0..4 {
        let begin = polygon[index];
        let end = polygon[(index + 1) % 4];
        let cross =
            (end.0 - begin.0).mul_add(point.1 - begin.1, -(end.1 - begin.1) * (point.0 - begin.0));
        minimum = minimum.min(cross);
        maximum = maximum.max(cross);
    }
    minimum >= -epsilon || maximum <= epsilon
}

struct ScanContext<'a> {
    source: &'a SourcePdf,
    page: PageNumber,
    summary: PagePaintSummary,
    paintings: Vec<ImagePaint>,
    active_forms: BTreeSet<ObjectId>,
    do_path: Vec<u32>,
}

fn numeric(value: &OwnedObject) -> Option<f64> {
    match value {
        OwnedObject::Real(real) if real.is_finite() => Some(*real),
        OwnedObject::Integer(value) => value.to_string().parse::<f64>().ok(),
        _ => None,
    }
}

fn resolved_number(source: &SourcePdf, value: &OwnedObject) -> Result<f64, Error> {
    numeric(&resolve_owned(source, value.clone())?)
        .ok_or_else(|| Error::Invalid("PDF numeric resource expected".into()))
}

fn array_numbers<const COUNT: usize>(
    source: &SourcePdf,
    value: &OwnedObject,
) -> Result<[f64; COUNT], Error> {
    let OwnedObject::Array(items) = resolve_owned(source, value.clone())? else {
        return Err(Error::Invalid("PDF numeric array expected".into()));
    };
    if items.len() != COUNT {
        return Err(Error::Invalid("PDF numeric array length mismatch".into()));
    }
    let out = items
        .iter()
        .map(|item| resolved_number(source, item))
        .collect::<Result<Vec<_>, _>>()?;
    out.try_into()
        .map_err(|_| Error::Invalid("incorrect PDF numeric array".into()))
}

pub(crate) fn resolve_dictionary(
    source: &SourcePdf,
    value: &OwnedObject,
) -> Result<OwnedDictionary, Error> {
    resolve_owned(source, value.clone())?
        .as_dictionary()
        .cloned()
        .ok_or_else(|| Error::Invalid("PDF dictionary resource expected".into()))
}

pub(crate) fn inherited_page_resources(
    source: &SourcePdf,
    first: ObjectId,
) -> Result<OwnedDictionary, Error> {
    let mut visited = BTreeSet::new();
    let mut next = Some(first);
    while let Some(id) = next {
        if !visited.insert(id) {
            return Err(Error::Invalid("cyclic page-tree resources".into()));
        }
        let object = source.materialize(id)?;
        let dict = object
            .as_dictionary()
            .ok_or_else(|| Error::Invalid("page must be a dictionary".into()))?;
        if let Some(value) = dict.get(b"Resources".as_slice()) {
            return resolve_dictionary(source, value);
        }
        next = match dict.get(b"Parent".as_slice()) {
            Some(OwnedObject::Reference(ObjectHandle::Existing(parent))) => Some(*parent),
            None => None,
            _ => return Err(Error::Invalid("malformed page-tree parent".into())),
        };
    }
    Ok(OwnedDictionary::new())
}

fn image_source_id(source: ObjectId) -> String {
    format!("{}:{}", source.number(), source.generation())
}

fn image_dimensions(source: &SourcePdf, dict: &OwnedDictionary) -> Result<(u32, u32), Error> {
    let dim = |key: &[u8]| -> Result<u32, Error> {
        let value = dict
            .get(key)
            .ok_or_else(|| Error::Invalid("missing image dimension".into()))?;
        let OwnedObject::Integer(value) = resolve_owned(source, value.clone())? else {
            return Err(Error::Invalid("non-integer image dimension".into()));
        };
        u32::try_from(value).map_err(|_| Error::Invalid("invalid image dimension".into()))
    };
    Ok((dim(b"Width")?, dim(b"Height")?))
}

fn subtype(source: &SourcePdf, dict: &OwnedDictionary) -> Result<Option<Vec<u8>>, Error> {
    let Some(value) = dict.get(b"Subtype".as_slice()) else {
        return Ok(None);
    };
    match resolve_owned(source, value.clone())? {
        OwnedObject::Name(name) => Ok(Some(name)),
        _ => Err(Error::Invalid("PDF `XObject` subtype not a name".into())),
    }
}

pub(crate) fn font_searchable(
    source: &SourcePdf,
    resources: &OwnedDictionary,
    font_name: &[u8],
) -> Result<bool, Error> {
    let Some(fonts) = resources.get(b"Font".as_slice()) else {
        return Ok(false);
    };
    let fonts = resolve_dictionary(source, fonts)?;
    let Some(font) = fonts.get(font_name) else {
        return Ok(false);
    };
    font_object_searchable(source, font)
}

fn font_object_searchable(source: &SourcePdf, font: &OwnedObject) -> Result<bool, Error> {
    let font = resolve_dictionary(source, font)?;
    let kind = subtype(source, &font)?.unwrap_or_default();
    if kind.as_slice() == b"Type3" {
        return Ok(false);
    }
    if font.contains_key(b"ToUnicode".as_slice()) {
        return Ok(true);
    }
    if let Some(encoding) = font.get(b"Encoding".as_slice()) {
        let name = resolve_owned(source, encoding.clone())?;
        return Ok(matches!(
            name,
            OwnedObject::Name(ref value)
                if matches!(value.as_slice(), b"WinAnsiEncoding" | b"MacRomanEncoding" | b"StandardEncoding")
        ));
    }
    let basefont = match font.get(b"BaseFont".as_slice()) {
        Some(raw) => resolve_owned(source, raw.clone())?,
        None => OwnedObject::Null,
    };
    let OwnedObject::Name(name) = basefont else {
        return Ok(false);
    };
    let base = name.rsplit(|byte| *byte == b'+').next().unwrap_or(&name);
    Ok(kind.as_slice() == b"Type1"
        && matches!(
            base,
            b"Courier"
                | b"Courier-Bold"
                | b"Courier-Oblique"
                | b"Courier-BoldOblique"
                | b"Helvetica"
                | b"Helvetica-Bold"
                | b"Helvetica-Oblique"
                | b"Helvetica-BoldOblique"
                | b"Times-Roman"
                | b"Times-Bold"
                | b"Times-Italic"
                | b"Times-BoldItalic"
        ))
}

/// A graphics-state `Font` entry selects a font *object* and size independently
/// of any name previously used by `Tf`. A missing entry leaves the font intact.
pub(crate) fn extgstate_font_searchability(
    source: &SourcePdf,
    gs: &OwnedDictionary,
) -> Result<Option<bool>, Error> {
    let Some(raw) = gs.get(b"Font".as_slice()) else {
        return Ok(None);
    };
    let OwnedObject::Array(values) = resolve_owned(source, raw.clone())? else {
        return Err(Error::Invalid("graphics-state Font must be a pair".into()));
    };
    let [font, size]: [OwnedObject; 2] = values
        .try_into()
        .map_err(|_| Error::Invalid("graphics-state Font must have two elements".into()))?;
    let _ = resolved_number(source, &size)?;
    font_object_searchable(source, &font).map(Some)
}

fn opaque_extgs(source: &SourcePdf, gs: &OwnedDictionary) -> Result<bool, Error> {
    for (key, expected) in [
        (b"SMask".as_slice(), b"None".as_slice()),
        (b"BM".as_slice(), b"Normal".as_slice()),
        (b"TR".as_slice(), b"Identity".as_slice()),
    ] {
        if let Some(value) = gs.get(key)
            && !matches!(resolve_owned(source, value.clone())?, OwnedObject::Name(name) if name == expected)
        {
            return Ok(false);
        }
    }
    if let Some(value) = gs.get(b"TR2".as_slice())
        && !matches!(resolve_owned(source, value.clone())?, OwnedObject::Name(name) if name == b"Identity" || name == b"Default")
    {
        return Ok(false);
    }
    if let Some(value) = gs.get(b"HT".as_slice())
        && !matches!(resolve_owned(source, value.clone())?, OwnedObject::Name(name) if name == b"Default")
    {
        return Ok(false);
    }
    for key in [b"ca".as_slice(), b"CA"] {
        if let Some(value) = gs.get(key)
            && (resolved_number(source, value)? - 1.0).abs() > 1e-6
        {
            return Ok(false);
        }
    }
    for key in [b"OP".as_slice(), b"op", b"AIS"] {
        if let Some(value) = gs.get(key)
            && !matches!(
                resolve_owned(source, value.clone())?,
                OwnedObject::Boolean(false)
            )
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn append_content(
    source: &SourcePdf,
    value: &OwnedObject,
    out: &mut Vec<u8>,
    active: &mut BTreeSet<ObjectId>,
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_CONTENT_STREAM_NEST {
        return Err(Error::Invalid(
            "cyclic or deeply nested PDF contents".into(),
        ));
    }
    let append = |out: &mut Vec<u8>, data: &[u8]| {
        if out
            .last()
            .is_some_and(|last| !matches!(*last, b'\n' | b' '))
        {
            out.push(b'\n');
        }
        out.extend_from_slice(data);
    };
    match value {
        OwnedObject::Reference(ObjectHandle::Existing(id)) => {
            if !active.insert(*id) {
                return Err(Error::Invalid("cyclic PDF content stream".into()));
            }
            let object = source.materialize(*id)?;
            if matches!(object, OwnedObject::Stream { .. }) {
                let decoded = source.decoded_stream_data(*id)?;
                append(out, &decoded);
            } else {
                append_content(source, &object, out, active, depth + 1)?;
            }
            active.remove(id);
        }
        OwnedObject::Stream {
            data: pdf_redox::StreamData::Source(id),
            ..
        } => {
            append(out, &source.decoded_stream_data(*id)?);
        }
        OwnedObject::Array(values) => {
            for value in values {
                append_content(source, value, out, active, depth + 1)?;
            }
        }
        OwnedObject::Null => {}
        _ => {
            return Err(Error::Invalid(
                "unsupported direct PDF content stream".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn page_content(source: &SourcePdf, page: ObjectId) -> Result<Vec<u8>, Error> {
    let object = source.materialize(page)?;
    let dict = object
        .as_dictionary()
        .ok_or_else(|| Error::Invalid("PDF page must be dictionary".into()))?;
    let Some(contents) = dict.get(b"Contents".as_slice()) else {
        return Ok(Vec::new());
    };
    let mut output = Vec::new();
    append_content(source, contents, &mut output, &mut BTreeSet::new(), 0)?;
    Ok(output)
}

fn named_resource(
    source: &SourcePdf,
    resources: &OwnedDictionary,
    table: &[u8],
    name: &[u8],
) -> Result<Option<OwnedObject>, Error> {
    let Some(table) = resources.get(table) else {
        return Ok(None);
    };
    let table = resolve_dictionary(source, table)?;
    let Some(resource) = table.get(name) else {
        return Ok(None);
    };
    Ok(Some(resource.clone()))
}

fn instruction_numbers(
    instruction: &hayro_syntax::content::Instruction<'_, '_>,
) -> Option<Vec<f64>> {
    instruction
        .operands()
        .map(|operand| match operand {
            HayroObject::Number(value) if value.as_f64().is_finite() => Some(value.as_f64()),
            _ => None,
        })
        .collect()
}

fn instruction_name(instruction: &hayro_syntax::content::Instruction<'_, '_>) -> Option<Vec<u8>> {
    let mut ops = instruction.operands();
    let first = ops.next()?;
    if ops.next().is_some() {
        return None;
    }
    match first {
        HayroObject::Name(name) => Some(name.as_ref().to_vec()),
        _ => None,
    }
}

fn image_mask(source: &SourcePdf, dict: &OwnedDictionary) -> Result<bool, Error> {
    let Some(mask) = dict.get(b"ImageMask".as_slice()) else {
        return Ok(false);
    };
    Ok(matches!(
        resolve_owned(source, mask.clone())?,
        OwnedObject::Boolean(true)
    ))
}

const fn source_id_from_ref(reference: &OwnedObject) -> Option<ObjectId> {
    match reference {
        OwnedObject::Reference(ObjectHandle::Existing(id)) => Some(*id),
        _ => None,
    }
}

impl ScanContext<'_> {
    fn visit_xobject(
        &mut self,
        resource_name: &[u8],
        resources: &OwnedDictionary,
        state: &State,
        depth: usize,
    ) -> Result<(), Error> {
        let Some(reference) = named_resource(self.source, resources, b"XObject", resource_name)?
        else {
            self.summary.unknown_paint += 1;
            return Ok(());
        };
        let Some(id) = source_id_from_ref(&reference) else {
            self.summary.unknown_paint += 1;
            return Ok(());
        };
        let object = resolve_owned(self.source, reference)?;
        let Some(dict) = object.as_dictionary() else {
            self.summary.unknown_paint += 1;
            return Ok(());
        };
        match subtype(self.source, dict)?.as_deref() {
            Some(b"Image") => self.visit_image(id, dict, state),
            Some(b"Form") => self.visit_form(id, dict, resources, state, depth),
            _ => {
                self.summary.unknown_paint += 1;
                Ok(())
            }
        }
    }

    fn visit_image(
        &mut self,
        id: ObjectId,
        dict: &OwnedDictionary,
        state: &State,
    ) -> Result<(), Error> {
        self.summary.image_paints += 1;
        let paint_index = u32::try_from(self.summary.image_paints - 1)
            .map_err(|_| Error::Invalid("excessive image paint count".into()))?;
        let Ok((width, height)) = image_dimensions(self.source, dict) else {
            self.summary.unsafe_image_paints += 1;
            self.summary.unknown_paint += 1;
            return Ok(());
        };
        let mut unsafe_reasons = Vec::new();
        if width == 0 || height == 0 || width > 100_000 || height > 100_000 {
            unsafe_reasons.push("invalid_source_dimensions");
        }
        if state.ctm.area_scale() < 1e-9 || !state.ctm.area_scale().is_finite() {
            unsafe_reasons.push("degenerate_image_placement");
        }
        if image_mask(self.source, dict)? {
            unsafe_reasons.push("image_stencil_mask");
        }
        if state.clip_unknown {
            unsafe_reasons.push("nonrectangular_or_untracked_clip");
        }
        if state.gs_unsafe {
            unsafe_reasons.push("unsupported_transparency_or_graphics_state");
        }
        if dict.contains_key(b"SMask".as_slice()) || dict.contains_key(b"Mask".as_slice()) {
            unsafe_reasons.push("source_alpha_or_color_key_requires_compositing");
        }
        if !unsafe_reasons.is_empty() {
            self.summary.unsafe_image_paints += 1;
        }
        let object_number = u32::try_from(id.number())
            .map_err(|_| Error::Invalid("invalid source object number".into()))?;
        let generation = u16::try_from(id.generation())
            .map_err(|_| Error::Invalid("invalid source generation".into()))?;
        self.paintings.push(ImagePaint {
            page: self.page.get(),
            paint_index,
            paint_path: self.do_path.clone(),
            source_id: image_source_id(id),
            object_number,
            generation,
            width,
            height,
            ctm: state.ctm.0,
            page_quad: state.ctm.rect_quad(PageBox::new([0.0, 0.0, 1.0, 1.0])?),
            clips: state.clips.clone(),
            unsafe_reasons,
        });
        Ok(())
    }

    fn visit_form(
        &mut self,
        id: ObjectId,
        dict: &OwnedDictionary,
        resources: &OwnedDictionary,
        state: &State,
        depth: usize,
    ) -> Result<(), Error> {
        if depth >= MAX_FORM_DEPTH || !self.active_forms.insert(id) {
            self.summary.unsafe_form_recursion += 1;
            return Ok(());
        }
        self.summary.forms_visited += 1;
        let mut child = state.clone();
        if let Some(matrix) = dict.get(b"Matrix".as_slice()) {
            child.ctm = child
                .ctm
                .concat(PdfMatrix(array_numbers::<6>(self.source, matrix)?));
        }
        if let Some(bounds) = dict.get(b"BBox".as_slice()) {
            match PageBox::from_unordered_corners(array_numbers::<4>(self.source, bounds)?)? {
                Some(page_box) => child.clips.push(child.ctm.rect_quad(page_box)),
                // A degenerate Form BBox cannot justify visibility proof:
                // still traverse the content but mark all contained paints
                // unsafe for source-only or independently admitted OCR.
                None => child.clip_unknown = true,
            }
        } else {
            child.clip_unknown = true;
        }
        if dict.contains_key(b"Group".as_slice()) {
            child.gs_unsafe = true;
        }
        let child_resources = match dict.get(b"Resources".as_slice()) {
            Some(value) => resolve_dictionary(self.source, value)?,
            None => resources.clone(),
        };
        let decoded = self.source.decoded_stream_data(id)?;
        let result = self.visit_stream(&decoded, &child_resources, child, depth + 1);
        self.active_forms.remove(&id);
        result
    }
}

impl ScanContext<'_> {
    /// Classify shows by the font already selected in graphics state, not
    /// by a name resolved against this stream's potentially different resources.
    const fn handle_text_show(&mut self, state: &State) {
        if state.render_mode >= 4 {
            self.summary.non_image_visual_paint += 1;
        } else if state.render_mode != 3 {
            if matches!(state.font, SelectedFont::Searchable) {
                self.summary.native_searchable_shows += 1;
            } else {
                self.summary.non_image_visual_paint += 1;
            }
        }
    }

    fn apply_extended_graphics_state(
        &mut self,
        instruction: &hayro_syntax::content::Instruction<'_, '_>,
        resources: &OwnedDictionary,
        state: &mut State,
    ) -> Result<(), Error> {
        let name = instruction_name(instruction);
        let raw = name
            .as_deref()
            .map(|name| named_resource(self.source, resources, b"ExtGState", name))
            .transpose()?
            .flatten();
        let safe = match raw {
            Some(raw) => {
                let gs = resolve_dictionary(self.source, &raw)?;
                if let Some(searchable) = extgstate_font_searchability(self.source, &gs)? {
                    state.font = if searchable {
                        SelectedFont::Searchable
                    } else {
                        SelectedFont::Unsearchable
                    };
                }
                opaque_extgs(self.source, &gs)?
            }
            None => false,
        };
        if !safe {
            self.summary.unsupported_extgs += 1;
            state.gs_unsafe = true;
        }
        Ok(())
    }

    fn visit_do(
        &mut self,
        instruction: &hayro_syntax::content::Instruction<'_, '_>,
        resources: &OwnedDictionary,
        state: &State,
        depth: usize,
        ordinal: u32,
    ) -> Result<(), Error> {
        self.do_path.push(ordinal);
        let result = if let Some(name) = instruction_name(instruction) {
            self.visit_xobject(&name, resources, state, depth)
        } else {
            self.summary.unknown_paint += 1;
            Ok(())
        };
        self.do_path.pop();
        result
    }

    fn visit_stream(
        &mut self,
        bytes: &[u8],
        resources: &OwnedDictionary,
        mut state: State,
        depth: usize,
    ) -> Result<(), Error> {
        let mut iter = UntypedIter::new(bytes);
        let mut saved_states = Vec::new();
        let mut path = PathState::default();
        let mut next_do = 0_u32;

        while let Some(instruction) = iter.next() {
            let op: &[u8] = &instruction.operator[..];
            let numbers = instruction_numbers(&instruction);
            if let Some(visual) = path.consume(op, numbers.as_deref(), &mut state) {
                if visual {
                    self.summary.non_image_visual_paint += 1;
                }
                continue;
            }
            match op {
                b"q" => saved_states.push(state.clone()),
                b"Q" => {
                    if let Some(previous) = saved_states.pop() {
                        state = previous;
                    } else {
                        self.summary.unbalanced_q += 1;
                        state.clip_unknown = true;
                    }
                }
                b"cm" => {
                    if let Some(args) = numbers.as_deref()
                        && let Ok(next) = PdfMatrix::try_from(args)
                    {
                        state.ctm = state.ctm.concat(next);
                    } else {
                        self.summary.unknown_paint += 1;
                        state.clip_unknown = true;
                    }
                }
                b"gs" => {
                    self.apply_extended_graphics_state(&instruction, resources, &mut state)?;
                }
                b"Tf" => {
                    state.font = match instruction.operands().next() {
                        Some(HayroObject::Name(name))
                            if font_searchable(self.source, resources, name.as_ref())? =>
                        {
                            SelectedFont::Searchable
                        }
                        _ => SelectedFont::Unsearchable,
                    };
                }
                b"Tr" => {
                    let render = numbers
                        .as_deref()
                        .and_then(|args| args.first())
                        .filter(|value| value.is_finite() && value.fract().abs() < 1e-9)
                        .and_then(|value| value.to_string().parse::<u8>().ok());
                    if let Some(value @ 0..=7) = render {
                        state.render_mode = value;
                    } else {
                        self.summary.unknown_paint += 1;
                        state.gs_unsafe = true;
                    }
                }
                b"BT" => state.text_open = true,
                b"ET" => state.text_open = false,
                b"Tj" | b"TJ" | b"'" | b"\"" => self.handle_text_show(&state),
                b"sh" => self.summary.non_image_visual_paint += 1,
                b"Do" => {
                    self.visit_do(&instruction, resources, &state, depth, next_do)?;
                    next_do = next_do
                        .checked_add(1)
                        .ok_or_else(|| Error::Invalid("excessive Do invocations".into()))?;
                }
                b"BI" | b"ID" | b"EI" => self.summary.unsupported_inline_images += 1,
                b"w" | b"J" | b"j" | b"M" | b"d" | b"ri" | b"i" | b"g" | b"G" | b"rg" | b"RG"
                | b"k" | b"K" | b"cs" | b"CS" | b"sc" | b"SC" | b"scn" | b"SCN" | b"Tc" | b"Tw"
                | b"Tz" | b"TL" | b"Ts" | b"Td" | b"TD" | b"Tm" | b"T*" | b"BMC" | b"BDC"
                | b"EMC" | b"MP" | b"DP" | b"BX" | b"EX" => {}
                _ => self.summary.unknown_paint += 1,
            }
        }
        if !saved_states.is_empty() || path.pending_clip || state.text_open {
            self.summary.unbalanced_stream += 1;
        }
        if !iter.is_at_end() {
            self.summary.unknown_paint += 1;
        }
        Ok(())
    }
}

/// Traverse native PDF content once and identify real image paints, including Forms.
///
/// # Errors
/// Returns an error when the source PDF, resources, decoded streams, page
/// geometry or object graph are malformed. Unsupported paint remains tracked in
/// the page safety report, rather than being silently dropped.
pub fn enumerate_image_paints(
    metadata: &PdfMetadata,
    selected: &[PageNumber],
) -> Result<PaintReport, Error> {
    let source = metadata.source();
    let page_ids = source.page_ids();
    let mut result = PaintReport::default();
    let mut seen = BTreeSet::new();
    for page in selected {
        if !seen.insert(*page) {
            return Err(Error::Invalid("duplicate requested page".into()));
        }
        let page_offset = usize::try_from(page.get() - 1)
            .map_err(|_| Error::Invalid("selected page outside index space".into()))?;
        let id = *page_ids
            .get(page_offset)
            .ok_or_else(|| Error::Invalid("requested page outside source".into()))?;
        let media_box = metadata.media_boxes()[page];
        let resources = inherited_page_resources(source, id)?;
        let content = page_content(source, id)?;
        let mut ctx = ScanContext {
            source,
            page: *page,
            summary: PagePaintSummary {
                page: page.get(),
                ..PagePaintSummary::default()
            },
            paintings: Vec::new(),
            active_forms: BTreeSet::new(),
            do_path: Vec::new(),
        };
        let page_object = source.materialize(id)?;
        if let Some(dict) = page_object.as_dictionary()
            && let Some(annotations) = dict.get(b"Annots".as_slice())
        {
            match resolve_owned(source, annotations.clone())? {
                OwnedObject::Array(items) => ctx.summary.annotations = items.len(),
                _ => ctx.summary.unknown_paint += 1,
            }
        }
        ctx.visit_stream(&content, &resources, State::for_page(media_box), 0)?;
        ctx.summary.certify();
        result.placements.extend(ctx.paintings);
        result.pages.push(ctx.summary);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{PdfMatrix, enumerate_image_paints};
    use crate::{PageNumber, geometry::Point, pdf_metadata::PdfMetadata};

    fn two_page_pdf_with_one_repeated_image() -> Vec<u8> {
        let mut source = b"%PDF-1.4\n".to_vec();
        let image_bytes = [255_u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let paint = b"q\n100 0 0 100 20 20 cm\n/Im0 Do\nQ\n";
        let dicts = [
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /MediaBox [0 0 600 800] >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im0 5 0 R >> >> /Contents 6 0 R >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im0 5 0 R >> >> /Contents 6 0 R >>".to_vec(),
            [&b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 12 >>\nstream\n"[..],
                &image_bytes, &b"\nendstream"[..]].concat(),
            [format!("<< /Length {} >>\nstream\n", paint.len()).as_bytes(), paint, b"endstream"].concat(),
        ];
        let mut offsets = vec![0_usize];
        for (index, body) in dicts.iter().enumerate() {
            offsets.push(source.len());
            source.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
            source.extend_from_slice(body);
            source.extend_from_slice(b"\nendobj\n");
        }
        let table = source.len();
        source.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            source.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        source.extend_from_slice(
            format!("trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{table}\n%%EOF\n").as_bytes(),
        );
        source
    }

    #[test]
    fn matrix_concatenation_matches_pdf_reference() {
        let a = PdfMatrix([2.0, 1.0, 3.0, 4.0, 5.0, 7.0]);
        let b = PdfMatrix([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let result = a.concat(b);
        let point = Point(0.2, 0.7);
        let direct = a.point(b.point(point));
        let composed = result.point(point);
        assert!((direct.0 - composed.0).abs() < 1e-8);
        assert!((direct.1 - composed.1).abs() < 1e-8);
    }

    #[test]
    fn reused_forms_inherit_the_actual_selected_font_not_local_aliases() -> Result<(), crate::Error>
    {
        // On page 1, a Form whose /F points at Type3 inherits the caller's
        // selected Type1 /F and paints searchable text. On page 2, the
        // reverse Form has a local /G pointing at Type1 but inherits a Type3
        // selected by the caller. The call-site /Tr modes and q/Q stacks
        // independently preserve invisible, clipping, and visual text.
        let fixture = include_bytes!("../tests/fixtures/native-font-invocation-context.pdf");
        let metadata = PdfMetadata::from_bytes(fixture.to_vec())?;
        let pages = [
            PageNumber::new(1)?,
            PageNumber::new(2)?,
            PageNumber::new(3)?,
        ];
        let report = enumerate_image_paints(&metadata, &pages)?;
        assert!(report.placements.is_empty());
        for (page, native, non_image) in [(1, 5, 2), (2, 2, 5), (3, 4, 2)] {
            let result = report
                .pages
                .iter()
                .find(|entry| entry.page == page)
                .ok_or_else(|| crate::Error::Invalid("scan omitted fixture page".into()))?;
            assert_eq!(result.page, page);
            assert_eq!(result.native_searchable_shows, native, "page {page}");
            assert_eq!(result.non_image_visual_paint, non_image, "page {page}");
            assert_eq!(result.unknown_paint, 0, "page {page}");
            assert_eq!(result.unbalanced_q, 0, "page {page}");
        }
        Ok(())
    }

    #[test]
    fn image_paints_repeat_per_page_without_repeated_source_decode() -> Result<(), crate::Error> {
        let metadata = PdfMetadata::from_bytes(two_page_pdf_with_one_repeated_image())?;
        let selected = [PageNumber::new(1)?, PageNumber::new(2)?];
        let scanned = enumerate_image_paints(&metadata, &selected)?;
        assert_eq!(scanned.placements.len(), 2);
        assert_eq!(
            scanned.placements[0].source_id,
            scanned.placements[1].source_id
        );
        assert_eq!(scanned.placements[0].width, 2);
        assert_eq!(scanned.placements[0].page_quad[0], Point(20.0, 20.0));
        assert_eq!(scanned.placements[0].page_quad[2], Point(120.0, 120.0));
        assert!(
            scanned
                .pages
                .iter()
                .all(|page| page.image_only_after_native)
        );
        Ok(())
    }
}
