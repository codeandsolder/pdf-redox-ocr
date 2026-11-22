//! Native raster extraction from the already parsed document's interpreter.
//!
//! A source-ID keyed consumer may reuse a decoded image only if every scanned
//! occurrence is accounted for and has the same opaque native RGB pixels.
//! Equal resolved decoding inputs reuse a previously verified raster.
//! The interpreter, not this module, resolves filters, color and Decode arrays.

use std::collections::{BTreeMap, BTreeSet, HashSet, btree_map::Entry};

use hayro_interpret::{
    BlendMode, ClipPath, Context, Device, DrawMode, DrawProps, Image, ImageData, ImageDrawProps,
    InterpreterCache, InterpreterSettings, RasterDecodeContext, RenderContent, SoftMask,
    font::GlyphRun,
    interpret_page,
    kurbo::{Affine, BezPath, Rect},
};
use hayro_syntax::{content::UntypedIter, object::Stream, page::Page};
use image::RgbImage;

use crate::{
    Error, PageNumber,
    native_images::NativeImageBudget,
    paint_scan::{ImagePaint, page_content},
    pdf_metadata::PdfMetadata,
};

#[derive(Debug, Default, serde::Serialize)]
pub struct DirectDecodeStats {
    pub requested_sources: usize,
    pub matched_occurrences: usize,
    pub decoded_occurrences: usize,
    pub decoded_pixels: u64,
    pub reused_occurrences: usize,
    pub reused_pixels: u64,
    pub accepted_sources: usize,
    pub rejected_sources: BTreeMap<String, &'static str>,
}

#[derive(Default)]
pub struct DirectImages {
    pub images: BTreeMap<String, RgbImage>,
    pub stats: DirectDecodeStats,
}

struct Collector<'a> {
    wanted: &'a BTreeSet<String>,
    budget: NativeImageBudget,
    result: DirectImages,
    stored_pixels: u64,
    verified_contexts: BTreeMap<String, HashSet<RasterDecodeContext>>,
}

struct Expected<'a> {
    paint: &'a ImagePaint,
    seen: bool,
}

struct PageDevice<'a, 'b> {
    collector: &'a mut Collector<'b>,
    expected: BTreeMap<&'b [u32], Expected<'b>>,
    groups: Vec<bool>,
    blocked_groups: usize,
    attempted_pixels: u64,
    malformed_groups: bool,
}

impl Collector<'_> {
    fn reject(&mut self, id: &str, reason: &'static str) {
        self.result
            .stats
            .rejected_sources
            .entry(id.into())
            .or_insert(reason);
        self.verified_contexts.remove(id);
        if let Some(image) = self.result.images.remove(id) {
            self.stored_pixels -= u64::from(image.width()) * u64::from(image.height());
        }
    }

    fn retain(&mut self, id: &str, rgb: RgbImage, context: Option<RasterDecodeContext>) {
        if let Some(previous) = self.result.images.get(id) {
            if previous != &rgb {
                self.reject(id, "resolved_pixels_differ");
                return;
            }
        } else {
            self.stored_pixels += u64::from(rgb.width()) * u64::from(rgb.height());
            self.result.images.insert(id.into(), rgb);
        }
        if let Some(context) = context {
            self.verified_contexts
                .entry(id.into())
                .or_default()
                .insert(context);
        }
    }
}

fn native_rgb(image: ImageData, dimensions: (u32, u32)) -> Option<RgbImage> {
    if (image.width(), image.height()) != dimensions {
        return None;
    }
    let data = match image {
        ImageData::Rgb(rgb) => rgb.data,
        ImageData::Luma(luma) => luma.data.into_iter().flat_map(|v| [v; 3]).collect(),
    };
    let length = (u64::from(dimensions.0) * u64::from(dimensions.1)).checked_mul(3)?;
    if u64::try_from(data.len()).ok()? != length {
        return None;
    }
    RgbImage::from_raw(dimensions.0, dimensions.1, data)
}

/// Hayro's image transform maps native top-left pixel coordinates to user
/// space. Convert it back to the PDF image unit square before comparison.
fn transform_matches(paint: &ImagePaint, transform: Affine) -> bool {
    let matrix = transform
        * Affine::new([
            f64::from(paint.width),
            0.,
            0.,
            -f64::from(paint.height),
            0.,
            f64::from(paint.height),
        ]);
    matrix
        .as_coeffs()
        .iter()
        .zip(paint.ctm)
        .all(|(actual, expected)| {
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= 1e-5 * actual.abs().max(expected.abs()).max(1.)
        })
}

impl PageDevice<'_, '_> {
    fn match_paint(
        &mut self,
        id: &str,
        path: Option<&[u32]>,
        dimensions: (u32, u32),
        transform: Affine,
    ) -> bool {
        let Some(entry) = path.and_then(|path| self.expected.get_mut(path)) else {
            return false;
        };
        if entry.seen
            || entry.paint.source_id != id
            || (entry.paint.width, entry.paint.height) != dimensions
            || !transform_matches(entry.paint, transform)
        {
            return false;
        }
        entry.seen = true;
        self.collector.result.stats.matched_occurrences += 1;
        true
    }

    fn can_decode(&mut self, id: &str, dimensions: (u32, u32)) -> bool {
        let pixels = u64::from(dimensions.0) * u64::from(dimensions.1);
        let budget = self.collector.budget;
        if pixels == 0 || pixels > budget.source_pixels {
            self.collector.reject(id, "source_pixel_budget");
            return false;
        }
        if !self.collector.result.images.contains_key(id)
            && (self.collector.result.images.len() >= budget.sources
                || pixels
                    > budget
                        .total_pixels
                        .saturating_sub(self.collector.stored_pixels))
        {
            self.collector.reject(id, "resident_source_budget");
            return false;
        }
        if pixels
            > budget
                .extraction_pixels
                .saturating_sub(self.attempted_pixels)
        {
            self.collector.reject(id, "page_decode_budget");
            return false;
        }
        self.attempted_pixels += pixels;
        self.collector.result.stats.decoded_pixels = self
            .collector
            .result
            .stats
            .decoded_pixels
            .saturating_add(pixels);
        true
    }

    fn finish(self) {
        for entry in self.expected.values() {
            if self.malformed_groups || !entry.seen {
                self.collector
                    .reject(&entry.paint.source_id, "unmatched_scanned_occurrence");
            }
        }
    }
}

impl Device<'_> for PageDevice<'_, '_> {
    fn draw_path(&mut self, _: &BezPath, _: DrawProps<'_>, _: &DrawMode) {}
    fn push_clip_path(&mut self, _: &ClipPath) {}
    fn pop_clip(&mut self) {}
    fn draw_glyph_run(&mut self, _: &GlyphRun<'_, '_>, _: DrawProps<'_>, _: &DrawMode) {}

    fn push_transparency_group(
        &mut self,
        opacity: f32,
        mask: Option<SoftMask<'_>>,
        mode: BlendMode,
        isolated: bool,
        knockout: bool,
    ) {
        let blocked = opacity.to_bits() != 1.0_f32.to_bits()
            || mask.is_some()
            || mode != BlendMode::Normal
            || isolated
            || knockout;
        self.groups.push(blocked);
        self.blocked_groups += usize::from(blocked);
    }

    fn pop_transparency_group(&mut self) {
        if let Some(blocked) = self.groups.pop() {
            self.blocked_groups -= usize::from(blocked);
        } else {
            self.malformed_groups = true;
        }
    }

    fn draw_image(&mut self, image: Image<'_, '_>, props: ImageDrawProps<'_>) {
        let stream = match &image {
            Image::Stencil(stencil) => stencil.stream(),
            Image::Raster(raster) => raster.stream(),
        };
        let Some(id) = stream.dict().obj_id() else {
            return;
        };
        let id = format!("{}:{}", id.obj_number, id.gen_number);
        if !self.collector.wanted.contains(&id)
            || self
                .collector
                .result
                .stats
                .rejected_sources
                .contains_key(&id)
        {
            return;
        }
        let dimensions = (image.width(), image.height());
        if !self.match_paint(
            &id,
            props.page_xobject_path.as_deref(),
            dimensions,
            props.transform,
        ) {
            self.collector
                .reject(&id, "unmatched_interpreter_occurrence");
            return;
        }
        if self.blocked_groups != 0
            || props.soft_mask.is_some()
            || props.blend_mode != BlendMode::Normal
            || [b"SMask".as_slice(), b"Mask", b"SMaskInData"]
                .iter()
                .any(|key| stream.dict().contains_key(*key))
        {
            self.collector.reject(&id, "requires_compositing");
            return;
        }
        let Image::Raster(raster) = image else {
            self.collector.reject(&id, "stencil");
            return;
        };
        let context = raster.decode_context(None);
        if context.as_ref().is_some_and(|context| {
            self.collector
                .verified_contexts
                .get(&id)
                .is_some_and(|verified| verified.contains(context))
        }) {
            let stats = &mut self.collector.result.stats;
            stats.reused_occurrences += 1;
            stats.reused_pixels = stats
                .reused_pixels
                .saturating_add(u64::from(dimensions.0) * u64::from(dimensions.1));
            return;
        }
        if !self.can_decode(&id, dimensions) {
            return;
        }
        let mut decoded = None;
        raster.with_rgba(
            |image, alpha| {
                if alpha.is_none() {
                    decoded = native_rgb(image, dimensions);
                }
            },
            None,
        );
        if let Some(rgb) = decoded {
            self.collector.result.stats.decoded_occurrences += 1;
            self.collector.retain(&id, rgb, context);
        } else {
            self.collector
                .reject(&id, "unsupported_or_nonopaque_decode");
        }
    }
}

// A Contents array can be joined differently by the scanner and interpreter.
// For a single stream both consume the same parsed stream's decoded bytes.
fn equivalent_page_program(metadata: &PdfMetadata, page: &Page<'_>) -> Result<bool, Error> {
    let Some(interpreted) = page.page_stream() else {
        return Ok(false);
    };
    if page.raw().get::<Stream<'_>>(b"Contents").is_some() {
        return Ok(true);
    }
    let id = page
        .raw()
        .obj_id()
        .ok_or_else(|| Error::Invalid("missing extraction page identity".into()))?;
    let scanned = page_content(metadata.source(), id.into())?;
    if scanned == interpreted {
        return Ok(true);
    }
    let (mut scanned, mut interpreted) =
        (UntypedIter::new(&scanned), UntypedIter::new(interpreted));
    loop {
        match (scanned.next(), interpreted.next()) {
            (Some(a), Some(b)) if *a.operator == *b.operator && a.operands == b.operands => {}
            (None, None) => return Ok(scanned.is_at_end() && interpreted.is_at_end()),
            _ => return Ok(false),
        }
    }
}

type PageExpectations<'a> = BTreeMap<PageNumber, BTreeMap<&'a [u32], Expected<'a>>>;

fn index_expected_paints<'a>(
    collector: &mut Collector<'_>,
    pages: &BTreeSet<PageNumber>,
    paints: &'a [ImagePaint],
) -> Result<PageExpectations<'a>, Error> {
    // Index exact invocation paths before decoding any pixels. A
    // known unsafe occurrence invalidates shared source reuse immediately.
    let mut expected_pages = PageExpectations::new();
    for paint in paints
        .iter()
        .filter(|paint| collector.wanted.contains(&paint.source_id))
    {
        let page = PageNumber::new(paint.page)?;
        if !pages.contains(&page) {
            collector.reject(&paint.source_id, "occurrence_outside_selection");
            continue;
        }
        if !paint.unsafe_reasons.is_empty() {
            collector.reject(&paint.source_id, "unsafe_scanned_occurrence");
        }
        match expected_pages
            .entry(page)
            .or_default()
            .entry(&paint.paint_path)
        {
            Entry::Vacant(entry) => {
                entry.insert(Expected { paint, seen: false });
            }
            Entry::Occupied(entry) => {
                collector.reject(&entry.get().paint.source_id, "duplicate_scanned_occurrence");
                collector.reject(&paint.source_id, "duplicate_scanned_occurrence");
            }
        }
    }
    Ok(expected_pages)
}

fn preflight_pages(
    metadata: &PdfMetadata,
    collector: &mut Collector<'_>,
    pages: &BTreeSet<PageNumber>,
    expected_pages: &PageExpectations<'_>,
) -> Result<(), Error> {
    // Preflight all selected page contexts before interpreting any of them.
    // A later incompatible occurrence must not consume earlier decode budgets.
    let pdf = metadata.source().syntax();
    for &page in pages {
        let index = usize::try_from(page.get() - 1)
            .map_err(|_| Error::Invalid("invalid extraction page".into()))?;
        let source_page = pdf
            .pages()
            .get(index)
            .ok_or_else(|| Error::Invalid("extraction page outside source".into()))?;
        let Some(expected) = expected_pages.get(&page) else {
            continue;
        };
        // interpret_page does not wrap the root page in its declared
        // transparency group. Do not assume that space is equivalent to
        // independently decoded source RGB.
        let reason = if source_page.raw().contains_key(b"Group".as_slice()) {
            Some("page_transparency_group")
        } else if expected.values().any(|entry| {
            !collector
                .result
                .stats
                .rejected_sources
                .contains_key(&entry.paint.source_id)
        }) && !equivalent_page_program(metadata, source_page)?
        {
            Some("page_content_path_mismatch")
        } else {
            None
        };
        if let Some(reason) = reason {
            for entry in expected.values() {
                collector.reject(&entry.paint.source_id, reason);
            }
        }
    }
    Ok(())
}

/// Decode selected source images without extracting unrelated page images.
///
/// Reuse requires full occurrence correspondence and either equal decoded
/// RGB or a previously verified exact interpreter decoding context.
///
/// # Errors
/// Rejects invalid budgets and page selections.
pub fn direct_native_images(
    metadata: &PdfMetadata,
    selected: &[PageNumber],
    wanted: &BTreeSet<String>,
    paints: &[ImagePaint],
    budget: NativeImageBudget,
) -> Result<DirectImages, Error> {
    let mut collector = Collector {
        wanted,
        budget: budget.validate()?,
        result: DirectImages::default(),
        stored_pixels: 0,
        verified_contexts: BTreeMap::new(),
    };
    collector.result.stats.requested_sources = wanted.len();
    if wanted.is_empty() {
        return Ok(collector.result);
    }
    let pdf = metadata.source().syntax();
    let pages = selected.iter().copied().collect::<BTreeSet<_>>();
    if pages.len() != selected.len() {
        return Err(Error::Invalid("duplicate extraction page".into()));
    }
    let expected_pages = index_expected_paints(&mut collector, &pages, paints)?;
    preflight_pages(metadata, &mut collector, &pages, &expected_pages)?;
    for (page, expected) in expected_pages {
        if expected.values().all(|entry| {
            collector
                .result
                .stats
                .rejected_sources
                .contains_key(&entry.paint.source_id)
        }) {
            continue;
        }
        let index = usize::try_from(page.get() - 1)
            .map_err(|_| Error::Invalid("invalid extraction page".into()))?;
        let source_page = &pdf.pages()[index];
        // Do not reuse Hayro's document-wide object cache between different
        // page contexts: independently resolve color resources for each page.
        let cache = InterpreterCache::new();
        let bbox = metadata
            .media_boxes()
            .get(&page)
            .ok_or_else(|| Error::Invalid("missing extraction MediaBox".into()))?;
        let settings = InterpreterSettings {
            render_content: RenderContent::PAGE,
            ..Default::default()
        };
        let mut context = Context::new(
            Affine::IDENTITY,
            Rect::new(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
            &cache,
            pdf.xref(),
            settings,
        );
        let mut device = PageDevice {
            collector: &mut collector,
            expected,
            groups: Vec::new(),
            blocked_groups: 0,
            attempted_pixels: 0,
            malformed_groups: false,
        };
        interpret_page(source_page, &mut context, &mut device);
        device.malformed_groups |= !device.groups.is_empty();
        device.finish();
    }
    for id in wanted {
        if !collector.result.images.contains_key(id)
            && !collector.result.stats.rejected_sources.contains_key(id)
        {
            collector.reject(id, "no_decoded_occurrence");
        }
    }
    collector.result.stats.accepted_sources = collector.result.images.len();
    Ok(collector.result)
}
