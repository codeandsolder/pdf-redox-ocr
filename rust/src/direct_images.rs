//! Native raster extraction from the already parsed document's interpreter.
//!
//! Bind each matched paint to its distinct, opaque native RGB appearance.
//! Equal resolved decoding inputs reuse a previously verified raster.
//! The interpreter, not this module, resolves filters, color and Decode arrays.

use std::collections::{BTreeMap, BTreeSet, HashMap, btree_map::Entry};

use hayro_interpret::{
    BlendMode, ClipPath, Context, Device, DrawMode, DrawProps, Image, ImageData, ImageDrawProps,
    InterpreterCache, InterpreterSettings, RasterDecodeContext, RenderContent, SoftMask,
    font::GlyphRun,
    interpret_page,
    kurbo::{Affine, BezPath, Rect},
};
use hayro_syntax::{
    content::UntypedIter,
    object::{ObjectIdentifier, Stream},
    page::Page,
};
use image::RgbImage;

use crate::{
    Error, PageNumber,
    native_images::NativeImageBudget,
    paint_scan::{ImagePaint, page_content},
    pdf_metadata::PdfMetadata,
    source_appearances::{AppearanceId, PaintAppearances, SourceAppearance},
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
    pub accepted_appearances: usize,
    pub accepted_occurrences: usize,
    pub rejected_paints: BTreeMap<u32, BTreeMap<u32, &'static str>>,
    pub rejected_pages: BTreeMap<u32, &'static str>,
}

#[derive(Default)]
pub struct DirectImages {
    pub appearances: PaintAppearances<RgbImage>,
    pub stats: DirectDecodeStats,
}

enum Wanted<'a> {
    Sources(&'a BTreeSet<String>),
    Paint(&'a ImagePaint),
}

impl Wanted<'_> {
    fn contains(&self, paint: &ImagePaint) -> bool {
        match self {
            Self::Sources(ids) => ids.contains(&paint.source_id),
            Self::Paint(target) => {
                target.page == paint.page
                    && target.paint_path == paint.paint_path
                    && target.source_id == paint.source_id
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Sources(ids) => ids.len(),
            Self::Paint(_) => 1,
        }
    }
}

struct Collector<'a> {
    wanted: Wanted<'a>,
    budget: NativeImageBudget,
    result: DirectImages,
    stored_pixels: u64,
    verified_contexts: HashMap<RasterDecodeContext, AppearanceId>,
    next_appearance: usize,
}

struct Expected<'a> {
    paint: &'a ImagePaint,
    seen: bool,
}

struct PageDevice<'a, 'b> {
    collector: &'a mut Collector<'b>,
    page: PageNumber,
    expected: BTreeMap<&'b [u32], Expected<'b>>,
    groups: Vec<bool>,
    blocked_groups: usize,
    attempted_pixels: u64,
    invalid_page: Option<&'static str>,
}

impl Collector<'_> {
    fn reject(&mut self, paint: &ImagePaint, reason: &'static str) {
        if self.wanted.contains(paint) {
            self.result
                .stats
                .rejected_paints
                .entry(paint.page)
                .or_default()
                .entry(paint.paint_index)
                .or_insert(reason);
        }
    }

    fn reject_page(
        &mut self,
        page: PageNumber,
        expected: &BTreeMap<&[u32], Expected<'_>>,
        reason: &'static str,
    ) {
        self.result
            .stats
            .rejected_pages
            .entry(page.get())
            .or_insert(reason);
        for entry in expected.values() {
            self.reject(entry.paint, reason);
        }
        self.result.appearances.paint_bindings.remove(&page.get());

        let referenced = self.result.appearances.referenced_ids();
        self.result.appearances.variants.retain(|id, appearance| {
            if referenced.contains(id) {
                true
            } else {
                self.stored_pixels -=
                    u64::from(appearance.value.width()) * u64::from(appearance.value.height());
                false
            }
        });
        self.verified_contexts
            .retain(|_, id| referenced.contains(id));
    }

    fn resident_capacity(&self, pixels: u64) -> bool {
        self.result.appearances.len() < self.budget.sources
            && pixels <= self.budget.total_pixels.saturating_sub(self.stored_pixels)
    }

    fn has_source(&self, id: &str) -> bool {
        self.result
            .appearances
            .variants
            .values()
            .any(|appearance| appearance.source_id == id)
    }

    fn retain(
        &mut self,
        paint: &ImagePaint,
        rgb: RgbImage,
        context: Option<RasterDecodeContext>,
    ) -> Option<AppearanceId> {
        let existing = self
            .result
            .appearances
            .variants
            .iter()
            .find_map(|(&id, appearance)| {
                (appearance.source_id == paint.source_id && appearance.value == rgb).then_some(id)
            });
        let id = if let Some(id) = existing {
            id
        } else {
            let pixels = u64::from(rgb.width()) * u64::from(rgb.height());
            if !self.resident_capacity(pixels) {
                self.reject(paint, "resident_appearance_budget");
                return None;
            }
            let id = AppearanceId(self.next_appearance);
            self.next_appearance += 1;
            self.stored_pixels += pixels;
            self.result.appearances.variants.insert(
                id,
                SourceAppearance {
                    source_id: paint.source_id.clone(),
                    value: rgb,
                },
            );
            id
        };
        if let Some(context) = context {
            self.verified_contexts.insert(context, id);
        }
        Some(id)
    }

    fn finalize(mut self) -> DirectImages {
        let appearances = &self.result.appearances;
        self.result.stats.accepted_sources = appearances
            .variants
            .values()
            .map(|appearance| appearance.source_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        self.result.stats.accepted_appearances = appearances.len();
        self.result.stats.accepted_occurrences =
            appearances.paint_bindings.values().map(BTreeMap::len).sum();
        self.result
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

impl<'b> PageDevice<'_, 'b> {
    fn match_paint(
        &mut self,
        id: ObjectIdentifier,
        path: Option<&[u32]>,
        dimensions: (u32, u32),
        transform: Affine,
    ) -> Option<&'b ImagePaint> {
        let entry = path.and_then(|path| self.expected.get_mut(path))?;
        if entry.seen
            || entry.paint.object_number != id.obj_number
            || i32::from(entry.paint.generation) != id.gen_number
            || (entry.paint.width, entry.paint.height) != dimensions
            || !transform_matches(entry.paint, transform)
        {
            return None;
        }
        entry.seen = true;
        Some(entry.paint)
    }

    fn can_decode(&mut self, paint: &ImagePaint, dimensions: (u32, u32)) -> bool {
        let pixels = u64::from(dimensions.0) * u64::from(dimensions.1);
        let budget = self.collector.budget;
        let reason = if pixels == 0 || pixels > budget.source_pixels {
            Some("source_pixel_budget")
        } else if !self.collector.has_source(&paint.source_id)
            && !self.collector.resident_capacity(pixels)
        {
            Some("resident_appearance_budget")
        } else if pixels
            > budget
                .extraction_pixels
                .saturating_sub(self.attempted_pixels)
        {
            Some("page_decode_budget")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.collector.reject(paint, reason);
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

    fn bind(&mut self, paint: &ImagePaint, id: AppearanceId) {
        if !self.collector.result.appearances.bind(paint, id) {
            self.invalid_page
                .get_or_insert("ambiguous_appearance_binding");
        }
    }

    fn finish(self) {
        if let Some(reason) = self.invalid_page {
            self.collector
                .reject_page(self.page, &self.expected, reason);
            return;
        }
        for entry in self.expected.values() {
            if !entry.seen {
                self.collector
                    .reject(entry.paint, "unmatched_scanned_occurrence");
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
            self.invalid_page
                .get_or_insert("malformed_interpreter_groups");
        }
    }

    fn draw_image(&mut self, image: Image<'_, '_>, props: ImageDrawProps<'_>) {
        if self.invalid_page.is_some() {
            return;
        }
        let stream = match &image {
            Image::Stencil(stencil) => stencil.stream(),
            Image::Raster(raster) => raster.stream(),
        };
        let path = props.page_xobject_path.as_deref();
        let Some(id) = stream.dict().obj_id() else {
            if path.is_some() {
                self.invalid_page = Some("unmatched_interpreter_occurrence");
            }
            return;
        };
        let dimensions = (image.width(), image.height());
        let Some(paint) = self.match_paint(id, path, dimensions, props.transform) else {
            self.invalid_page = Some("unmatched_interpreter_occurrence");
            return;
        };
        if !self.collector.wanted.contains(paint) {
            return;
        }
        self.collector.result.stats.matched_occurrences += 1;
        if !paint.unsafe_reasons.is_empty() {
            self.collector.reject(paint, "unsafe_scanned_occurrence");
            return;
        }
        if self.blocked_groups != 0
            || props.soft_mask.is_some()
            || props.blend_mode != BlendMode::Normal
            || [b"SMask".as_slice(), b"Mask", b"SMaskInData"]
                .iter()
                .any(|key| stream.dict().contains_key(*key))
        {
            self.collector.reject(paint, "requires_compositing");
            return;
        }
        let Image::Raster(raster) = image else {
            self.collector.reject(paint, "stencil");
            return;
        };
        let context = raster.decode_context(None);
        if let Some(&id) = context
            .as_ref()
            .and_then(|context| self.collector.verified_contexts.get(context))
        {
            let stats = &mut self.collector.result.stats;
            stats.reused_occurrences += 1;
            stats.reused_pixels = stats
                .reused_pixels
                .saturating_add(u64::from(dimensions.0) * u64::from(dimensions.1));
            self.bind(paint, id);
            return;
        }
        if !self.can_decode(paint, dimensions) {
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
            if let Some(id) = self.collector.retain(paint, rgb, context) {
                self.bind(paint, id);
            }
        } else {
            self.collector
                .reject(paint, "unsupported_or_nonopaque_decode");
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
    let mut expected_pages = PageExpectations::new();
    let mut indices = BTreeSet::new();
    for paint in paints {
        let page = PageNumber::new(paint.page)?;
        if !pages.contains(&page) {
            continue;
        }
        if !paint.unsafe_reasons.is_empty() {
            let reason = if paint.unsafe_reasons.contains(&"page_transparency_group") {
                "page_transparency_group"
            } else {
                "unsafe_scanned_occurrence"
            };
            collector.reject(paint, reason);
        }
        let expected = expected_pages.entry(page).or_default();
        if !indices.insert((page, paint.paint_index)) {
            collector.reject_page(page, expected, "duplicate_scanned_occurrence");
            collector.reject(paint, "duplicate_scanned_occurrence");
            continue;
        }
        match expected.entry(&paint.paint_path) {
            Entry::Vacant(entry) => {
                entry.insert(Expected { paint, seen: false });
            }
            Entry::Occupied(_) => {
                collector.reject_page(page, expected, "duplicate_scanned_occurrence");
                collector.reject(paint, "duplicate_scanned_occurrence");
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
        // Root transparency color conversion is not implemented by interpret_page.
        let reason = if let Some(&reason) = collector.result.stats.rejected_pages.get(&page.get()) {
            Some(reason)
        } else if source_page.raw().contains_key(b"Group".as_slice()) {
            Some("page_transparency_group")
        } else if expected.values().any(|entry| {
            collector.wanted.contains(entry.paint) && entry.paint.unsafe_reasons.is_empty()
        }) && !equivalent_page_program(metadata, source_page)?
        {
            Some("page_content_path_mismatch")
        } else {
            None
        };
        if let Some(reason) = reason {
            collector.reject_page(page, expected, reason);
        }
    }
    Ok(())
}

/// Decode selected source images without extracting unrelated page images.
///
/// Bind each exact, independently safe occurrence to its decoded appearance.
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
    decode_selected(metadata, selected, Wanted::Sources(wanted), paints, budget)
}

/// Decode exactly one matched native occurrence, not its sibling invocations.
pub(crate) fn decode_exact_paint(
    metadata: &PdfMetadata,
    paints: &[ImagePaint],
    paint: &ImagePaint,
) -> Result<Option<RgbImage>, Error> {
    let pixels = u64::from(paint.width) * u64::from(paint.height);
    let budget = NativeImageBudget {
        source_pixels: pixels,
        total_pixels: pixels,
        extraction_pixels: pixels,
        sources: 1,
    };
    let mut result = decode_selected(
        metadata,
        &[PageNumber::new(paint.page)?],
        Wanted::Paint(paint),
        paints,
        budget,
    )?;
    let Some((id, _)) = result.appearances.get(paint) else {
        return Ok(None);
    };
    Ok(result
        .appearances
        .variants
        .remove(&id)
        .map(|appearance| appearance.value))
}

fn decode_selected(
    metadata: &PdfMetadata,
    selected: &[PageNumber],
    wanted: Wanted<'_>,
    paints: &[ImagePaint],
    budget: NativeImageBudget,
) -> Result<DirectImages, Error> {
    let mut collector = Collector {
        wanted,
        budget: budget.validate()?,
        result: DirectImages::default(),
        stored_pixels: 0,
        verified_contexts: HashMap::new(),
        next_appearance: 0,
    };
    collector.result.stats.requested_sources = collector.wanted.len();
    if collector.wanted.len() == 0 {
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
        if collector
            .result
            .stats
            .rejected_pages
            .contains_key(&page.get())
            || !expected.values().any(|entry| {
                collector.wanted.contains(entry.paint) && entry.paint.unsafe_reasons.is_empty()
            })
        {
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
            page,
            expected,
            groups: Vec::new(),
            blocked_groups: 0,
            attempted_pixels: 0,
            invalid_page: None,
        };
        interpret_page(source_page, &mut context, &mut device);
        if !device.groups.is_empty() {
            device
                .invalid_page
                .get_or_insert("malformed_interpreter_groups");
        }
        device.finish();
    }
    Ok(collector.finalize())
}
