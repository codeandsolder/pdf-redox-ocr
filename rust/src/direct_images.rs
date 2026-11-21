//! Native raster extraction from the already parsed document's interpreter.
//!
//! A source-ID keyed consumer may reuse a decoded image only if every scanned
//! occurrence is accounted for and has the same opaque native RGB pixels.
//! The interpreter, not this module, resolves filters, color and Decode arrays.

use std::collections::{BTreeMap, BTreeSet};

use hayro_interpret::{
    BlendMode, ClipPath, Context, Device, DrawMode, DrawProps, Image, ImageData, ImageDrawProps,
    InterpreterCache, InterpreterSettings, RenderContent, SoftMask,
    font::GlyphRun,
    interpret_page,
    kurbo::{Affine, BezPath, Rect},
};
use image::RgbImage;

use crate::{
    Error, PageNumber, native_images::NativeImageBudget, paint_scan::ImagePaint,
    pdf_metadata::PdfMetadata,
};

#[derive(Debug, Default, serde::Serialize)]
pub struct DirectDecodeStats {
    pub requested_sources: usize,
    pub matched_occurrences: usize,
    pub decoded_occurrences: usize,
    pub decoded_pixels: u64,
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
}

struct Expected<'a> {
    paint: &'a ImagePaint,
    seen: bool,
}

struct PageDevice<'a, 'b> {
    collector: &'a mut Collector<'b>,
    expected: BTreeMap<String, Vec<Expected<'b>>>,
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
        if let Some(image) = self.result.images.remove(id) {
            self.stored_pixels -= u64::from(image.width()) * u64::from(image.height());
        }
    }

    fn retain(&mut self, id: &str, rgb: RgbImage) {
        if let Some(previous) = self.result.images.get(id) {
            if previous != &rgb {
                self.reject(id, "resolved_pixels_differ");
            }
        } else {
            self.stored_pixels += u64::from(rgb.width()) * u64::from(rgb.height());
            self.result.images.insert(id.into(), rgb);
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
    fn match_paint(&mut self, id: &str, dimensions: (u32, u32), transform: Affine) -> bool {
        let Some(expected) = self.expected.get_mut(id) else {
            return false;
        };
        let Some(entry) = expected.iter_mut().find(|entry| {
            !entry.seen
                && (entry.paint.width, entry.paint.height) == dimensions
                && transform_matches(entry.paint, transform)
        }) else {
            return false;
        };
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
        for (id, entries) in &self.expected {
            if self.malformed_groups || entries.iter().any(|entry| !entry.seen) {
                self.collector.reject(id, "unmatched_scanned_occurrence");
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
        if !self.match_paint(&id, dimensions, props.transform) {
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
            self.collector.retain(&id, rgb);
        } else {
            self.collector
                .reject(&id, "unsupported_or_nonopaque_decode");
        }
    }
}

/// Decode selected source images without extracting unrelated page images.
/// Reuse is based on full occurrence correspondence plus decoded RGB equality.
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
    // Index the expected placements once, before decoding any pixels. A
    // known unsafe occurrence invalidates shared source reuse immediately.
    let mut expected_pages = BTreeMap::<PageNumber, BTreeMap<String, Vec<Expected>>>::new();
    for paint in paints
        .iter()
        .filter(|paint| wanted.contains(&paint.source_id))
    {
        let page = PageNumber::new(paint.page)?;
        if !pages.contains(&page) {
            collector.reject(&paint.source_id, "occurrence_outside_selection");
            continue;
        }
        if !paint.unsafe_reasons.is_empty() {
            collector.reject(&paint.source_id, "unsafe_scanned_occurrence");
        }
        expected_pages
            .entry(page)
            .or_default()
            .entry(paint.source_id.clone())
            .or_default()
            .push(Expected { paint, seen: false });
    }
    for page in pages {
        let index = usize::try_from(page.get() - 1)
            .map_err(|_| Error::Invalid("invalid extraction page".into()))?;
        let source_page = pdf
            .pages()
            .get(index)
            .ok_or_else(|| Error::Invalid("extraction page outside source".into()))?;
        let Some(expected) = expected_pages.remove(&page) else {
            continue;
        };
        // interpret_page does not wrap the root page in its declared
        // transparency group. Do not assume that space is equivalent to
        // independently decoded source RGB.
        if source_page.raw().contains_key(b"Group".as_slice()) {
            for id in expected.keys() {
                collector.reject(id, "page_transparency_group");
            }
            continue;
        }
        if expected
            .keys()
            .all(|id| collector.result.stats.rejected_sources.contains_key(id))
        {
            continue;
        }
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
