//! Source-specific counterfactual image-text masks for PDF visibility proofs.
//!
//! The PDF is never edited in place: copy-on-write changes one image `XObject`.
//! Bit-exact packed samples retain the original `ColorSpace` and `Decode` mapping,
//! including Indexed palettes and 16-bit samples. Alpha and unknown color
//! spaces retain their normal page-OCR fallback.

use pdf_redox::{EditDocument, ObjectId, OwnedObject, SourcePdf, StreamData};

use crate::{
    Error,
    draft::OcrLine,
    geometry::{Point, Quad},
    paint_scan::ImagePaint,
    polygon::convex,
    sample_layout::color_components,
    source_samples::PackedSamples,
};

const MASK_PADDING: u32 = 3;

/// A standalone intrinsic image or a particular resolved paint.
#[derive(Debug, Clone, Copy)]
pub enum MaskSource<'a> {
    Object(ObjectId),
    Paint(&'a ImagePaint),
}

impl MaskSource<'_> {
    fn object_id(self) -> ObjectId {
        match self {
            Self::Object(id) => id,
            Self::Paint(paint) => ObjectId::new(paint.object_number, i32::from(paint.generation)),
        }
    }

    fn components(self, source: &SourcePdf, object: &OwnedObject) -> Result<Option<usize>, Error> {
        match self {
            Self::Paint(paint) => Ok(paint.sample_layout.components),
            Self::Object(_) => object
                .as_dictionary()
                .and_then(|dict| dict.get(b"ColorSpace".as_slice()))
                .map(|raw| color_components(source, raw))
                .transpose()
                .map(Option::flatten),
        }
    }
}

/// Quantifies the specific source text modified by a counterfactual.
#[derive(Debug, Clone)]
pub struct MaskMetric {
    /// OCR text whose image pixels were suppressed.
    pub text: String,
    /// Native pixels with any changed packed sample inside the OCR polygon.
    /// This is a count, not a color-distance measurement.
    pub changed_source_pixels: u64,
    /// OCR quad in source-local coordinates, not the page placement.
    pub source_quad: Quad,
}

/// Immutable temporary PDF with only one source `XObject` masked.
pub struct MaskedSourcePdf {
    /// A rewritten standalone PDF with the selected source text masked.
    pub bytes: Vec<u8>,
    /// Per-line source pixel-change expectations.
    pub metrics: Vec<MaskMetric>,
}

/// An isolated replacement image `XObject`, not yet attached to a page paint.
pub struct MaskedSourceObject {
    /// Replacement image COS stream with decoded and changed native pixels.
    pub object: OwnedObject,
    /// Per-line source pixel-change expectations.
    pub metrics: Vec<MaskMetric>,
}

/// Create one source-specific counterfactual in Rust.
///
/// Only source streams with resolved native sample layouts are modified.
/// The original image `ColorSpace` and `Decode` mapping remain untouched;
/// unsupported image encodings remain eligible for ordinary page OCR.
///
/// # Errors
/// Returns an error for malformed PDF objects or source OCR polygons that
/// lie outside the exact embedded image.
pub fn mask_source_text(
    original: &[u8],
    source_id: ObjectId,
    lines: &[OcrLine],
    min_confidence: f64,
) -> Result<Option<MaskedSourcePdf>, Error> {
    let mut doc = EditDocument::from_bytes(original.to_vec())?;
    let Some(masked) = mask_source_object(
        doc.source(),
        MaskSource::Object(source_id),
        &lines.iter().collect::<Vec<_>>(),
        min_confidence,
    )?
    else {
        return Ok(None);
    };
    *doc.edit_object(source_id)? = masked.object;
    Ok(Some(MaskedSourcePdf {
        bytes: doc.write_compact()?,
        metrics: masked.metrics,
    }))
}

/// Mask an original image into an unattached `XObject`, leaving all paint sites intact.
///
/// Returns None for image encodings that cannot be safely decoded to exact pixels.
///
/// # Errors
/// Returns an error for invalid source polygons or a malformed source PDF.
pub fn mask_source_object(
    source: &SourcePdf,
    image: MaskSource<'_>,
    lines: &[&OcrLine],
    min_confidence: f64,
) -> Result<Option<MaskedSourceObject>, Error> {
    if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
        return Err(Error::Invalid("invalid source mask confidence".into()));
    }
    let selected = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| mask_candidate(line, min_confidence).then_some(index))
        .collect::<Vec<_>>();
    mask_source_subset(source, image, lines, &selected)
}

pub(crate) fn mask_candidate(line: &OcrLine, min_confidence: f64) -> bool {
    line.confidence >= min_confidence && line.text.trim().chars().count() >= 2
}

/// Mask an explicit subset of source observations.
///
/// All other recognized regions, including those excluded by confidence or page-specific
/// admission, retain their original samples even if mask padding overlaps.
///
/// The selected indices must be unique and in ascending order.
///
/// # Errors
/// Returns an error for invalid indices, malformed PDF objects or invalid source
/// geometry in selected and protected observations.
pub fn mask_source_subset(
    source: &SourcePdf,
    image: MaskSource<'_>,
    lines: &[&OcrLine],
    selected: &[usize],
) -> Result<Option<MaskedSourceObject>, Error> {
    if selected.is_empty() {
        return Ok(None);
    }
    if selected.windows(2).any(|pair| pair[0] >= pair[1])
        || selected.last().is_some_and(|&last| last >= lines.len())
    {
        return Err(Error::Invalid(
            "invalid selected source mask indices".into(),
        ));
    }
    let eligible = selected
        .iter()
        .map(|&index| lines[index])
        .collect::<Vec<_>>();
    let object = source.materialize(image.object_id())?;
    let Some(mut image) = decode_native_samples(source, image, &object)? else {
        return Ok(None);
    };
    let before = image.clone();
    let width = u32::try_from(image.width())
        .map_err(|_| Error::Invalid("source width exceeds u32".into()))?;
    let height = u32::try_from(image.height())
        .map_err(|_| Error::Invalid("source height exceeds u32".into()))?;
    let mut polygons = Vec::with_capacity(eligible.len());
    for line in &eligible {
        polygons.push(to_top_left(&line.quad, width, height)?);
    }
    for polygon in &polygons {
        cover_text(&mut image, polygon, MASK_PADDING);
    }
    let pad = f64::from(MASK_PADDING);
    let affected = polygons
        .iter()
        .map(|polygon| extent(polygon, image.width(), image.height(), pad))
        .collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        if selected.binary_search(&index).is_ok() || line.text.trim().is_empty() {
            continue;
        }
        let protected = to_top_left(&line.quad, width, height)?;
        let bounds = extent(&protected, image.width(), image.height(), pad);
        if affected
            .iter()
            .any(|&target| bounds_overlap(bounds, target))
        {
            restore_region(&before, &mut image, &protected, bounds, pad);
        }
    }
    let metrics = eligible
        .iter()
        .zip(&polygons)
        .map(|(line, polygon)| MaskMetric {
            text: line.text.clone(),
            changed_source_pixels: changed_in_polygon(&before, &image, polygon),
            source_quad: line.quad,
        })
        .collect::<Vec<_>>();
    if metrics
        .iter()
        .all(|metric| metric.changed_source_pixels == 0)
    {
        return Ok(None);
    }
    let payload = image.into_data();
    let length = i64::try_from(payload.len())
        .map_err(|_| Error::Invalid("source stream length exceeds i64".into()))?;
    let mut rewritten = object;
    let OwnedObject::Stream { dictionary, data } = &mut rewritten else {
        return Err(Error::Invalid("image object is not a PDF stream".into()));
    };
    dictionary.retain(|key, _| {
        !matches!(
            key.as_slice(),
            b"Filter" | b"F" | b"FFilter" | b"FDecodeParms" | b"DecodeParms" | b"DP"
        )
    });
    dictionary.insert(b"Length".to_vec(), OwnedObject::Integer(length));
    *data = StreamData::Owned(payload.into());
    Ok(Some(MaskedSourceObject {
        object: rewritten,
        metrics,
    }))
}

/// Decode samples without changing their color-space interpretation.
fn decode_native_samples(
    source: &SourcePdf,
    image: MaskSource<'_>,
    object: &OwnedObject,
) -> Result<Option<PackedSamples>, Error> {
    let Some(dictionary) = object.as_dictionary() else {
        return Ok(None);
    };
    let entry = |key: &[u8]| dictionary.get(key);
    if !matches!(entry(b"Subtype"), Some(OwnedObject::Name(name)) if name == b"Image")
        || [
            b"SMask".as_slice(),
            b"Mask",
            b"Alternates",
            b"F",
            b"SMaskInData",
        ]
        .iter()
        .any(|key| entry(key).is_some())
        || matches!(entry(b"ImageMask"), Some(OwnedObject::Boolean(true)))
    {
        return Ok(None);
    }
    let Some(components) = image.components(source, object)? else {
        return Ok(None);
    };
    let Some(bits) = entry(b"BitsPerComponent").and_then(|raw| match raw {
        OwnedObject::Integer(value) => u8::try_from(*value).ok(),
        _ => None,
    }) else {
        return Ok(None);
    };
    let width = match entry(b"Width") {
        Some(OwnedObject::Integer(value)) => u32::try_from(*value).ok(),
        _ => None,
    };
    let height = match entry(b"Height") {
        Some(OwnedObject::Integer(value)) => u32::try_from(*value).ok(),
        _ => None,
    };
    let (Some(width), Some(height)) = (width, height) else {
        return Ok(None);
    };
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > 30_000_000 {
        return Ok(None);
    }
    // Validate the expected decoded size before invoking a stream codec.
    let Some(row_bits) = u64::from(width)
        .checked_mul(u64::try_from(components).unwrap_or(u64::MAX))
        .and_then(|value| value.checked_mul(u64::from(bits)))
    else {
        return Ok(None);
    };
    let Some(expected_bytes) = row_bits
        .checked_add(7)
        .map(|value| value / 8)
        .and_then(|value| value.checked_mul(u64::from(height)))
    else {
        return Ok(None);
    };
    if expected_bytes > 128 * 1024 * 1024 {
        return Ok(None);
    }
    // Image codecs that cannot expose exact native samples are eligible for
    // page OCR; unsupported source decoding must not abort the whole document.
    let Ok(decoded) = source.decoded_stream_data(image.object_id()) else {
        return Ok(None);
    };
    Ok(PackedSamples::new(
        width,
        height,
        components,
        bits,
        decoded.into_owned(),
    ))
}

fn to_top_left(quad: &Quad, width: u32, height: u32) -> Result<Quad, Error> {
    let width = f64::from(width);
    let height = f64::from(height);
    let result = quad.map(|p| Point(p.0, height - p.1));
    if !convex(&result)
        || result.iter().any(|p| {
            !p.0.is_finite()
                || !p.1.is_finite()
                || p.0 < 0.0
                || p.0 > width
                || p.1 < 0.0
                || p.1 > height
        })
    {
        return Err(Error::Invalid(
            "invalid OCR region outside image source".into(),
        ));
    }
    Ok(result)
}

fn within_quad(quad: &Quad, point: Point, pad: f64) -> bool {
    let mut positive = false;
    let mut negative = false;
    for i in 0..4 {
        let from = quad[i];
        let to = quad[(i + 1) % 4];
        let dx = to.0 - from.0;
        let dy = to.1 - from.1;
        let cross = dx.mul_add(point.1 - from.1, -dy * (point.0 - from.0));
        let tolerance = pad * dx.hypot(dy);
        if cross > tolerance {
            positive = true;
        }
        if cross < -tolerance {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

fn extent(quad: &Quad, width: usize, height: usize, pad: f64) -> PixelBounds {
    let min_x = quad.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = quad.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = quad.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = quad.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let clamp = |value: f64, limit: usize| {
        let bound = num_traits::ToPrimitive::to_f64(&limit).unwrap_or(f64::MAX);
        num_traits::ToPrimitive::to_usize(&value.clamp(0.0, bound)).unwrap_or(limit)
    };
    (
        clamp((min_x - pad).floor(), width),
        clamp((min_y - pad).floor(), height),
        clamp((max_x + pad).ceil(), width),
        clamp((max_y + pad).ceil(), height),
    )
}

type PixelBounds = (usize, usize, usize, usize);

const fn bounds_overlap(a: PixelBounds, b: PixelBounds) -> bool {
    a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
}

/// Restore native component values rather than round-tripping RGB, so
/// protected ink and packed row padding are bit-identical to the source.
fn restore_region(
    original: &PackedSamples,
    image: &mut PackedSamples,
    polygon: &Quad,
    (x0, y0, x1, y1): PixelBounds,
    padding: f64,
) {
    for y in y0..y1 {
        for x in x0..x1 {
            if within_quad(polygon, pixel_center(x, y), padding) {
                for channel in 0..image.components() {
                    image.set(x, y, channel, original.get(x, y, channel));
                }
            }
        }
    }
}

fn pixel_center(x: usize, y: usize) -> Point {
    Point(
        num_traits::ToPrimitive::to_f64(&x).unwrap_or(f64::INFINITY) + 0.5,
        num_traits::ToPrimitive::to_f64(&y).unwrap_or(f64::INFINITY) + 0.5,
    )
}

fn cover_text(image: &mut PackedSamples, polygon: &Quad, padding: u32) {
    let pad = f64::from(padding);
    let (x0, y0, x1, y1) = extent(polygon, image.width(), image.height(), pad + 5.0);
    let mut channels = vec![Vec::<u16>::new(); image.components()];
    for y in y0..y1 {
        for x in x0..x1 {
            let point = pixel_center(x, y);
            if within_quad(polygon, point, pad + 5.0) && !within_quad(polygon, point, pad) {
                for (channel, samples) in channels.iter_mut().enumerate() {
                    samples.push(image.get(x, y, channel));
                }
            }
        }
    }
    if channels[0].is_empty() {
        return;
    }
    let color = channels
        .iter_mut()
        .map(|samples| {
            let middle = samples.len() / 2;
            *samples.select_nth_unstable(middle).1
        })
        .collect::<Vec<_>>();
    for y in y0..y1 {
        for x in x0..x1 {
            if within_quad(polygon, pixel_center(x, y), pad) {
                for (channel, &sample) in color.iter().enumerate() {
                    image.set(x, y, channel, sample);
                }
            }
        }
    }
}

fn changed_in_polygon(before: &PackedSamples, after: &PackedSamples, quad: &Quad) -> u64 {
    count_in_polygon(before.width(), before.height(), quad, |x, y| {
        before.pixel_changed(after, x, y)
    })
}

/// Significant contrast in actual decoded RGB, not in packed sample codes.
pub(crate) fn native_rgb_change_count(
    before: &image::RgbImage,
    after: &image::RgbImage,
    source_quad: &Quad,
) -> Result<u64, Error> {
    if before.dimensions() != after.dimensions() {
        return Err(Error::Invalid("native contrast dimensions changed".into()));
    }
    let quad = to_top_left(source_quad, before.width(), before.height())?;
    Ok(count_in_polygon(
        before.width() as usize,
        before.height() as usize,
        &quad,
        |x, y| {
            let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
                return false;
            };
            crate::rasterize::significant_rgb_change(
                *before.get_pixel(x, y),
                *after.get_pixel(x, y),
            )
        },
    ))
}

fn count_in_polygon(
    width: usize,
    height: usize,
    quad: &Quad,
    mut changed: impl FnMut(usize, usize) -> bool,
) -> u64 {
    let (x0, y0, x1, y1) = extent(quad, width, height, 0.0);
    let mut count = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            if within_quad(quad, pixel_center(x, y), 0.0) && changed(x, y) {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::{to_top_left, within_quad};
    use crate::geometry::Point;

    #[test]
    fn source_quads_are_winding_independent() -> Result<(), crate::Error> {
        let quad = [
            Point(10.0, 10.0),
            Point(110.0, 10.0),
            Point(110.0, 40.0),
            Point(10.0, 40.0),
        ];
        let transformed = to_top_left(&quad, 200, 100)?;
        assert!(within_quad(&transformed, Point(20.0, 75.0), 0.0));
        assert!(!within_quad(&transformed, Point(20.0, 20.0), 0.0));
        let reversed = [
            transformed[3],
            transformed[2],
            transformed[1],
            transformed[0],
        ];
        assert!(within_quad(&reversed, Point(20.0, 75.0), 0.0));
        assert!(to_top_left(&quad, 60, 100).is_err());
        Ok(())
    }
}
