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
    pdf_metadata::resolve_owned,
    polygon::area,
    source_samples::PackedSamples,
};

/// Quantifies the specific source text modified by a counterfactual.
#[derive(Debug, Clone)]
pub struct MaskMetric {
    /// OCR text whose image pixels were suppressed.
    pub text: String,
    /// Source pixels changed inside the original OCR polygon.
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
    let Some(masked) = mask_source_object(doc.source(), source_id, lines, min_confidence)? else {
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
    source_id: ObjectId,
    lines: &[OcrLine],
    min_confidence: f64,
) -> Result<Option<MaskedSourceObject>, Error> {
    if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
        return Err(Error::Invalid("invalid source mask confidence".into()));
    }
    let eligible = lines
        .iter()
        .filter(|line| line.confidence >= min_confidence && line.text.trim().chars().count() >= 2)
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return Ok(None);
    }
    let object = source.materialize(source_id)?;
    let Some(mut image) = decode_native_samples(source, source_id, &object)? else {
        return Ok(None);
    };
    let before = image.clone();
    let width = u32::try_from(image.width)
        .map_err(|_| Error::Invalid("source width exceeds u32".into()))?;
    let height = u32::try_from(image.height)
        .map_err(|_| Error::Invalid("source height exceeds u32".into()))?;
    let mut polygons = Vec::with_capacity(eligible.len());
    for line in &eligible {
        polygons.push(to_top_left(&line.quad, width, height)?);
    }
    for polygon in &polygons {
        cover_text(&mut image, polygon, 3);
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
    let payload = image.data;
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
    *data = StreamData::Owned(payload);
    Ok(Some(MaskedSourceObject {
        object: rewritten,
        metrics,
    }))
}

/// Decode samples without changing their color-space interpretation.
fn decode_native_samples(
    source: &SourcePdf,
    source_id: ObjectId,
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
    let Some(components) = entry(b"ColorSpace")
        .map(|raw| color_components(source, raw))
        .transpose()?
        .flatten()
    else {
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
    let Ok(decoded) = source.decoded_stream_data(source_id) else {
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

/// Count PDF color-space components without performing color conversion.
/// Unknown named aliases require page/Form resources and cannot be guessed.
fn color_components(source: &SourcePdf, raw: &OwnedObject) -> Result<Option<usize>, Error> {
    let resolved = resolve_owned(source, raw.clone())?;
    let OwnedObject::Array(items) = &resolved else {
        return Ok(match resolved {
            OwnedObject::Name(name) => match name.as_slice() {
                b"DeviceGray" => Some(1),
                b"DeviceRGB" => Some(3),
                b"DeviceCMYK" => Some(4),
                _ => None,
            },
            _ => None,
        });
    };
    let Some(first) = items.first() else {
        return Ok(None);
    };
    let OwnedObject::Name(kind) = resolve_owned(source, first.clone())? else {
        return Ok(None);
    };
    let count = match kind.as_slice() {
        b"CalGray" if items.len() == 2 => Some(1),
        b"CalRGB" | b"Lab" if items.len() == 2 => Some(3),
        b"Indexed" | b"I" | b"Separation" if items.len() == 4 => Some(1),
        b"DeviceN" if items.len() >= 4 => match resolve_owned(source, items[1].clone())? {
            OwnedObject::Array(names)
                if names.iter().all(|n| matches!(n, OwnedObject::Name(_))) =>
            {
                Some(names.len())
            }
            _ => None,
        },
        b"ICCBased" if items.len() == 2 => {
            let profile = resolve_owned(source, items[1].clone())?;
            profile
                .as_dictionary()
                .and_then(|dict| dict.get(b"N".as_slice()))
                .and_then(|value| match value {
                    OwnedObject::Integer(value) => usize::try_from(*value).ok(),
                    _ => None,
                })
                .filter(|channels| matches!(*channels, 1 | 3 | 4))
        }
        _ => None,
    };
    Ok(count.filter(|channels| (1..=32).contains(channels)))
}

fn to_top_left(quad: &Quad, width: u32, height: u32) -> Result<Quad, Error> {
    let width = f64::from(width);
    let height = f64::from(height);
    let result = quad.map(|p| Point(p.0, height - p.1));
    if area(&result) <= 1e-8
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

fn extent(quad: &Quad, width: usize, height: usize, pad: f64) -> (usize, usize, usize, usize) {
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

fn pixel_center(x: usize, y: usize) -> Point {
    Point(
        num_traits::ToPrimitive::to_f64(&x).unwrap_or(f64::INFINITY) + 0.5,
        num_traits::ToPrimitive::to_f64(&y).unwrap_or(f64::INFINITY) + 0.5,
    )
}

fn cover_text(image: &mut PackedSamples, polygon: &Quad, padding: u32) {
    let pad = f64::from(padding);
    let (x0, y0, x1, y1) = extent(polygon, image.width, image.height, pad + 5.0);
    let mut channels = vec![Vec::<u16>::new(); image.components];
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
    let (x0, y0, x1, y1) = extent(quad, before.width, before.height, 0.0);
    let mut count = 0_u64;
    for y in y0..y1 {
        for x in x0..x1 {
            if within_quad(quad, pixel_center(x, y), 0.0) && before.pixel_changed(after, x, y) {
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
