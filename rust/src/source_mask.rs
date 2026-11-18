//! Source-specific counterfactual image-text masks for PDF visibility proofs.
//!
//! The PDF is never edited in place: copy-on-write changes one image `XObject`,
//! preserving all original paint operations and shared Form resource references.
//! Unknown color decoding, transparency, alpha and non-8-bit pixels are refused.

use image::{Rgb, RgbImage};
use pdf_redox::{EditDocument, ObjectId, OwnedObject, StreamData};

use crate::{
    Error,
    draft::OcrLine,
    geometry::{Point, Quad},
    polygon::area,
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

/// Create one source-specific counterfactual in Rust.
///
/// The accepted source image must be an 8-bit device-RGB `XObject` whose
/// stream decodes to exact native RGB pixels without color conversions.
/// Other encodings return None and remain eligible for ordinary page OCR.
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

    let mut doc = EditDocument::from_bytes(original.to_vec())?;
    let source = doc.source();
    let object = source.materialize(source_id)?;
    let OwnedObject::Stream { dictionary, .. } = &object else {
        return Ok(None);
    };
    let entry = |key: &[u8]| dictionary.get(key);
    if !matches!(entry(b"Subtype"), Some(OwnedObject::Name(name)) if name == b"Image")
        || !matches!(entry(b"ColorSpace"), Some(OwnedObject::Name(name)) if name == b"DeviceRGB")
        || !matches!(entry(b"BitsPerComponent"), Some(OwnedObject::Integer(8)))
        || [b"SMask".as_slice(), b"Mask", b"Decode", b"Alternates", b"F"]
            .iter()
            .any(|key| entry(key).is_some())
        || matches!(entry(b"ImageMask"), Some(OwnedObject::Boolean(true)))
    {
        return Ok(None);
    }
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
    let decoded = source.decoded_stream_data(source_id)?;
    let Some(byte_count) = pixels
        .checked_mul(3)
        .and_then(|count| usize::try_from(count).ok())
    else {
        return Ok(None);
    };
    if decoded.len() != byte_count {
        return Ok(None);
    }
    let Some(mut image) = RgbImage::from_raw(width, height, decoded.into_owned()) else {
        return Ok(None);
    };
    let before = image.clone();
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
    let payload = image.into_raw();
    let length = i64::try_from(payload.len())
        .map_err(|_| Error::Invalid("source stream length exceeds i64".into()))?;
    let object = doc.edit_object(source_id)?;
    let OwnedObject::Stream { dictionary, data } = object else {
        return Err(Error::Invalid(
            "image object changed during copy-on-write edit".into(),
        ));
    };
    dictionary.retain(|key, _| {
        !matches!(
            key.as_slice(),
            b"Filter" | b"F" | b"FFilter" | b"FDecodeParms" | b"DecodeParms" | b"DP"
        )
    });
    dictionary.insert(b"Length".to_vec(), OwnedObject::Integer(length));
    *data = StreamData::Owned(payload);
    Ok(Some(MaskedSourcePdf {
        bytes: doc.write_compact()?,
        metrics,
    }))
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

fn extent(quad: &Quad, width: u32, height: u32, pad: f64) -> (u32, u32, u32, u32) {
    let min_x = quad.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = quad.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = quad.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = quad.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let clamp = |v: f64, limit: u32| {
        num_traits::ToPrimitive::to_u32(&v.clamp(0.0, f64::from(limit))).unwrap_or(limit)
    };
    (
        clamp((min_x - pad).floor(), width),
        clamp((min_y - pad).floor(), height),
        clamp((max_x + pad).ceil(), width),
        clamp((max_y + pad).ceil(), height),
    )
}

fn cover_text(image: &mut RgbImage, polygon: &Quad, padding: u32) {
    let pad = f64::from(padding);
    let (x0, y0, x1, y1) = extent(polygon, image.width(), image.height(), pad + 5.0);
    let mut channels: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());
    for y in y0..y1 {
        for x in x0..x1 {
            let point = Point(f64::from(x) + 0.5, f64::from(y) + 0.5);
            if within_quad(polygon, point, pad + 5.0) && !within_quad(polygon, point, pad) {
                let pixel = image.get_pixel(x, y);
                for (index, samples) in channels.iter_mut().enumerate() {
                    samples.push(pixel[index]);
                }
            }
        }
    }
    if channels[0].is_empty() {
        return;
    }
    for samples in &mut channels {
        samples.sort_unstable();
    }
    let median = channels[0].len() / 2;
    let color = Rgb([
        channels[0][median],
        channels[1][median],
        channels[2][median],
    ]);
    for y in y0..y1 {
        for x in x0..x1 {
            if within_quad(polygon, Point(f64::from(x) + 0.5, f64::from(y) + 0.5), pad) {
                *image.get_pixel_mut(x, y) = color;
            }
        }
    }
}

fn changed_in_polygon(before: &RgbImage, after: &RgbImage, quad: &Quad) -> u64 {
    let (x0, y0, x1, y1) = extent(quad, before.width(), before.height(), 0.0);
    let mut count = 0_u64;
    for y in y0..y1 {
        for x in x0..x1 {
            if !within_quad(quad, Point(f64::from(x) + 0.5, f64::from(y) + 0.5), 0.0) {
                continue;
            }
            if before
                .get_pixel(x, y)
                .0
                .iter()
                .zip(after.get_pixel(x, y).0)
                .any(|(left, right)| left.abs_diff(right) > 8)
            {
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
