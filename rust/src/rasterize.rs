//! Render exact PDF pages with Poppler without writing temporary page images.
//!
//! Recognition receives one bounded, 8-bit RGB page. Its image pixel geometry
//! is returned to the caller for conversion into PDF user-space coordinates.

use std::{
    path::Path,
    process::{Command, Stdio},
};

use image::Rgb;
use image::{ImageFormat, RgbImage};

use crate::{
    Error, PageNumber,
    geometry::{DisplayMapper, PageBox, Point},
};

/// Empirically validated default page render limit, below the 2048-pixel detector bound.
pub const DEFAULT_PAGE_MAX_SIDE: u32 = 1920;

/// Rendering policy shared by full-page fallback and regression measurements.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RenderPolicy {
    /// Maximum rendered pixel dimension.
    pub max_side: u32,
    /// Base rendering DPI before the max-side constraint is applied.
    pub dpi: u32,
}

impl Default for RenderPolicy {
    fn default() -> Self {
        Self {
            max_side: DEFAULT_PAGE_MAX_SIDE,
            dpi: 72,
        }
    }
}

impl RenderPolicy {
    /// Reject pathological or silently degraded image settings.
    ///
    /// # Errors
    /// Returns an error for dimensions or DPI outside the supported Poppler range.
    pub fn validate(self) -> Result<Self, Error> {
        if !(128..=16_384).contains(&self.max_side) || !(36..=2400).contains(&self.dpi) {
            return Err(Error::Invalid("invalid page raster resolution".into()));
        }
        Ok(self)
    }
}

/// Render a single PDF page to in-memory RGB pixels.
///
/// # Errors
/// Returns an error for a malformed PDF, a failed renderer, or invalid pixels.
pub fn render_rgb_page(
    pdf: &Path,
    number: PageNumber,
    policy: RenderPolicy,
) -> Result<RgbImage, Error> {
    let policy = policy.validate()?;
    let output = Command::new("pdftoppm")
        .arg("-f")
        .arg(number.get().to_string())
        .arg("-l")
        .arg(number.get().to_string())
        .arg("-singlefile")
        .arg("-r")
        .arg(policy.dpi.to_string())
        .arg("-scale-to")
        .arg(policy.max_side.to_string())
        .arg(pdf)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        return Err(Error::Poppler(format!(
            "pdftoppm failed for page {}: {}",
            number.get(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let bitmap = image::load_from_memory_with_format(&output.stdout, ImageFormat::Pnm)?;
    let rgb = bitmap.into_rgb8();
    if rgb.width() == 0 || rgb.height() == 0 || rgb.width().max(rgb.height()) > policy.max_side + 1
    {
        return Err(Error::Invalid(
            "unexpected Poppler raster dimensions".into(),
        ));
    }
    Ok(rgb)
}

/// Native-pitch Poppler raster cropped in device pixels, with an exact
/// mapping from *cropped image pixel coordinates* into original PDF space.
pub struct RenderedCrop {
    /// Rasterized visible PDF paint in the selected source-resolution region.
    pub rgb: RgbImage,
    mapper: CropPixelMapper,
}

/// Copyable coordinate transform that outlives moving the cropped raster.
#[derive(Clone, Copy)]
pub struct CropPixelMapper {
    origin_x: u32,
    origin_y: u32,
    point_per_pixel: f64,
    display_width: f64,
    display_height: f64,
    mapper: DisplayMapper,
}

impl CropPixelMapper {
    /// Inverse map from original PDF user space to cropped raster pixel frame.
    #[must_use]
    pub fn to_raster(self, point: Point) -> Point {
        let Point(u, v) = self.mapper.to_raster(point);
        Point(
            u * self.display_width / self.point_per_pixel - f64::from(self.origin_x),
            v * self.display_height / self.point_per_pixel - f64::from(self.origin_y),
        )
    }

    /// Map raster pixel corners directly into original PDF user coordinates.
    #[must_use]
    pub fn to_pdf(self, Point(x, y): Point) -> Point {
        let u = (f64::from(self.origin_x) + x) * self.point_per_pixel / self.display_width;
        let v = (f64::from(self.origin_y) + y) * self.point_per_pixel / self.display_height;
        self.mapper.to_pdf(u, v)
    }
}

impl RenderedCrop {
    /// Preserve the exact pixel-to-page mapping when transferring the image
    /// buffer to the OCR backend.
    #[must_use]
    pub const fn pixel_mapper(&self) -> CropPixelMapper {
        self.mapper
    }

    /// Project cropped image pixel coordinates to original PDF user space.
    #[must_use]
    pub fn pixel_to_pdf(&self, point: Point) -> Point {
        self.pixel_mapper().to_pdf(point)
    }
}

pub(crate) struct CropRequest {
    dpi: u32,
    width: u32,
    height: u32,
    min_dimensions: (u32, u32),
    mapper: CropPixelMapper,
}

/// Enclose the two device-scale computations, including their rounding:
/// points * (dpi / 72) and points / (72 / dpi). A PDF page exactly at an
/// integer device coordinate can yield either neighboring ceiling in
/// different renderers. Preserve that uncertainty only at the page edge.
fn device_extent(points: f64, dpi: u32) -> [f64; 2] {
    let scale = f64::from(dpi) / 72.0;
    let pitch = 72.0 / f64::from(dpi);
    [
        (points * scale.next_down())
            .min(points / pitch.next_up())
            .next_down()
            .ceil(),
        (points * scale.next_up())
            .max(points / pitch.next_down())
            .next_up()
            .ceil(),
    ]
}

struct CropAxis {
    origin: u32,
    extent: u32,
    min_extent: u32,
}

fn crop_axis(start: f64, end: f64, page_points: f64, dpi: u32) -> Result<CropAxis, Error> {
    let [page_min, page_max] = device_extent(page_points, dpi);
    let start = (start.floor() - 1.0).max(0.0);
    let end = (end.ceil() + 1.0).min(page_max);
    let to_u32 = |v: f64| num_traits::ToPrimitive::to_u32(&v);
    let (Some(origin), Some(end), Some(min_end)) =
        (to_u32(start), to_u32(end), to_u32(end.min(page_min)))
    else {
        return Err(Error::Invalid(
            "native mosaic crop coordinates exceed u32".into(),
        ));
    };
    let extent = end
        .checked_sub(origin)
        .ok_or_else(|| Error::Invalid("invalid native mosaic crop extent".into()))?;
    Ok(CropAxis {
        origin,
        extent,
        min_extent: min_end.saturating_sub(origin),
    })
}

/// Calculate a bounded device-space rectangle before invoking Poppler.
pub(crate) fn calculate_crop(
    mapper: DisplayMapper,
    region: PageBox,
    dpi: u32,
    max_pixels: u64,
) -> Result<CropRequest, Error> {
    if !(36..=2400).contains(&dpi) || max_pixels == 0 {
        return Err(Error::Invalid("invalid native mosaic render budget".into()));
    }
    let Point(display_width, display_height) = mapper.display_dimensions();
    let points_per_pixel = 72.0 / f64::from(dpi);
    let (mut left, mut top, mut max_x, mut max_y) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for corner in [
        Point(region.x0, region.y0),
        Point(region.x1, region.y0),
        Point(region.x1, region.y1),
        Point(region.x0, region.y1),
    ] {
        let Point(u, v) = mapper.to_raster(corner);
        let x = u * display_width / points_per_pixel;
        let y = v * display_height / points_per_pixel;
        left = left.min(x);
        top = top.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    // One source pixel of guard space at region boundaries helps OCR retain
    // glyph edges. It is included in both budgets and the coordinate mapping.
    let x = crop_axis(left, max_x, display_width, dpi)?;
    let y = crop_axis(top, max_y, display_height, dpi)?;
    let (width, height) = (x.extent, y.extent);
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > max_pixels {
        return Err(Error::Invalid(
            "native mosaic raster exceeds pixel budget".into(),
        ));
    }
    Ok(CropRequest {
        dpi,
        width,
        height,
        min_dimensions: (x.min_extent, y.min_extent),
        mapper: CropPixelMapper {
            origin_x: x.origin,
            origin_y: y.origin,
            point_per_pixel: points_per_pixel,
            display_width,
            display_height,
            mapper,
        },
    })
}

/// Render a source-resolution crop using Poppler's complete PDF paint stack.
///
/// Honors image codecs, alpha masks, ICC colors, arbitrary clips, page
/// rotations and nonzero page origins, without a full-resolution page raster.
///
/// # Errors
/// Returns an error for invalid or oversized crop geometry or Poppler failure.
pub fn render_rgb_region(
    pdf: &Path,
    page: PageNumber,
    mapper: DisplayMapper,
    region: PageBox,
    dpi: u32,
    max_pixels: u64,
) -> Result<RenderedCrop, Error> {
    render_crop(pdf, page, &calculate_crop(mapper, region, dpi, max_pixels)?)
}

impl CropRequest {
    pub(crate) fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

pub(crate) fn render_crop(
    pdf: &Path,
    page: PageNumber,
    crop: &CropRequest,
) -> Result<RenderedCrop, Error> {
    let pixels = crop.pixels();
    let CropRequest {
        dpi,
        width,
        height,
        min_dimensions,
        mapper,
    } = *crop;
    let CropPixelMapper {
        origin_x, origin_y, ..
    } = mapper;
    let output = Command::new("pdftoppm")
        .args(["-f", &page.get().to_string()])
        .args(["-l", &page.get().to_string()])
        .arg("-singlefile")
        .args(["-r", &dpi.to_string()])
        .args(["-x", &origin_x.to_string()])
        .args(["-y", &origin_y.to_string()])
        .args(["-W", &width.to_string()])
        .args(["-H", &height.to_string()])
        .arg(pdf)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        return Err(Error::Poppler(format!(
            "native crop render failed on page {}: {}",
            page.get(),
            String::from_utf8_lossy(&output.stderr),
        )));
    }
    if u64::try_from(output.stdout.len()).unwrap_or(u64::MAX)
        > pixels.saturating_mul(3).saturating_add(4096)
    {
        return Err(Error::Invalid(
            "Poppler native crop exceeded output budget".into(),
        ));
    }
    let rgb = image::load_from_memory_with_format(&output.stdout, ImageFormat::Pnm)?.into_rgb8();
    // The requested guard pixel can be clipped at an uncertain fractional
    // page edge. The actual raster keeps exactly the original origin and
    // native pitch; no resizing or synthetic extra column is needed.
    if rgb.width() == 0
        || rgb.height() == 0
        || !(min_dimensions.0..=width).contains(&rgb.width())
        || !(min_dimensions.1..=height).contains(&rgb.height())
    {
        return Err(Error::Invalid(
            "Poppler returned wrong native crop dimensions".into(),
        ));
    }
    Ok(RenderedCrop { rgb, mapper })
}

/// Sample pixel-center RGB coordinates with a white exterior.
pub(crate) fn sample_whitened(image: &RgbImage, Point(x, y): Point) -> Rgb<u8> {
    // Pixel centers are at i+0.5; sample in that convention. Sampling
    // outside the region contributes white exactly as Poppler's backdrop.
    let x = x - 0.5;
    let y = y - 0.5;
    let left = x.floor();
    let top = y.floor();
    let dx = x - left;
    let dy = y - top;
    let mut channels = [0.0_f64; 3];
    for (offset_y, weight_y) in [(0.0, 1.0 - dy), (1.0, dy)] {
        for (offset_x, weight_x) in [(0.0, 1.0 - dx), (1.0, dx)] {
            let rgba = num_traits::ToPrimitive::to_u32(&(left + offset_x))
                .zip(num_traits::ToPrimitive::to_u32(&(top + offset_y)))
                .filter(|(column, row)| *column < image.width() && *row < image.height())
                .map_or([255, 255, 255], |(column, row)| {
                    image.get_pixel(column, row).0
                });
            let weight = weight_x * weight_y;
            for index in 0..3 {
                channels[index] = f64::from(rgba[index]).mul_add(weight, channels[index]);
            }
        }
    }
    Rgb(channels.map(|channel| num_traits::ToPrimitive::to_u8(&channel.round()).unwrap_or(255)))
}

#[cfg(feature = "pdf-integration")]
/// Shared native and page RGB contrast criterion.
pub(crate) fn significant_rgb_change(before: Rgb<u8>, after: Rgb<u8>) -> bool {
    before.0.iter().zip(after.0).any(|(a, b)| a.abs_diff(b) > 8)
}

#[cfg(test)]
mod tests {
    use super::{RenderPolicy, crop_axis};

    #[test]
    fn only_page_edge_can_clip_rounding_guard() -> Result<(), crate::Error> {
        let edge = crop_axis(0.0, 42.24 / 0.12, 42.24, 600)?;
        assert_eq!((edge.origin, edge.min_extent, edge.extent), (0, 352, 353));
        let inside = crop_axis(20.0, 100.0, 42.24, 600)?;
        assert_eq!(
            (inside.origin, inside.min_extent, inside.extent),
            (19, 82, 82)
        );
        let fractional = crop_axis(0.0, 42.25 / 0.12, 42.25, 600)?;
        assert_eq!(fractional.min_extent, fractional.extent);
        assert_eq!(fractional.extent, 353);
        Ok(())
    }

    #[test]
    fn validates_rendering_limits() {
        assert!(RenderPolicy::default().validate().is_ok());
        assert!(
            RenderPolicy {
                max_side: 0,
                dpi: 72
            }
            .validate()
            .is_err()
        );
        assert!(
            RenderPolicy {
                max_side: 1600,
                dpi: 1
            }
            .validate()
            .is_err()
        );
    }
}
