//! Render exact PDF pages with Poppler without writing temporary page images.
//!
//! Recognition receives one bounded, 8-bit RGB page. Its image pixel geometry
//! is returned to the caller for conversion into PDF user-space coordinates.

use std::{
    path::Path,
    process::{Command, Stdio},
};

use image::{ImageFormat, RgbImage};

use crate::{
    Error, PageNumber,
    geometry::{DisplayMapper, PageBox, Point},
};

/// Rendering policy shared by full-page fallback and regression measurements.
#[derive(Clone, Copy, Debug)]
pub struct RenderPolicy {
    /// Maximum rendered pixel dimension.
    pub max_side: u32,
    /// Base rendering DPI before the max-side constraint is applied.
    pub dpi: u32,
}

impl Default for RenderPolicy {
    fn default() -> Self {
        Self {
            max_side: 1600,
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

struct CropRequest {
    width: u32,
    height: u32,
    mapper: CropPixelMapper,
}

/// Calculate a bounded device-space rectangle before invoking Poppler.
fn calculate_crop(
    mapper: DisplayMapper,
    region: PageBox,
    dpi: u32,
    max_pixels: u64,
) -> Result<CropRequest, Error> {
    if !(36..=2400).contains(&dpi) || max_pixels == 0 {
        return Err(Error::Invalid("invalid native mosaic render budget".into()));
    }
    let right = mapper.to_pdf(1.0, 0.0);
    let bottom = mapper.to_pdf(0.0, 1.0);
    let first = mapper.to_pdf(0.0, 0.0);
    let display_width = (right.0 - first.0).hypot(right.1 - first.1);
    let display_height = (bottom.0 - first.0).hypot(bottom.1 - first.1);
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
    let start_x = (left.floor() - 1.0).max(0.0);
    let start_y = (top.floor() - 1.0).max(0.0);
    let end_x = (max_x.ceil() + 1.0).min((display_width / points_per_pixel).ceil());
    let end_y = (max_y.ceil() + 1.0).min((display_height / points_per_pixel).ceil());
    let to_u32 = |value: f64| num_traits::ToPrimitive::to_u32(&value);
    let (Some(origin_x), Some(origin_y), Some(end_x), Some(end_y)) = (
        to_u32(start_x),
        to_u32(start_y),
        to_u32(end_x),
        to_u32(end_y),
    ) else {
        return Err(Error::Invalid(
            "native mosaic crop coordinates exceed u32".into(),
        ));
    };
    let (Some(width), Some(height)) = (end_x.checked_sub(origin_x), end_y.checked_sub(origin_y))
    else {
        return Err(Error::Invalid("invalid native mosaic crop extent".into()));
    };
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > max_pixels {
        return Err(Error::Invalid(
            "native mosaic raster exceeds pixel budget".into(),
        ));
    }
    Ok(CropRequest {
        width,
        height,
        mapper: CropPixelMapper {
            origin_x,
            origin_y,
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
    let crop = calculate_crop(mapper, region, dpi, max_pixels)?;
    let CropPixelMapper {
        origin_x, origin_y, ..
    } = crop.mapper;
    let (width, height) = (crop.width, crop.height);
    let pixels = u64::from(width) * u64::from(height);
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
    if rgb.width() != width || rgb.height() != height {
        return Err(Error::Invalid(
            "Poppler returned wrong native crop dimensions".into(),
        ));
    }
    Ok(RenderedCrop {
        rgb,
        mapper: crop.mapper,
    })
}

#[cfg(test)]
mod tests {
    use super::RenderPolicy;

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
