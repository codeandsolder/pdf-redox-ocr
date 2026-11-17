//! Render exact PDF pages with Poppler without writing temporary page images.
//!
//! Recognition receives one bounded, 8-bit RGB page. Its image pixel geometry
//! is returned to the caller for conversion into PDF user-space coordinates.

use std::{
    path::Path,
    process::{Command, Stdio},
};

use image::{ImageFormat, RgbImage};

use crate::{Error, PageNumber};

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
