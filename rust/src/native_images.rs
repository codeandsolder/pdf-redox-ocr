//! Poppler-native source raster extraction with exact PDF `XObject` identities.
//!
//! The PDF is parsed only once by pdf-redox for paint geometry. Poppler's
//! pdfimages handles PDF image filters and color conversion without fragile,
//! duplicated image color-space implementations. Source raster OCR is cached by
//! source object ID, while paint geometry remains per occurrence.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    process::{Command, Stdio},
};

use image::RgbImage;
use tempfile::TempDir;

use crate::{Error, PageNumber, native_words::contiguous_runs};

/// Item in Poppler's native image inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInventoryEntry {
    /// One-based PDF page number.
    pub page: PageNumber,
    /// Extraction file ordinal within one Poppler invocation.
    pub number: u32,
    /// Stable PDF `XObject` ID.
    pub source_id: String,
    /// Image type (ordinary image, alpha mask, stencil or inline).
    pub kind: String,
    /// Source-native width.
    pub width: u32,
    /// Source-native height.
    pub height: u32,
}

/// Parse a single Poppler image inventory entry.
///
/// # Errors
/// Returns an error when required fields are malformed. Unknown image types
/// remain ordinary entries, but only `XObject` type "image" can be recognized.
pub fn parse_inventory_line(line: &str) -> Result<Option<ImageInventoryEntry>, Error> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 12
        || fields[0] == "page"
        || fields[0].starts_with('-')
        || !fields[0].bytes().all(|ch| ch.is_ascii_digit())
    {
        return Ok(None);
    }
    let number = fields[1]
        .parse::<u32>()
        .map_err(|_| Error::Invalid("non-numeric pdfimages ordinal".into()))?;
    let page = PageNumber::new(
        fields[0]
            .parse::<u32>()
            .map_err(|_| Error::Invalid("non-numeric pdfimages page".into()))?,
    )?;
    let width = fields[3]
        .parse::<u32>()
        .map_err(|_| Error::Invalid("non-numeric pdfimages width".into()))?;
    let height = fields[4]
        .parse::<u32>()
        .map_err(|_| Error::Invalid("non-numeric pdfimages height".into()))?;
    // Inline images use object ID -1 and have no reusable XObject identity.
    if fields[10] == "-1" {
        return Ok(None);
    }
    let object = fields[10]
        .parse::<u32>()
        .map_err(|_| Error::Invalid("non-numeric pdfimages `XObject` ID".into()))?;
    let generation = fields[11]
        .parse::<u16>()
        .map_err(|_| Error::Invalid("non-numeric pdfimages `XObject` generation".into()))?;
    Ok(Some(ImageInventoryEntry {
        page,
        number,
        source_id: format!("{object}:{generation}"),
        kind: fields[2].to_owned(),
        width,
        height,
    }))
}

/// Independent resource limits for reusable native source raster decoding.
#[derive(Debug, Clone, Copy)]
pub struct NativeImageBudget {
    /// Maximum pixels for one source `XObject`.
    pub max_source_pixels: u64,
    /// Maximum pixels stored across all distinct sources.
    pub max_total_pixels: u64,
    /// Maximum number of distinct source objects.
    pub max_sources: usize,
    /// Upper bound on all images a single pdfimages invocation may extract.
    pub max_extraction_pixels: u64,
}

impl Default for NativeImageBudget {
    fn default() -> Self {
        Self {
            max_source_pixels: 8_000_000,
            max_total_pixels: 16_000_000,
            max_sources: 12,
            max_extraction_pixels: 64_000_000,
        }
    }
}

impl NativeImageBudget {
    /// Check that external and resident image decode limits are nonzero.
    ///
    /// # Errors
    /// Returns an error for a zero resource ceiling.
    pub fn validate(self) -> Result<Self, Error> {
        if self.max_source_pixels == 0
            || self.max_total_pixels == 0
            || self.max_sources == 0
            || self.max_extraction_pixels == 0
        {
            return Err(Error::Invalid("native image budget must be nonzero".into()));
        }
        Ok(self)
    }
}

fn list_native_images(pdf: &Path, arguments: &[String]) -> Result<Vec<ImageInventoryEntry>, Error> {
    let mut args = arguments.to_vec();
    args.push("-list".to_owned());
    let result = Command::new("pdfimages")
        .args(&args)
        .arg(pdf)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !result.status.success() {
        return Err(Error::Poppler(format!(
            "pdfimages inventory failed: {}",
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    let lines = std::str::from_utf8(&result.stdout)
        .map_err(|error| Error::Invalid(format!("pdfimages inventory is not UTF-8: {error}")))?;
    lines
        .lines()
        .map(parse_inventory_line)
        .collect::<Result<Vec<_>, _>>()
        .map(|entries| entries.into_iter().flatten().collect())
}

/// Extract and normalize native source images for selected pages.
///
/// The same PDF image resource may appear on hundreds of pages. Its pixels are
/// retained exactly once in the returned map, keyed by the indirect object.
/// Unsupported inline images and transparency mask objects are never emitted.
/// Candidate images exceeding resource ceilings are skipped rather than loaded.
///
/// # Errors
/// Returns an error when Poppler fails, the image inventory is malformed or a
/// a source raster with matching metadata cannot be decoded.
pub fn native_source_images(
    pdf: &Path,
    selected: &[PageNumber],
    wanted_ids: &BTreeSet<String>,
    budget: NativeImageBudget,
) -> Result<BTreeMap<String, RgbImage>, Error> {
    let budget = budget.validate()?;
    let mut decoded = BTreeMap::new();
    let mut used_pixels = 0_u64;
    if wanted_ids.is_empty() {
        return Ok(decoded);
    }
    // One page per call bounds Poppler's temporary output and allows early
    // termination once all requested sources or the memory budget are used.
    for run in contiguous_runs(selected, 1)? {
        if decoded.len() >= budget.max_sources
            || used_pixels >= budget.max_total_pixels
            || wanted_ids.iter().all(|id| decoded.contains_key(id))
        {
            break;
        }
        let first = run[0].get().to_string();
        let end_page = run[run.len() - 1].get().to_string();
        let mut arguments = vec!["-f".to_owned(), first, "-l".to_owned(), end_page];
        let inventory = list_native_images(pdf, &arguments)?;
        let extraction_pixels = inventory.iter().fold(0_u64, |acc, entry| {
            acc.saturating_add(u64::from(entry.width) * u64::from(entry.height))
        });
        if extraction_pixels > budget.max_extraction_pixels {
            // pdfimages emits every source on the requested page, including
            // ones we don't want, so even rejected sources consume resources.
            continue;
        }
        let candidates = inventory
            .iter()
            .filter(|entry| {
                wanted_ids.contains(&entry.source_id)
                    && !decoded.contains_key(&entry.source_id)
                    && entry.kind == "image"
                    && u64::from(entry.width) * u64::from(entry.height) <= budget.max_source_pixels
                    && entry.width > 0
                    && entry.height > 0
                    && u64::from(entry.width) * u64::from(entry.height)
                        <= budget.max_total_pixels.saturating_sub(used_pixels)
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            continue;
        }
        let temp = TempDir::new()?;
        let prefix = temp.path().join("image");
        arguments.push("-png".into());
        let extract = Command::new("pdfimages")
            .args(&arguments)
            .arg(pdf)
            .arg(&prefix)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;
        if !extract.status.success() {
            return Err(Error::Poppler(format!(
                "pdfimages native extraction failed: {}",
                String::from_utf8_lossy(&extract.stderr)
            )));
        }
        for entry in candidates {
            if decoded.contains_key(&entry.source_id) || decoded.len() >= budget.max_sources {
                continue;
            }
            let pixels = u64::from(entry.width) * u64::from(entry.height);
            if pixels > budget.max_total_pixels.saturating_sub(used_pixels) {
                continue;
            }
            // Poppler indexes all emitted images including masks. The summary
            // ordinal is the same ordinal used for generated PNG filenames.
            let file = prefix.with_file_name(format!("image-{:03}.png", entry.number));
            let rgb = image::open(&file)
                .map_err(|error| {
                    Error::Invalid(format!(
                        "unable to decode source {}: {error}",
                        entry.source_id
                    ))
                })?
                .into_rgb8();
            if rgb.width() != entry.width || rgb.height() != entry.height {
                return Err(Error::Invalid(format!(
                    "pdfimages source {} raster dimensions do not match inventory",
                    entry.source_id
                )));
            }
            used_pixels += pixels;
            decoded.insert(entry.source_id.clone(), rgb);
        }
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::parse_inventory_line;

    #[test]
    fn inventory_preserves_pdf_source_identity_across_repeated_pages() -> Result<(), crate::Error> {
        let first = parse_inventory_line(
            "   1     0 image    1200   300  rgb     3   8  image  no         9  0   173   173 13.8K 1.3%"
        )?.ok_or_else(|| crate::Error::Invalid("missing inventory".into()))?;
        let second = parse_inventory_line(
            "   2     1 image    1200   300  rgb     3   8  image  no         9  0   173   173 13.8K 1.3%"
        )?.ok_or_else(|| crate::Error::Invalid("missing inventory".into()))?;
        assert_eq!(first.source_id, second.source_id);
        assert_eq!(first.source_id, "9:0");
        assert_eq!(first.width, 1200);
        assert_eq!(first.height, 300);
        assert_eq!(first.page.get(), 1);
        assert_eq!(second.page.get(), 2);
        assert_eq!(second.number, 1);
        assert!(parse_inventory_line("-------")?.is_none());
        assert!(parse_inventory_line("page num type width")?.is_none());
        assert!(
            parse_inventory_line("1 1 image 100 100 rgb 3 8 image no -1 0 300 300 1K 1%")?
                .is_none()
        );
        Ok(())
    }
}
