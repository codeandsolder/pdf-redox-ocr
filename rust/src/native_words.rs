//! Batched Poppler XHTML extraction of native searchable word geometry.
//!
//! Empty and invalid words do not block OCR; invalid page dimensions fail.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    process::{Command, Stdio},
};

use crate::{
    Error, PageNumber,
    geometry::{DisplayMapper, Point, Quad},
};
use quick_xml::{Reader, events::Event};

fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn xml_attribute(event: &quick_xml::events::BytesStart<'_>, name: &str) -> Result<f64, Error> {
    for attribute in event.attributes() {
        let attribute =
            attribute.map_err(|e| Error::Invalid(format!("invalid XML attribute: {e}")))?;
        if local_name(attribute.key.as_ref()) == name {
            let value = attribute.value.as_ref();
            let parsed = value.parse::<f64>()?;
            if parsed.is_finite() {
                return Ok(parsed);
            }
            return Err(Error::Invalid("nonfinite native word coordinate".into()));
        }
    }
    Err(Error::Invalid(format!(
        "missing native word attribute {name}"
    )))
}

#[derive(Clone, Copy)]
struct WordBounds {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl WordBounds {
    fn new(event: &quick_xml::events::BytesStart<'_>) -> Result<Self, Error> {
        Ok(Self {
            x0: xml_attribute(event, "xMin")?,
            y0: xml_attribute(event, "yMin")?,
            x1: xml_attribute(event, "xMax")?,
            y1: xml_attribute(event, "yMax")?,
        })
    }

    fn to_quad(self, width: f64, height: f64, mapper: DisplayMapper) -> Option<Quad> {
        if self.x1 - self.x0 < 0.25 || self.y1 - self.y0 < 0.25 {
            return None;
        }
        let to_point = |x: f64, y: f64| -> Point { mapper.to_pdf(x / width, y / height) };
        Some([
            to_point(self.x0, self.y1),
            to_point(self.x1, self.y1),
            to_point(self.x1, self.y0),
            to_point(self.x0, self.y0),
        ])
    }
}

/// Parse Poppler XHTML into one vector of PDF-space quads per page.
///
/// # Errors
/// Returns an error for malformed XML, invalid numeric positions or missing pages.
///
/// XML C0 controls are stripped because malformed PDF text encodings may
/// otherwise cause a complete loss of native-word geometry.
pub fn parse_bbox_xhtml(
    data: &[u8],
    transforms: &[DisplayMapper],
) -> Result<Vec<Vec<Quad>>, Error> {
    let cleaned = data
        .iter()
        .copied()
        .filter(|byte| !matches!(byte, 0..=8 | 11 | 12 | 14..=31))
        .collect::<Vec<_>>();
    let mut reader = Reader::from_reader(cleaned.as_slice());
    let mut buffer = Vec::new();
    let mut pages: Vec<Vec<Quad>> = Vec::new();
    let mut page_width = 0.0;
    let mut page_height = 0.0;
    let mut word: Option<WordBounds> = None;
    let mut word_has_text = false;

    loop {
        match reader.read_event_into(&mut buffer)? {
            Event::Start(ref start) if local_name(start.name().as_ref()) == "page" => {
                page_width = xml_attribute(start, "width")?;
                page_height = xml_attribute(start, "height")?;
                if page_width <= 0.0 || page_height <= 0.0 {
                    return Err(Error::Invalid("invalid native-text page dimensions".into()));
                }
                if pages.len() >= transforms.len() {
                    return Err(Error::Invalid("more Poppler pages than requested".into()));
                }
                pages.push(Vec::new());
            }
            Event::Start(ref start) if local_name(start.name().as_ref()) == "word" => {
                word = Some(WordBounds::new(start)?);
                word_has_text = false;
            }
            Event::Text(ref text) if word.is_some() => {
                let unescaped = quick_xml::escape::unescape(text.as_ref())
                    .map_err(|e| Error::Invalid(format!("invalid XHTML text entity: {e}")))?;
                word_has_text |= !unescaped.trim().is_empty();
            }
            Event::GeneralRef(ref reference) if word.is_some() => {
                let name = reference.as_ref();
                let entity = format!("&{name};");
                let unescaped = quick_xml::escape::unescape(&entity)
                    .map_err(|e| Error::Invalid(format!("invalid XHTML text entity: {e}")))?;
                word_has_text |= !unescaped.trim().is_empty();
            }
            Event::End(ref end) if local_name(end.name().as_ref()) == "word" => {
                if let Some(bounds) = word.take()
                    && word_has_text
                {
                    let index = pages
                        .len()
                        .checked_sub(1)
                        .ok_or_else(|| Error::Invalid("word outside Poppler page".into()))?;
                    if let Some(quad) = bounds.to_quad(page_width, page_height, transforms[index]) {
                        pages[index].push(quad);
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }

    if pages.len() != transforms.len() {
        return Err(Error::Invalid(format!(
            "unexpected native text pages: requested {}, got {}",
            transforms.len(),
            pages.len()
        )));
    }
    Ok(pages)
}

/// Partition arbitrary one-based pages into bounded contiguous extraction runs.
///
/// # Errors
/// Returns an error for no pages, duplicate pages, or a zero chunk size.
pub fn contiguous_runs(
    pages: &[PageNumber],
    chunk_size: usize,
) -> Result<Vec<Vec<PageNumber>>, Error> {
    if chunk_size == 0 || pages.is_empty() {
        return Err(Error::Invalid(
            "empty page selection or zero chunk size".into(),
        ));
    }
    let ordered = pages.iter().copied().collect::<BTreeSet<_>>();
    if ordered.len() != pages.len() {
        return Err(Error::Invalid("duplicate page selection".into()));
    }
    let mut runs: Vec<Vec<PageNumber>> = Vec::new();
    for page in ordered {
        if let Some(last) = runs.last_mut()
            && last.len() < chunk_size
            && last
                .last()
                .is_some_and(|last_page| last_page.get().checked_add(1) == Some(page.get()))
        {
            last.push(page);
        } else {
            runs.push(vec![page]);
        }
    }
    Ok(runs)
}

/// Extract all native searchable words via bounded Poppler calls.
///
/// The caller supplies a `MediaBox`-based mapper for the standard pdftotext
/// mode, or a `CropBox` mapper only if it requested cropbox rendering itself.
///
/// # Errors
/// Returns an error when the external extractor fails, or its output has bad PDF coordinates.
pub fn native_words_for_pages(
    pdf: &Path,
    selected: &BTreeMap<PageNumber, DisplayMapper>,
    chunk_size: usize,
) -> Result<BTreeMap<PageNumber, Vec<Quad>>, Error> {
    let pages = selected.keys().copied().collect::<Vec<_>>();
    let mut output = BTreeMap::new();
    for run in contiguous_runs(&pages, chunk_size)? {
        let first = run[0];
        let last = run[run.len() - 1];
        let result = Command::new("pdftotext")
            .args(["-bbox", "-f"])
            .arg(first.get().to_string())
            .arg("-l")
            .arg(last.get().to_string())
            .arg(pdf)
            .arg("-")
            .stderr(Stdio::piped())
            .output()?;
        if !result.status.success() {
            return Err(Error::Poppler(
                String::from_utf8_lossy(&result.stderr).into_owned(),
            ));
        }
        let mut transforms = Vec::with_capacity(run.len());
        for page in &run {
            transforms.push(
                *selected
                    .get(page)
                    .ok_or_else(|| Error::Invalid("missing selected-page transform".into()))?,
            );
        }
        let quads = parse_bbox_xhtml(&result.stdout, &transforms)?;
        for (page, regions) in run.into_iter().zip(quads) {
            output.insert(page, regions);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{contiguous_runs, parse_bbox_xhtml};
    use crate::{
        PageNumber,
        geometry::{DisplayMapper, PageBox, Point, Rotation},
    };
    use std::collections::BTreeSet;

    #[test]
    fn strips_invalid_controls_and_ignores_empty_words() -> Result<(), crate::Error> {
        let page_box = PageBox::new([30.0, 40.0, 830.0, 640.0])?;
        let mapper = DisplayMapper::new(page_box, Rotation::Deg0);
        let xml = br#"<html><body><page width="800" height="600"><word xMin="170" yMin="170" xMax="220" yMax="185">Visible</word><word xMin="10" yMin="10" xMax="30" yMax="30">  </word></page></body></html>"#;
        let mut noisy = xml.to_vec();
        noisy.insert(20, 0x01);
        let pages = parse_bbox_xhtml(&noisy, &[mapper])?;
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].len(), 1);
        assert_eq!(pages[0][0][0], Point(200.0, 455.0));
        assert!((pages[0][0][2].0 - 250.0).abs() < 1e-9);
        assert!((pages[0][0][2].1 - 470.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn parses_namespaced_elements_and_decodes_text_entities() -> Result<(), crate::Error> {
        let box_ = PageBox::new([0.0, 0.0, 100.0, 100.0])?;
        let mapper = DisplayMapper::new(box_, Rotation::Deg0);
        let xml = br#"<x:page xmlns:x="urn:poppler" width="100" height="100"><x:word x:xMin="10" x:yMin="20" x:xMax="30" x:yMax="40">&#65; &amp; &#233;</x:word><x:word xMin="50" yMin="60" xMax="80" yMax="90">&#32;</x:word></x:page>"#;
        let pages = parse_bbox_xhtml(xml, &[mapper])?;
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].len(), 1);
        assert_eq!(pages[0][0][0], Point(10.0, 60.0));
        Ok(())
    }

    #[test]
    fn chunks_contiguous_page_runs_without_duplicates() -> Result<(), crate::Error> {
        let selection = [1, 3, 4, 5, 6, 8]
            .into_iter()
            .map(PageNumber::new)
            .collect::<Result<Vec<_>, _>>()?;
        let runs = contiguous_runs(&selection, 2)?;
        let expected: &[&[u32]] = &[&[1], &[3, 4], &[5, 6], &[8]];
        assert_eq!(runs.iter().map(Vec::len).sum::<usize>(), selection.len());
        for (run, numbers) in runs.iter().zip(expected.iter()) {
            assert_eq!(
                run.iter().map(|p| p.get()).collect::<BTreeSet<_>>(),
                numbers.iter().copied().collect()
            );
        }
        assert!(contiguous_runs(&selection, 0).is_err());
        assert!(contiguous_runs(&[PageNumber::new(1)?, PageNumber::new(1)?], 24).is_err());
        Ok(())
    }
}
