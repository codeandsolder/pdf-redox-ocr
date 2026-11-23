//! PDF page geometry from the same lazy Hayro parser used by pdf-redox.
//!
//! The page-tree inheritance rules are resolved explicitly: page dictionaries
//! need not contain their own `MediaBox`, `CropBox` or `Rotate` entries.

use std::{collections::BTreeMap, fs, path::Path};

use pdf_redox::{ObjectHandle, ObjectId, OwnedObject, SourcePdf};

use crate::{
    Error, PageNumber,
    geometry::{DisplayMapper, PageBox, Rotation},
};

/// Metadata required to project image detections into the original PDF.
pub struct PdfMetadata {
    /// Shared parsed original source bytes.
    source: SourcePdf,
    /// Effective per-page display transforms.
    mappers: BTreeMap<PageNumber, DisplayMapper>,
    /// Effective unrotated `MediaBoxes`.
    media_boxes: BTreeMap<PageNumber, PageBox>,
}

impl PdfMetadata {
    /// Parse one PDF once and resolve the effective `MediaBox` and rotation.
    ///
    /// # Errors
    /// Returns an error if the PDF is malformed or a page-tree entry is unusable.
    pub fn open(path: &Path) -> Result<Self, Error> {
        Self::from_bytes(fs::read(path)?)
    }

    /// Parse one owned PDF byte buffer without copying it.
    ///
    /// # Errors
    /// Returns an error if the PDF or its inherited geometry is malformed.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, Error> {
        let source = SourcePdf::from_bytes(bytes)?;
        let page_ids = source.page_ids();
        if source.page_count() != page_ids.len() {
            return Err(Error::Invalid(
                "a source page has no indirect identity".into(),
            ));
        }
        let mut mappers = BTreeMap::new();
        let mut media_boxes = BTreeMap::new();
        for (index, &object) in page_ids.iter().enumerate() {
            let page_number = u32::try_from(index + 1)
                .map_err(|_| Error::Invalid("PDF has too many pages".into()))?;
            let page = PageNumber::new(page_number)?;
            let (box_, rotation) = inherited_geometry(&source, object)?;
            media_boxes.insert(page, box_);
            mappers.insert(page, DisplayMapper::new(box_, rotation));
        }
        Ok(Self {
            source,
            mappers,
            media_boxes,
        })
    }

    /// Number of pages in PDF page-tree order.
    #[must_use = "use the page count to bound source-page operations"]
    pub fn page_count(&self) -> usize {
        self.mappers.len()
    }

    /// Effective normalized raster-to-PDF transform on a page.
    ///
    /// # Errors
    /// Returns an error for an out-of-range page index.
    pub fn mapper(&self, page: PageNumber) -> Result<DisplayMapper, Error> {
        self.mappers
            .get(&page)
            .copied()
            .ok_or_else(|| Error::Invalid(format!("missing PDF page {}", page.get())))
    }

    /// Validated effective page geometries in PDF page-tree order.
    #[must_use = "use source media boxes when reconstructing page geometry"]
    pub const fn media_boxes(&self) -> &BTreeMap<PageNumber, PageBox> {
        &self.media_boxes
    }

    /// Reuse the original Hayro-backed PDF source to examine content streams.
    #[must_use = "use the owning source PDF when resolving document objects"]
    pub const fn source(&self) -> &SourcePdf {
        &self.source
    }

    /// Original bytes, identical to those hashed in an OCR sidecar.
    #[must_use = "retain the original PDF bytes when applying byte-preserving edits"]
    pub fn original_bytes(&self) -> &[u8] {
        self.source.bytes()
    }

    /// Native searchable text transforms for a selected subset of pages.
    ///
    /// # Errors
    /// Returns an error if any selected page does not exist.
    pub fn select_mappers(
        &self,
        selected: &[PageNumber],
    ) -> Result<BTreeMap<PageNumber, DisplayMapper>, Error> {
        selected
            .iter()
            .map(|&page| Ok((page, self.mapper(page)?)))
            .collect()
    }
}

pub(crate) fn resolve_owned(
    source: &SourcePdf,
    mut value: OwnedObject,
) -> Result<OwnedObject, Error> {
    // Bound recursive indirection independently of page-tree parent ancestry.
    for _ in 0..64 {
        let OwnedObject::Reference(reference) = value else {
            return Ok(value);
        };
        let ObjectHandle::Existing(object_id) = reference else {
            return Err(Error::Invalid(
                "new-object handle in immutable PDF source".into(),
            ));
        };
        value = source.materialize(object_id)?;
    }
    Err(Error::Invalid(
        "cyclic or excessively nested PDF reference chain".into(),
    ))
}

fn inherited_geometry(source: &SourcePdf, first: ObjectId) -> Result<(PageBox, Rotation), Error> {
    let mut visited = std::collections::BTreeSet::new();
    let mut current = Some(first);
    let mut media_box = None;
    let mut rotation = None;
    while let Some(page_id) = current {
        if !visited.insert(page_id) {
            return Err(Error::Invalid("cycle in PDF page tree".into()));
        }
        let object = source.materialize(page_id)?;
        let dict = object
            .as_dictionary()
            .ok_or_else(|| Error::Invalid("PDF page or ancestor is not a dictionary".into()))?;
        if media_box.is_none()
            && let Some(value) = dict.get(b"MediaBox".as_slice())
        {
            media_box = Some(resolve_box(source, value)?);
        }
        if rotation.is_none()
            && let Some(value) = dict.get(b"Rotate".as_slice())
        {
            rotation = Some(resolve_rotation(source, value)?);
        }
        current = match dict.get(b"Parent".as_slice()) {
            Some(OwnedObject::Reference(ObjectHandle::Existing(id))) => Some(*id),
            None => None,
            Some(_) => return Err(Error::Invalid("invalid Parent page-tree reference".into())),
        };
    }
    let box_ = media_box.ok_or_else(|| Error::Invalid("page has no inherited MediaBox".into()))?;
    Ok((box_, rotation.unwrap_or(Rotation::Deg0)))
}

fn resolve_box(source: &SourcePdf, raw: &OwnedObject) -> Result<PageBox, Error> {
    let owned = resolve_owned(source, raw.clone())?;
    let OwnedObject::Array(values) = owned else {
        return Err(Error::Invalid("PDF MediaBox must be an array".into()));
    };
    let coords = values
        .into_iter()
        .map(|value| match resolve_owned(source, value)? {
            OwnedObject::Integer(int) => Ok(int.to_string().parse::<f64>()?),
            OwnedObject::Real(real) => Ok(real),
            _ => Err(Error::Invalid("non-numeric PDF MediaBox value".into())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let coords: [f64; 4] = coords
        .try_into()
        .map_err(|_| Error::Invalid("PDF MediaBox must have four coordinates".into()))?;
    PageBox::new(coords)
}

fn resolve_rotation(source: &SourcePdf, raw: &OwnedObject) -> Result<Rotation, Error> {
    let OwnedObject::Integer(degrees) = resolve_owned(source, raw.clone())? else {
        return Err(Error::Invalid("PDF Rotate must be an integer".into()));
    };
    Rotation::try_from(
        i32::try_from(degrees)
            .map_err(|_| Error::Invalid("PDF Rotate outside i32 range".into()))?,
    )
}

#[cfg(test)]
mod tests {
    use super::PdfMetadata;
    use crate::{PageNumber, geometry::Point};

    fn minimal_pdf() -> Vec<u8> {
        let mut data = b"%PDF-1.4\n".to_vec();
        let bodies = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [30 40 830 640] /Rotate 90 >>",
            "<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>",
            "<< /Length 0 >>\nstream\n\nendstream",
        ];
        let mut offsets = vec![0_usize];
        for (index, body) in bodies.iter().enumerate() {
            offsets.push(data.len());
            data.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", index + 1).as_bytes());
        }
        let xref = data.len();
        data.extend_from_slice(b"xref\n0 5\n0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            data.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        data.extend_from_slice(
            format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
        );
        data
    }

    #[test]
    fn inherited_geometry_maps_rotated_pages() -> Result<(), crate::Error> {
        let doc = PdfMetadata::from_bytes(minimal_pdf())?;
        assert_eq!(doc.page_count(), 1);
        let page = PageNumber::new(1)?;
        let mapper = doc.mapper(page)?;
        assert_eq!(mapper.to_pdf(0.0, 0.0), Point(30.0, 40.0));
        assert_eq!(mapper.to_pdf(1.0, 1.0), Point(830.0, 640.0));
        assert_eq!(doc.media_boxes()[&page].width(), 800.0);
        assert!(doc.mapper(PageNumber::new(2)?).is_err());
        Ok(())
    }

    #[test]
    fn rejects_unresolvable_page_parent() {
        let mut bytes = minimal_pdf();
        bytes.truncate(10);
        assert!(PdfMetadata::from_bytes(bytes).is_err());
    }
}
