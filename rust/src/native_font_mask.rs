//! Render-only copy of a PDF with searchable native font shows made invisible.
//!
//! An original text show still executes with *exactly* the same operands and
//! font state, so advances, kerning, text matrices and clipping are preserved.
//! Type3, unresolved fonts, pre-invisible text and clipping modes 4..7 are
//! never hidden. Shared Form `XObjects` are cloned per page/resource context,
//! preventing a font inherited on one page from affecting another page.

use std::collections::{BTreeMap, BTreeSet};

use hayro_syntax::{content::UntypedIter, object::Object as HayroObject};
use pdf_redox::{EditDocument, ObjectHandle, ObjectId, OwnedDictionary, OwnedObject, SourcePdf};

use crate::{
    Error, PageNumber,
    paint_isolation::make_decoded_stream,
    paint_scan::{font_searchable, inherited_page_resources, page_content, resolve_dictionary},
    pdf_metadata::PdfMetadata,
};

const MAX_FORM_DEPTH: usize = 32;
const MAX_FORM_VISITS: usize = 20_000;

/// Number of original native text shows and copies rewritten for OCR rasterization.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeFontMaskStats {
    /// Visible searchable shows hidden with temporary text rendering mode 3.
    pub hidden_native_shows: u64,
    /// Searchable native shows preserved because they establish text clipping.
    pub preserved_clipping_shows: u64,
    /// Native shows already invisible in the original PDF.
    pub already_invisible_shows: u64,
    /// Type3 or unresolved-font shows kept visible for OCR.
    pub preserved_nonsearchable_shows: u64,
    /// Source pages whose own content or Form resource references were changed.
    pub rewritten_pages: u64,
    /// Independently cloned Form `XObjects` whose content or descendants changed.
    pub cloned_forms: u64,
}

/// A temporary render-only PDF, or no PDF when nothing needs hiding.
#[derive(Debug)]
pub struct NativeFontMask {
    /// None means that original and render-only PDFs would be identical.
    pub bytes: Option<Vec<u8>>,
    /// Evidence of exactly which graphics operations were hidden.
    pub stats: NativeFontMaskStats,
}

#[derive(Clone, Default)]
struct TextState {
    font: Option<Vec<u8>>,
    render_mode: u8,
}

fn set_render_mode(instruction: &hayro_syntax::content::Instruction<'_, '_>) -> Result<u8, Error> {
    let mut operands = instruction.operands();
    let Some(HayroObject::Number(number)) = operands.next() else {
        return Err(Error::Invalid("native text Tr has invalid operand".into()));
    };
    let value = number.as_f64();
    if operands.next().is_some()
        || !value.is_finite()
        || !(0.0..=7.0).contains(&value)
        || value.fract() != 0.0
    {
        return Err(Error::Invalid("native text Tr outside 0..=7".into()));
    }
    num_traits::ToPrimitive::to_u8(&value)
        .ok_or_else(|| Error::Invalid("native text Tr conversion failed".into()))
}

fn set_font(instruction: &hayro_syntax::content::Instruction<'_, '_>) -> Option<Vec<u8>> {
    match instruction.operands().next() {
        Some(HayroObject::Name(name)) => Some(name.as_ref().to_vec()),
        _ => None,
    }
}

/// Insert temporary text modes around original byte spans, preserving all
/// original operands, clipping, `CMaps` and matrix operators byte-for-byte.
fn rewrite_native_stream(
    source: &SourcePdf,
    stream: &[u8],
    resources: &OwnedDictionary,
    stats: &mut NativeFontMaskStats,
) -> Result<Option<Vec<u8>>, Error> {
    let mut iter = UntypedIter::new(stream);
    let mut current = TextState::default();
    let mut stack = Vec::new();
    let mut cached_fonts = BTreeMap::<Vec<u8>, bool>::new();
    let mut replacement = Vec::with_capacity(stream.len());
    let mut cursor = 0_usize;
    while let Some(instruction) = iter.next() {
        let op: &[u8] = &instruction.operator[..];
        match op {
            b"q" => stack.push(current.clone()),
            b"Q" => {
                current = stack
                    .pop()
                    .ok_or_else(|| Error::Invalid("unbalanced Q in native font content".into()))?;
            }
            b"Tf" => current.font = set_font(&instruction),
            b"Tr" => current.render_mode = set_render_mode(&instruction)?,
            b"Tj" | b"TJ" | b"'" | b"\"" => {
                // Fonts are immutable inside this content stream's resource
                // context; resolve each indirect dictionary only once.
                let searchable = if let Some(font) = current.font.as_deref() {
                    if let Some(&cached) = cached_fonts.get(font) {
                        cached
                    } else {
                        let resolved = font_searchable(source, resources, font)?;
                        cached_fonts.insert(font.to_vec(), resolved);
                        resolved
                    }
                } else {
                    false
                };
                if !searchable {
                    stats.preserved_nonsearchable_shows += 1;
                } else if current.render_mode >= 4 {
                    stats.preserved_clipping_shows += 1;
                } else if current.render_mode == 3 {
                    stats.already_invisible_shows += 1;
                } else {
                    let span = instruction.span();
                    if span.start < cursor || span.end > stream.len() {
                        return Err(Error::Invalid(
                            "overlapping native text operator spans".into(),
                        ));
                    }
                    replacement.extend_from_slice(&stream[cursor..span.start]);
                    replacement.extend_from_slice(b"\n3 Tr\n");
                    replacement.extend_from_slice(&stream[span]);
                    replacement
                        .extend_from_slice(format!("\n{} Tr\n", current.render_mode).as_bytes());
                    cursor = instruction.span().end;
                    stats.hidden_native_shows += 1;
                }
            }
            _ => {}
        }
    }
    if !iter.is_at_end() || !stack.is_empty() {
        return Err(Error::Invalid(
            "native text stream did not parse completely".into(),
        ));
    }
    if cursor == 0 {
        return Ok(None);
    }
    replacement.extend_from_slice(&stream[cursor..]);
    Ok(Some(replacement))
}

/// Only invoked Form names are relevant: a Form inheriting page resources
/// commonly inherits its own `XObject` entry, but that is *not* recursive paint.
fn invoked_xobjects(stream: &[u8]) -> Result<BTreeSet<Vec<u8>>, Error> {
    let mut iter = UntypedIter::new(stream);
    let mut names = BTreeSet::new();
    while let Some(instruction) = iter.next() {
        if &instruction.operator[..] != b"Do" {
            continue;
        }
        let mut operands = instruction.operands();
        let Some(HayroObject::Name(name)) = operands.next() else {
            return Err(Error::Invalid(
                "native Form Do missing resource name".into(),
            ));
        };
        if operands.next().is_some() {
            return Err(Error::Invalid(
                "native Form Do has excessive operands".into(),
            ));
        }
        names.insert(name.as_ref().to_vec());
    }
    if !iter.is_at_end() {
        return Err(Error::Invalid(
            "native Form Do stream did not parse completely".into(),
        ));
    }
    Ok(names)
}

/// One page's complete Form graph is cloned only along changed resources.
/// Inherited resources are materialized per invocation context, not changed
/// at their original shared indirect objects.
struct FormRewriter<'a> {
    source: &'a SourcePdf,
    edited: &'a mut EditDocument,
    active: BTreeSet<ObjectId>,
    visits: usize,
    stats: &'a mut NativeFontMaskStats,
}

impl FormRewriter<'_> {
    fn resource_forms(
        &mut self,
        mut resources: OwnedDictionary,
        stream: &[u8],
        depth: usize,
    ) -> Result<(OwnedDictionary, bool), Error> {
        let names = invoked_xobjects(stream)?;
        if names.is_empty() {
            return Ok((resources, false));
        }
        let Some(table) = resources.get(b"XObject".as_slice()) else {
            return Ok((resources, false));
        };
        let mut xobjects = resolve_dictionary(self.source, table)?;
        let mut changed = false;
        for name in names {
            let Some(OwnedObject::Reference(ObjectHandle::Existing(id))) =
                xobjects.get(name.as_slice())
            else {
                continue;
            };
            let id = *id;
            let obj = self.source.materialize(id)?;
            if !matches!(
                obj.as_dictionary().and_then(|dict| dict.get(b"Subtype".as_slice())),
                Some(OwnedObject::Name(name)) if name == b"Form"
            ) {
                continue;
            }
            if let Some(new_id) = self.rewrite_form(id, &resources, depth + 1)? {
                xobjects.insert(name, OwnedObject::Reference(ObjectHandle::New(new_id)));
                changed = true;
            }
        }
        if changed {
            resources.insert(b"XObject".to_vec(), OwnedObject::Dictionary(xobjects));
        }
        Ok((resources, changed))
    }

    fn rewrite_form(
        &mut self,
        id: ObjectId,
        inherited: &OwnedDictionary,
        depth: usize,
    ) -> Result<Option<pdf_redox::NewObjectId>, Error> {
        if depth > MAX_FORM_DEPTH || self.visits >= MAX_FORM_VISITS || !self.active.insert(id) {
            return Err(Error::Invalid(
                "cyclic or excessive native Form resource graph".into(),
            ));
        }
        self.visits += 1;
        let result = self.rewrite_form_inner(id, inherited, depth);
        self.active.remove(&id);
        result
    }

    fn rewrite_form_inner(
        &mut self,
        id: ObjectId,
        inherited: &OwnedDictionary,
        depth: usize,
    ) -> Result<Option<pdf_redox::NewObjectId>, Error> {
        let existing = self.source.materialize(id)?;
        let OwnedObject::Stream {
            dictionary: mut form,
            ..
        } = existing
        else {
            return Err(Error::Invalid("native font Form not a stream".into()));
        };
        let resources = match form.get(b"Resources".as_slice()) {
            Some(raw) => resolve_dictionary(self.source, raw)?,
            None => inherited.clone(),
        };
        let source_bytes = self.source.decoded_stream_data(id)?;
        let (resources, descendants_changed) =
            self.resource_forms(resources, &source_bytes, depth)?;
        let rewritten = rewrite_native_stream(self.source, &source_bytes, &resources, self.stats)?;
        if rewritten.is_none() && !descendants_changed {
            return Ok(None);
        }
        form.insert(b"Resources".to_vec(), OwnedObject::Dictionary(resources));
        let object =
            make_decoded_stream(form, rewritten.unwrap_or_else(|| source_bytes.into_owned()))?;
        self.stats.cloned_forms += 1;
        Ok(Some(self.edited.add_object(object)))
    }
}

/// Hide searchable non-Type3 native text only in a render-only PDF copy.
///
/// All selected page content and recursively referenced Form `XObjects` are
/// handled; a Form inheriting different resources on separate pages is cloned
/// separately. Nonselected pages and shared original objects remain unchanged.
///
/// # Errors
/// Returns an error on invalid page selection, malformed content operators,
/// cyclic Form graphs, or PDF encoding failure.
pub fn mask_native_fonts(
    original: &PdfMetadata,
    selected: &[PageNumber],
) -> Result<NativeFontMask, Error> {
    if selected.is_empty() {
        return Err(Error::Invalid("empty native mask page selection".into()));
    }
    let source = original.source();
    let mut edited = EditDocument::from_bytes(original.original_bytes().to_vec())?;
    let mut stats = NativeFontMaskStats::default();
    let mut seen = BTreeSet::new();
    for &page in selected {
        if !seen.insert(page) {
            return Err(Error::Invalid("duplicate native font mask page".into()));
        }
        let index = usize::try_from(page.get() - 1)
            .map_err(|_| Error::Invalid("page outside usize".into()))?;
        let page_id = *source
            .page_ids()
            .get(index)
            .ok_or_else(|| Error::Invalid("native mask page out of range".into()))?;
        let resources = inherited_page_resources(source, page_id)?;
        let mut rewriter = FormRewriter {
            source,
            edited: &mut edited,
            active: BTreeSet::new(),
            visits: 0,
            stats: &mut stats,
        };
        let page_data = page_content(source, page_id)?;
        let (resources, forms_changed) = rewriter.resource_forms(resources, &page_data, 0)?;
        let rewritten = rewrite_native_stream(source, &page_data, &resources, rewriter.stats)?;
        if rewritten.is_none() && !forms_changed {
            continue;
        }
        let content = rewritten
            .map(|bytes| {
                make_decoded_stream(OwnedDictionary::new(), bytes)
                    .map(|stream| rewriter.edited.add_object(stream))
            })
            .transpose()?;
        let page_object = rewriter.edited.edit_object(page_id)?;
        let dict = page_object.as_dictionary_mut().ok_or_else(|| {
            Error::Invalid("native font mask page object is not a dictionary".into())
        })?;
        if forms_changed {
            dict.insert(b"Resources".to_vec(), OwnedObject::Dictionary(resources));
        }
        if let Some(id) = content {
            dict.insert(
                b"Contents".to_vec(),
                OwnedObject::Reference(ObjectHandle::New(id)),
            );
        }
        stats.rewritten_pages += 1;
    }
    let bytes = if stats.hidden_native_shows > 0 {
        Some(edited.write_compact()?)
    } else {
        None
    };
    Ok(NativeFontMask { bytes, stats })
}

#[cfg(test)]
mod tests {
    use super::rewrite_native_stream;
    use crate::paint_scan::inherited_page_resources;
    use pdf_redox::SourcePdf;

    #[test]
    fn every_pdf_text_show_preserves_operands_and_graphics_state() -> Result<(), crate::Error> {
        let source = SourcePdf::from_bytes(
            include_bytes!("../tests/fixtures/native-font-form-masking.pdf").to_vec(),
        )?;
        let resources = inherited_page_resources(&source, source.page_ids()[0])?;
        let input = br#"q BT /Normal 12 Tf 2 Tr (A) Tj [(B) -120 (C)] TJ (D) ' 0 0 (E) " /Glyphs 20 Tf (X) Tj /Normal 12 Tf 7 Tr (CLIP) Tj 3 Tr (HIDDEN) Tj ET Q"#;
        let mut stats = super::NativeFontMaskStats::default();
        let rewritten = rewrite_native_stream(&source, input, &resources, &mut stats)?
            .ok_or_else(|| crate::Error::Invalid("native text was not masked".into()))?;
        assert_eq!(stats.hidden_native_shows, 4);
        assert_eq!(stats.preserved_clipping_shows, 1);
        assert_eq!(stats.already_invisible_shows, 1);
        assert_eq!(stats.preserved_nonsearchable_shows, 1);
        let text = String::from_utf8_lossy(&rewritten);
        for show in [
            "(A) Tj",
            "[(B) -120 (C)] TJ",
            "(D) '",
            "0 0 (E) \"",
            "(X) Tj",
            "(CLIP) Tj",
            "(HIDDEN) Tj",
        ] {
            assert!(text.contains(show), "original PDF show lost: {show}");
        }
        assert_eq!(text.matches("3 Tr").count(), 5);
        assert_eq!(text.matches("2 Tr").count(), 5);
        assert!(text.contains("7 Tr (CLIP) Tj"));
        Ok(())
    }

    #[test]
    fn pdf_content_operators_preserve_original_bytes_and_kerning() -> Result<(), crate::Error> {
        let pdf = include_bytes!("../tests/fixtures/repeated-nested-paint-occlusion.pdf").to_vec();
        let source = SourcePdf::from_bytes(pdf)?;
        let resources = inherited_page_resources(&source, source.page_ids()[0])?;
        let content = b"BT /Normal 12 Tf 0 Tr (HELLO) Tj [<41> -120 (B)] TJ 7 Tr (CLIP) Tj 3 Tr (INVISIBLE) Tj ET";
        let mut stats = super::NativeFontMaskStats::default();
        let changed = rewrite_native_stream(&source, content, &resources, &mut stats)?;
        // The chosen fixture intentionally does not define /Normal: unknown
        // fonts are preserved, not erased on the assumption they are valid.
        assert!(changed.is_none());
        assert_eq!(stats.hidden_native_shows, 0);
        assert_eq!(stats.preserved_nonsearchable_shows, 4);
        Ok(())
    }
}
