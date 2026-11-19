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
    paint_scan::{
        extgstate_font_searchability, font_searchable, inherited_page_resources, page_content,
        resolve_dictionary,
    },
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
    // PDF graphics state holds an already resolved font object, *not* the
    // name from the caller's resource dictionary. A Form with its own /F
    // alias cannot change the inherited font until it executes another Tf.
    font_searchable: bool,
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

/// Apply any font selection executed by a named extended graphics state.
/// An unresolved resource leaves the selected font untrusted, so native ink
/// will remain visible rather than accidentally masking a graphical font.
fn update_selected_font_from_gs(
    source: &SourcePdf,
    resources: &OwnedDictionary,
    instruction: &hayro_syntax::content::Instruction<'_, '_>,
    selected: &mut bool,
) -> Result<(), Error> {
    let mut operands = instruction.operands();
    let name = match (operands.next(), operands.next()) {
        (Some(HayroObject::Name(name)), None) => name,
        _ => {
            *selected = false;
            return Ok(());
        }
    };
    let Some(table) = resources.get(b"ExtGState".as_slice()) else {
        *selected = false;
        return Ok(());
    };
    let table = resolve_dictionary(source, table)?;
    let Some(raw) = table.get(name.as_ref()) else {
        *selected = false;
        return Ok(());
    };
    let gs = resolve_dictionary(source, raw)?;
    if let Some(searchable) = extgstate_font_searchability(source, &gs)? {
        *selected = searchable;
    }
    Ok(())
}

/// One source-content instruction replaced without losing neighboring bytes.
struct ByteEdit {
    span: std::ops::Range<usize>,
    replacement: Vec<u8>,
}

struct StreamRewrite {
    /// None means the original contents and resource bindings are untouched.
    bytes: Option<Vec<u8>>,
    /// The call-site-local resource table, including any new Form aliases.
    resources: OwnedDictionary,
    /// True only if some exact Form invocation was retargeted.
    aliased: bool,
}

/// Every rewritten Form is specific to its *actual* invocation's inherited
/// text/graphics state and resources; no global Form-ID rewrite cache.
struct FormRewriter<'a> {
    source: &'a SourcePdf,
    edited: &'a mut EditDocument,
    active: BTreeSet<ObjectId>,
    visits: usize,
    stats: &'a mut NativeFontMaskStats,
}

impl FormRewriter<'_> {
    fn rewrite_stream(
        &mut self,
        bytes: &[u8],
        mut resources: OwnedDictionary,
        initial: TextState,
        depth: usize,
    ) -> Result<StreamRewrite, Error> {
        let mut iter = UntypedIter::new(bytes);
        let mut current = initial;
        let mut saved = Vec::new();
        let mut cached_fonts = BTreeMap::<Vec<u8>, bool>::new();
        let mut edits = Vec::new();
        let mut aliased = false;
        while let Some(instruction) = iter.next() {
            match &instruction.operator[..] {
                b"q" => saved.push(current.clone()),
                b"Q" => {
                    current = saved.pop().ok_or_else(|| {
                        Error::Invalid("unbalanced Q in native font content".into())
                    })?;
                }
                b"Tf" => {
                    current.font_searchable = match set_font(&instruction) {
                        Some(name) => {
                            if let Some(&cached) = cached_fonts.get(&name) {
                                cached
                            } else {
                                let resolved = font_searchable(self.source, &resources, &name)?;
                                cached_fonts.insert(name, resolved);
                                resolved
                            }
                        }
                        None => false,
                    };
                }
                b"Tr" => current.render_mode = set_render_mode(&instruction)?,
                b"gs" => update_selected_font_from_gs(
                    self.source,
                    &resources,
                    &instruction,
                    &mut current.font_searchable,
                )?,
                b"Tj" | b"TJ" | b"'" | b"\"" => {
                    self.rewrite_show(&instruction, bytes, &current, &mut edits);
                }
                b"Do" => {
                    if let Some(name) =
                        self.rewrite_do(&instruction, &mut resources, &current, depth)?
                    {
                        edits.push(ByteEdit {
                            span: instruction.operand_spans().next().ok_or_else(|| {
                                Error::Invalid("native Form Do lacks operand byte span".into())
                            })?,
                            replacement: name,
                        });
                        aliased = true;
                    }
                }
                _ => {}
            }
        }
        if !iter.is_at_end() || !saved.is_empty() {
            return Err(Error::Invalid(
                "native font stream did not parse completely".into(),
            ));
        }
        let mut cursor = 0_usize;
        let mut rewritten = Vec::with_capacity(bytes.len() + edits.len() * 12);
        for edit in &edits {
            if edit.span.start < cursor || edit.span.end > bytes.len() {
                return Err(Error::Invalid("overlapping native font byte spans".into()));
            }
            rewritten.extend_from_slice(&bytes[cursor..edit.span.start]);
            rewritten.extend_from_slice(&edit.replacement);
            cursor = edit.span.end;
        }
        if edits.is_empty() {
            return Ok(StreamRewrite {
                bytes: None,
                resources,
                aliased: false,
            });
        }
        rewritten.extend_from_slice(&bytes[cursor..]);
        Ok(StreamRewrite {
            bytes: Some(rewritten),
            resources,
            aliased,
        })
    }

    fn rewrite_show(
        &mut self,
        instruction: &hayro_syntax::content::Instruction<'_, '_>,
        bytes: &[u8],
        state: &TextState,
        edits: &mut Vec<ByteEdit>,
    ) {
        if !state.font_searchable {
            self.stats.preserved_nonsearchable_shows += 1;
        } else if state.render_mode >= 4 {
            self.stats.preserved_clipping_shows += 1;
        } else if state.render_mode == 3 {
            self.stats.already_invisible_shows += 1;
        } else {
            let mut replacement = Vec::new();
            replacement.extend_from_slice(b"\n3 Tr\n");
            // Preserve exact original show operands, kerning, glyph offsets
            // and source token encoding in the selected PDF stream.
            replacement.extend_from_slice(&bytes[instruction.span()]);
            replacement.extend_from_slice(format!("\n{} Tr\n", state.render_mode).as_bytes());
            edits.push(ByteEdit {
                span: instruction.span(),
                replacement,
            });
            self.stats.hidden_native_shows += 1;
        }
    }

    fn rewrite_do(
        &mut self,
        instruction: &hayro_syntax::content::Instruction<'_, '_>,
        resources: &mut OwnedDictionary,
        state: &TextState,
        depth: usize,
    ) -> Result<Option<Vec<u8>>, Error> {
        let mut operands = instruction.operands();
        let Some(HayroObject::Name(name)) = operands.next() else {
            return Err(Error::Invalid("native Form Do lacks resource name".into()));
        };
        if operands.next().is_some() || instruction.operand_spans().len() != 1 {
            return Err(Error::Invalid(
                "native Form Do has malformed operands".into(),
            ));
        }
        let name = name.as_ref();
        let Some(table) = resources.get(b"XObject".as_slice()) else {
            return Ok(None);
        };
        let mut xobjects = resolve_dictionary(self.source, table)?;
        let Some(OwnedObject::Reference(ObjectHandle::Existing(id))) = xobjects.get(name) else {
            return Ok(None);
        };
        let object = *id;
        let original = self.source.materialize(object)?;
        if !matches!(
            original.as_dictionary().and_then(|dict| dict.get(b"Subtype".as_slice())),
            Some(OwnedObject::Name(subtype)) if subtype == b"Form"
        ) {
            return Ok(None);
        }
        let Some(new_object) = self.rewrite_form(object, resources, state.clone(), depth + 1)?
        else {
            return Ok(None);
        };
        let mut suffix = 0_u32;
        let alias = loop {
            let candidate = format!("OcrNativeFont{suffix}").into_bytes();
            if !xobjects.contains_key(candidate.as_slice()) {
                break candidate;
            }
            suffix = suffix
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("native font alias space exhausted".into()))?;
        };
        let handle = ObjectHandle::New(new_object);
        xobjects.insert(alias.clone(), OwnedObject::Reference(handle));
        resources.insert(b"XObject".to_vec(), OwnedObject::Dictionary(xobjects));
        let mut replacement = Vec::with_capacity(alias.len() + 1);
        replacement.push(b'/');
        replacement.extend_from_slice(&alias);
        Ok(Some(replacement))
    }

    fn rewrite_form(
        &mut self,
        id: ObjectId,
        inherited: &OwnedDictionary,
        state: TextState,
        depth: usize,
    ) -> Result<Option<pdf_redox::NewObjectId>, Error> {
        if depth > MAX_FORM_DEPTH || self.visits >= MAX_FORM_VISITS || !self.active.insert(id) {
            return Err(Error::Invalid(
                "cyclic or excessive native Form invocation graph".into(),
            ));
        }
        self.visits += 1;
        let result = self.rewrite_form_inner(id, inherited, state, depth);
        self.active.remove(&id);
        result
    }

    fn rewrite_form_inner(
        &mut self,
        id: ObjectId,
        inherited: &OwnedDictionary,
        state: TextState,
        depth: usize,
    ) -> Result<Option<pdf_redox::NewObjectId>, Error> {
        let existing = self.source.materialize(id)?;
        let OwnedObject::Stream {
            dictionary: mut form,
            ..
        } = existing
        else {
            return Err(Error::Invalid("native font Form is not a stream".into()));
        };
        let resources = match form.get(b"Resources".as_slice()) {
            Some(raw) => resolve_dictionary(self.source, raw)?,
            None => inherited.clone(),
        };
        let source_bytes = self.source.decoded_stream_data(id)?;
        let result = self.rewrite_stream(&source_bytes, resources, state, depth)?;
        let Some(contents) = result.bytes else {
            return Ok(None);
        };
        if result.aliased {
            form.insert(
                b"Resources".to_vec(),
                OwnedObject::Dictionary(result.resources),
            );
        }
        self.stats.cloned_forms += 1;
        Ok(Some(
            self.edited.add_object(make_decoded_stream(form, contents)?),
        ))
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
        let result = rewriter.rewrite_stream(&page_data, resources, TextState::default(), 0)?;
        let Some(contents) = result.bytes else {
            continue;
        };
        let content = rewriter
            .edited
            .add_object(make_decoded_stream(OwnedDictionary::new(), contents)?);
        let page_object = rewriter.edited.edit_object(page_id)?;
        let dict = page_object.as_dictionary_mut().ok_or_else(|| {
            Error::Invalid("native font mask page object is not a dictionary".into())
        })?;
        if result.aliased {
            dict.insert(
                b"Resources".to_vec(),
                OwnedObject::Dictionary(result.resources),
            );
        }
        dict.insert(
            b"Contents".to_vec(),
            OwnedObject::Reference(ObjectHandle::New(content)),
        );
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
    use super::{FormRewriter, NativeFontMaskStats, TextState};
    use crate::paint_scan::inherited_page_resources;
    use pdf_redox::{EditDocument, SourcePdf};

    fn rewrite_fixture(
        pdf: &[u8],
        content: &[u8],
    ) -> Result<(Option<Vec<u8>>, NativeFontMaskStats), crate::Error> {
        let source = SourcePdf::from_bytes(pdf.to_vec())?;
        let resources = inherited_page_resources(&source, source.page_ids()[0])?;
        let mut stats = NativeFontMaskStats::default();
        let mut edited = EditDocument::from_bytes(source.bytes().to_vec())?;
        let mut rewriter = FormRewriter {
            source: &source,
            edited: &mut edited,
            active: std::collections::BTreeSet::new(),
            visits: 0,
            stats: &mut stats,
        };
        let changed = rewriter
            .rewrite_stream(content, resources, TextState::default(), 0)?
            .bytes;
        Ok((changed, stats))
    }

    #[test]
    fn every_pdf_text_show_preserves_operands_and_graphics_state() -> Result<(), crate::Error> {
        let input = br#"q BT /Normal 12 Tf 2 Tr (A) Tj [(B) -120 (C)] TJ (D) ' 0 0 (E) " /Glyphs 20 Tf (X) Tj /Normal 12 Tf 7 Tr (CLIP) Tj 3 Tr (HIDDEN) Tj ET Q"#;
        let (changed, stats) = rewrite_fixture(
            include_bytes!("../tests/fixtures/native-font-form-masking.pdf"),
            input,
        )?;
        let rewritten =
            changed.ok_or_else(|| crate::Error::Invalid("native text was not masked".into()))?;
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
        let content = b"BT /Normal 12 Tf 0 Tr (HELLO) Tj [<41> -120 (B)] TJ 7 Tr (CLIP) Tj 3 Tr (INVISIBLE) Tj ET";
        let (changed, stats) = rewrite_fixture(
            include_bytes!("../tests/fixtures/repeated-nested-paint-occlusion.pdf"),
            content,
        )?;
        // The chosen fixture intentionally does not define /Normal: unknown
        // fonts are preserved, not erased on the assumption they are valid.
        assert!(changed.is_none());
        assert_eq!(stats.hidden_native_shows, 0);
        assert_eq!(stats.preserved_nonsearchable_shows, 4);
        Ok(())
    }
}
