//! Isolate actual Image Do paints through arbitrarily reused nested Forms.
//!
//! A page-scoped invocation tree groups replacements by their shared path.
//! Each affected stream is parsed and copied once; unrelated Form calls,
//! resources and image paints retain their original bindings.

use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    ops::Range,
};

use hayro_syntax::{
    content::{Instruction, UntypedIter},
    object::Object as HayroObject,
};
use pdf_redox::{EditDocument, ObjectHandle, ObjectId, OwnedDictionary, OwnedObject, StreamData};

use crate::{
    Error, PageNumber,
    paint_scan::{
        ImagePaint, MAX_FORM_DEPTH, inherited_page_resources, page_content, resolve_dictionary,
        subtype,
    },
    pdf_metadata::PdfMetadata,
};

/// One replacement image shared by any number of exact original paint calls.
pub struct PaintReplacement<'a> {
    /// Unattached replacement image stream.
    pub image: OwnedObject,
    /// Original occurrences to retarget to that replacement.
    pub paints: Vec<&'a ImagePaint>,
}

#[derive(Default)]
struct InvocationPlan {
    targets: BTreeMap<u32, InvocationTarget>,
}

enum InvocationTarget {
    Image {
        source: ObjectId,
        replacement: ObjectHandle,
    },
    Form(InvocationPlan),
}

impl InvocationPlan {
    fn insert(
        &mut self,
        path: &[u32],
        source: ObjectId,
        replacement: ObjectHandle,
    ) -> Result<(), Error> {
        let (&ordinal, rest) = path
            .split_first()
            .ok_or_else(|| Error::Invalid("empty image invocation path".into()))?;
        if rest.is_empty() {
            return match self.targets.entry(ordinal) {
                Entry::Vacant(entry) => {
                    entry.insert(InvocationTarget::Image {
                        source,
                        replacement,
                    });
                    Ok(())
                }
                Entry::Occupied(_) => Err(Error::Invalid(
                    "duplicate or conflicting image invocation path".into(),
                )),
            };
        }
        let target = self
            .targets
            .entry(ordinal)
            .or_insert_with(|| InvocationTarget::Form(Self::default()));
        match target {
            InvocationTarget::Form(child) => child.insert(rest, source, replacement),
            InvocationTarget::Image { .. } => Err(Error::Invalid(
                "image path cannot enclose another invocation".into(),
            )),
        }
    }
}

/// Replace exact expanded image invocations in one copy-on-write document.
/// A shared replacement image is allocated once, and common Form invocation
/// prefixes are cloned only once. Empty input returns the original source.
/// # Errors
/// Returns an error for duplicate or conflicting paths, malformed target
/// streams/resources, or a path that does not resolve to its original image.
pub fn isolate_source_paints<'a>(
    original: &PdfMetadata,
    replacements: impl IntoIterator<Item = PaintReplacement<'a>>,
) -> Result<Vec<u8>, Error> {
    let mut doc = EditDocument::from_bytes(original.original_bytes().to_vec())?;
    let mut pages = BTreeMap::<PageNumber, InvocationPlan>::new();
    for PaintReplacement { image, paints } in replacements {
        if paints.is_empty() || !is_image(doc.source(), &image)? {
            return Err(Error::Invalid("invalid isolated image replacement".into()));
        }
        let replacement = ObjectHandle::New(doc.add_object(image));
        for paint in paints {
            if paint.paint_path.len() > MAX_FORM_DEPTH + 1 {
                return Err(Error::Invalid(
                    "image invocation path exceeds Form depth limit".into(),
                ));
            }
            let source = ObjectId::new(paint.object_number, i32::from(paint.generation));
            pages
                .entry(PageNumber::new(paint.page)?)
                .or_default()
                .insert(&paint.paint_path, source, replacement)?;
        }
    }
    if pages.is_empty() {
        return Ok(original.original_bytes().to_vec());
    }
    for (page, plan) in pages {
        rewrite_page(&mut doc, page, &plan)?;
    }
    Ok(doc.write_compact()?)
}

fn rewrite_page(
    doc: &mut EditDocument,
    page: PageNumber,
    plan: &InvocationPlan,
) -> Result<(), Error> {
    let page_index = usize::try_from(page.get() - 1)
        .map_err(|_| Error::Invalid("page index exceeds usize".into()))?;
    let page_id = *doc
        .source()
        .page_ids()
        .get(page_index)
        .ok_or_else(|| Error::Invalid("paint refers to missing page".into()))?;
    let resources = inherited_page_resources(doc.source(), page_id)?;
    let contents = page_content(doc.source(), page_id)?;
    let (contents, resources) = rewrite_invocations(doc, &contents, resources, plan)?;
    let content_id = doc.add_object(make_decoded_stream(OwnedDictionary::new(), contents)?);
    let page = doc.edit_object(page_id)?;
    let dict = page
        .as_dictionary_mut()
        .ok_or_else(|| Error::Invalid("page is not a dictionary".into()))?;
    dict.insert(
        b"Contents".to_vec(),
        OwnedObject::Reference(ObjectHandle::New(content_id)),
    );
    dict.insert(b"Resources".to_vec(), OwnedObject::Dictionary(resources));
    Ok(())
}

fn rewrite_form(
    doc: &mut EditDocument,
    source: ObjectId,
    inherited: &OwnedDictionary,
    plan: &InvocationPlan,
) -> Result<ObjectHandle, Error> {
    let existing = doc.source().materialize(source)?;
    let OwnedObject::Stream {
        dictionary: mut form_dict,
        ..
    } = existing
    else {
        return Err(Error::Invalid("target Form is not a stream".into()));
    };
    if !matches!(form_dict.get(b"Subtype".as_slice()), Some(OwnedObject::Name(name)) if name == b"Form")
    {
        return Err(Error::Invalid("target XObject is not a Form".into()));
    }
    let child_resources = match form_dict.get(b"Resources".as_slice()) {
        Some(raw) => resolve_dictionary(doc.source(), raw)?,
        None => inherited.clone(),
    };
    let bytes = doc.source().decoded_stream_data(source)?.into_owned();
    let (rewritten, child_resources) = rewrite_invocations(doc, &bytes, child_resources, plan)?;
    form_dict.insert(
        b"Resources".to_vec(),
        OwnedObject::Dictionary(child_resources),
    );
    Ok(ObjectHandle::New(
        doc.add_object(make_decoded_stream(form_dict, rewritten)?),
    ))
}

fn is_image(source: &pdf_redox::SourcePdf, object: &OwnedObject) -> Result<bool, Error> {
    match object {
        OwnedObject::Stream { dictionary, .. } => {
            Ok(subtype(source, dictionary)?.as_deref() == Some(b"Image"))
        }
        _ => Ok(false),
    }
}

fn replacement_handle(
    doc: &mut EditDocument,
    source: ObjectId,
    inherited: &OwnedDictionary,
    target: &InvocationTarget,
) -> Result<ObjectHandle, Error> {
    match target {
        InvocationTarget::Image {
            source: expected,
            replacement,
        } if *expected == source => {
            if !is_image(doc.source(), &doc.source().materialize(source)?)? {
                return Err(Error::Invalid("target XObject is not an image".into()));
            }
            Ok(*replacement)
        }
        InvocationTarget::Image { .. } => {
            Err(Error::Invalid("paint path resolved to wrong image".into()))
        }
        InvocationTarget::Form(plan) => rewrite_form(doc, source, inherited, plan),
    }
}

fn fresh_alias(
    xobjects: &mut OwnedDictionary,
    next_suffix: &mut u32,
    reserved_names: &BTreeSet<Vec<u8>>,
    replacement: ObjectHandle,
) -> Result<Vec<u8>, Error> {
    loop {
        let candidate = format!("OcrIsolatedPaint{next_suffix}").into_bytes();
        *next_suffix = next_suffix
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("resource alias space exhausted".into()))?;
        if reserved_names.contains(&candidate) {
            continue;
        }
        if let Entry::Vacant(entry) = xobjects.entry(candidate.clone()) {
            entry.insert(OwnedObject::Reference(replacement));
            return Ok(candidate);
        }
    }
}

fn rewrite_invocations(
    doc: &mut EditDocument,
    stream: &[u8],
    mut resources: OwnedDictionary,
    plan: &InvocationPlan,
) -> Result<(Vec<u8>, OwnedDictionary), Error> {
    let original_xobjects = match resources.get(b"XObject".as_slice()) {
        Some(raw) => resolve_dictionary(doc.source(), raw)?,
        None => return Err(Error::Invalid("paint lacks XObject resources".into())),
    };
    // Resolve all original calls against immutable source bindings. A new
    // alias must never accidentally satisfy another malformed original Do.
    let mut xobjects = original_xobjects.clone();
    let mut iter = UntypedIter::new(stream);
    let mut ordinal = 0_u32;
    let mut reserved_names = BTreeSet::new();
    let mut targets = Vec::with_capacity(plan.targets.len());
    while let Some(instruction) = iter.next() {
        if &instruction.operator[..] != b"Do" {
            continue;
        }
        for operand in instruction.operands() {
            if let HayroObject::Name(name) = operand {
                reserved_names.insert(name.as_ref().to_vec());
            }
        }
        let current = ordinal;
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("excessive Do count".into()))?;
        if let Some(target) = plan.targets.get(&current) {
            let (name, span) = target_do_operand(&instruction)?;
            targets.push((name, span, target));
        }
    }
    if !iter.is_at_end() || targets.len() != plan.targets.len() {
        return Err(Error::Invalid(
            "incomplete PDF content parse or missing target Do".into(),
        ));
    }
    let mut next_suffix = 0_u32;
    let mut cursor = 0_usize;
    let mut changed = Vec::with_capacity(stream.len());
    for (name, span, target) in targets {
        let source = match original_xobjects.get(name.as_slice()) {
            Some(OwnedObject::Reference(ObjectHandle::Existing(id))) => *id,
            _ => {
                return Err(Error::Invalid(
                    "target resource is not a source XObject".into(),
                ));
            }
        };
        let replacement = replacement_handle(doc, source, &resources, target)?;
        let alias = fresh_alias(
            &mut xobjects,
            &mut next_suffix,
            &reserved_names,
            replacement,
        )?;
        if span.start < cursor || span.end > stream.len() {
            return Err(Error::Invalid(
                "overlapping or invalid Do operand spans".into(),
            ));
        }
        changed.extend_from_slice(&stream[cursor..span.start]);
        changed.push(b'/');
        changed.extend_from_slice(&alias);
        cursor = span.end;
    }
    changed.extend_from_slice(&stream[cursor..]);
    resources.insert(b"XObject".to_vec(), OwnedObject::Dictionary(xobjects));
    Ok((changed, resources))
}

fn target_do_operand(instruction: &Instruction<'_, '_>) -> Result<(Vec<u8>, Range<usize>), Error> {
    let Some(HayroObject::Name(name)) = instruction.operands().next() else {
        return Err(Error::Invalid("target Do has no name".into()));
    };
    if instruction.operand_spans().len() != 1 {
        return Err(Error::Invalid("target Do has malformed operands".into()));
    }
    let operand = instruction
        .operand_spans()
        .next()
        .ok_or_else(|| Error::Invalid("target Do missing name span".into()))?;
    Ok((name.as_ref().to_vec(), operand))
}

pub(crate) fn make_decoded_stream(
    mut dict: OwnedDictionary,
    data: Vec<u8>,
) -> Result<OwnedObject, Error> {
    dict.retain(|name, _| {
        !matches!(
            name.as_slice(),
            b"Filter" | b"F" | b"FFilter" | b"FDecodeParms" | b"DecodeParms" | b"DP"
        )
    });
    dict.insert(
        b"Length".to_vec(),
        OwnedObject::Integer(
            i64::try_from(data.len())
                .map_err(|_| Error::Invalid("content stream too large".into()))?,
        ),
    );
    Ok(OwnedObject::Stream {
        dictionary: dict,
        data: StreamData::Owned(data),
    })
}
