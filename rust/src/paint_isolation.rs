//! Isolate one actual Image Do paint through arbitrarily reused nested Forms.
//!
//! Only nodes on the chosen page-to-image invocation path are copy-on-write
//! cloned. The target image gains a fresh resource alias, while all other Do
//! operators and the original `XObject` graph retain their original bindings.

use hayro_syntax::{content::UntypedIter, object::Object as HayroObject};
use pdf_redox::{EditDocument, ObjectHandle, ObjectId, OwnedDictionary, OwnedObject, StreamData};

use crate::{
    Error,
    paint_scan::{ImagePaint, inherited_page_resources, page_content, resolve_dictionary},
    pdf_metadata::PdfMetadata,
};

/// Replace exactly one expanded paint invocation with an independently masked image.
///
/// A selected image paint carries the ordered, zero-based Do indices of its
/// enclosing page stream and nested Form streams. At every level, only the
/// selected instruction is retargeted to a new name and new indirect object.
/// This preserves shared resources, Form reuse and independent page siblings.
///
/// # Errors
/// Returns an error if the paint path no longer resolves to the selected image,
/// any affected stream fails complete parsing, or a required Form is malformed.
pub fn isolate_source_paint(
    original: &PdfMetadata,
    paint: &ImagePaint,
    masked_image: OwnedObject,
) -> Result<Vec<u8>, Error> {
    if paint.paint_path.is_empty() || !matches!(masked_image, OwnedObject::Stream { .. }) {
        return Err(Error::Invalid("invalid isolated image paint".into()));
    }
    let page_index = usize::try_from(paint.page - 1)
        .map_err(|_| Error::Invalid("page index exceeds usize".into()))?;
    let source_id = ObjectId::new(paint.object_number, i32::from(paint.generation));
    let mut doc = EditDocument::from_bytes(original.original_bytes().to_vec())?;
    let page_id = *doc
        .source()
        .page_ids()
        .get(page_index)
        .ok_or_else(|| Error::Invalid("paint refers to missing page".into()))?;
    let resources = inherited_page_resources(doc.source(), page_id)?;
    let page_bytes = page_content(doc.source(), page_id)?;
    let masked_ref = ObjectHandle::New(doc.add_object(masked_image));
    let (contents, resources) = alias_invocation(
        &mut doc,
        &page_bytes,
        resources,
        &paint.paint_path,
        source_id,
        masked_ref,
    )?;
    let stream = make_decoded_stream(OwnedDictionary::new(), contents)?;
    let content_id = doc.add_object(stream);
    let page = doc.edit_object(page_id)?;
    let dict = page
        .as_dictionary_mut()
        .ok_or_else(|| Error::Invalid("page is not a dictionary".into()))?;
    dict.insert(
        b"Contents".to_vec(),
        OwnedObject::Reference(ObjectHandle::New(content_id)),
    );
    dict.insert(b"Resources".to_vec(), OwnedObject::Dictionary(resources));
    Ok(doc.write_compact()?)
}

fn alias_invocation(
    doc: &mut EditDocument,
    stream: &[u8],
    mut resources: OwnedDictionary,
    path: &[u32],
    target_source: ObjectId,
    masked_ref: ObjectHandle,
) -> Result<(Vec<u8>, OwnedDictionary), Error> {
    let (&target, rest) = path
        .split_first()
        .ok_or_else(|| Error::Invalid("empty image invocation path".into()))?;
    let (resource_name, span) = target_do_operand(stream, target)?;
    let mut xobjects = match resources.get(b"XObject".as_slice()) {
        Some(raw) => resolve_dictionary(doc.source(), raw)?,
        None => return Err(Error::Invalid("paint lacks XObject resources".into())),
    };
    let source = match xobjects.get(resource_name.as_slice()) {
        Some(OwnedObject::Reference(ObjectHandle::Existing(id))) => *id,
        _ => {
            return Err(Error::Invalid(
                "target resource is not a source XObject".into(),
            ));
        }
    };
    let replacement = if rest.is_empty() {
        if source != target_source {
            return Err(Error::Invalid("paint path resolved to wrong image".into()));
        }
        masked_ref
    } else {
        let existing = doc.source().materialize(source)?;
        let OwnedObject::Stream {
            dictionary: mut form_dict,
            ..
        } = existing
        else {
            return Err(Error::Invalid("target Form is not a stream".into()));
        };
        if !matches!(
            form_dict.get(b"Subtype".as_slice()),
            Some(OwnedObject::Name(name)) if name == b"Form"
        ) {
            return Err(Error::Invalid("target XObject is not a Form".into()));
        }
        let child_resources = match form_dict.get(b"Resources".as_slice()) {
            Some(raw) => resolve_dictionary(doc.source(), raw)?,
            None => resources.clone(),
        };
        let bytes = doc.source().decoded_stream_data(source)?.into_owned();
        let (rewritten, child_resources) = alias_invocation(
            doc,
            &bytes,
            child_resources,
            rest,
            target_source,
            masked_ref,
        )?;
        form_dict.insert(
            b"Resources".to_vec(),
            OwnedObject::Dictionary(child_resources),
        );
        ObjectHandle::New(doc.add_object(make_decoded_stream(form_dict, rewritten)?))
    };
    let mut suffix = 0_u32;
    let alias = loop {
        let candidate = format!("OcrIsolatedPaint{suffix}").into_bytes();
        if !xobjects.contains_key(candidate.as_slice()) {
            break candidate;
        }
        suffix = suffix
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("resource alias space exhausted".into()))?;
    };
    xobjects.insert(alias.clone(), OwnedObject::Reference(replacement));
    resources.insert(b"XObject".to_vec(), OwnedObject::Dictionary(xobjects));
    let mut changed = Vec::with_capacity(stream.len() + alias.len());
    changed.extend_from_slice(&stream[..span.start]);
    changed.push(b'/');
    changed.extend_from_slice(&alias);
    changed.extend_from_slice(&stream[span.end..]);
    Ok((changed, resources))
}

/// Resolve the exact original name byte span using Hayro's parsed operands.
fn target_do_operand(
    stream: &[u8],
    target: u32,
) -> Result<(Vec<u8>, std::ops::Range<usize>), Error> {
    let mut iter = UntypedIter::new(stream);
    let mut do_index = 0_u32;
    let mut chosen = None;
    while let Some(instruction) = iter.next() {
        if &instruction.operator[..] != b"Do" {
            continue;
        }
        if do_index == target {
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
            chosen = Some((name.as_ref().to_vec(), operand));
        }
        do_index = do_index
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("excessive Do count".into()))?;
    }
    if !iter.is_at_end() {
        return Err(Error::Invalid("incomplete PDF content parse".into()));
    }
    let (resource_name, span) =
        chosen.ok_or_else(|| Error::Invalid("paint path Do not found".into()))?;
    Ok((resource_name, span))
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
