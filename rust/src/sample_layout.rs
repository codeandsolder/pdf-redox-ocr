//! Structural native image sample layouts, independent of color conversion.

use pdf_redox::{OwnedDictionary, OwnedObject, SourcePdf};

use crate::{Error, pdf_metadata::resolve_owned};

/// Native packed-sample layout in the actual paint's resource environment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct ImageSampleLayout {
    pub components: Option<usize>,
    /// A named resource, rather than a self-contained image definition.
    pub contextual: bool,
}

pub(crate) fn image_sample_layout(
    source: &SourcePdf,
    dict: &OwnedDictionary,
    resources: &OwnedDictionary,
) -> ImageSampleLayout {
    let Some(raw) = dict.get(b"ColorSpace".as_slice()) else {
        return ImageSampleLayout::default();
    };
    let Some(resolved) = resolve_owned(source, raw.clone()).ok() else {
        return ImageSampleLayout {
            components: None,
            contextual: true,
        };
    };
    let components = color_components(source, &resolved).ok().flatten();
    if components.is_some() {
        return ImageSampleLayout {
            components,
            contextual: false,
        };
    }
    let OwnedObject::Name(alias) = resolved else {
        return ImageSampleLayout::default();
    };
    let components = resources
        .get(b"ColorSpace".as_slice())
        .and_then(|raw| resolve_owned(source, raw.clone()).ok())
        .and_then(|table| table.as_dictionary()?.get(alias.as_slice()).cloned())
        .and_then(|definition| color_components(source, &definition).ok().flatten());
    ImageSampleLayout {
        components,
        contextual: true,
    }
}

/// Count PDF color-space components without performing color conversion.
/// Unknown named aliases require page/Form resources and cannot be guessed.
pub(crate) fn color_components(
    source: &SourcePdf,
    raw: &OwnedObject,
) -> Result<Option<usize>, Error> {
    let resolved = resolve_owned(source, raw.clone())?;
    let OwnedObject::Array(items) = &resolved else {
        return Ok(match resolved {
            OwnedObject::Name(name) => match name.as_slice() {
                b"DeviceGray" => Some(1),
                b"DeviceRGB" => Some(3),
                b"DeviceCMYK" => Some(4),
                _ => None,
            },
            _ => None,
        });
    };
    let Some(first) = items.first() else {
        return Ok(None);
    };
    let OwnedObject::Name(kind) = resolve_owned(source, first.clone())? else {
        return Ok(None);
    };
    let count = match kind.as_slice() {
        b"CalGray" if items.len() == 2 => Some(1),
        b"CalRGB" | b"Lab" if items.len() == 2 => Some(3),
        b"Indexed" | b"I" | b"Separation" if items.len() == 4 => Some(1),
        b"DeviceN" if items.len() >= 4 => match resolve_owned(source, items[1].clone())? {
            OwnedObject::Array(names)
                if names.iter().all(|n| matches!(n, OwnedObject::Name(_))) =>
            {
                Some(names.len())
            }
            _ => None,
        },
        b"ICCBased" if items.len() == 2 => {
            let profile = resolve_owned(source, items[1].clone())?;
            profile
                .as_dictionary()
                .and_then(|dict| dict.get(b"N".as_slice()))
                .and_then(|value| match value {
                    OwnedObject::Integer(value) => usize::try_from(*value).ok(),
                    _ => None,
                })
                .filter(|channels| matches!(*channels, 1 | 3 | 4))
        }
        _ => None,
    };
    Ok(count.filter(|channels| (1..=32).contains(channels)))
}
