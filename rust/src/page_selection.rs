//! Shared one-based PDF page selection grammar.

use std::collections::{BTreeMap, BTreeSet};

use crate::{Error, PageNumber};

/// Parse comma-separated page numbers and inclusive ranges (for example 1,3-5).
/// Default to every page when no selection is supplied.
///
/// # Errors
/// Rejects empty, duplicate, reversed or out-of-document selections.
pub fn parse_page_selection(
    specification: Option<&str>,
    count: usize,
) -> Result<Vec<PageNumber>, Error> {
    let count = u32::try_from(count).map_err(|_| Error::Invalid("too many PDF pages".into()))?;
    let Some(specification) = specification else {
        return (1..=count).map(PageNumber::new).collect();
    };
    let mut pages = BTreeSet::new();
    for part in specification.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(Error::Invalid("empty page in selection".into()));
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start
                .trim()
                .parse::<u32>()
                .map_err(|e| Error::Invalid(e.to_string()))?;
            let end = end
                .trim()
                .parse::<u32>()
                .map_err(|e| Error::Invalid(e.to_string()))?;
            if start == 0 || start > end || end > count {
                return Err(Error::Invalid("invalid page range".into()));
            }
            for number in start..=end {
                if !pages.insert(number) {
                    return Err(Error::Invalid("duplicate page selection".into()));
                }
            }
        } else {
            let number = part
                .parse::<u32>()
                .map_err(|e| Error::Invalid(e.to_string()))?;
            if number == 0 || number > count || !pages.insert(number) {
                return Err(Error::Invalid("invalid or duplicate selected page".into()));
            }
        }
    }
    if pages.is_empty() {
        return Err(Error::Invalid("empty page selection".into()));
    }
    pages.into_iter().map(PageNumber::new).collect()
}

/// Parse explicit original-draft page and zero-based line indices.
///
/// An empty specification means no explicit selection.
///
/// # Errors
/// Rejects malformed or duplicate targets.
pub fn parse_exact_lines(
    specifications: &[String],
) -> Result<BTreeMap<PageNumber, BTreeSet<usize>>, Error> {
    let mut targets = BTreeMap::<PageNumber, BTreeSet<usize>>::new();
    for spec in specifications {
        let (page, index) = spec
            .split_once(':')
            .ok_or_else(|| Error::Invalid(format!("expected PAGE:INDEX, got {spec:?}")))?;
        let number = page
            .parse::<u32>()
            .map_err(|_| Error::Invalid(format!("invalid OCR target page {page:?}")))?;
        let number = PageNumber::new(number)?;
        let index = index
            .parse::<usize>()
            .map_err(|_| Error::Invalid(format!("invalid OCR target index {index:?}")))?;
        if !targets.entry(number).or_default().insert(index) {
            return Err(Error::Invalid(format!(
                "duplicate explicit OCR target {spec:?}"
            )));
        }
    }
    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::parse_page_selection;

    #[test]
    fn full_and_partial_selections_preserve_identity() -> Result<(), crate::Error> {
        assert_eq!(
            parse_page_selection(None, 3)?
                .iter()
                .map(|p| p.get())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            parse_page_selection(Some("3,1-2"), 3)?
                .iter()
                .map(|p| p.get())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        for input in ["", "0", "5", "2-1", "1,1", "1-2,2", "1,,2"] {
            assert!(parse_page_selection(Some(input), 3).is_err(), "{input}");
        }
        Ok(())
    }
}
