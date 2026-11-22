//! Document-local native image appearances, bound to actual page paints.

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use crate::paint_scan::ImagePaint;

/// One distinct decoded appearance of an original indirect image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(transparent)]
pub struct AppearanceId(pub usize);

/// A raster or recognition value associated with its original image object.
#[derive(Debug, serde::Serialize)]
pub struct SourceAppearance<T> {
    pub source_id: String,
    pub value: T,
}

/// Exact appearance bindings indexed by one-based page and expanded paint index.
#[derive(Debug, serde::Serialize)]
pub struct PaintAppearances<T> {
    pub variants: BTreeMap<AppearanceId, SourceAppearance<T>>,
    pub paint_bindings: BTreeMap<u32, BTreeMap<u32, AppearanceId>>,
}

impl<T> Default for PaintAppearances<T> {
    fn default() -> Self {
        Self {
            variants: BTreeMap::new(),
            paint_bindings: BTreeMap::new(),
        }
    }
}

impl<T> PaintAppearances<T> {
    /// Import an object-only raster inventory, as produced by pdfimages.
    /// Named resource-dependent images require an actual appearance binding.
    #[must_use]
    pub fn from_object_inventory(values: BTreeMap<String, T>, paints: &[ImagePaint]) -> Self {
        let mut variants = values
            .into_iter()
            .enumerate()
            .map(|(index, (source_id, value))| {
                (AppearanceId(index), SourceAppearance { source_id, value })
            })
            .collect::<BTreeMap<_, _>>();
        let ids = variants
            .iter()
            .map(|(&id, appearance)| (appearance.source_id.as_str(), id))
            .collect::<BTreeMap<_, _>>();
        let mut paint_bindings = BTreeMap::<u32, BTreeMap<u32, AppearanceId>>::new();
        for paint in paints {
            if paint.unsafe_reasons.is_empty()
                && !paint.sample_layout.contextual
                && let Some(&id) = ids.get(paint.source_id.as_str())
            {
                paint_bindings
                    .entry(paint.page)
                    .or_default()
                    .insert(paint.paint_index, id);
            }
        }
        let referenced: BTreeSet<_> = paint_bindings
            .values()
            .flat_map(|page| page.values().copied())
            .collect();
        variants.retain(|id, _| referenced.contains(id));
        Self {
            variants,
            paint_bindings,
        }
    }

    /// Look up only an observation associated with the exact paint and source.
    #[must_use]
    pub fn get(&self, paint: &ImagePaint) -> Option<(AppearanceId, &T)> {
        let id = *self
            .paint_bindings
            .get(&paint.page)?
            .get(&paint.paint_index)?;
        let appearance = self.variants.get(&id)?;
        (appearance.source_id == paint.source_id).then_some((id, &appearance.value))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.variants.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }

    pub(crate) fn bind(&mut self, paint: &ImagePaint, id: AppearanceId) -> bool {
        if self
            .variants
            .get(&id)
            .is_none_or(|appearance| appearance.source_id != paint.source_id)
        {
            return false;
        }
        match self
            .paint_bindings
            .entry(paint.page)
            .or_default()
            .entry(paint.paint_index)
        {
            Entry::Vacant(entry) => {
                entry.insert(id);
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    pub(crate) fn referenced_ids(&self) -> BTreeSet<AppearanceId> {
        self.paint_bindings
            .values()
            .flat_map(|page| page.values().copied())
            .collect()
    }

    pub(crate) fn discard_unavailable_bindings(&mut self) {
        self.paint_bindings.retain(|_, page| {
            page.retain(|_, id| self.variants.contains_key(id));
            !page.is_empty()
        });
    }

    /// Measure or project each distinct appearance without changing its bindings.
    pub(crate) fn map_values<U>(&self, f: impl Fn(&T) -> U) -> PaintAppearances<U> {
        PaintAppearances {
            variants: self
                .variants
                .iter()
                .map(|(&id, appearance)| {
                    (
                        id,
                        SourceAppearance {
                            source_id: appearance.source_id.clone(),
                            value: f(&appearance.value),
                        },
                    )
                })
                .collect(),
            paint_bindings: self.paint_bindings.clone(),
        }
    }
}
