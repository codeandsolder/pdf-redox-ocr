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

/// A binding retains the invocation evidence, not merely its ordinal.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct PaintBinding {
    pub(crate) id: AppearanceId,
    path: Vec<u32>,
    object_number: u32,
    generation: u16,
    dimensions: [u32; 2],
    ctm: [f64; 6],
    sample_layout: crate::sample_layout::ImageSampleLayout,
}

impl PaintBinding {
    fn new(paint: &ImagePaint, id: AppearanceId) -> Self {
        Self {
            id,
            path: paint.paint_path.clone(),
            object_number: paint.object_number,
            generation: paint.generation,
            dimensions: [paint.width, paint.height],
            ctm: paint.ctm,
            sample_layout: paint.sample_layout,
        }
    }

    fn matches(&self, paint: &ImagePaint) -> bool {
        self.path == paint.paint_path
            && self.object_number == paint.object_number
            && self.generation == paint.generation
            && self.dimensions == [paint.width, paint.height]
            && self.ctm.map(f64::to_bits) == paint.ctm.map(f64::to_bits)
            && self.sample_layout == paint.sample_layout
            && paint.unsafe_reasons.is_empty()
    }
}

/// Exact appearance bindings indexed by one-based page and expanded paint index.
#[derive(Debug, serde::Serialize)]
pub struct PaintAppearances<T> {
    pub(crate) variants: BTreeMap<AppearanceId, SourceAppearance<T>>,
    pub(crate) paint_bindings: BTreeMap<u32, BTreeMap<u32, PaintBinding>>,
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
    #[must_use = "use the appearance inventory for source-paint lookup"]
    pub fn from_object_inventory(values: BTreeMap<String, T>, paints: &[ImagePaint]) -> Self {
        let variants = values
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
        // Duplicate expanded indices or paths mean the scan cannot identify
        // actual invocations uniquely on that page, even for an object-only
        // inventory. Do not silently overwrite an earlier binding.
        let mut indices = BTreeSet::new();
        let mut paths = BTreeSet::new();
        let mut ambiguous_pages = BTreeSet::new();
        for paint in paints {
            if !indices.insert((paint.page, paint.paint_index))
                || !paths.insert((paint.page, paint.paint_path.as_slice()))
            {
                ambiguous_pages.insert(paint.page);
            }
        }
        let mut paint_bindings = BTreeMap::<u32, BTreeMap<u32, PaintBinding>>::new();
        for paint in paints {
            if !ambiguous_pages.contains(&paint.page)
                && paint.unsafe_reasons.is_empty()
                && !paint.sample_layout.contextual
                && let Some(&id) = ids.get(paint.source_id.as_str())
            {
                paint_bindings
                    .entry(paint.page)
                    .or_default()
                    .insert(paint.paint_index, PaintBinding::new(paint, id));
            }
        }
        let mut result = Self {
            variants,
            paint_bindings,
        };
        let referenced = result.referenced_ids();
        result.variants.retain(|id, _| referenced.contains(id));
        result
    }

    /// The distinct decoded or recognized appearances, without mutable
    /// access to the verified invocation bindings.
    #[must_use = "inspect distinct appearances before consuming the inventory"]
    pub const fn variants(&self) -> &BTreeMap<AppearanceId, SourceAppearance<T>> {
        &self.variants
    }

    /// Look up only an observation associated with the exact paint and source.
    #[must_use = "inspect the resolved appearance before matching source paint"]
    pub fn get(&self, paint: &ImagePaint) -> Option<(AppearanceId, &T)> {
        let binding = self
            .paint_bindings
            .get(&paint.page)?
            .get(&paint.paint_index)?;
        if !binding.matches(paint) {
            return None;
        }
        let appearance = self.variants.get(&binding.id)?;
        (appearance.source_id == paint.source_id).then_some((binding.id, &appearance.value))
    }

    #[must_use = "use the inventory length when accounting for source appearances"]
    pub fn len(&self) -> usize {
        self.variants.len()
    }

    #[must_use = "check whether source appearances are available before matching"]
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
                entry.insert(PaintBinding::new(paint, id));
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    pub(crate) fn referenced_ids(&self) -> BTreeSet<AppearanceId> {
        self.paint_bindings
            .values()
            .flat_map(|page| page.values().map(|binding| binding.id))
            .collect()
    }

    pub(crate) fn discard_unavailable_bindings(&mut self) {
        self.paint_bindings.retain(|_, page| {
            page.retain(|_, binding| self.variants.contains_key(&binding.id));
            !page.is_empty()
        });
    }

    /// Measure or project each distinct appearance without changing its bindings.
    #[must_use = "use the mapped appearances with their original paint bindings"]
    pub fn map_values<U>(&self, f: impl Fn(&T) -> U) -> PaintAppearances<U> {
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
