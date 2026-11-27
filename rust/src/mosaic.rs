//! Source-resolution image-fragment clustering from actual PDF paint geometry.
//!
//! Use one spatial index to form connected components of compatible native
//! image footprints. A mosaic is a *group of paint invocations*, not a group
//! of `XObject` IDs: the same source can occur repeatedly, with different CTMs.
//! A caller only replaces page OCR when every visible image paint belongs to
//! a safely reconstructed image-only page.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    Error, PageNumber,
    disjoint_set::DisjointSet,
    geometry::{DisplayMapper, PageBox, Point},
    paint_scan::{ImagePaint, PagePaintSummary, PdfMatrix},
};

const MAX_BUCKETS_PER_PAINT: u64 = 256;

/// Limits for high-resolution regional raster reconstruction.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct MosaicBudget {
    /// Maximum pixels in one native-resolution group crop.
    pub group_pixels: u64,
    /// Maximum pixels rendered across all groups on a page.
    pub page_pixels: u64,
    /// Maximum number of groups to render on one page.
    pub groups: usize,
    /// Upper native raster DPI bound for local Poppler crop rendering.
    pub dpi: u32,
}

impl Default for MosaicBudget {
    fn default() -> Self {
        Self {
            group_pixels: 12_000_000,
            page_pixels: 24_000_000,
            groups: 24,
            dpi: 2400,
        }
    }
}

impl MosaicBudget {
    /// Validate bounded image allocation and renderer resolution.
    ///
    /// # Errors
    /// Returns an error for zero limits or invalid renderer DPI.
    pub fn validate(self) -> Result<Self, Error> {
        if self.group_pixels == 0
            || self.page_pixels == 0
            || self.groups == 0
            || !(36..=2400).contains(&self.dpi)
        {
            return Err(Error::Invalid(
                "invalid native fragment mosaic budget".into(),
            ));
        }
        Ok(self)
    }
}

/// One geometrically connected group of actual PDF image paint invocations.
#[derive(Clone, Debug)]
pub struct MosaicGroup {
    /// One-based page containing every group member.
    pub page: PageNumber,
    /// Expanded paint invocation IDs, in original page painting order.
    pub paint_indices: Vec<u32>,
    /// Selected source-resolution PDF-space rectangle, page-clipped.
    pub region: PageBox,
    /// Median native sample pitch, in PDF user-space units per pixel.
    pub pitch: f64,
    /// Original image's native x/y axes in PDF display coordinates.
    pub axes: MosaicAxes,
    /// Conservative estimated target pixels before rasterization.
    pub estimated_pixels: u64,
}

/// Complete bounded partition of an image-only page's image paint operations.
#[derive(Debug)]
pub struct MosaicPlan {
    /// All image paints, including independent singleton groups.
    pub groups: Vec<MosaicGroup>,
    /// Estimated aggregate raster pixels for the page.
    pub estimated_pixels: u64,
}

#[derive(Clone, Copy)]
struct Item {
    paint_index: u32,
    box_: PageBox,
    pitch: f64,
    axes: MosaicAxes,
}

/// Source image directions in page display coordinates, where Y increases
/// downward and each basis column has unit Euclidean length.
///
/// An affine image's native horizontal/vertical axes are independent; this
/// also represents reflected and sheared paints without discrete rotations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MosaicAxes {
    /// Direction of increasing native image column.
    pub x: Point,
    /// Direction of increasing native image row.
    pub y: Point,
}

impl MosaicAxes {
    /// Project a page-display vector into source-aligned coordinates.
    #[must_use = "use transformed coordinates when locating mosaic elements"]
    pub fn project(self, Point(x, y): Point) -> Point {
        let det = self.x.0.mul_add(self.y.1, -(self.x.1 * self.y.0));
        Point(
            self.y.1.mul_add(x, -(self.y.0 * y)) / det,
            self.x.0.mul_add(y, -(self.x.1 * x)) / det,
        )
    }

    /// Convert source-aligned coordinates to page-display vector.
    #[must_use = "use inverse-transformed coordinates when mapping mosaic edits"]
    pub fn unproject(self, Point(x, y): Point) -> Point {
        Point(
            self.x.0.mul_add(x, self.y.0 * y),
            self.x.1.mul_add(x, self.y.1 * y),
        )
    }

    fn similar(self, other: Self) -> bool {
        const MIN_ALIGNMENT: f64 = 0.992_546_151_641_322; // cos(7°)
        self.x.0.mul_add(other.x.0, self.x.1 * other.x.1) >= MIN_ALIGNMENT
            && self.y.0.mul_add(other.y.0, self.y.1 * other.y.1) >= MIN_ALIGNMENT
    }
}

fn native_axes(paint: &ImagePaint, mapper: DisplayMapper) -> Option<MosaicAxes> {
    let transform = PdfMatrix(paint.ctm);
    // PDF image sample rows point down from user-space (0,1) to (0,0).
    let origin = mapper.to_raster(transform.point(Point(0.0, 1.0)));
    let source_right = mapper.to_raster(transform.point(Point(1.0, 1.0)));
    let source_down = mapper.to_raster(transform.point(Point(0.0, 0.0)));
    let Point(width, height) = mapper.display_dimensions();
    let vector = |point: Point| Point((point.0 - origin.0) * width, (point.1 - origin.1) * height);
    let unit = |Point(x, y): Point| {
        let length = x.hypot(y);
        (length > 1e-9 && length.is_finite()).then_some(Point(x / length, y / length))
    };
    let axes = MosaicAxes {
        x: unit(vector(source_right))?,
        y: unit(vector(source_down))?,
    };
    let determinant = axes.x.0.mul_add(axes.y.1, -(axes.x.1 * axes.y.0));
    (determinant.abs() > 0.15).then_some(axes)
}

/// Composite-only image effects can be rendered without interpreting
/// the source samples ourselves. Unknown source geometry remains unsupported.
pub(crate) fn compositor_handles_reason(reason: &str) -> bool {
    matches!(
        reason,
        "source_alpha_or_color_key_requires_compositing"
            | "nonrectangular_or_untracked_clip"
            | "unsupported_transparency_or_graphics_state"
            | "image_stencil_mask"
            | "page_transparency_group"
            | "resource_default_color_space_requires_resolution"
    )
}

/// A page containing only actual source image paints can be reconstructed
/// through Poppler even when the raw-source-only path rejects alpha or blends.
///
/// # Notes
/// Unknown content, image placements, inline images and annotations remain
/// excluded because the raster-only merger requires every painted mark to
/// be accounted for by an image paint.
#[must_use = "check paint eligibility before using mosaic recognition"]
pub fn mosaic_page_eligible(summary: &PagePaintSummary, paints: &[ImagePaint]) -> bool {
    summary.image_paints > 0
        && summary.image_paints == paints.len()
        && summary.native_searchable_shows == 0
        && summary.non_image_visual_paint == 0
        && summary.unknown_paint == 0
        && summary.unsupported_inline_images == 0
        && summary.unsafe_form_recursion == 0
        && summary.unbalanced_q == 0
        && summary.unbalanced_stream == 0
        && summary.annotations == 0
        && paints.iter().all(|paint| {
            paint.page == summary.page
                && paint
                    .unsafe_reasons
                    .iter()
                    .all(|reason| compositor_handles_reason(reason))
        })
}

fn clipped_item(paint: &ImagePaint, page: PageBox, mapper: DisplayMapper) -> Option<Item> {
    if paint
        .unsafe_reasons
        .iter()
        .any(|reason| !compositor_handles_reason(reason))
        || paint.width == 0
        || paint.height == 0
    {
        return None;
    }
    let [a, b, c, d, _, _] = paint.ctm;
    if !paint.ctm.iter().all(|x| x.is_finite()) {
        return None;
    }
    let pitch_x = a.hypot(b) / f64::from(paint.width);
    let pitch_y = c.hypot(d) / f64::from(paint.height);
    let pitch = f64::midpoint(pitch_x, pitch_y);
    if !pitch.is_finite()
        || pitch <= 0.0
        || (pitch_x - pitch_y).abs() > 0.075 * pitch
        || a.mul_add(c, b * d).abs() > 0.15 * a.hypot(b) * c.hypot(d)
        || PdfMatrix(paint.ctm).area_scale() <= 1e-9
    {
        return None;
    }
    let x0 = paint
        .page_quad
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .max(page.x0());
    let x1 = paint
        .page_quad
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(page.x1());
    let y0 = paint
        .page_quad
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .max(page.y0());
    let y1 = paint
        .page_quad
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(page.y1());
    Some(Item {
        paint_index: paint.paint_index,
        box_: PageBox::new([x0, y0, x1, y1]).ok()?,
        pitch,
        axes: native_axes(paint, mapper)?,
    })
}

fn compatible(a: Item, b: Item) -> bool {
    if (a.pitch - b.pitch).abs() > 0.075 * a.pitch.max(b.pitch) {
        return false;
    }
    let gap_x = 0.0_f64
        .max(a.box_.x0() - b.box_.x1())
        .max(b.box_.x0() - a.box_.x1());
    let gap_y = 0.0_f64
        .max(a.box_.y0() - b.box_.y1())
        .max(b.box_.y0() - a.box_.y1());
    gap_x <= 2.5 * a.pitch.max(b.pitch) && gap_y <= 2.5 * a.pitch.max(b.pitch)
}

fn bucket_cells(box_: PageBox, page: PageBox, cell: f64) -> Option<Vec<(i64, i64)>> {
    let start_x = num_traits::ToPrimitive::to_i64(&((box_.x0() - page.x0()) / cell).floor())?;
    let end_x = num_traits::ToPrimitive::to_i64(&((box_.x1() - page.x0()) / cell).floor())?;
    let start_y = num_traits::ToPrimitive::to_i64(&((box_.y0() - page.y0()) / cell).floor())?;
    let end_y = num_traits::ToPrimitive::to_i64(&((box_.y1() - page.y0()) / cell).floor())?;
    let span_x = u64::try_from(end_x.checked_sub(start_x)?.checked_add(1)?).ok()?;
    let span_y = u64::try_from(end_y.checked_sub(start_y)?.checked_add(1)?).ok()?;
    if span_x.checked_mul(span_y)? > MAX_BUCKETS_PER_PAINT {
        return None;
    }
    Some(
        (start_x..=end_x)
            .flat_map(|x| (start_y..=end_y).map(move |y| (x, y)))
            .collect(),
    )
}

fn connected_components(items: &[Item], page: PageBox) -> BTreeMap<usize, Vec<Item>> {
    let mut components = DisjointSet::new(items.len());
    let cell = (page.width() * page.height()
        / num_traits::ToPrimitive::to_f64(&items.len().max(1)).unwrap_or(f64::MAX))
    .sqrt();
    let mut buckets = HashMap::<(i64, i64), Vec<usize>>::new();
    let mut oversized = Vec::<usize>::new();
    for (i, &item) in items.iter().enumerate() {
        let query = bucket_cells(item.box_, page, cell);
        let mut candidate_indices = HashSet::new();
        if let Some(cells) = query {
            candidate_indices.extend(oversized.iter().copied());
            for key in cells {
                if let Some(candidates) = buckets.get(&key) {
                    candidate_indices.extend(candidates.iter().copied());
                }
            }
        } else {
            candidate_indices.extend(0..i);
        }
        for j in candidate_indices {
            if compatible(item, items[j]) {
                components.connect(i, j);
            }
        }
        // A compatible future source can be up to 1/(1-tolerance)
        // times coarser than this source. Expand old boxes accordingly,
        // then query future boxes without missing symmetric pairs.
        let margin = 2.5 * item.pitch / (1.0 - 0.075);
        let expanded = PageBox::new([
            item.box_.x0() - margin,
            item.box_.y0() - margin,
            item.box_.x1() + margin,
            item.box_.y1() + margin,
        ]);
        if let Some(cells) = expanded
            .ok()
            .and_then(|bounds| bucket_cells(bounds, page, cell))
        {
            for key in cells {
                buckets.entry(key).or_default().push(i);
            }
        } else {
            oversized.push(i);
        }
    }
    let mut groups = BTreeMap::<usize, Vec<Item>>::new();
    for (i, &item) in items.iter().enumerate() {
        groups.entry(components.root(i)).or_default().push(item);
    }
    groups
}

fn group_region(items: &[Item], page: PageBox, pitch: f64) -> Option<PageBox> {
    let pad = pitch * 5.0;
    PageBox::new([
        (items
            .iter()
            .map(|p| p.box_.x0())
            .fold(f64::INFINITY, f64::min)
            - pad)
            .max(page.x0()),
        (items
            .iter()
            .map(|p| p.box_.y0())
            .fold(f64::INFINITY, f64::min)
            - pad)
            .max(page.y0()),
        (items
            .iter()
            .map(|p| p.box_.x1())
            .fold(f64::NEG_INFINITY, f64::max)
            + pad)
            .min(page.x1()),
        (items
            .iter()
            .map(|p| p.box_.y1())
            .fold(f64::NEG_INFINITY, f64::max)
            + pad)
            .min(page.y1()),
    ])
    .ok()
}

/// Determine when independent image recognition must use composited inputs.
///
/// This holds even when the subsequent crop plan fails a budget or geometry
/// gate: deliberately avoid the raw-source shortcut in that case.
#[must_use = "select composited recognition when separated paints are insufficient"]
pub fn requires_composited_recognition(
    paints: &[ImagePaint],
    page: PageBox,
    mapper: DisplayMapper,
) -> bool {
    if paints.iter().any(|paint| !paint.unsafe_reasons.is_empty()) {
        return true;
    }
    let items = paints
        .iter()
        .filter_map(|paint| clipped_item(paint, page, mapper))
        .collect::<Vec<_>>();
    connected_components(&items, page)
        .values()
        .any(|group| group.len() > 1)
}

/// Partition every image invocation into source-scale connected mosaics.
///
/// A page is eligible only if every image paint has sane, compatible
/// geometry, with either a fragment cluster or an independent-compositing
/// blocker. Singleton groups then share the same rendered-pixel path.
/// Returns None when rendering would exceed either independent pixel budget
/// or when source text orientations differ within a connected component.
///
/// # Errors
/// Returns an error for invalid resource ceilings.
pub fn plan_raster_mosaics(
    page: PageNumber,
    paints: &[ImagePaint],
    box_: PageBox,
    mapper: DisplayMapper,
    budget: MosaicBudget,
) -> Result<Option<MosaicPlan>, Error> {
    let budget = budget.validate()?;
    if paints.is_empty() || paints.iter().any(|paint| paint.page != page.get()) {
        return Ok(None);
    }
    let Some(items) = paints
        .iter()
        .map(|paint| clipped_item(paint, box_, mapper))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let components = connected_components(&items, box_);
    let requires_compositing = paints.iter().any(|paint| !paint.unsafe_reasons.is_empty());
    if (!requires_compositing && !components.values().any(|group| group.len() >= 2))
        || components.len() > budget.groups
    {
        return Ok(None);
    }
    let mut groups = Vec::with_capacity(components.len());
    let mut total_pixels = 0_u64;
    for component in components.values() {
        let axes = component[0].axes;
        if component.iter().any(|item| !axes.similar(item.axes)) {
            return Ok(None);
        }
        let mut pitches = component.iter().map(|item| item.pitch).collect::<Vec<_>>();
        pitches.sort_unstable_by(f64::total_cmp);
        let pitch = pitches[pitches.len() / 2];
        if component
            .iter()
            .any(|item| (item.pitch - pitch).abs() > 0.075 * item.pitch.max(pitch))
        {
            return Ok(None);
        }
        let Some(region) = group_region(component, box_, pitch) else {
            return Ok(None);
        };
        let estimate_x = num_traits::ToPrimitive::to_u64(&(region.width() / pitch).ceil());
        let estimate_y = num_traits::ToPrimitive::to_u64(&(region.height() / pitch).ceil());
        let Some(pixels) = estimate_x
            .zip(estimate_y)
            .and_then(|(x, y)| x.checked_mul(y))
        else {
            return Ok(None);
        };
        if pixels == 0 || pixels > budget.group_pixels {
            return Ok(None);
        }
        total_pixels = total_pixels.saturating_add(pixels);
        if total_pixels > budget.page_pixels {
            return Ok(None);
        }
        groups.push(MosaicGroup {
            page,
            paint_indices: component.iter().map(|p| p.paint_index).collect(),
            region,
            pitch,
            axes,
            estimated_pixels: pixels,
        });
    }
    Ok(Some(MosaicPlan {
        groups,
        estimated_pixels: total_pixels,
    }))
}

#[cfg(test)]
mod tests {
    use super::{MosaicBudget, plan_raster_mosaics};
    use crate::{
        PageNumber,
        geometry::{DisplayMapper, PageBox, Rotation},
        paint_scan::{ImagePaint, PdfMatrix},
    };

    fn paint(id: u32, x: f64, y: f64, w: u32, h: u32, pitch: f64) -> ImagePaint {
        let ctm = [f64::from(w) * pitch, 0.0, 0.0, f64::from(h) * pitch, x, y];
        ImagePaint {
            page: 1,
            paint_index: id,
            paint_path: vec![id],
            source_id: format!("{}:0", id + 10),
            object_number: id + 10,
            generation: 0,
            width: w,
            height: h,
            sample_layout: crate::sample_layout::ImageSampleLayout::default(),
            ctm,
            page_quad: PdfMatrix(ctm).rect_quad(PageBox::UNIT),
            clips: vec![],
            unsafe_reasons: vec![],
        }
    }

    #[test]
    fn arbitrary_affine_axes_roundtrip_rotation_reflection_and_shear() {
        use super::MosaicAxes;
        for degrees in [0.0_f64, 11.0, 17.0, 31.0, 45.0, 89.0, 173.0, 270.0] {
            let angle = degrees.to_radians();
            let (s, c) = angle.sin_cos();
            for reflect in [1.0, -1.0] {
                for shear in [0.0, 0.07] {
                    let axes = MosaicAxes {
                        x: crate::geometry::Point(c, s),
                        y: crate::geometry::Point(
                            c.mul_add(shear, -s) * reflect,
                            s.mul_add(shear, c) * reflect,
                        ),
                    };
                    for point in [
                        crate::geometry::Point(0.0, 0.0),
                        crate::geometry::Point(3.5, 17.25),
                        crate::geometry::Point(-135.0, 42.0),
                    ] {
                        let rebuilt = axes.unproject(axes.project(point));
                        assert!((rebuilt.0 - point.0).abs() < 1e-10);
                        assert!((rebuilt.1 - point.1).abs() < 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn transitive_native_clusters_are_order_independent() -> Result<(), crate::Error> {
        let page = PageBox::new([0.0, 0.0, 900.0, 800.0])?;
        let mapper = DisplayMapper::new(page, Rotation::Deg0);
        let mut source = vec![
            paint(0, 20.0, 50.0, 10, 20, 1.0),
            paint(1, 30.5, 50.0, 10, 20, 1.05),
            paint(2, 43.0, 50.0, 10, 20, 1.0),
            paint(3, 600.0, 500.0, 20, 20, 1.0),
        ];
        let result = plan_raster_mosaics(
            PageNumber::new(1)?,
            &source,
            page,
            mapper,
            MosaicBudget::default(),
        )?
        .ok_or(crate::Error::Invalid("clusters absent".into()))?;
        assert_eq!(result.groups.len(), 2);
        assert_eq!(result.groups[0].paint_indices, [0, 1, 2]);
        assert_eq!(result.groups[1].paint_indices, [3]);
        source.reverse();
        let reverse = plan_raster_mosaics(
            PageNumber::new(1)?,
            &source,
            page,
            mapper,
            MosaicBudget::default(),
        )?
        .ok_or(crate::Error::Invalid("reverse clusters absent".into()))?;
        let mut actual = reverse
            .groups
            .iter()
            .map(|g| {
                let mut paints = g.paint_indices.clone();
                paints.sort_unstable();
                paints
            })
            .collect::<Vec<_>>();
        actual.sort();
        assert_eq!(actual, vec![vec![0, 1, 2], vec![3]]);
        Ok(())
    }

    fn draw(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        u32::try_from(*state >> 32).unwrap_or(u32::MAX)
    }

    fn sorted_components(items: &[super::Item], page: PageBox, indexed: bool) -> Vec<Vec<u32>> {
        let mut components = if indexed {
            super::connected_components(items, page)
        } else {
            let mut components = crate::disjoint_set::DisjointSet::new(items.len());
            for (i, &item) in items.iter().enumerate() {
                for (j, &other) in items.iter().enumerate().skip(i + 1) {
                    if super::compatible(item, other) {
                        components.connect(i, j);
                    }
                }
            }
            let mut result = std::collections::BTreeMap::new();
            for (i, &item) in items.iter().enumerate() {
                result
                    .entry(components.root(i))
                    .or_insert_with(Vec::new)
                    .push(item);
            }
            result
        }
        .into_values()
        .map(|group| {
            let mut ids = group
                .into_iter()
                .map(|item| item.paint_index)
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids
        })
        .collect::<Vec<_>>();
        components.sort();
        components
    }

    #[test]
    fn spatial_union_matches_quadratic_reference_under_permutation() -> Result<(), crate::Error> {
        let page = PageBox::new([-50.0, -50.0, 850.0, 650.0])?;
        let mapper = DisplayMapper::new(page, Rotation::Deg0);
        let mut seed = 314_159_u64;
        for trial in 0..12 {
            let mut paints = Vec::new();
            for index in 0..220 {
                let pitch = [0.6, 0.62, 0.65, 0.75, 1.0, 1.05, 1.1, 1.3]
                    [usize::try_from(draw(&mut seed) % 8).unwrap_or(0)];
                let width = [12, 24, 60, 180][usize::try_from(draw(&mut seed) % 4).unwrap_or(0)];
                let height = [6, 12, 40, 200][usize::try_from(draw(&mut seed) % 4).unwrap_or(0)];
                let x = f64::from(i32::try_from(draw(&mut seed) % 1100).unwrap_or(0) - 150);
                let y = f64::from(i32::try_from(draw(&mut seed) % 900).unwrap_or(0) - 150);
                let mut candidate = paint(index, x, y, width, height, pitch);
                if trial % 3 == 0 && index % 23 == 0 {
                    candidate.unsafe_reasons.push("unsupported");
                }
                paints.push(candidate);
            }
            let mut items = paints
                .iter()
                .filter_map(|paint| super::clipped_item(paint, page, mapper))
                .collect::<Vec<_>>();
            let expected = sorted_components(&items, page, false);
            assert_eq!(
                sorted_components(&items, page, true),
                expected,
                "trial {trial}"
            );
            items.reverse();
            assert_eq!(
                sorted_components(&items, page, true),
                expected,
                "reverse trial {trial}"
            );
        }
        Ok(())
    }

    #[test]
    fn incompatible_pitch_and_out_of_budget_keep_page_fallback() -> Result<(), crate::Error> {
        let page = PageBox::new([0.0, 0.0, 100.0, 100.0])?;
        let mapper = DisplayMapper::new(page, Rotation::Deg0);
        let left = paint(0, 20.0, 20.0, 40, 30, 0.75);
        let distant = paint(1, 75.0, 20.0, 10, 30, 1.1);
        let budget = MosaicBudget::default();
        assert!(
            plan_raster_mosaics(
                PageNumber::new(1)?,
                &[left.clone(), distant],
                page,
                mapper,
                budget
            )?
            .is_none()
        );
        assert!(
            plan_raster_mosaics(
                PageNumber::new(1)?,
                &[left.clone(), paint(1, 50.0, 20.0, 40, 30, 0.75)],
                page,
                mapper,
                MosaicBudget {
                    group_pixels: 10,
                    ..budget
                }
            )?
            .is_none()
        );
        let mut invalid = paint(1, 50.0, 20.0, 40, 30, 0.75);
        invalid.unsafe_reasons.push("unsupported_blending");
        assert!(
            plan_raster_mosaics(PageNumber::new(1)?, &[left, invalid], page, mapper, budget)?
                .is_none()
        );
        Ok(())
    }
}
