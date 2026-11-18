//! Source-resolution image-fragment clustering from actual PDF paint geometry.
//!
//! Use one spatial index to form connected components of compatible native
//! image footprints. A mosaic is a *group of paint invocations*, not a group
//! of `XObject` IDs: the same source can occur repeatedly, with different CTMs.
//! A caller only replaces page OCR when every visible image paint belongs to
//! a safely reconstructed image-only page.

use std::collections::{BTreeMap, HashMap, HashSet};

use image::{RgbImage, imageops};

use crate::{
    Error, PageNumber,
    geometry::{DisplayMapper, PageBox, Point},
    paint_scan::{ImagePaint, PdfMatrix},
};

const MAX_BUCKETS_PER_PAINT: u64 = 256;

/// Limits for high-resolution regional raster reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct MosaicBudget {
    /// Maximum pixels in one native-resolution group crop.
    pub max_group_pixels: u64,
    /// Maximum pixels rendered across all groups on a page.
    pub max_page_pixels: u64,
    /// Maximum number of groups to render on one page.
    pub max_groups: usize,
    /// Upper native raster DPI bound for local Poppler crop rendering.
    pub max_dpi: u32,
}

impl Default for MosaicBudget {
    fn default() -> Self {
        Self {
            max_group_pixels: 12_000_000,
            max_page_pixels: 24_000_000,
            max_groups: 24,
            max_dpi: 2400,
        }
    }
}

impl MosaicBudget {
    /// Validate bounded image allocation and renderer resolution.
    ///
    /// # Errors
    /// Returns an error for zero limits or invalid renderer DPI.
    pub fn validate(self) -> Result<Self, Error> {
        if self.max_group_pixels == 0
            || self.max_page_pixels == 0
            || self.max_groups == 0
            || !(36..=2400).contains(&self.max_dpi)
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
    /// Rotation applied clockwise to the resulting PDF display crop.
    pub clockwise_turns: u8,
    /// PDF image sample axes are reflected relative to the display frame.
    pub flip_vertical: bool,
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
    turn: u8,
    flip_vertical: bool,
}

fn oriented_turn(paint: &ImagePaint, mapper: DisplayMapper) -> Option<(u8, bool)> {
    let transform = PdfMatrix(paint.ctm);
    let first = mapper.to_raster(transform.point(Point(0.0, 0.0)));
    let second = mapper.to_raster(transform.point(Point(1.0, 0.0)));
    let page_right = mapper.to_pdf(1.0, 0.0);
    let page_down = mapper.to_pdf(0.0, 1.0);
    let page_origin = mapper.to_pdf(0.0, 0.0);
    let width = (page_right.0 - page_origin.0).hypot(page_right.1 - page_origin.1);
    let height = (page_down.0 - page_origin.0).hypot(page_down.1 - page_origin.1);
    let direction = Point((second.0 - first.0) * width, (second.1 - first.1) * height);
    if !direction.0.is_finite()
        || !direction.1.is_finite()
        || direction.0.hypot(direction.1) <= 1e-9
    {
        return None;
    }
    let quarter = direction.1.atan2(direction.0) / std::f64::consts::FRAC_PI_2;
    let nearest = quarter.round();
    if (quarter - nearest).abs() > 7.0 / 90.0 {
        return None;
    }
    // Source x-axis pointing down requires a counterclockwise raster turn
    // (three clockwise quarter-turns) to make OCR source text upright.
    let clockwise = (-(num_traits::ToPrimitive::to_i32(&nearest)?)).rem_euclid(4);
    let [a, b, c, d, _, _] = paint.ctm;
    Some((u8::try_from(clockwise).ok()?, a.mul_add(d, -b * c) < 0.0))
}

fn clipped_item(paint: &ImagePaint, page: PageBox, mapper: DisplayMapper) -> Option<Item> {
    if !paint.unsafe_reasons.is_empty() || paint.width == 0 || paint.height == 0 {
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
        .max(page.x0);
    let x1 = paint
        .page_quad
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(page.x1);
    let y0 = paint
        .page_quad
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .max(page.y0);
    let y1 = paint
        .page_quad
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(page.y1);
    let (turn, flip_vertical) = oriented_turn(paint, mapper)?;
    Some(Item {
        paint_index: paint.paint_index,
        box_: PageBox::new([x0, y0, x1, y1]).ok()?,
        pitch,
        turn,
        flip_vertical,
    })
}

fn compatible(a: Item, b: Item) -> bool {
    if (a.pitch - b.pitch).abs() > 0.075 * a.pitch.max(b.pitch) {
        return false;
    }
    let gap_x = 0.0_f64
        .max(a.box_.x0 - b.box_.x1)
        .max(b.box_.x0 - a.box_.x1);
    let gap_y = 0.0_f64
        .max(a.box_.y0 - b.box_.y1)
        .max(b.box_.y0 - a.box_.y1);
    gap_x <= 2.5 * a.pitch.max(b.pitch) && gap_y <= 2.5 * a.pitch.max(b.pitch)
}

fn bucket_cells(box_: PageBox, page: PageBox, cell: f64) -> Option<Vec<(i64, i64)>> {
    let start_x = num_traits::ToPrimitive::to_i64(&((box_.x0 - page.x0) / cell).floor())?;
    let end_x = num_traits::ToPrimitive::to_i64(&((box_.x1 - page.x0) / cell).floor())?;
    let start_y = num_traits::ToPrimitive::to_i64(&((box_.y0 - page.y0) / cell).floor())?;
    let end_y = num_traits::ToPrimitive::to_i64(&((box_.y1 - page.y0) / cell).floor())?;
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

const fn root(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn connect(parents: &mut [usize], left: usize, right: usize) {
    let a = root(parents, left);
    let b = root(parents, right);
    if a != b {
        parents[a.max(b)] = a.min(b);
    }
}

fn connected_components(items: &[Item], page: PageBox) -> BTreeMap<usize, Vec<Item>> {
    let mut parents = (0..items.len()).collect::<Vec<_>>();
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
                connect(&mut parents, i, j);
            }
        }
        // A compatible future source can be up to 1/(1-tolerance)
        // times coarser than this source. Expand old boxes accordingly,
        // then query future boxes without missing symmetric pairs.
        let margin = 2.5 * item.pitch / (1.0 - 0.075);
        let expanded = PageBox {
            x0: item.box_.x0 - margin,
            y0: item.box_.y0 - margin,
            x1: item.box_.x1 + margin,
            y1: item.box_.y1 + margin,
        };
        if let Some(cells) = bucket_cells(expanded, page, cell) {
            for key in cells {
                buckets.entry(key).or_default().push(i);
            }
        } else {
            oversized.push(i);
        }
    }
    let mut groups = BTreeMap::<usize, Vec<Item>>::new();
    for (i, &item) in items.iter().enumerate() {
        groups.entry(root(&mut parents, i)).or_default().push(item);
    }
    groups
}

fn group_region(items: &[Item], page: PageBox, pitch: f64) -> Option<PageBox> {
    let pad = pitch * 5.0;
    PageBox::new([
        (items
            .iter()
            .map(|p| p.box_.x0)
            .fold(f64::INFINITY, f64::min)
            - pad)
            .max(page.x0),
        (items
            .iter()
            .map(|p| p.box_.y0)
            .fold(f64::INFINITY, f64::min)
            - pad)
            .max(page.y0),
        (items
            .iter()
            .map(|p| p.box_.x1)
            .fold(f64::NEG_INFINITY, f64::max)
            + pad)
            .min(page.x1),
        (items
            .iter()
            .map(|p| p.box_.y1)
            .fold(f64::NEG_INFINITY, f64::max)
            + pad)
            .min(page.y1),
    ])
    .ok()
}

/// A geometrically adjacent source pair needs mosaic reconstruction
/// even if the page later fails a budget, clipping, or orientation gate.
#[must_use]
pub fn contains_native_fragments(
    paints: &[ImagePaint],
    page: PageBox,
    mapper: DisplayMapper,
) -> bool {
    let items = paints
        .iter()
        .filter_map(|paint| clipped_item(paint, page, mapper))
        .collect::<Vec<_>>();
    connected_components(&items, page)
        .values()
        .any(|group| group.len() > 1)
}

/// Rotate the actual PDF-painted crop until its source x-axis points right.
#[must_use]
pub fn orient_raster(rgb: RgbImage, clockwise_turns: u8, flip_vertical: bool) -> RgbImage {
    let rotated = match clockwise_turns % 4 {
        0 => rgb,
        1 => imageops::rotate90(&rgb),
        2 => imageops::rotate180(&rgb),
        _ => imageops::rotate270(&rgb),
    };
    if flip_vertical {
        imageops::flip_vertical(&rotated)
    } else {
        rotated
    }
}

/// Map an oriented OCR pixel corner back to the original PDF-painted crop.
#[must_use]
pub fn unrotate_pixel(
    point: Point,
    original_width: u32,
    original_height: u32,
    clockwise_turns: u8,
    flip_vertical: bool,
) -> Point {
    let (w, h) = (f64::from(original_width), f64::from(original_height));
    let oriented_height = if clockwise_turns % 2 == 0 { h } else { w };
    let point = if flip_vertical {
        Point(point.0, oriented_height - point.1)
    } else {
        point
    };
    match clockwise_turns % 4 {
        0 => point,
        1 => Point(point.1, h - point.0),
        2 => Point(w - point.0, h - point.1),
        _ => Point(w - point.1, point.0),
    }
}

/// Partition every image invocation into source-scale connected mosaics.
///
/// A page is eligible only if every image paint has sane, compatible
/// geometry and at least one actual multi-paint fragment cluster exists.
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
    if !components.values().any(|group| group.len() >= 2) || components.len() > budget.max_groups {
        return Ok(None);
    }
    let mut groups = Vec::with_capacity(components.len());
    let mut total_pixels = 0_u64;
    for component in components.values() {
        let turn = component[0].turn;
        let flip_vertical = component[0].flip_vertical;
        if component
            .iter()
            .any(|item| item.turn != turn || item.flip_vertical != flip_vertical)
        {
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
        if pixels == 0 || pixels > budget.max_group_pixels {
            return Ok(None);
        }
        total_pixels = total_pixels.saturating_add(pixels);
        if total_pixels > budget.max_page_pixels {
            return Ok(None);
        }
        groups.push(MosaicGroup {
            page,
            paint_indices: component.iter().map(|p| p.paint_index).collect(),
            region,
            pitch,
            clockwise_turns: turn,
            flip_vertical,
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
            ctm,
            page_quad: PdfMatrix(ctm).rect_quad(PageBox {
                x0: 0.0,
                y0: 0.0,
                x1: 1.0,
                y1: 1.0,
            }),
            clips: vec![],
            unsafe_reasons: vec![],
        }
    }

    #[test]
    fn pixel_corner_rotation_mapping_covers_all_quarter_turns() {
        use super::{orient_raster, unrotate_pixel};
        use image::{Rgb, RgbImage};
        let original = RgbImage::from_fn(3, 5, |x, y| {
            Rgb([
                u8::try_from(x).unwrap_or(0),
                u8::try_from(y).unwrap_or(0),
                255,
            ])
        });
        for flip_vertical in [false, true] {
            for turns in 0..=3 {
                let rotated = orient_raster(original.clone(), turns, flip_vertical);
                for y in 0..rotated.height() {
                    for x in 0..rotated.width() {
                        let source = unrotate_pixel(
                            crate::geometry::Point(f64::from(x) + 0.5, f64::from(y) + 0.5),
                            original.width(),
                            original.height(),
                            turns,
                            flip_vertical,
                        );
                        let orig_x =
                            num_traits::ToPrimitive::to_u32(&source.0.floor()).unwrap_or(999);
                        let orig_y =
                            num_traits::ToPrimitive::to_u32(&source.1.floor()).unwrap_or(999);
                        assert_eq!(rotated.get_pixel(x, y), original.get_pixel(orig_x, orig_y));
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
                    max_group_pixels: 10,
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
