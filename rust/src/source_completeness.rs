//! Prove that an apparently cropped ink run contains complete foreground.
//!
//! The unmodified padded sample and the recognition crop share a threshold,
//! pixel lattice, and connected-component identity. A whole-component
//! partition with equal exact pixel count proves that no unobserved part of
//! the run crosses the crop boundary. A bounding-box heuristic alone would
//! not prove this.

use serde::Serialize;

use crate::{
    candidate_ink::SourceInk,
    glyph_isolation::InkGroup,
    pixel_regions::{InkBounds, PixelRegion},
};

#[derive(Debug, Serialize)]
pub struct CompleteRun {
    pub run: usize,
    pub component_indices: Vec<usize>,
    pub ink_pixels: u64,
}

const fn contained(outer: InkBounds, inner: InkBounds) -> bool {
    outer.left <= inner.left
        && outer.top <= inner.top
        && inner.right <= outer.right
        && inner.bottom <= outer.bottom
}

fn prove_one(
    run: usize,
    group: &InkGroup,
    observation: InkBounds,
    sample_size: (u32, u32),
    components: &[PixelRegion],
) -> Option<CompleteRun> {
    if !group.touches_line_edge || group.run.observation_support.inside_pixels == 0 {
        return None;
    }
    let local = group.run.ink;
    let box_in_sample = InkBounds {
        left: local.left.checked_add(observation.left)?,
        right: local.right.checked_add(observation.left)?,
        top: local.top.checked_add(observation.top)?,
        bottom: local.bottom.checked_add(observation.top)?,
    };
    if !contained(observation, box_in_sample) {
        return None;
    }
    let mut ink_pixels = 0_u64;
    let mut component_indices = Vec::new();
    for (index, component) in components.iter().enumerate() {
        let b = component.bounds;
        if component.observation_support.outside_pixels != 0
            || b.touches_edge(sample_size.0, sample_size.1)
            || !contained(box_in_sample, b)
            || !contained(observation, b)
        {
            continue;
        }
        let pixels = u64::try_from(component.area_pixels).ok()?;
        if pixels == 0 || component.observation_support.inside_pixels != pixels {
            return None;
        }
        ink_pixels = ink_pixels.checked_add(pixels)?;
        component_indices.push(index);
    }
    (ink_pixels == group.run.observation_support.inside_pixels && !component_indices.is_empty())
        .then_some(CompleteRun {
            run,
            component_indices,
            ink_pixels,
        })
}

#[must_use]
pub fn prove(
    source_ink: &SourceInk,
    observation: InkBounds,
    sample_size: (u32, u32),
    components: &[PixelRegion],
) -> Vec<CompleteRun> {
    source_ink
        .groups
        .iter()
        .enumerate()
        .filter_map(|(run, group)| prove_one(run, group, observation, sample_size, components))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, imageops};

    use crate::glyph_isolation::isolate_ink;

    fn bounds(left: u32, top: u32, right: u32, bottom: u32) -> InkBounds {
        InkBounds {
            left,
            top,
            right,
            bottom,
        }
    }

    fn regions_on_both(
        image: &GrayImage,
        observation: InkBounds,
    ) -> (Vec<PixelRegion>, Vec<InkGroup>) {
        let full = isolate_ink(image, observation, 128, &[128]).unwrap();
        let crop = imageops::crop_imm(
            image,
            observation.left,
            observation.top,
            observation.right - observation.left,
            observation.bottom - observation.top,
        )
        .to_image();
        let local = bounds(0, 0, crop.width(), crop.height());
        let groups = isolate_ink(&crop, local, 128, &[128]).unwrap().groups;
        (full.components, groups)
    }

    #[test]
    fn complete_boundary_touch_is_not_truncation() {
        let o = bounds(5, 5, 15, 15);
        let mut image = GrayImage::from_pixel(25, 25, Luma([255]));
        for x in 5..7 {
            for y in 6..14 {
                image.put_pixel(x, y, Luma([0]));
            }
        }
        let (components, groups) = regions_on_both(&image, o);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].touches_line_edge);
        let proof = prove_one(0, &groups[0], o, image.dimensions(), &components).unwrap();
        assert_eq!(proof.component_indices, vec![0]);
        assert_eq!(proof.ink_pixels, 16);
        // Exactly the same ink at the same apparent crop edge, but
        // physically joined to exterior foreground: now not complete.
        image.put_pixel(4, 10, Luma([0]));
        let (components, groups) = regions_on_both(&image, o);
        assert!(prove_one(0, &groups[0], o, image.dimensions(), &components).is_none());
    }

    #[test]
    fn rejects_unobserved_sample_edge_and_partial_component() {
        let o = bounds(0, 5, 10, 15);
        let mut image = GrayImage::from_pixel(25, 25, Luma([255]));
        for y in 6..14 {
            image.put_pixel(0, y, Luma([0]));
        }
        let (components, groups) = regions_on_both(&image, o);
        assert!(prove_one(0, &groups[0], o, image.dimensions(), &components).is_none());

        let o = bounds(5, 5, 15, 15);
        let mut image = GrayImage::from_pixel(25, 25, Luma([255]));
        for y in 6..14 {
            image.put_pixel(5, y, Luma([0]));
        }
        let (mut components, groups) = regions_on_both(&image, o);
        assert!(prove_one(0, &groups[0], o, image.dimensions(), &components).is_some());
        components[0].area_pixels += 1;
        assert!(prove_one(0, &groups[0], o, image.dimensions(), &components).is_none());
        components[0].area_pixels -= 1;
        components[0].observation_support.outside_pixels += 1;
        assert!(prove_one(0, &groups[0], o, image.dimensions(), &components).is_none());
    }
}
