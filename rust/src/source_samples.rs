//! Bit-exact PDF sample planes for source-local OCR counterfactuals.
//!
//! Samples are packed most-significant bit first, with each scanline padded to
//! a whole byte. Never convert source samples through RGB: indexed palette
//! entries, inverted decode arrays and ICC-based components must retain their
//! original numeric meaning in the unchanged image dictionary.

/// One native-image sample buffer with up to 32 interleaved color components.
#[derive(Clone)]
pub struct PackedSamples {
    pub width: usize,
    pub height: usize,
    pub components: usize,
    bpc: usize,
    row_bytes: usize,
    pub data: Vec<u8>,
}

impl PackedSamples {
    /// Validate the complete byte layout, including per-row bit padding.
    pub fn new(width: u32, height: u32, components: usize, bpc: u8, data: Vec<u8>) -> Option<Self> {
        if !matches!(bpc, 1 | 2 | 4 | 8 | 16) || !(1..=32).contains(&components) {
            return None;
        }
        let width = usize::try_from(width).ok()?;
        let height = usize::try_from(height).ok()?;
        let bpc = usize::from(bpc);
        let row_bytes = width
            .checked_mul(components)?
            .checked_mul(bpc)?
            .checked_add(7)?
            / 8;
        if width == 0 || height == 0 || row_bytes.checked_mul(height)? != data.len() {
            return None;
        }
        Some(Self {
            width,
            height,
            components,
            bpc,
            row_bytes,
            data,
        })
    }

    #[inline]
    const fn start_bit(&self, x: usize, y: usize, channel: usize) -> usize {
        y * self.row_bytes * 8 + (x * self.components + channel) * self.bpc
    }

    /// Read a PDF component value without truncating 16-bit samples.
    pub fn get(&self, x: usize, y: usize, channel: usize) -> u16 {
        let start = self.start_bit(x, y, channel);
        let mut value = 0_u16;
        for offset in 0..self.bpc {
            let index = start + offset;
            let bit = (self.data[index / 8] >> (7 - index % 8)) & 1;
            value = (value << 1) | u16::from(bit);
        }
        value
    }

    /// Replace only a sample's exact bits, leaving neighbors and row padding.
    pub fn set(&mut self, x: usize, y: usize, channel: usize, value: u16) {
        let start = self.start_bit(x, y, channel);
        for offset in 0..self.bpc {
            let index = start + offset;
            let bit = (value >> (self.bpc - offset - 1)) & 1;
            let mask = 1_u8 << (7 - index % 8);
            if bit == 0 {
                self.data[index / 8] &= !mask;
            } else {
                self.data[index / 8] |= mask;
            }
        }
    }

    /// Compare channel changes using the *same relative 8/255 threshold*
    /// across packed 1/2/4/8-bit and wide 16-bit component depths.
    pub fn pixel_changed(&self, other: &Self, x: usize, y: usize) -> bool {
        let maximum = (1_u32 << self.bpc) - 1;
        (0..self.components).any(|channel| {
            let left = self.get(x, y, channel);
            let right = other.get(x, y, channel);
            u32::from(left.abs_diff(right)) * 255 > 8 * maximum
        })
    }
}

#[cfg(test)]
mod tests {
    use super::PackedSamples;

    #[test]
    fn packed_samples_preserve_neighbor_channels_and_row_padding() -> Result<(), crate::Error> {
        for depth in [1_u8, 2, 4, 8, 16] {
            for channels in [1, 3, 4, 5] {
                let bits_per_pixel = usize::from(depth) * channels;
                let row_bytes = (7 * bits_per_pixel).div_ceil(8);
                let initial = vec![0b1010_0101_u8; row_bytes * 3];
                let mut image = PackedSamples::new(7, 3, channels, depth, initial.clone())
                    .ok_or_else(|| crate::Error::Invalid("valid bit layout rejected".into()))?;
                let before = image.clone();
                let changed = u16::MAX >> (16 - depth);
                image.set(4, 1, channels - 1, changed);
                assert_eq!(image.get(4, 1, channels - 1), changed);
                for y in 0..3 {
                    for x in 0..7 {
                        for channel in 0..channels {
                            if (x, y, channel) != (4, 1, channels - 1) {
                                assert_eq!(image.get(x, y, channel), before.get(x, y, channel),);
                            }
                        }
                    }
                }
                if bits_per_pixel * 7 % 8 != 0 {
                    let last_byte = row_bytes + row_bytes - 1;
                    let pad = 8 - (bits_per_pixel * 7 % 8);
                    assert_eq!(
                        image.data[last_byte] & ((1 << pad) - 1),
                        initial[last_byte] & ((1 << pad) - 1),
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_incomplete_or_overlong_rows_and_unsupported_depths() {
        assert!(PackedSamples::new(10, 10, 3, 8, vec![0; 300]).is_some());
        assert!(PackedSamples::new(10, 10, 3, 8, vec![0; 299]).is_none());
        assert!(PackedSamples::new(10, 10, 3, 8, vec![0; 301]).is_none());
        assert!(PackedSamples::new(10, 10, 3, 12, vec![0; 450]).is_none());
        assert!(PackedSamples::new(u32::MAX, u32::MAX, 32, 16, vec![]).is_none());
    }
}
