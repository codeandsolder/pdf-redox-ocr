//! Linear-space Unicode Levenshtein alignment for OCR confusion auditing.
//! Hirschberg division keeps memory proportional to the shorter input length.

/// A reference character and aligned OCR prediction; None is an alignment gap.
pub(super) type Aligned = (Option<char>, Option<char>);

fn fill_last_row<'a>(
    reference: impl Iterator<Item = &'a char>,
    predicted: impl ExactSizeIterator<Item = &'a char> + Clone,
    row: &mut Vec<usize>,
    scratch: &mut Vec<usize>,
) {
    row.clear();
    row.extend(0..=predicted.len());
    scratch.resize(row.len(), 0);
    for (i, &expected) in reference.enumerate() {
        scratch[0] = i + 1;
        for (j, &observed) in predicted.clone().enumerate() {
            scratch[j + 1] = (row[j + 1] + 1)
                .min(scratch[j] + 1)
                .min(row[j] + usize::from(expected != observed));
        }
        std::mem::swap(row, scratch);
    }
}

pub(super) fn last_row(reference: &[char], predicted: &[char]) -> Vec<usize> {
    let mut row = Vec::with_capacity(predicted.len() + 1);
    let mut scratch = Vec::with_capacity(row.capacity());
    fill_last_row(reference.iter(), predicted.iter(), &mut row, &mut scratch);
    row
}

/// Three reusable DP rows sized by the shorter original input. Descendants
/// reuse the same storage after the parent has selected its split.
struct Rows {
    forward: Vec<usize>,
    backward: Vec<usize>,
    scratch: Vec<usize>,
}

impl Rows {
    fn new(shorter_len: usize) -> Self {
        let capacity = shorter_len + 1;
        Self {
            forward: Vec::with_capacity(capacity),
            backward: Vec::with_capacity(capacity),
            scratch: Vec::with_capacity(capacity),
        }
    }

    fn split(&mut self, reference: &[char], predicted: &[char], midpoint: usize) -> usize {
        fill_last_row(
            reference[..midpoint].iter(),
            predicted.iter(),
            &mut self.forward,
            &mut self.scratch,
        );
        fill_last_row(
            reference[midpoint..].iter().rev(),
            predicted.iter().rev(),
            &mut self.backward,
            &mut self.scratch,
        );
        // Preserve the original deterministic earliest-column tie-break.
        (0..=predicted.len())
            .min_by_key(|&index| {
                (
                    self.forward[index] + self.backward[predicted.len() - index],
                    index,
                )
            })
            .unwrap_or(0)
    }
}

fn align_one(reference: &[char], predicted: &[char], result: &mut Vec<Aligned>) {
    debug_assert!(reference.len() == 1 || predicted.len() == 1);
    if reference.len() == 1 {
        let expected = reference[0];
        if let Some(position) = predicted.iter().position(|&value| value == expected) {
            result.extend(
                predicted[..position]
                    .iter()
                    .copied()
                    .map(|c| (None, Some(c))),
            );
            result.push((Some(expected), Some(expected)));
            result.extend(
                predicted[position + 1..]
                    .iter()
                    .copied()
                    .map(|c| (None, Some(c))),
            );
        } else {
            result.push((Some(expected), Some(predicted[0])));
            result.extend(predicted[1..].iter().copied().map(|c| (None, Some(c))));
        }
    } else {
        let observed = predicted[0];
        if let Some(position) = reference.iter().position(|&value| value == observed) {
            result.extend(
                reference[..position]
                    .iter()
                    .copied()
                    .map(|c| (Some(c), None)),
            );
            result.push((Some(observed), Some(observed)));
            result.extend(
                reference[position + 1..]
                    .iter()
                    .copied()
                    .map(|c| (Some(c), None)),
            );
        } else {
            result.push((Some(reference[0]), Some(observed)));
            result.extend(reference[1..].iter().copied().map(|c| (Some(c), None)));
        }
    }
}

fn align_into(reference: &[char], predicted: &[char], result: &mut Vec<Aligned>, rows: &mut Rows) {
    if reference.is_empty() {
        result.extend(predicted.iter().copied().map(|c| (None, Some(c))));
    } else if predicted.is_empty() {
        result.extend(reference.iter().copied().map(|c| (Some(c), None)));
    } else if reference.len() == 1 || predicted.len() == 1 {
        align_one(reference, predicted, result);
    } else {
        let midpoint = reference.len() / 2;
        let split = rows.split(reference, predicted, midpoint);
        align_into(&reference[..midpoint], &predicted[..split], result, rows);
        align_into(&reference[midpoint..], &predicted[split..], result, rows);
    }
}

/// Minimal Unicode edit alignment with linear auxiliary memory.
pub(super) fn alignment(reference: &str, predicted: &str) -> Vec<Aligned> {
    let reference = reference.chars().collect::<Vec<_>>();
    let predicted = predicted.chars().collect::<Vec<_>>();
    let mut out = Vec::with_capacity(reference.len().max(predicted.len()));
    let mut rows = Rows::new(reference.len().min(predicted.len()));
    if reference.len() >= predicted.len() {
        align_into(&reference, &predicted, &mut out, &mut rows);
    } else {
        align_into(&predicted, &reference, &mut out, &mut rows);
        for (before, after) in &mut out {
            std::mem::swap(before, after);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Rows, align_into, alignment, last_row};
    use crate::quality_eval::character_edits;

    fn corpus(alphabet: &[char], max_len: usize) -> Vec<String> {
        let mut values = vec![String::new()];
        let mut layer = vec![String::new()];
        for _ in 0..max_len {
            layer = layer
                .iter()
                .flat_map(|prefix| {
                    alphabet.iter().map(move |c| {
                        let mut text = prefix.clone();
                        text.push(*c);
                        text
                    })
                })
                .collect();
            values.extend(layer.iter().cloned());
        }
        values
    }

    #[test]
    fn alignment_is_minimal_and_roundtrips_all_short_unicode_strings() {
        let values = corpus(&['Ø', '0', '温'], 4);
        for reference in &values {
            for prediction in &values {
                let aligned = alignment(reference, prediction);
                let edits = aligned.iter().filter(|(a, b)| a != b).count();
                assert_eq!(
                    edits,
                    character_edits(reference, prediction),
                    "reference {reference:?}, OCR {prediction:?}"
                );
                let original = aligned.iter().filter_map(|(a, _)| *a).collect::<String>();
                let observed = aligned.iter().filter_map(|(_, b)| *b).collect::<String>();
                assert_eq!(&original, reference);
                assert_eq!(&observed, prediction);
            }
        }
    }

    #[test]
    fn splits_match_independently_reversed_rows_and_reuse_fixed_storage() {
        let values = corpus(&['Ø', '0', '温'], 3);
        let mut rows = Rows::new(3);
        for reference in &values {
            let reference = reference.chars().collect::<Vec<_>>();
            for prediction in &values {
                let prediction = prediction.chars().collect::<Vec<_>>();
                let midpoint = reference.len() / 2;
                let forward = last_row(&reference[..midpoint], &prediction);
                let reversed = reference[midpoint..]
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<_>>();
                let reversed_prediction = prediction.iter().rev().copied().collect::<Vec<_>>();
                let backward = last_row(&reversed, &reversed_prediction);
                let expected = (0..=prediction.len())
                    .min_by_key(|&i| (forward[i] + backward[prediction.len() - i], i))
                    .unwrap_or(0);
                assert_eq!(rows.split(&reference, &prediction, midpoint), expected);
            }
        }
        let reference = vec!['温'; 8192];
        let prediction = ['0', '0', '0'];
        let capacities = [
            rows.forward.capacity(),
            rows.backward.capacity(),
            rows.scratch.capacity(),
        ];
        let mut output = Vec::new();
        align_into(&reference, &prediction, &mut output, &mut rows);
        assert_eq!(
            [
                rows.forward.capacity(),
                rows.backward.capacity(),
                rows.scratch.capacity()
            ],
            capacities
        );
        assert_eq!(output.iter().filter(|(a, b)| a != b).count(), 8192);
    }

    #[test]
    fn long_lines_have_linear_auxiliary_memory_and_all_edits() {
        let reference = "温".repeat(2000);
        let predicted = format!("{}0{}", "温".repeat(1000), "温".repeat(1000));
        let aligned = alignment(&reference, &predicted);
        assert_eq!(aligned.iter().filter(|(a, b)| a != b).count(), 1);
        assert_eq!(aligned.len(), 2001);
    }
}
