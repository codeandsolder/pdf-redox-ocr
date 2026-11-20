//! Linear-space Unicode Levenshtein alignment for OCR confusion auditing.
//! Hirschberg division keeps memory proportional to the shorter input length.

/// A reference character and aligned OCR prediction; None is an alignment gap.
pub(super) type Aligned = (Option<char>, Option<char>);

fn last_row(reference: &[char], predicted: &[char]) -> Vec<usize> {
    let mut prev = (0..=predicted.len()).collect::<Vec<_>>();
    let mut curr = vec![0; predicted.len() + 1];
    for (i, &expected) in reference.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &observed) in predicted.iter().enumerate() {
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + usize::from(expected != observed));
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev
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

fn align_into(reference: &[char], predicted: &[char], result: &mut Vec<Aligned>) {
    if reference.is_empty() {
        result.extend(predicted.iter().copied().map(|c| (None, Some(c))));
    } else if predicted.is_empty() {
        result.extend(reference.iter().copied().map(|c| (Some(c), None)));
    } else if reference.len() == 1 || predicted.len() == 1 {
        align_one(reference, predicted, result);
    } else {
        let midpoint = reference.len() / 2;
        let forward = last_row(&reference[..midpoint], predicted);
        let reversed_reference = reference[midpoint..]
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let reversed_predicted = predicted.iter().rev().copied().collect::<Vec<_>>();
        let backward = last_row(&reversed_reference, &reversed_predicted);
        // Deterministic tie-break: earliest OCR prediction position.
        let split = (0..=predicted.len())
            .min_by_key(|&index| (forward[index] + backward[predicted.len() - index], index))
            .unwrap_or(0);
        align_into(&reference[..midpoint], &predicted[..split], result);
        align_into(&reference[midpoint..], &predicted[split..], result);
    }
}

/// Minimal Unicode edit alignment with linear auxiliary memory.
pub(super) fn alignment(reference: &str, predicted: &str) -> Vec<Aligned> {
    let reference = reference.chars().collect::<Vec<_>>();
    let predicted = predicted.chars().collect::<Vec<_>>();
    let mut out = Vec::with_capacity(reference.len().max(predicted.len()));
    if reference.len() >= predicted.len() {
        align_into(&reference, &predicted, &mut out);
    } else {
        align_into(&predicted, &reference, &mut out);
        for (before, after) in &mut out {
            std::mem::swap(before, after);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::alignment;
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
    fn long_lines_have_linear_auxiliary_memory_and_all_edits() {
        let reference = "温".repeat(2000);
        let predicted = format!("{}0{}", "温".repeat(1000), "温".repeat(1000));
        let aligned = alignment(&reference, &predicted);
        assert_eq!(aligned.iter().filter(|(a, b)| a != b).count(), 1);
        assert_eq!(aligned.len(), 2001);
    }
}
