//! Bounded diagnostic decoding of actual CTC class distributions.
//!
//! The greedy path remains the production observation. Prefix search sums
//! distinct retained blank/nonblank paths rather than ranking single frames.
//! Row-level pruning limits new prefixes, not transitions into retained
//! prefixes. Beam pruning still omits probability mass, so weights are lower bounds, not
//! calibrated probabilities of transcription correctness.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    Error,
    ctc_alignment::{SequenceAlignment, align},
};

const ROW_CLASSES: usize = 8;
const BEAM_WIDTH: usize = 12;
const REPORTED_HYPOTHESES: usize = 5;

/// An actual model class at one CTC timestep; class zero is blank.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ClassScore {
    pub class_index: usize,
    pub character: Option<char>,
    pub probability: f32,
}

/// Top classes at the actual first timestep of a collapsed greedy emission.
#[derive(Debug, Serialize)]
pub struct EmissionEvidence {
    pub column: usize,
    pub classes: Vec<ClassScore>,
    /// Report the blank even when it falls outside the top classes.
    pub blank_probability: f32,
}

/// Lower bound on the path mass for one retained class sequence.
#[derive(Debug, Serialize)]
pub struct SequenceHypothesis {
    pub text: String,
    /// Distinct model class sequences can decode to the same Unicode text.
    pub classes: Vec<usize>,
    /// Natural logarithm of retained path probability mass before normalization.
    pub log_mass_lower_bound: f64,
    pub alignment: SequenceAlignment,
}

/// Diagnostic evidence from the same inference as the ordinary greedy output.
#[derive(Debug, Serialize)]
pub struct CtcEvidence {
    pub sequence_length: usize,
    pub vocabulary_size: usize,
    pub row_class_limit: usize,
    pub beam_width: usize,
    pub greedy_text: String,
    pub emissions: Vec<EmissionEvidence>,
    pub hypotheses: Vec<SequenceHypothesis>,
}

/// Preserve an explicit failure rather than silently substituting other crops.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CtcOutcome {
    Measured { evidence: CtcEvidence },
    Unavailable { reason: String },
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

fn stronger(a: ClassScore, b: ClassScore) -> bool {
    a.probability > b.probability
        || (a.probability.to_bits() == b.probability.to_bits() && a.class_index < b.class_index)
}

fn top_classes(row: &[f32], labels: &[char]) -> Result<Vec<ClassScore>, Error> {
    if row.len() != labels.len() || row.len() < 2 {
        return Err(invalid("CTC vocabulary does not match model output"));
    }
    let mut top = Vec::with_capacity(ROW_CLASSES);
    let mut sum = 0.0;
    for (class_index, (&probability, &label)) in row.iter().zip(labels).enumerate() {
        if !(0.0..=1.0).contains(&probability) {
            return Err(invalid("invalid CTC class probability"));
        }
        sum += f64::from(probability);
        let score = ClassScore {
            class_index,
            character: (class_index != 0).then_some(label),
            probability,
        };
        if top.len() == ROW_CLASSES && top.last().is_some_and(|&last| !stronger(score, last)) {
            continue;
        }
        let position = top.partition_point(|&current| stronger(current, score));
        top.insert(position, score);
        top.truncate(ROW_CLASSES);
    }
    if (sum - 1.0).abs() > 0.01 {
        return Err(invalid("CTC output is not a probability distribution"));
    }
    Ok(top)
}

#[derive(Clone, Copy, Debug, Default)]
struct Mass {
    blank: f64,
    nonblank: f64,
}

impl Mass {
    fn total(self) -> f64 {
        self.blank + self.nonblank
    }

    fn extension(self, repeated: bool) -> f64 {
        if repeated { self.blank } else { self.total() }
    }
}

type Prefixes = BTreeMap<Vec<usize>, Mass>;

struct PrefixBeam {
    paths: Prefixes,
    log_scale: f64,
}

impl PrefixBeam {
    fn new() -> Self {
        Self {
            paths: BTreeMap::from([(
                Vec::new(),
                Mass {
                    blank: 1.0,
                    nonblank: 0.0,
                },
            )]),
            log_scale: 0.0,
        }
    }

    fn step(&mut self, row: &[f32], classes: &[ClassScore]) -> Result<(), Error> {
        let mut next = Prefixes::new();
        // Keep every transition whose destination is an already retained
        // prefix, even when its final class is outside the row's top classes.
        for (prefix, &mass) in &self.paths {
            let nonblank = prefix.split_last().map_or(0.0, |(&last, parent)| {
                let extend = self.paths.get(parent).map_or(0.0, |parent_mass| {
                    parent_mass.extension(parent.last() == Some(&last))
                });
                (mass.nonblank + extend) * f64::from(row[last])
            });
            next.insert(
                prefix.clone(),
                Mass {
                    blank: mass.total() * f64::from(row[0]),
                    nonblank,
                },
            );
        }
        // Only the creation of previously unretained prefixes needs the
        // row-class limit. Contributions from distinct parents cannot
        // overlap: removing the final class identifies the unique parent.
        for (prefix, &mass) in &self.paths {
            for class in classes {
                let index = class.class_index;
                if index == 0 || class.probability == 0.0 {
                    continue;
                }
                let extend = mass.extension(prefix.last() == Some(&index));
                if extend > 0.0 {
                    let mut longer = prefix.clone();
                    longer.push(index);
                    next.entry(longer).or_insert_with(|| Mass {
                        blank: 0.0,
                        nonblank: extend * f64::from(class.probability),
                    });
                }
            }
        }
        let mut ranked = next
            .into_iter()
            .filter(|(_, mass)| mass.total() > 0.0)
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|(a, x), (b, y)| {
            y.total().total_cmp(&x.total()).then_with(|| a.cmp(b))
        });
        ranked.truncate(BEAM_WIDTH);
        let scale = ranked
            .first()
            .ok_or_else(|| invalid("empty CTC beam"))?
            .1
            .total();
        self.log_scale += scale.ln();
        self.paths = ranked
            .into_iter()
            .map(|(prefix, mass)| {
                (
                    prefix,
                    Mass {
                        blank: mass.blank / scale,
                        nonblank: mass.nonblank / scale,
                    },
                )
            })
            .collect();
        Ok(())
    }

    fn finish(self, labels: &[char], rows: &[&[f32]]) -> Vec<SequenceHypothesis> {
        let mut ranked = self.paths.into_iter().collect::<Vec<_>>();
        ranked.sort_unstable_by(|(a, x), (b, y)| {
            y.total().total_cmp(&x.total()).then_with(|| a.cmp(b))
        });
        ranked.truncate(REPORTED_HYPOTHESES);
        ranked
            .into_iter()
            .map(|(classes, mass)| SequenceHypothesis {
                text: classes.iter().map(|&index| labels[index]).collect(),
                alignment: align(rows, &classes),
                classes,
                log_mass_lower_bound: self.log_scale + mass.total().ln(),
            })
            .collect()
    }
}

impl CtcEvidence {
    /// Keep the top classes and prefix hypotheses from a probability tensor.
    /// The vocabulary is indexed by class, including blank at index zero.
    /// Verify the independent greedy collapse against the model's actual output.
    ///
    /// # Errors
    /// Rejects malformed distributions, empty tensors, or a disagreement with
    /// the recognizer's greedy text or emission columns.
    pub fn measure<'a>(
        labels: &[char],
        rows: impl IntoIterator<Item = &'a [f32]>,
        expected_text: &str,
        expected_columns: &[usize],
    ) -> Result<Self, Error> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let mut beam = PrefixBeam::new();
        let mut emissions = Vec::new();
        let mut greedy_text = String::new();
        let mut previous = 0;
        let mut sequence_length = 0;
        for (column, &row) in rows.iter().enumerate() {
            let classes = top_classes(row, labels)?;
            let winner = classes.first().ok_or_else(|| invalid("empty CTC column"))?;
            let index = winner.class_index;
            beam.step(row, &classes)?;
            if index != 0 && previous != index {
                greedy_text.push(labels[index]);
                emissions.push(EmissionEvidence {
                    column,
                    classes,
                    blank_probability: row[0],
                });
            }
            previous = index;
            sequence_length = column + 1;
        }
        if sequence_length == 0
            || greedy_text != expected_text
            || emissions
                .iter()
                .map(|emission| emission.column)
                .ne(expected_columns.iter().copied())
        {
            return Err(invalid("CTC evidence disagrees with primary greedy decode"));
        }
        Ok(Self {
            sequence_length,
            vocabulary_size: labels.len(),
            row_class_limit: ROW_CLASSES,
            beam_width: BEAM_WIDTH,
            greedy_text,
            emissions,
            hypotheses: beam.finish(labels, &rows),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beam_sums_paths_and_preserves_repeated_labels() -> Result<(), Error> {
        let labels = ['\0', 'A', 'B'];
        let rows = [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.1, 0.7, 0.2]];
        let evidence =
            CtcEvidence::measure(&labels, rows.iter().map(<[_; 3]>::as_slice), "AA", &[0, 2])?;
        assert_eq!(evidence.emissions.len(), 2);
        assert_eq!(evidence.emissions[0].classes[0].character, Some('A'));
        // Exhaustively enumerate all 27 paths, with no beam pruning for this short example.
        let mut mass = BTreeMap::<Vec<usize>, f64>::new();
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let mut collapsed = Vec::new();
                    let mut previous = 0;
                    for index in [a, b, c] {
                        if index != 0 && index != previous {
                            collapsed.push(index);
                        }
                        previous = index;
                    }
                    let retained = mass.entry(collapsed).or_default();
                    *retained = (f64::from(rows[0][a]) * f64::from(rows[1][b]))
                        .mul_add(f64::from(rows[2][c]), *retained);
                }
            }
        }
        for entry in &evidence.hypotheses {
            assert!((entry.log_mass_lower_bound.exp() - mass[&entry.classes]).abs() < 1e-12);
        }
        assert!(evidence.hypotheses.iter().any(|h| h.text == "AA"));
        Ok(())
    }

    #[test]
    fn row_pruning_preserves_all_transitions_into_retained_prefixes() -> Result<(), Error> {
        let labels = ['\0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];
        let first = [0.2, 0.8, 0., 0., 0., 0., 0., 0., 0., 0., 0.];
        let second = [
            0.2, 0.008, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088,
        ];
        let classes = top_classes(&second, &labels)?;
        assert!(classes.iter().all(|class| class.class_index != 1));

        let evidence =
            CtcEvidence::measure(&labels, [first.as_slice(), second.as_slice()], "A", &[0])?;
        let candidate = evidence
            .hypotheses
            .iter()
            .find(|candidate| candidate.text == "A")
            .ok_or_else(|| invalid("retained A candidate missing"))?;
        let expected = f64::from(first[1]).mul_add(
            f64::from(second[0]),
            (f64::from(first[1]) + f64::from(first[0])) * f64::from(second[1]),
        );
        assert!((candidate.log_mass_lower_bound.exp() - expected).abs() < 1e-12);
        assert!(
            matches!(candidate.alignment, SequenceAlignment::Measured { log_mass, .. }
            if (log_mass.exp() - expected).abs() < 1e-12)
        );
        Ok(())
    }

    #[test]
    fn retained_repeat_only_extends_from_parent_blank_mass() -> Result<(), Error> {
        let labels = ['\0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];
        let row = [
            0.2, 0.008, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088, 0.088,
        ];
        let mut beam = PrefixBeam {
            paths: BTreeMap::from([
                (
                    vec![1],
                    Mass {
                        blank: 0.2,
                        nonblank: 0.5,
                    },
                ),
                (
                    vec![1, 1],
                    Mass {
                        blank: 0.1,
                        nonblank: 0.2,
                    },
                ),
            ]),
            log_scale: 0.0,
        };
        beam.step(&row, &top_classes(&row, &labels)?)?;
        let mass = beam
            .paths
            .get(&[1, 1][..])
            .ok_or_else(|| invalid("retained AA missing"))?;
        let scale = beam.log_scale.exp();
        let expected_blank = 0.3 * f64::from(row[0]);
        let expected_nonblank = 0.4 * f64::from(row[1]);
        assert!(mass.blank.mul_add(scale, -expected_blank).abs() < 1e-12);
        assert!(mass.nonblank.mul_add(scale, -expected_nonblank).abs() < 1e-12);
        assert!(!beam.paths.contains_key(&[1, 1, 1][..]));
        Ok(())
    }

    #[test]
    fn greedy_is_not_necessarily_best_collapsed_sequence() -> Result<(), Error> {
        let labels = ['\0', 'X'];
        let rows = [[0.6, 0.4], [0.6, 0.4]];
        let evidence = CtcEvidence::measure(&labels, rows.iter().map(<[_; 2]>::as_slice), "", &[])?;
        assert_eq!(evidence.hypotheses[0].text, "X");
        assert!((evidence.hypotheses[0].log_mass_lower_bound.exp() - 0.64).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn malformed_or_mismatched_distributions_are_not_evidence() {
        let labels = ['\0', 'A'];
        for row in [[f32::NAN, 0.5], [0.7, 0.7], [0.2, -0.1]] {
            assert!(CtcEvidence::measure(&labels, [row.as_slice()], "A", &[0]).is_err());
        }
        assert!(CtcEvidence::measure(&labels, [&[0.2, 0.8][..]], "B", &[0]).is_err());
        assert!(CtcEvidence::measure(&labels, [&[0.2, 0.8][..]], "A", &[1]).is_err());
    }

    #[test]
    fn scaled_beam_mass_survives_a_long_sequence() -> Result<(), Error> {
        let labels = ['\0', 'A'];
        let rows = [[0.5, 0.5]; 400];
        let evidence = CtcEvidence::measure(&labels, rows.iter().map(<[_; 2]>::as_slice), "", &[])?;
        assert!(evidence.hypotheses.iter().all(
            |entry| entry.log_mass_lower_bound.is_finite() && entry.log_mass_lower_bound <= 0.0
        ));
        Ok(())
    }
}
