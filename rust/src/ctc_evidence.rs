//! Bounded diagnostic decoding of actual CTC class distributions.
//!
//! The greedy path remains the production observation. Prefix search sums
//! distinct retained blank/nonblank paths rather than ranking single frames.
//! Pruning omits probability mass, so beam weights are lower bounds, not
//! calibrated probabilities of transcription correctness.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::Error;

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

    fn step(&mut self, blank: f32, classes: &[ClassScore]) -> Result<(), Error> {
        let mut next = Prefixes::new();
        for (prefix, mass) in std::mem::take(&mut self.paths) {
            let retained = &mut next.entry(prefix.clone()).or_default().blank;
            *retained = mass.total().mul_add(f64::from(blank), *retained);
            for class in classes {
                let index = class.class_index;
                if index == 0 || class.probability == 0.0 {
                    continue;
                }
                let probability = f64::from(class.probability);
                // A repeated nonblank class collapses unless preceded by blank.
                let extend = if prefix.last() == Some(&index) {
                    let retained = &mut next.entry(prefix.clone()).or_default().nonblank;
                    *retained = mass.nonblank.mul_add(probability, *retained);
                    mass.blank
                } else {
                    mass.total()
                };
                if extend > 0.0 {
                    let mut longer = prefix.clone();
                    longer.push(index);
                    let retained = &mut next.entry(longer).or_default().nonblank;
                    *retained = extend.mul_add(probability, *retained);
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

    fn finish(self, labels: &[char]) -> Vec<SequenceHypothesis> {
        let mut ranked = self.paths.into_iter().collect::<Vec<_>>();
        ranked.sort_unstable_by(|(a, x), (b, y)| {
            y.total().total_cmp(&x.total()).then_with(|| a.cmp(b))
        });
        ranked.truncate(REPORTED_HYPOTHESES);
        ranked
            .into_iter()
            .map(|(classes, mass)| SequenceHypothesis {
                text: classes.iter().map(|&index| labels[index]).collect(),
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
        let mut beam = PrefixBeam::new();
        let mut emissions = Vec::new();
        let mut greedy_text = String::new();
        let mut previous = 0;
        let mut sequence_length = 0;
        for (column, row) in rows.into_iter().enumerate() {
            let classes = top_classes(row, labels)?;
            let winner = classes.first().ok_or_else(|| invalid("empty CTC column"))?;
            let index = winner.class_index;
            beam.step(row[0], &classes)?;
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
            hypotheses: beam.finish(labels),
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
