//! Bounded diagnostic decoding of actual CTC class distributions.
//!
//! The greedy path remains the production observation. Prefix search sums
//! distinct retained blank/nonblank paths rather than ranking single frames.
//! Every beam step retains the strongest successors of the previous beam,
//! without an additional vocabulary-pruning approximation. Earlier beam
//! pruning still omits probability mass, so weights are lower bounds, not
//! calibrated probabilities of transcription correctness.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    Error,
    ctc_alignment::{SequenceAlignment, align},
};

const EMISSION_CLASSES: usize = 8;
const BEAM_WIDTH: usize = 12;
// One blank and one repeated last class can have different transition
// weights. All remaining classes share the same parent mass.
const SEARCH_CLASSES: usize = BEAM_WIDTH + 2;
const REPORTED_HYPOTHESES: usize = 5;

/// An actual model class at one CTC timestep; class zero is blank.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassScore {
    pub class_index: usize,
    pub character: Option<char>,
    pub probability: f32,
}

/// Top classes at the actual first timestep of a collapsed greedy emission.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmissionEvidence {
    pub column: usize,
    pub classes: Vec<ClassScore>,
    /// Report the blank even when it falls outside the top classes.
    pub blank_probability: f32,
}

/// Lower bound on the path mass for one retained class sequence.
#[derive(Debug, Serialize, Deserialize)]
pub struct SequenceHypothesis {
    pub text: String,
    /// Distinct model class sequences can decode to the same Unicode text.
    pub classes: Vec<usize>,
    /// Natural logarithm of retained path probability mass before normalization.
    pub log_mass_lower_bound: f64,
    pub alignment: SequenceAlignment,
}

/// An explicit sequence query on the complete original class distribution.
/// No beam rank or retained-path mass is implied.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExactSequence {
    pub text: String,
    pub classes: Vec<usize>,
    pub alignment: SequenceAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SequenceReference {
    Beam { index: usize },
    Query { index: usize },
}

#[derive(Clone, Copy)]
pub(crate) struct SequenceData<'a> {
    pub text: &'a str,
    pub classes: &'a [usize],
    pub alignment: &'a SequenceAlignment,
}

#[cfg(any(feature = "native-inference", test))]
impl SequenceData<'_> {
    pub(crate) fn log_mass(self) -> Option<f64> {
        let SequenceAlignment::Measured { log_mass, .. } = *self.alignment else {
            return None;
        };
        log_mass.is_finite().then_some(log_mass)
    }
}

impl SequenceHypothesis {
    pub(crate) fn data(&self) -> SequenceData<'_> {
        SequenceData {
            text: &self.text,
            classes: &self.classes,
            alignment: &self.alignment,
        }
    }
}

impl ExactSequence {
    pub(crate) fn data(&self) -> SequenceData<'_> {
        SequenceData {
            text: &self.text,
            classes: &self.classes,
            alignment: &self.alignment,
        }
    }
}

/// Diagnostic evidence from the same inference as the ordinary greedy output.
#[derive(Debug, Serialize, Deserialize)]
pub struct CtcEvidence {
    pub sequence_length: usize,
    pub vocabulary_size: usize,
    /// Maximum number of classes listed per greedy emission, not a beam cutoff.
    pub row_class_limit: usize,
    pub beam_width: usize,
    pub greedy_text: String,
    pub emissions: Vec<EmissionEvidence>,
    pub hypotheses: Vec<SequenceHypothesis>,
    /// Exact conditional scores of requested sequences absent from the beam.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub queries: Vec<ExactSequence>,
}

/// Preserve an explicit failure rather than silently substituting other crops.
#[derive(Debug, Serialize, Deserialize)]
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
    let mut top = Vec::with_capacity(SEARCH_CLASSES);
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
        if top.len() == SEARCH_CLASSES && top.last().is_some_and(|&last| !stronger(score, last)) {
            continue;
        }
        if top.len() == SEARCH_CLASSES {
            top.pop();
        }
        let position = top.partition_point(|&current| stronger(current, score));
        top.insert(position, score);
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

    // Select by the actual rounded product when probability order alone
    // cannot establish the lexical tie break. Keep only BEAM_WIDTH entries
    // even if the vocabulary has thousands of tied subnormal products.
    fn strongest_rounded_extensions(self, row: &[f32], last: Option<usize>) -> Vec<(usize, f64)> {
        let mut top: Vec<(usize, f64)> = Vec::with_capacity(BEAM_WIDTH);
        for (index, &probability) in row.iter().enumerate().skip(1) {
            let weight = self.extension(last == Some(index)) * f64::from(probability);
            if weight == 0.0 {
                continue;
            }
            // Visit classes in ascending index order. Existing ties precede
            // this class, so insertion after equal weights is deterministic.
            let position = top.partition_point(|&(_, existing)| existing >= weight);
            if position >= BEAM_WIDTH {
                continue;
            }
            if top.len() == BEAM_WIDTH {
                top.pop();
            }
            top.insert(position, (index, weight));
        }
        top
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
        // An unretained child has exactly one parent. Of the strongest
        // BEAM_WIDTH + 2 classes, at least BEAM_WIDTH are ordinary extensions
        // (neither blank nor repeat). An omitted child therefore has at
        // least BEAM_WIDTH siblings ahead of it, including the deterministic
        // class-index tie break. Existing children only gain mass.
        for (prefix, &mass) in &self.paths {
            let mut add = |index, weight| {
                if weight > 0.0 {
                    let mut longer = prefix.clone();
                    longer.push(index);
                    next.entry(longer).or_insert(Mass {
                        blank: 0.0,
                        nonblank: weight,
                    });
                }
            };
            // Distinct positive f32 probabilities multiplied by a common
            // f64 mass remain distinguishable in the normal range. For
            // subnormal products, rounding can introduce new ties; enumerate
            // the full row rather than break those ties by probability.
            let subnormal_cutoff = classes.len() == SEARCH_CLASSES
                && classes.last().is_some_and(|class| {
                    class.probability > 0.0
                        && mass.total() * f64::from(class.probability) < f64::MIN_POSITIVE
                });
            if subnormal_cutoff {
                for (index, weight) in
                    mass.strongest_rounded_extensions(row, prefix.last().copied())
                {
                    add(index, weight);
                }
            } else {
                for class in classes.iter().filter(|class| class.class_index != 0) {
                    let index = class.class_index;
                    add(
                        index,
                        mass.extension(prefix.last() == Some(&index))
                            * f64::from(class.probability),
                    );
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
    #[cfg(any(feature = "native-inference", test))]
    pub(crate) fn sequences(&self) -> impl Iterator<Item = (SequenceReference, SequenceData<'_>)> {
        self.hypotheses
            .iter()
            .enumerate()
            .map(|(index, h)| (SequenceReference::Beam { index }, h.data()))
            .chain(
                self.queries
                    .iter()
                    .enumerate()
                    .map(|(index, h)| (SequenceReference::Query { index }, h.data())),
            )
    }

    #[cfg(any(feature = "native-inference", test))]
    pub(crate) fn sequence(&self, reference: SequenceReference) -> Option<SequenceData<'_>> {
        Some(match reference {
            SequenceReference::Beam { index } => self.hypotheses.get(index)?.data(),
            SequenceReference::Query { index } => self.queries.get(index)?.data(),
        })
    }

    #[cfg(any(feature = "native-inference", test))]
    pub(crate) fn find_sequence(&self, classes: &[usize]) -> Option<SequenceReference> {
        self.sequences()
            .find_map(|(r, h)| (h.classes == classes).then_some(r))
    }

    #[cfg(any(feature = "native-inference", test))]
    pub(crate) fn strongest_matching(
        &self,
        matches: impl Fn(&str) -> bool,
    ) -> Option<SequenceReference> {
        self.sequences()
            .filter_map(|(r, h)| {
                matches(h.text)
                    .then(|| h.log_mass().map(|m| (r, m)))
                    .flatten()
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(r, _)| r)
    }

    // The caller supplies the very same already validated tensor as measure().
    // Queries are independent of the retained beam and cannot change its rank.
    #[cfg(any(feature = "native-inference", test))]
    fn query(&mut self, labels: &[char], rows: &[&[f32]], classes: &[usize]) {
        if self.find_sequence(classes).is_some()
            || rows.len() != self.sequence_length
            || labels.len() != self.vocabulary_size
            || classes.iter().any(|&c| c == 0 || c >= labels.len())
        {
            return;
        }
        self.queries.push(ExactSequence {
            text: classes.iter().map(|&c| labels[c]).collect(),
            classes: classes.to_vec(),
            alignment: align(rows, classes),
        });
    }

    /// Score the same actual primary and revised sequences on both sides of
    /// a counterfactual, even when one spelling falls outside the beam.
    /// The probability slices are borrowed from each original inference.
    #[cfg(any(feature = "native-inference", test))]
    pub(crate) fn pair_queries(
        labels: &[char],
        before: &mut Self,
        before_rows: &[&[f32]],
        after: &mut Self,
        after_rows: &[&[f32]],
        primary: &str,
    ) {
        let key = crate::refinement_evidence::canonical(primary);
        let matches = |text: &str| crate::refinement_evidence::canonical(text) == key;
        let Some(winner) = after.hypotheses.first() else {
            return;
        };
        if matches(&winner.text) {
            return;
        }
        let revised = winner.classes.clone();
        if before.strongest_matching(matches).is_none() {
            // Preserve the actual greedy sequence when it represents the
            // primary observation; otherwise use a uniquely encodable literal.
            let primary_classes = if matches(&before.greedy_text) {
                Some(
                    before
                        .emissions
                        .iter()
                        .filter_map(|e| e.classes.first().map(|c| c.class_index))
                        .collect::<Vec<_>>(),
                )
            } else {
                primary
                    .chars()
                    .map(|ch| {
                        let mut indices = labels
                            .iter()
                            .enumerate()
                            .skip(1)
                            .filter_map(|(i, &c)| (c == ch).then_some(i));
                        let index = indices.next()?;
                        indices.next().is_none().then_some(index)
                    })
                    .collect()
            };
            if let Some(classes) = primary_classes {
                before.query(labels, before_rows, &classes);
            }
        }
        let Some(primary_classes) = before
            .strongest_matching(matches)
            .and_then(|r| before.sequence(r))
            .map(|h| h.classes.to_vec())
        else {
            return;
        };
        before.query(labels, before_rows, &revised);
        after.query(labels, after_rows, &primary_classes);
    }

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
            let mut classes = top_classes(row, labels)?;
            let winner = classes.first().ok_or_else(|| invalid("empty CTC column"))?;
            let index = winner.class_index;
            beam.step(row, &classes)?;
            if index != 0 && previous != index {
                greedy_text.push(labels[index]);
                classes.truncate(EMISSION_CLASSES);
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
            row_class_limit: EMISSION_CLASSES,
            beam_width: BEAM_WIDTH,
            greedy_text,
            emissions,
            hypotheses: beam.finish(labels, &rows),
            queries: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_exact_queries_do_not_depend_on_retained_beam()
    -> Result<(), Box<dyn std::error::Error>> {
        let labels = ['\0', 'A', 'B', 'C', 'D', 'E', 'F', 'G'];
        let before_row = [0.01, 0.90, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01];
        let after_row = [0.01, 0.0001, 0.90, 0.025, 0.025, 0.02, 0.01, 0.0099];
        let before_rows = [before_row.as_slice()];
        let after_rows = [after_row.as_slice()];
        let mut before = CtcEvidence::measure(&labels, before_rows, "A", &[0])?;
        let mut after = CtcEvidence::measure(&labels, after_rows, "B", &[0])?;
        assert!(after.find_sequence(&[1]).is_none());
        let rank = serde_json::to_value(&after.hypotheses)?;
        CtcEvidence::pair_queries(
            &labels,
            &mut before,
            &before_rows,
            &mut after,
            &after_rows,
            "A",
        );
        assert_eq!(serde_json::to_value(&after.hypotheses)?, rank);
        let path = after
            .find_sequence(&[1])
            .and_then(|r| after.sequence(r))
            .ok_or_else(|| invalid("missing exact primary query"))?;
        assert_eq!(path.text, "A");
        let probability = path
            .log_mass()
            .ok_or_else(|| invalid("missing exact primary mass"))?
            .exp();
        assert!((probability - f64::from(after_row[1])).abs() < 1e-12);
        assert!(matches!(
            after.find_sequence(&[1]),
            Some(SequenceReference::Query { .. })
        ));
        Ok(())
    }

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
        let labels = [
            '\0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q',
        ];
        let mut first = [0.; 18];
        first[0] = 0.2;
        first[1] = 0.8;
        let mut second = [0.0495; 18];
        second[0] = 0.2;
        second[1] = 0.008;
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
        let labels = [
            '\0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q',
        ];
        let mut row = [0.0495; 18];
        row[0] = 0.2;
        row[1] = 0.008;
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
                        nonblank: 0.8,
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
        let expected_blank = 0.9 * f64::from(row[0]);
        let expected_nonblank = f64::from(row[1]);
        assert!(mass.blank.mul_add(scale, -expected_blank).abs() < 1e-12);
        assert!(mass.nonblank.mul_add(scale, -expected_nonblank).abs() < 1e-12);
        assert!(!beam.paths.contains_key(&[1, 1, 1][..]));
        Ok(())
    }

    fn full_step(paths: &Prefixes, row: &[f32]) -> Result<(Prefixes, f64), Error> {
        let mut all = std::collections::BTreeSet::new();
        for prefix in paths.keys() {
            all.insert(prefix.clone());
            for index in 1..row.len() {
                let mut child = prefix.clone();
                child.push(index);
                all.insert(child);
            }
        }
        let mut ranked = all
            .into_iter()
            .filter_map(|prefix| {
                let prior = paths.get(&prefix).copied().unwrap_or_default();
                let blank = (prior.blank + prior.nonblank) * f64::from(row[0]);
                let nonblank = prefix.split_last().map_or(0.0, |(&last, parent)| {
                    let parent_mass = paths.get(parent).copied().unwrap_or_default();
                    let extend = if parent.last() == Some(&last) {
                        parent_mass.blank
                    } else {
                        parent_mass.blank + parent_mass.nonblank
                    };
                    (prior.nonblank + extend) * f64::from(row[last])
                });
                let mass = Mass { blank, nonblank };
                (mass.total() > 0.0).then_some((prefix, mass))
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|(a, x), (b, y)| {
            y.total().total_cmp(&x.total()).then_with(|| a.cmp(b))
        });
        ranked.truncate(BEAM_WIDTH);
        let scale = ranked
            .first()
            .ok_or_else(|| invalid("empty reference beam"))?
            .1
            .total();
        let normalized = ranked
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
        Ok((normalized, scale.ln()))
    }

    fn compare_complete_step(
        beam: &mut PrefixBeam,
        row: &[f32],
        labels: &[char],
    ) -> Result<(), Error> {
        let (expected, log_scale) = full_step(&beam.paths, row)?;
        let expected_log = beam.log_scale + log_scale;
        beam.step(row, &top_classes(row, labels)?)?;
        assert!(beam.paths.keys().eq(expected.keys()));
        for (actual, expected) in beam.paths.values().zip(expected.values()) {
            assert_eq!(actual.blank.to_bits(), expected.blank.to_bits());
            assert_eq!(actual.nonblank.to_bits(), expected.nonblank.to_bits());
        }
        assert_eq!(beam.log_scale.to_bits(), expected_log.to_bits());
        Ok(())
    }

    #[test]
    fn bounded_expansion_matches_full_vocabulary_beam() -> Result<(), Error> {
        let labels = ['a'; 33];
        let mut beam = PrefixBeam::new();
        let mut state = 1_u32;
        for step in 0..96 {
            let mut weights = [0_u16; 33];
            for weight in &mut weights {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *weight = u16::try_from((state >> 24) % 31)
                    .map_err(|_| invalid("fixture weight overflow"))?;
            }
            if step % 3 == 0 {
                weights.fill(1); // equal-probability cutoff and lexical ties
            } else if step % 3 == 1 {
                weights[0] = 300; // substantial blank mass
            }
            let sum = f32::from(weights.iter().sum::<u16>());
            let row = weights.map(|weight| f32::from(weight) / sum);
            compare_complete_step(&mut beam, &row, &labels)?;
        }
        Ok(())
    }

    #[test]
    fn subnormal_products_preserve_full_vocabulary_tie_break() -> Result<(), Error> {
        let labels = ['a'; 33];
        let mut beam = PrefixBeam {
            paths: BTreeMap::from([(
                vec![1],
                Mass {
                    blank: f64::from_bits(100),
                    nonblank: 0.0,
                },
            )]),
            log_scale: 0.0,
        };
        let mut row = [0.0325; 33];
        row[0] = 0.0;
        row[1..17].fill(0.03);
        // All child products round to three units of the smallest positive
        // subnormal: the lower-probability classes win by lexical order.
        assert!(
            top_classes(&row, &labels)?
                .iter()
                .all(|class| class.class_index > 16)
        );
        compare_complete_step(&mut beam, &row, &labels)?;
        assert!(beam.paths.contains_key(&[1, 1][..]));
        assert!(!beam.paths.contains_key(&[1, 17][..]));
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
