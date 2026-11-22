//! Exact conditional localization of retained CTC class sequences.
//!
//! A collapsed sequence defines a monotone blank/interleaved-token automaton.
//! The entry transition of each token occurs exactly once on every complete
//! path. Forward/backward path sums therefore give its conditional first-
//! emission distribution, without treating a model timestep as a glyph box.

use serde::Serialize;

const MAX_CELLS: usize = 2_000_000;
const LOG_ZERO: f64 = f64::NEG_INFINITY;

#[derive(Debug, Serialize)]
pub struct TokenLocalization {
    pub class_index: usize,
    /// Modal first-emission column, conditional on the complete class sequence.
    pub mode: usize,
    /// Conditional probability of the modal entry column, not label confidence.
    pub mode_probability: f64,
    /// Inclusive equal-tail 95% interval of the conditional entry distribution.
    pub interval: [usize; 2],
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SequenceAlignment {
    Measured {
        /// Full, unpruned path mass of this exact sequence of model classes.
        log_mass: f64,
        tokens: Vec<TokenLocalization>,
    },
    Unavailable {
        reason: &'static str,
    },
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == LOG_ZERO {
        return b;
    }
    if b == LOG_ZERO {
        return a;
    }
    a.max(b) + (a.min(b) - a.max(b)).exp().ln_1p()
}

struct Lattice<'a> {
    rows: &'a [&'a [f32]],
    states: Vec<usize>,
    width: usize,
}

impl<'a> Lattice<'a> {
    fn new(rows: &'a [&'a [f32]], tokens: &[usize]) -> Option<Self> {
        let width = tokens.len().checked_mul(2)?.checked_add(1)?;
        if rows.is_empty()
            || rows.len().checked_mul(width)? > MAX_CELLS
            || tokens.iter().any(|&i| i == 0 || i >= rows[0].len())
            || rows
                .iter()
                .any(|row| row.len() != rows[0].len() || row.is_empty())
        {
            return None;
        }
        let mut states = Vec::with_capacity(width);
        states.push(0);
        for &token in tokens {
            states.extend([token, 0]);
        }
        Some(Self {
            rows,
            states,
            width,
        })
    }

    fn skips_to(&self, state: usize) -> bool {
        state >= 2 && self.states[state] != 0 && self.states[state] != self.states[state - 2]
    }

    fn emit(&self, col: usize, state: usize) -> f64 {
        f64::from(self.rows[col][self.states[state]]).ln()
    }

    fn forward(&self) -> Vec<f64> {
        let mut table = vec![LOG_ZERO; self.rows.len() * self.width];
        table[0] = self.emit(0, 0);
        if self.width > 1 {
            table[1] = self.emit(0, 1);
        }
        for col in 1..self.rows.len() {
            let base = col * self.width;
            let prev = base - self.width;
            for state in 0..self.width {
                let mut incoming = table[prev + state];
                if state > 0 {
                    incoming = log_add(incoming, table[prev + state - 1]);
                }
                if self.skips_to(state) {
                    incoming = log_add(incoming, table[prev + state - 2]);
                }
                table[base + state] = incoming + self.emit(col, state);
            }
        }
        table
    }

    fn final_mass(&self, forward: &[f64]) -> f64 {
        let last = forward.len() - 1;
        if self.width == 1 {
            forward[last]
        } else {
            log_add(forward[last], forward[last - 1])
        }
    }

    fn entry_mass(&self, forward: &[f64], col: usize, state: usize) -> f64 {
        if col == 0 {
            return if state == 1 {
                self.emit(0, state)
            } else {
                LOG_ZERO
            };
        }
        let prev = (col - 1) * self.width;
        let mut incoming = forward[prev + state - 1];
        if self.skips_to(state) {
            incoming = log_add(incoming, forward[prev + state - 2]);
        }
        incoming + self.emit(col, state)
    }

    fn previous_backward(&self, col: usize, future: &[f64], previous: &mut [f64]) {
        for state in 0..self.width {
            let mut outgoing = self.emit(col, state) + future[state];
            if state + 1 < self.width {
                outgoing = log_add(outgoing, self.emit(col, state + 1) + future[state + 1]);
            }
            if state + 2 < self.width && self.skips_to(state + 2) {
                outgoing = log_add(outgoing, self.emit(col, state + 2) + future[state + 2]);
            }
            previous[state] = outgoing;
        }
    }

    fn entry_distributions(&self, forward: &[f64], log_mass: f64) -> Vec<Vec<f64>> {
        let count = self.width / 2;
        let mut entries = vec![vec![0.0; self.rows.len()]; count];
        let mut backward = vec![LOG_ZERO; self.width];
        let mut previous = backward.clone();
        backward[self.width - 1] = 0.0;
        if self.width > 1 {
            backward[self.width - 2] = 0.0;
        }
        for col in (0..self.rows.len()).rev() {
            for (i, distribution) in entries.iter_mut().enumerate() {
                let state = 2 * i + 1;
                distribution[col] =
                    (self.entry_mass(forward, col, state) + backward[state] - log_mass).exp();
            }
            if col > 0 {
                self.previous_backward(col, &backward, &mut previous);
                std::mem::swap(&mut backward, &mut previous);
            }
        }
        entries
    }
}

fn summarize(class_index: usize, entries: &[f64]) -> Option<TokenLocalization> {
    let total = entries.iter().sum::<f64>();
    // Every successful automaton path enters the token exactly once.
    // Reject a broken calculation rather than normalize arbitrary evidence.
    if !total.is_finite() || (total - 1.0).abs() > 1e-6 {
        return None;
    }
    let (mode, probability) = entries
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))?;
    let quantile = |q: f64| {
        let mut accumulated = 0.0;
        entries
            .iter()
            .position(|p| {
                accumulated += p;
                accumulated >= q * total
            })
            .unwrap_or(entries.len() - 1)
    };
    Some(TokenLocalization {
        class_index,
        mode,
        mode_probability: probability / total,
        interval: [quantile(0.025), quantile(0.975)],
    })
}

/// Rows have already been checked as probability distributions by the CTC
/// diagnostic decoder. This function reads the unpruned classes, not top-k
/// emission scores. Infeasible or oversized sequences explicitly abstain.
pub(crate) fn align(rows: &[&[f32]], tokens: &[usize]) -> SequenceAlignment {
    let Some(lattice) = Lattice::new(rows, tokens) else {
        return SequenceAlignment::Unavailable {
            reason: "invalid or oversized CTC alignment lattice",
        };
    };
    let forward = lattice.forward();
    let log_mass = lattice.final_mass(&forward);
    if !log_mass.is_finite() {
        return SequenceAlignment::Unavailable {
            reason: "no complete CTC path",
        };
    }
    let distributions = lattice.entry_distributions(&forward, log_mass);
    let Some(tokens) = tokens
        .iter()
        .zip(distributions)
        .map(|(&class, entries)| summarize(class, &entries))
        .collect::<Option<Vec<_>>>()
    else {
        return SequenceAlignment::Unavailable {
            reason: "invalid conditional CTC entry distribution",
        };
    };
    SequenceAlignment::Measured { log_mass, tokens }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_entry_distributions_for_repeated_and_distinct_tokens() -> Result<(), crate::Error>
    {
        let rows = [
            [0.3, 0.5, 0.2],
            [0.6, 0.2, 0.2],
            [0.2, 0.4, 0.4],
            [0.3, 0.4, 0.3],
        ];
        let refs = rows.iter().map(<[_; 3]>::as_slice).collect::<Vec<_>>();
        for tokens in [vec![], vec![1], vec![1, 1], vec![1, 2], vec![2, 1, 2]] {
            let lattice = Lattice::new(&refs, &tokens)
                .ok_or_else(|| crate::Error::Invalid("fixture lattice".into()))?;
            let forward = lattice.forward();
            let actual = lattice.final_mass(&forward);
            let expected_entries = lattice.entry_distributions(&forward, actual);
            let mut mass = 0.0;
            let mut entries = vec![vec![0.0; rows.len()]; tokens.len()];
            for a in 0..3 {
                for b in 0..3 {
                    for c in 0..3 {
                        for d in 0..3 {
                            let path = [a, b, c, d];
                            let mut collapsed = Vec::new();
                            let mut columns = Vec::new();
                            let mut previous = 0;
                            let mut p = 1.0;
                            for (col, index) in path.into_iter().enumerate() {
                                p *= f64::from(rows[col][index]);
                                if index != 0 && previous != index {
                                    collapsed.push(index);
                                    columns.push(col);
                                }
                                previous = index;
                            }
                            if collapsed == tokens {
                                mass += p;
                                for (i, col) in columns.into_iter().enumerate() {
                                    entries[i][col] += p;
                                }
                            }
                        }
                    }
                }
            }
            assert!((actual.exp() - mass).abs() < 1e-12);
            for (measured, expected) in expected_entries.iter().zip(entries) {
                assert!(
                    measured
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b / mass).abs() < 1e-12)
                );
            }
            assert!(matches!(
                align(&refs, &tokens),
                SequenceAlignment::Measured { .. }
            ));
        }
        Ok(())
    }

    #[test]
    fn blank_separates_repeated_classes_and_zero_mass_abstains() -> Result<(), crate::Error> {
        let rows = [[0., 1.], [1., 0.], [0., 1.]];
        let refs = rows.iter().map(<[_; 2]>::as_slice).collect::<Vec<_>>();
        let SequenceAlignment::Measured { log_mass, tokens } = align(&refs, &[1, 1]) else {
            return Err(crate::Error::Invalid(
                "repeated class path must exist".into(),
            ));
        };
        assert!(log_mass.abs() < 1e-12);
        assert_eq!(tokens[0].interval, [0, 0]);
        assert_eq!(tokens[1].interval, [2, 2]);
        assert!(matches!(
            align(&refs, &[1]),
            SequenceAlignment::Unavailable { .. }
        ));
        assert!(matches!(
            align(&refs, &[0]),
            SequenceAlignment::Unavailable { .. }
        ));
        Ok(())
    }

    #[test]
    fn tiny_path_mass_stays_finite_in_log_domain() {
        let rows = [[0.999_999, 0.000_001]; 900];
        let refs = rows.iter().map(<[_; 2]>::as_slice).collect::<Vec<_>>();
        let tokens = vec![1; 400];
        assert!(
            matches!(align(&refs,&tokens),SequenceAlignment::Measured{log_mass,..} if log_mass < -1000.0)
        );
        assert!(matches!(
            align(&refs, &vec![1; 2000]),
            SequenceAlignment::Unavailable { .. }
        ));
    }
}
