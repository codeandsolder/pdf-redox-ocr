//! Deterministic connected components with path compression.

pub struct DisjointSet {
    parents: Vec<usize>,
}

impl DisjointSet {
    pub fn new(size: usize) -> Self {
        Self {
            parents: (0..size).collect(),
        }
    }

    /// Smallest member index is the stable representative of a component.
    pub fn root(&mut self, mut index: usize) -> usize {
        while self.parents[index] != index {
            self.parents[index] = self.parents[self.parents[index]];
            index = self.parents[index];
        }
        index
    }

    pub fn connect(&mut self, left: usize, right: usize) {
        let a = self.root(left);
        let b = self.root(right);
        self.parents[a.max(b)] = a.min(b);
    }
}
