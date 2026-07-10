#[derive(Clone, Debug)]
pub(crate) struct Matrix {
    nrow: usize,
    ncol: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub(crate) fn zeros(nrow: usize, ncol: usize) -> Self {
        Self {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        }
    }

    pub(crate) fn from_col_major(data: &[f64], nrow: usize, ncol: usize) -> Self {
        debug_assert_eq!(data.len(), nrow * ncol);
        let mut result = Self::zeros(nrow, ncol);
        for col in 0..ncol {
            for row in 0..nrow {
                result.data[row * ncol + col] = data[row + nrow * col];
            }
        }
        result
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.nrow);
        debug_assert!(col < self.ncol);
        row * self.ncol + col
    }

    #[inline]
    pub(crate) fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.index(row, col)]
    }

    #[inline]
    pub(crate) fn set(&mut self, row: usize, col: usize, value: f64) {
        let index = self.index(row, col);
        self.data[index] = value;
    }

    #[inline]
    pub(crate) fn add(&mut self, row: usize, col: usize, value: f64) {
        let index = self.index(row, col);
        self.data[index] += value;
    }

    #[inline]
    pub(crate) fn row(&self, row: usize) -> &[f64] {
        debug_assert!(row < self.nrow);
        let start = row * self.ncol;
        &self.data[start..start + self.ncol]
    }

    #[inline]
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [f64] {
        debug_assert!(row < self.nrow);
        let start = row * self.ncol;
        &mut self.data[start..start + self.ncol]
    }

    pub(crate) fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.nrow, other.nrow);
        debug_assert_eq!(self.ncol, other.ncol);
        self.data.copy_from_slice(&other.data);
    }

    pub(crate) fn copy_negated_from_symmetric(&mut self, other: &SymmetricMatrix) {
        debug_assert_eq!(self.nrow, other.n);
        debug_assert_eq!(self.ncol, other.n);
        for row in 0..other.n {
            for col in row..other.n {
                let value = -other.get(row, col);
                self.set(row, col, value);
                if row != col {
                    self.set(col, row, value);
                }
            }
        }
    }

    pub(crate) fn row_dot(&self, row: usize, vector: &[f64]) -> f64 {
        debug_assert_eq!(vector.len(), self.ncol);
        let mut value = 0.0;
        for (&matrix_value, &element) in self.row(row).iter().zip(vector) {
            value += matrix_value * element;
        }
        value
    }
}

/// Packed upper-triangular storage for a symmetric matrix.
#[derive(Clone, Debug)]
pub(crate) struct SymmetricMatrix {
    n: usize,
    data: Vec<f64>,
}

impl SymmetricMatrix {
    pub(crate) fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * (n + 1) / 2],
        }
    }

    pub(crate) fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.n);
        debug_assert!(col < self.n);
        let (first, second) = if row <= col { (row, col) } else { (col, row) };
        symmetric_pair_index(self.n, first, second)
    }

    #[inline]
    pub(crate) fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.index(row, col)]
    }

    #[inline]
    pub(crate) fn add(&mut self, row: usize, col: usize, value: f64) {
        let index = self.index(row, col);
        self.data[index] += value;
    }
}

/// Packed storage for a tensor symmetric under every permutation of its three
/// indices. Each unique `(first <= second <= third)` entry is stored once.
#[derive(Clone, Debug)]
pub(crate) struct SymmetricCube {
    n: usize,
    line_starts: Vec<usize>,
    data: Vec<f64>,
}

impl SymmetricCube {
    pub(crate) fn zeros(n: usize) -> Self {
        let pair_count = n * (n + 1) / 2;
        let mut line_starts = vec![0usize; pair_count];
        let mut data_len = 0usize;
        for first in 0..n {
            for second in first..n {
                line_starts[symmetric_pair_index(n, first, second)] = data_len;
                data_len += n - second;
            }
        }
        debug_assert_eq!(data_len, n * (n + 1) * (n + 2) / 6);
        Self {
            n,
            line_starts,
            data: vec![0.0; data_len],
        }
    }

    pub(crate) fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    #[inline]
    pub(crate) fn get(&self, first: usize, second: usize, third: usize) -> f64 {
        debug_assert!(first < self.n);
        debug_assert!(second < self.n);
        debug_assert!(third < self.n);
        let (first, second, third) = sorted_triple(first, second, third);
        let start = self.line_starts[symmetric_pair_index(self.n, first, second)];
        self.data[start + third - second]
    }

    #[inline]
    pub(crate) fn tail_mut(&mut self, first: usize, second: usize) -> &mut [f64] {
        debug_assert!(first <= second);
        debug_assert!(second < self.n);
        let start = self.line_starts[symmetric_pair_index(self.n, first, second)];
        &mut self.data[start..start + self.n - second]
    }
}

#[inline]
fn symmetric_pair_index(n: usize, first: usize, second: usize) -> usize {
    debug_assert!(first <= second);
    first * (2 * n - first + 1) / 2 + second - first
}

#[inline]
fn sorted_triple(mut first: usize, mut second: usize, mut third: usize) -> (usize, usize, usize) {
    if first > second {
        std::mem::swap(&mut first, &mut second);
    }
    if second > third {
        std::mem::swap(&mut second, &mut third);
    }
    if first > second {
        std::mem::swap(&mut first, &mut second);
    }
    (first, second, third)
}

/// Literal translation of the `vert`/`INVERT` routines in `coxphf.f90`.
#[cfg(test)]
pub(crate) fn invert(input: &Matrix) -> Matrix {
    let n = input.nrow;
    debug_assert_eq!(n, input.ncol);
    let mut value = input.clone();
    let mut workspace = vec![0usize; n];
    invert_in_place(&mut value, &mut workspace);
    value
}

pub(crate) fn invert_into(input: &Matrix, output: &mut Matrix, workspace: &mut Vec<usize>) {
    let n = input.nrow;
    debug_assert_eq!(n, input.ncol);
    output.copy_from(input);
    workspace.resize(n, 0);
    invert_in_place(output, workspace);
}

fn invert_in_place(value: &mut Matrix, workspace: &mut [usize]) {
    let n = value.nrow;

    if n == 1 {
        if value.get(0, 0) != 0.0 {
            value.set(0, 0, 1.0 / value.get(0, 0));
        }
        return;
    }

    // The scalar indices below remain one-based to mirror the original
    // control flow. Matrix accesses convert them to zero-based indices.
    let mut k = 1usize;
    let mut l = 0usize;
    let mut m = 1usize;

    loop {
        if l == n {
            break;
        }

        k = l;
        l = m;
        m += 1;
        let mut p = l;

        if m <= n {
            let mut largest = value.get(l - 1, l - 1).abs();
            for i in m..=n {
                let candidate = value.get(i - 1, l - 1).abs();
                if candidate > largest {
                    p = i;
                    largest = candidate;
                }
            }
            workspace[l - 1] = p;
        }

        let pivot = value.get(p - 1, l - 1);
        value.set(p - 1, l - 1, value.get(l - 1, l - 1));
        if pivot == 0.0 {
            return;
        }

        value.set(l - 1, l - 1, -1.0);
        let reciprocal = 1.0 / pivot;
        for i in 1..=n {
            value.set(i - 1, l - 1, -reciprocal * value.get(i - 1, l - 1));
        }

        let mut j = l;
        loop {
            j += 1;
            if j > n {
                j = 1;
            }
            if j == l {
                break;
            }

            let element = value.get(p - 1, j - 1);
            value.set(p - 1, j - 1, value.get(l - 1, j - 1));
            value.set(l - 1, j - 1, element);
            if element == 0.0 {
                continue;
            }

            if k != 0 {
                for i in 1..=k {
                    value.add(i - 1, j - 1, element * value.get(i - 1, l - 1));
                }
            }
            value.set(l - 1, j - 1, reciprocal * element);
            if m <= n {
                for i in m..=n {
                    value.add(i - 1, j - 1, element * value.get(i - 1, l - 1));
                }
            }
        }
    }

    while k > 0 {
        l = workspace[k - 1];
        for i in 1..=n {
            let saved = value.get(i - 1, l - 1);
            value.set(i - 1, l - 1, value.get(i - 1, k - 1));
            value.set(i - 1, k - 1, saved);
        }
        k -= 1;
    }
}

/// Literal translation of `FindDet`. The input is copied because the Fortran
/// routine destroys its working matrix.
#[cfg(test)]
pub(crate) fn determinant(input: &Matrix) -> f64 {
    let mut matrix = input.clone();
    determinant_in_place(&mut matrix)
}

pub(crate) fn determinant_with_workspace(input: &Matrix, workspace: &mut Matrix) -> f64 {
    workspace.copy_from(input);
    determinant_in_place(workspace)
}

fn determinant_in_place(matrix: &mut Matrix) -> f64 {
    let n = matrix.nrow;
    debug_assert_eq!(n, matrix.ncol);
    if n == 0 {
        return 1.0;
    }

    let mut sign = 1.0;

    for k in 0..n.saturating_sub(1) {
        if matrix.get(k, k) == 0.0 {
            let mut exists = false;
            for i in (k + 1)..n {
                if matrix.get(i, k) != 0.0 {
                    for j in 0..n {
                        let saved = matrix.get(i, j);
                        matrix.set(i, j, matrix.get(k, j));
                        matrix.set(k, j, saved);
                    }
                    exists = true;
                    sign = -sign;
                    break;
                }
            }
            if !exists {
                return 0.0;
            }
        }

        for j in (k + 1)..n {
            let multiplier = matrix.get(j, k) / matrix.get(k, k);
            for i in (k + 1)..n {
                matrix.set(j, i, matrix.get(j, i) - multiplier * matrix.get(k, i));
            }
        }
    }

    let mut value = sign;
    for i in 0..n {
        value *= matrix.get(i, i);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::{determinant, invert, sorted_triple, Matrix, SymmetricCube, SymmetricMatrix};

    #[test]
    fn inverse_and_determinant_match_small_examples() {
        // Column-major representation of [[4, 7], [2, 6]].
        let matrix = Matrix::from_col_major(&[4.0, 2.0, 7.0, 6.0], 2, 2);
        let inverse = invert(&matrix);
        let expected = [0.6, -0.2, -0.7, 0.4];
        for (index, &wanted) in expected.iter().enumerate() {
            let actual = inverse.get(index % 2, index / 2);
            assert!((actual - wanted).abs() < 1e-14);
        }
        assert!((determinant(&matrix) - 10.0).abs() < 1e-14);
    }

    #[test]
    fn packed_symmetric_storage_maps_every_permutation() {
        let mut matrix = SymmetricMatrix::zeros(4);
        matrix.add(3, 1, 7.0);
        assert_eq!(matrix.get(1, 3), 7.0);
        assert_eq!(matrix.get(3, 1), 7.0);
        assert_eq!(matrix.data.len(), 10);

        let mut cube = SymmetricCube::zeros(4);
        for first in 0..4 {
            for second in first..4 {
                let tail = cube.tail_mut(first, second);
                for third in second..4 {
                    tail[third - second] = (100 * first + 10 * second + third) as f64;
                }
            }
        }
        assert_eq!(cube.data.len(), 20);
        for first in 0..4 {
            for second in 0..4 {
                for third in 0..4 {
                    let sorted = sorted_triple(first, second, third);
                    let expected = (100 * sorted.0 + 10 * sorted.1 + sorted.2) as f64;
                    assert_eq!(cube.get(first, second, third), expected);
                }
            }
        }
    }
}
