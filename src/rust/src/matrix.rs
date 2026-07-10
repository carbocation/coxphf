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
        Self {
            nrow,
            ncol,
            data: data.to_vec(),
        }
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.nrow);
        debug_assert!(col < self.ncol);
        row + self.nrow * col
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

    pub(crate) fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub(crate) fn negated(&self) -> Self {
        let mut result = self.clone();
        for value in result.as_mut_slice() {
            *value = -*value;
        }
        result
    }

    pub(crate) fn row_dot(&self, row: usize, vector: &[f64]) -> f64 {
        debug_assert_eq!(vector.len(), self.ncol);
        let mut value = 0.0;
        for (col, &element) in vector.iter().enumerate() {
            value += self.get(row, col) * element;
        }
        value
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Cube {
    n: usize,
    data: Vec<f64>,
}

impl Cube {
    pub(crate) fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n * n],
        }
    }

    #[inline]
    fn index(&self, first: usize, second: usize, third: usize) -> usize {
        debug_assert!(first < self.n);
        debug_assert!(second < self.n);
        debug_assert!(third < self.n);
        first + self.n * (second + self.n * third)
    }

    #[inline]
    pub(crate) fn get(&self, first: usize, second: usize, third: usize) -> f64 {
        self.data[self.index(first, second, third)]
    }

    #[inline]
    pub(crate) fn add(&mut self, first: usize, second: usize, third: usize, value: f64) {
        let index = self.index(first, second, third);
        self.data[index] += value;
    }
}

/// Literal translation of the `vert`/`INVERT` routines in `coxphf.f90`.
pub(crate) fn invert(input: &Matrix) -> Matrix {
    let n = input.nrow;
    debug_assert_eq!(n, input.ncol);
    let mut value = input.clone();

    if n == 1 {
        if value.get(0, 0) != 0.0 {
            value.set(0, 0, 1.0 / value.get(0, 0));
        }
        return value;
    }

    // The scalar indices below remain one-based to mirror the original
    // control flow. Matrix accesses convert them to zero-based indices.
    let mut workspace = vec![0usize; n];
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
            return value;
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

    value
}

/// Literal translation of `FindDet`. The input is copied because the Fortran
/// routine destroys its working matrix.
pub(crate) fn determinant(input: &Matrix) -> f64 {
    let n = input.nrow;
    debug_assert_eq!(n, input.ncol);
    if n == 0 {
        return 1.0;
    }

    let mut matrix = input.clone();
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
    use super::{determinant, invert, Matrix};

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
}
