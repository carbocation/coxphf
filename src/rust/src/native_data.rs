use crate::matrix::Matrix;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) struct PatternData {
    pub(crate) row_pattern: Vec<u32>,
    pub(crate) patterns: Matrix,
}

#[derive(Clone, Debug)]
pub(crate) struct NativeData {
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) ntde: usize,
    pub(crate) p_total: usize,
    pub(crate) x: Matrix,
    pub(crate) t1: Vec<f64>,
    pub(crate) t2: Vec<f64>,
    pub(crate) ic: Vec<i32>,
    pub(crate) start_order: Vec<usize>,
    score_weights: Option<Matrix>,
    unit_score_weights: Vec<f64>,
    pub(crate) bresx: Matrix,
    pub(crate) ibresc: Vec<i32>,
    pub(crate) event_rows: Vec<usize>,
    pub(crate) ft: Matrix,
    pub(crate) ftmap: Vec<usize>,
    pub(crate) sparse_moments: bool,
    pub(crate) pattern_data: Option<PatternData>,
}

impl NativeData {
    pub(crate) fn from_native_arrays(
        cards: &[f64],
        parms: &[f64],
        ioarray: &[f64],
        io_nrow: usize,
    ) -> Self {
        let n = parms[0] as usize;
        let p = parms[1] as usize;
        let ntde = parms[13] as usize;
        let p_total = p + ntde;
        let card_ncol = 2 * p + 4 + 2 * ntde;
        assert_eq!(cards.len(), n * card_ncol);

        let column = |col: usize| -> &[f64] {
            let start = n * col;
            &cards[start..start + n]
        };

        let x = Matrix::from_col_major(&cards[..n * p], n, p);
        let t1 = column(p).to_vec();
        let t2 = column(p + 1).to_vec();
        let ic: Vec<i32> = column(p + 2).iter().map(|&value| value as i32).collect();
        let start_order = column(p + 3)
            .iter()
            .map(|&value| to_zero_based_index(value))
            .collect();

        let score_start = n * (p + 4);
        let score_end = score_start + n * p_total;
        let score_weights = Matrix::from_col_major(&cards[score_start..score_end], n, p_total);

        let ft = if ntde == 0 {
            Matrix::zeros(n, 0)
        } else {
            let ft_start = n * (2 * p + 4 + ntde);
            Matrix::from_col_major(&cards[ft_start..ft_start + n * ntde], n, ntde)
        };

        let ftmap = (0..ntde)
            .map(|j| to_zero_based_index(ioarray[3 + io_nrow * (p + j)]))
            .collect();

        Self::finish(
            n,
            p,
            ntde,
            x,
            t1,
            t2,
            ic,
            start_order,
            Some(score_weights),
            ft,
            ftmap,
        )
    }

    pub(crate) fn from_compact_arrays(
        x: &[f64],
        response: &[f64],
        start_order: &[i32],
        timedata: &[f64],
        parms: &[f64],
        ioarray: &[f64],
        io_nrow: usize,
    ) -> Self {
        let n = parms[0] as usize;
        let p = parms[1] as usize;
        let ntde = parms[13] as usize;
        assert_eq!(x.len(), n * p);
        assert_eq!(response.len(), n * 3);
        assert_eq!(start_order.len(), n);
        assert_eq!(timedata.len(), n * ntde);

        let response_column = |col: usize| -> &[f64] {
            let start = n * col;
            &response[start..start + n]
        };
        let x = Matrix::from_col_major(x, n, p);
        let t1 = response_column(0).to_vec();
        let t2 = response_column(1).to_vec();
        let ic: Vec<i32> = response_column(2)
            .iter()
            .map(|&value| value as i32)
            .collect();
        let start_order = start_order
            .iter()
            .map(|&value| to_zero_based_integer(value))
            .collect();
        let ft = Matrix::from_col_major(timedata, n, ntde);
        let ftmap = (0..ntde)
            .map(|j| to_zero_based_index(ioarray[3 + io_nrow * (p + j)]))
            .collect();

        Self::finish(n, p, ntde, x, t1, t2, ic, start_order, None, ft, ftmap)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        n: usize,
        p: usize,
        ntde: usize,
        x: Matrix,
        t1: Vec<f64>,
        t2: Vec<f64>,
        ic: Vec<i32>,
        start_order: Vec<usize>,
        score_weights: Option<Matrix>,
        ft: Matrix,
        ftmap: Vec<usize>,
    ) -> Self {
        let p_total = p + ntde;
        let sparse_moments = ntde == 0 && has_sparse_column(&x, n, p);
        let pattern_data = if ntde == 0 {
            build_pattern_data(&x, n, p)
        } else {
            None
        };
        let mut score_weights = score_weights;
        let (event_rows, bresx, ibresc) =
            aggregate_breslow_events(&x, p, &t2, &ic, score_weights.as_mut());
        Self {
            n,
            p,
            ntde,
            p_total,
            x,
            t1,
            t2,
            ic,
            start_order,
            score_weights,
            unit_score_weights: vec![1.0; p_total],
            bresx,
            ibresc,
            event_rows,
            ft,
            ftmap,
            sparse_moments,
            pattern_data,
        }
    }

    #[inline]
    pub(crate) fn score_weights(&self, row: usize) -> &[f64] {
        self.score_weights
            .as_ref()
            .map_or(&self.unit_score_weights, |weights| weights.row(row))
    }

    #[inline]
    pub(crate) fn score_weight(&self, row: usize, col: usize) -> f64 {
        self.score_weights
            .as_ref()
            .map_or(1.0, |weights| weights.get(row, col))
    }
}

fn build_pattern_data(x: &Matrix, n: usize, p: usize) -> Option<PatternData> {
    // Pattern lookup is worthwhile only when each distinct row is reused many
    // times. Sparse binary exposures combined with discrete nuisance
    // covariates are the primary high-throughput case.
    let maximum_patterns = (n / 8).max(1).min(u32::MAX as usize);
    if p == 3 {
        return build_three_column_pattern_data(x, n, maximum_patterns);
    }

    let mut lookup: HashMap<Vec<u64>, u32> = HashMap::new();
    let mut row_pattern = Vec::with_capacity(n);
    let mut pattern_rows: Vec<Vec<f64>> = Vec::new();

    for row in 0..n {
        let key: Vec<u64> = x
            .row(row)
            .iter()
            .map(|&value| if value == 0.0 { 0 } else { value.to_bits() })
            .collect();
        let pattern = if let Some(&pattern) = lookup.get(&key) {
            pattern
        } else {
            if pattern_rows.len() >= maximum_patterns {
                return None;
            }
            let pattern = pattern_rows.len() as u32;
            pattern_rows.push(x.row(row).to_vec());
            lookup.insert(key, pattern);
            pattern
        };
        row_pattern.push(pattern);
    }

    finish_pattern_data(row_pattern, pattern_rows, p)
}

fn build_three_column_pattern_data(
    x: &Matrix,
    n: usize,
    maximum_patterns: usize,
) -> Option<PatternData> {
    let mut lookup: HashMap<[u64; 3], u32> = HashMap::new();
    let mut row_pattern = Vec::with_capacity(n);
    let mut pattern_rows: Vec<Vec<f64>> = Vec::new();

    for row in 0..n {
        let values = x.row(row);
        let key = [
            normalized_float_bits(values[0]),
            normalized_float_bits(values[1]),
            normalized_float_bits(values[2]),
        ];
        let pattern = if let Some(&pattern) = lookup.get(&key) {
            pattern
        } else {
            if pattern_rows.len() >= maximum_patterns {
                return None;
            }
            let pattern = pattern_rows.len() as u32;
            pattern_rows.push(values.to_vec());
            lookup.insert(key, pattern);
            pattern
        };
        row_pattern.push(pattern);
    }

    finish_pattern_data(row_pattern, pattern_rows, 3)
}

fn normalized_float_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn finish_pattern_data(
    row_pattern: Vec<u32>,
    pattern_rows: Vec<Vec<f64>>,
    p: usize,
) -> Option<PatternData> {
    let mut patterns = Matrix::zeros(pattern_rows.len(), p);
    for (row, values) in pattern_rows.iter().enumerate() {
        patterns.row_mut(row).copy_from_slice(values);
    }
    Some(PatternData {
        row_pattern,
        patterns,
    })
}

fn has_sparse_column(x: &Matrix, n: usize, p: usize) -> bool {
    (0..p).any(|column| {
        let zero_count = (0..n).filter(|&row| x.get(row, column) == 0.0).count();
        zero_count as f64 / n as f64 >= 0.8
    })
}

fn aggregate_breslow_events(
    x: &Matrix,
    p: usize,
    t2: &[f64],
    ic: &[i32],
    mut score_weights: Option<&mut Matrix>,
) -> (Vec<usize>, Matrix, Vec<i32>) {
    let n = ic.len();
    let mut groups = Vec::new();
    let mut start = 0usize;

    while start < n {
        let mut end = start;
        while end + 1 < n && ic[end + 1] >= 1 && (t2[end] - t2[end + 1]).abs() < 0.0001 {
            end += 1;
        }

        if let Some(weights) = score_weights.as_deref_mut() {
            if end > start {
                let final_weights = weights.row(end).to_vec();
                for row in start..end {
                    weights.row_mut(row).copy_from_slice(&final_weights);
                }
            }
        }

        let event_count = ic[start..=end].iter().sum();
        if event_count != 0 {
            groups.push((start, end, event_count));
        }
        start = end + 1;
    }

    let mut event_rows = Vec::with_capacity(groups.len());
    let mut event_counts = Vec::with_capacity(groups.len());
    let mut event_covariates = Matrix::zeros(groups.len(), p);
    for (event_index, &(start, end, event_count)) in groups.iter().enumerate() {
        event_rows.push(start);
        event_counts.push(event_count);
        for column in 0..p {
            let mut value = x.get(end, column);
            for row in (start..end).rev() {
                value += x.get(row, column);
            }
            event_covariates.set(event_index, column, value);
        }
    }

    (event_rows, event_covariates, event_counts)
}

fn to_zero_based_index(value: f64) -> usize {
    let index = value as isize;
    assert!(index >= 1);
    (index - 1) as usize
}

fn to_zero_based_integer(value: i32) -> usize {
    assert!(value >= 1);
    (value - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_columns_are_detected_from_structural_zeros() {
        let mut x = Matrix::zeros(5, 2);
        for row in 0..5 {
            x.set(row, 0, row as f64 + 1.0);
        }
        x.set(4, 1, 2.0);

        assert!(has_sparse_column(&x, 5, 2));
        x.set(0, 1, 1.0);
        assert!(!has_sparse_column(&x, 5, 2));
    }

    #[test]
    fn repeated_design_rows_are_compacted_into_patterns() {
        let mut repeated = Matrix::zeros(32, 2);
        for row in 0..32 {
            repeated.set(row, 0, (row % 2) as f64);
            repeated.set(row, 1, ((row / 2) % 2) as f64);
        }
        let patterns = build_pattern_data(&repeated, 32, 2).unwrap();
        assert_eq!(patterns.patterns.nrow(), 4);
        assert_eq!(patterns.row_pattern.len(), 32);

        let mut unique = Matrix::zeros(32, 2);
        for row in 0..32 {
            unique.set(row, 0, row as f64);
            unique.set(row, 1, row as f64 + 0.5);
        }
        assert!(build_pattern_data(&unique, 32, 2).is_none());

        let mut three_column = Matrix::zeros(64, 3);
        for row in 0..64 {
            three_column.set(row, 0, (row % 2) as f64);
            three_column.set(row, 1, ((row / 2) % 2) as f64);
            three_column.set(row, 2, ((row / 4) % 2) as f64);
        }
        let three_column_patterns = build_pattern_data(&three_column, 64, 3).unwrap();
        assert_eq!(three_column_patterns.patterns.nrow(), 8);
        assert_eq!(three_column_patterns.row_pattern.len(), 64);
    }

    #[test]
    fn breslow_events_are_compacted_without_changing_tie_aggregation() {
        let mut x = Matrix::zeros(5, 2);
        let mut weights = Matrix::zeros(5, 2);
        for row in 0..5 {
            x.set(row, 0, row as f64 + 1.0);
            x.set(row, 1, 10.0 * (row as f64 + 1.0));
            weights.set(row, 0, 100.0 + row as f64);
            weights.set(row, 1, 200.0 + row as f64);
        }

        let (event_rows, bresx, event_counts) = aggregate_breslow_events(
            &x,
            2,
            &[1.0, 1.0, 2.0, 2.0, 3.0],
            &[1, 1, 0, 1, 0],
            Some(&mut weights),
        );

        assert_eq!(event_rows, vec![0, 2]);
        assert_eq!(event_counts, vec![2, 1]);
        assert_eq!(bresx.row(0), &[3.0, 30.0]);
        assert_eq!(bresx.row(1), &[7.0, 70.0]);
        assert_eq!(weights.row(0), &[101.0, 201.0]);
        assert_eq!(weights.row(2), &[103.0, 203.0]);
    }
}
