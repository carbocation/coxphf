use crate::matrix::Matrix;

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
