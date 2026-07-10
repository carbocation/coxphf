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
        let mut result = Self {
            n,
            p,
            ntde,
            p_total,
            x: x.clone(),
            t1,
            t2,
            ic: ic.clone(),
            start_order,
            score_weights,
            unit_score_weights: vec![1.0; p_total],
            bresx: x,
            ibresc: ic,
            ft,
            ftmap,
        };
        result.combine_breslow_ties();
        result
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

    fn combine_breslow_ties(&mut self) {
        if self.n < 2 {
            return;
        }

        for i in (0..(self.n - 1)).rev() {
            if self.ic[i + 1] >= 1 && (self.t2[i] - self.t2[i + 1]).abs() < 0.0001 {
                if let Some(score_weights) = &mut self.score_weights {
                    for j in 0..self.p_total {
                        score_weights.set(i, j, score_weights.get(i + 1, j));
                    }
                }
                for j in 0..self.p {
                    let next = self.bresx.get(i + 1, j);
                    self.bresx.add(i, j, next);
                }
                self.ibresc[i] += self.ibresc[i + 1];
                self.ibresc[i + 1] = 0;
            }
        }
    }
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
