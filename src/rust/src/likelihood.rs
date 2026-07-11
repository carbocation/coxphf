#![allow(clippy::needless_range_loop)]

use crate::matrix::{
    determinant_with_workspace, invert_into, Matrix, SymmetricCube, SymmetricMatrix,
};
use crate::native_data::NativeData;

const LOWEST: f64 = 1.0e-44;

#[derive(Clone, Copy)]
enum LikelihoodMode {
    General,
    IntervalFast,
    Fast,
}

pub(crate) struct LikeWorkspace {
    mode: LikelihoodMode,
    pub(crate) score: Vec<f64>,
    pub(crate) hessian: SymmetricMatrix,
    derivative_hessian: SymmetricCube,
    x_all: Matrix,
    mask: Vec<bool>,
    bresx_all: Vec<f64>,
    risk: Vec<f64>,
    first_moment: Vec<f64>,
    second_moment: SymmetricMatrix,
    third_moment: SymmetricCube,
    information: Matrix,
    information_inverse: Matrix,
    determinant_workspace: Matrix,
    inverse_workspace: Vec<usize>,
    pattern_moments: Matrix,
}

impl LikeWorkspace {
    pub(crate) fn new(data: &NativeData) -> Self {
        let n = data.n;
        let p = data.p_total;
        let fast_mode = data.ntde == 0
            && data.t1.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                < data.t2.iter().copied().fold(f64::INFINITY, f64::min);
        let mode = if fast_mode {
            LikelihoodMode::Fast
        } else if data.ntde == 0 {
            LikelihoodMode::IntervalFast
        } else {
            LikelihoodMode::General
        };

        let mut x_all = if data.ntde == 0 {
            Matrix::zeros(0, 0)
        } else {
            Matrix::zeros(n, p)
        };
        if data.ntde != 0 {
            for row in 0..n {
                x_all.row_mut(row)[..data.p].copy_from_slice(data.x.row(row));
            }
        }

        Self {
            mode,
            score: vec![0.0; p],
            hessian: SymmetricMatrix::zeros(p),
            derivative_hessian: SymmetricCube::zeros(p),
            x_all,
            mask: vec![false; n],
            bresx_all: vec![0.0; p],
            risk: vec![0.0; n],
            first_moment: vec![0.0; p],
            second_moment: SymmetricMatrix::zeros(p),
            third_moment: SymmetricCube::zeros(p),
            information: Matrix::zeros(p, p),
            information_inverse: Matrix::zeros(p, p),
            determinant_workspace: Matrix::zeros(p, p),
            inverse_workspace: vec![0; p],
            pattern_moments: Matrix::zeros(
                data.pattern_data
                    .as_ref()
                    .map_or(0, |patterns| patterns.patterns.nrow()),
                moment_count(p),
            ),
        }
    }

    pub(crate) fn reset_derivatives(&mut self) {
        self.score.fill(0.0);
        self.hessian.fill(0.0);
        self.derivative_hessian.fill(0.0);
    }
}

pub(crate) struct LikeResult {
    pub(crate) loglik: f64,
    pub(crate) jcode: i32,
}

pub(crate) fn evaluate(
    data: &NativeData,
    coefficients: &[f64],
    ifirth: i32,
    ngv: i32,
    penalty: f64,
    initial_jcode: i32,
    workspace: &mut LikeWorkspace,
) -> LikeResult {
    let p = data.p_total;
    let mut loglik = 0.0;
    workspace.score.fill(0.0);
    workspace.hessian.fill(0.0);
    if ifirth != 0 {
        workspace.derivative_hessian.fill(0.0);
    }
    if let Some(patterns) = &data.pattern_data {
        prepare_pattern_moments(
            &patterns.patterns,
            coefficients,
            ifirth,
            &mut workspace.pattern_moments,
        );
    }

    let mode = workspace.mode;
    let LikeWorkspace {
        score,
        hessian,
        derivative_hessian,
        x_all,
        mask,
        bresx_all,
        risk,
        first_moment,
        second_moment,
        third_moment,
        information,
        information_inverse,
        determinant_workspace,
        inverse_workspace,
        pattern_moments,
        ..
    } = workspace;
    match mode {
        LikelihoodMode::General => evaluate_general(
            data,
            coefficients,
            ifirth,
            ngv,
            x_all,
            &mut loglik,
            score,
            hessian,
            derivative_hessian,
            mask,
            bresx_all,
            risk,
            first_moment,
            second_moment,
            third_moment,
        ),
        LikelihoodMode::IntervalFast => evaluate_interval_fast(
            data,
            coefficients,
            ifirth,
            ngv,
            &data.x,
            &mut loglik,
            score,
            hessian,
            derivative_hessian,
            bresx_all,
            risk,
            first_moment,
            second_moment,
            third_moment,
            pattern_moments,
        ),
        LikelihoodMode::Fast => evaluate_fast(
            data,
            coefficients,
            ifirth,
            ngv,
            &data.x,
            &mut loglik,
            score,
            hessian,
            derivative_hessian,
            first_moment,
            second_moment,
            third_moment,
            pattern_moments,
        ),
    }

    let mut jcode = initial_jcode;
    if ifirth != 0 {
        information.copy_negated_from_symmetric(hessian);
        invert_into(information, information_inverse, inverse_workspace);

        for j in 0..p {
            let mut trace = 0.0;
            for k in 0..p {
                let information_row = information_inverse.row(k);
                for l in 0..p {
                    trace -= information_row[l] * derivative_hessian.get(j, l, k);
                }
            }
            score[j] += trace * penalty;
        }

        let mut det = if p == 1 {
            information.get(0, 0)
        } else {
            determinant_with_workspace(information, determinant_workspace)
        };
        if det < LOWEST {
            det = LOWEST;
            jcode = 2;
        }
        loglik += penalty * det.ln();
    }

    LikeResult { loglik, jcode }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_general(
    data: &NativeData,
    coefficients: &[f64],
    ifirth: i32,
    ngv: i32,
    x_all: &mut Matrix,
    loglik: &mut f64,
    score: &mut [f64],
    hessian: &mut SymmetricMatrix,
    derivative_hessian: &mut SymmetricCube,
    mask: &mut [bool],
    bresx_all: &mut [f64],
    risk: &mut [f64],
    first_moment: &mut [f64],
    second_moment: &mut SymmetricMatrix,
    third_moment: &mut SymmetricCube,
) {
    let n = data.n;
    let p = data.p_total;

    for event_index in 0..data.event_rows.len() {
        let event_row = data.event_rows[event_index];
        let event_count = data.ibresc[event_index];

        let event_time = data.t2[event_row] - 0.00001;
        for row in 0..n {
            mask[row] = data.t1[row] < event_time && data.t2[row] >= event_time;
        }

        for j in 0..data.p {
            bresx_all[j] = data.bresx.get(event_index, j);
        }
        for j in data.p..p {
            let time_column = j - data.p;
            let source_column = data.ftmap[time_column];
            let time_value = data.ft.get(event_row, time_column);
            for row in 0..n {
                if mask[row] {
                    x_all.set(row, j, data.x.get(row, source_column) * time_value);
                }
            }
            bresx_all[j] = time_value * data.bresx.get(event_index, source_column);
        }

        let mut risk_sum = 0.0;
        first_moment.fill(0.0);
        second_moment.fill(0.0);
        if ifirth != 0 {
            third_moment.fill(0.0);
        }
        for row in 0..n {
            if mask[row] {
                risk[row] = x_all.row_dot(row, coefficients).exp();
                risk_sum += risk[row];
                let x_row = x_all.row(row);
                for j in 0..p {
                    let weighted_x = x_row[j] * risk[row];
                    first_moment[j] += weighted_x;
                    for k in j..p {
                        second_moment.add(j, k, weighted_x * x_row[k]);
                        if ifirth != 0 {
                            let third_tail = third_moment.tail_mut(j, k);
                            for l in k..p {
                                third_tail[l - k] += x_row[l] * x_row[k] * x_row[j] * risk[row];
                            }
                        }
                    }
                }
            }
        }

        let log_risk_sum = risk_sum.max(LOWEST).ln();
        let mut numerator = 0.0;
        for j in 0..p {
            numerator += bresx_all[j] * coefficients[j];
        }
        let contribution = numerator - f64::from(event_count) * log_risk_sum;
        if ngv == p as i32 {
            *loglik += contribution * data.score_weight(event_row, 0);
        } else {
            *loglik += contribution;
        }

        let weights = data.score_weights(event_row);
        for j in 0..p {
            let weight_j = weights[j];
            score[j] +=
                (bresx_all[j] - f64::from(event_count) * first_moment[j] / risk_sum) * weight_j;

            for k in j..p {
                let weight_k = weights[k];
                let centered_second = (second_moment.get(j, k)
                    - first_moment[j] / risk_sum * first_moment[k])
                    / risk_sum;
                hessian.add(
                    j,
                    k,
                    -f64::from(event_count) * centered_second * weight_j * weight_k,
                );

                if ifirth != 0 {
                    let derivative_tail = derivative_hessian.tail_mut(j, k);
                    for l in k..p {
                        let centered_third = centered_third_moment(
                            third_moment.get(j, k, l),
                            j,
                            k,
                            l,
                            risk_sum,
                            first_moment,
                            second_moment,
                        );
                        derivative_tail[l - k] += -f64::from(event_count)
                            * centered_third
                            * weight_j
                            * weight_k
                            * weights[l];
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_interval_fast(
    data: &NativeData,
    coefficients: &[f64],
    ifirth: i32,
    ngv: i32,
    x_all: &Matrix,
    loglik: &mut f64,
    score: &mut [f64],
    hessian: &mut SymmetricMatrix,
    derivative_hessian: &mut SymmetricCube,
    bresx_all: &mut [f64],
    risk: &mut [f64],
    first_moment: &mut [f64],
    second_moment: &mut SymmetricMatrix,
    third_moment: &mut SymmetricCube,
    pattern_moments: &Matrix,
) {
    let n = data.n;
    first_moment.fill(0.0);
    second_moment.fill(0.0);
    if ifirth != 0 {
        third_moment.fill(0.0);
    }

    let mut risk_sum = 0.0;
    let mut add_row = n as isize - 1;
    let mut start_position = 0usize;

    for event_index in (0..data.event_rows.len()).rev() {
        let event_row = data.event_rows[event_index];
        let event_time = data.t2[event_row] - 0.00001;

        while add_row >= 0 {
            let row = add_row as usize;
            if data.t2[row] < event_time {
                break;
            }
            if let Some(patterns) = &data.pattern_data {
                let moments = pattern_moments.row(patterns.row_pattern[row] as usize);
                risk[row] = moments[0];
                update_precomputed_moments(
                    moments,
                    1.0,
                    ifirth,
                    &mut risk_sum,
                    first_moment,
                    second_moment,
                    third_moment,
                );
            } else {
                risk[row] = x_all.row_dot(row, coefficients).exp();
                update_risk_moments(
                    x_all.row(row),
                    1.0,
                    risk[row],
                    ifirth,
                    data.sparse_moments,
                    &mut risk_sum,
                    first_moment,
                    second_moment,
                    third_moment,
                );
            }
            add_row -= 1;
        }

        while start_position < n {
            let row = data.start_order[start_position];
            if data.t1[row] < event_time {
                break;
            }
            if let Some(patterns) = &data.pattern_data {
                update_precomputed_moments(
                    pattern_moments.row(patterns.row_pattern[row] as usize),
                    -1.0,
                    ifirth,
                    &mut risk_sum,
                    first_moment,
                    second_moment,
                    third_moment,
                );
            } else {
                update_risk_moments(
                    x_all.row(row),
                    -1.0,
                    risk[row],
                    ifirth,
                    data.sparse_moments,
                    &mut risk_sum,
                    first_moment,
                    second_moment,
                    third_moment,
                );
            }
            start_position += 1;
        }

        for j in 0..data.p {
            bresx_all[j] = data.bresx.get(event_index, j);
        }
        accumulate_aggregated_event(
            data,
            event_index,
            event_row,
            coefficients,
            ngv,
            ifirth,
            bresx_all,
            risk_sum,
            first_moment,
            second_moment,
            third_moment,
            loglik,
            score,
            hessian,
            derivative_hessian,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_fast(
    data: &NativeData,
    coefficients: &[f64],
    ifirth: i32,
    ngv: i32,
    x_all: &Matrix,
    loglik: &mut f64,
    score: &mut [f64],
    hessian: &mut SymmetricMatrix,
    derivative_hessian: &mut SymmetricCube,
    first_moment: &mut [f64],
    second_moment: &mut SymmetricMatrix,
    third_moment: &mut SymmetricCube,
    pattern_moments: &Matrix,
) {
    let n = data.n;
    let p = data.p_total;
    first_moment.fill(0.0);
    second_moment.fill(0.0);
    if ifirth != 0 {
        third_moment.fill(0.0);
    }

    let mut risk_sum = 0.0;
    let mut tied_events = 0i32;
    for row in (0..n).rev() {
        let x_row = x_all.row(row);
        let linear_predictor = x_all.row_dot(row, coefficients);
        let row_risk = data.pattern_data.as_ref().map_or_else(
            || linear_predictor.exp(),
            |patterns| pattern_moments.get(patterns.row_pattern[row] as usize, 0),
        );
        let weights = data.score_weights(row);
        let current_events;
        if row > 0 {
            if data.t2[row] == data.t2[row - 1] {
                current_events = 0;
                tied_events += data.ic[row];
            } else {
                current_events = tied_events + data.ic[row];
                tied_events = 0;
            }
        } else {
            current_events = tied_events + data.ic[row];
            tied_events = 0;
        }

        if let Some(patterns) = &data.pattern_data {
            update_precomputed_moments(
                pattern_moments.row(patterns.row_pattern[row] as usize),
                1.0,
                ifirth,
                &mut risk_sum,
                first_moment,
                second_moment,
                third_moment,
            );
        } else {
            update_risk_moments(
                x_row,
                1.0,
                row_risk,
                ifirth,
                data.sparse_moments,
                &mut risk_sum,
                first_moment,
                second_moment,
                third_moment,
            );
        }

        let contribution = linear_predictor * f64::from(data.ic[row])
            - f64::from(current_events) * risk_sum.max(LOWEST).ln();
        if ngv == p as i32 {
            *loglik += contribution * data.score_weight(row, 0);
        } else {
            *loglik += contribution;
        }

        for j in 0..p {
            let weight_j = weights[j];
            score[j] += (x_row[j] * f64::from(data.ic[row])
                - f64::from(current_events) * first_moment[j] / risk_sum)
                * weight_j;
            for k in j..p {
                let weight_k = weights[k];
                let centered_second = (second_moment.get(j, k)
                    - first_moment[j] / risk_sum * first_moment[k])
                    / risk_sum;
                hessian.add(
                    j,
                    k,
                    -f64::from(current_events) * centered_second * weight_j * weight_k,
                );

                if ifirth != 0 {
                    let derivative_tail = derivative_hessian.tail_mut(j, k);
                    for l in k..p {
                        let centered_third = centered_third_moment(
                            third_moment.get(j, k, l),
                            j,
                            k,
                            l,
                            risk_sum,
                            first_moment,
                            second_moment,
                        );
                        derivative_tail[l - k] += -f64::from(current_events)
                            * centered_third
                            * weight_j
                            * weight_k
                            * weights[l];
                    }
                }
            }
        }
    }
}

fn moment_count(p: usize) -> usize {
    1 + p + p * (p + 1) / 2 + p * (p + 1) * (p + 2) / 6
}

fn prepare_pattern_moments(
    patterns: &Matrix,
    coefficients: &[f64],
    ifirth: i32,
    output: &mut Matrix,
) {
    let p = coefficients.len();
    for pattern in 0..patterns.nrow() {
        let x_row = patterns.row(pattern);
        let row_risk = patterns.row_dot(pattern, coefficients).exp();
        let moments = output.row_mut(pattern);
        moments[0] = row_risk;
        let mut position = 1;
        for j in 0..p {
            moments[position] = x_row[j] * row_risk;
            position += 1;
        }
        for j in 0..p {
            let first = x_row[j] * row_risk;
            for k in j..p {
                moments[position] = first * x_row[k];
                position += 1;
            }
        }
        if ifirth == 1 {
            for j in 0..p {
                let first = x_row[j] * row_risk;
                for k in j..p {
                    let second = first * x_row[k];
                    for l in k..p {
                        moments[position] = second * x_row[l];
                        position += 1;
                    }
                }
            }
            debug_assert_eq!(position, moment_count(p));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_precomputed_moments(
    moments: &[f64],
    direction: f64,
    ifirth: i32,
    risk_sum: &mut f64,
    first_moment: &mut [f64],
    second_moment: &mut SymmetricMatrix,
    third_moment: &mut SymmetricCube,
) {
    let p = first_moment.len();
    *risk_sum += direction * moments[0];
    let mut position = 1;
    for target in first_moment.iter_mut() {
        *target += direction * moments[position];
        position += 1;
    }
    for target in second_moment.data_mut() {
        *target += direction * moments[position];
        position += 1;
    }
    if ifirth == 1 {
        for target in third_moment.data_mut() {
            *target += direction * moments[position];
            position += 1;
        }
        debug_assert_eq!(position, moment_count(p));
    }
}

#[allow(clippy::too_many_arguments)]
fn update_risk_moments(
    x_row: &[f64],
    direction: f64,
    row_risk: f64,
    ifirth: i32,
    sparse_moments: bool,
    risk_sum: &mut f64,
    first_moment: &mut [f64],
    second_moment: &mut SymmetricMatrix,
    third_moment: &mut SymmetricCube,
) {
    let p = first_moment.len();
    *risk_sum += direction * row_risk;

    if sparse_moments {
        let mut active_p = p;
        while active_p > 0 && x_row[active_p - 1] == 0.0 {
            active_p -= 1;
        }
        for j in 0..active_p {
            if x_row[j] == 0.0 {
                continue;
            }
            let first = x_row[j] * row_risk;
            first_moment[j] += direction * first;
            for k in j..active_p {
                if x_row[k] == 0.0 {
                    continue;
                }
                let second = first * x_row[k];
                second_moment.add(j, k, direction * second);
                if ifirth == 1 {
                    let third_tail = third_moment.tail_mut(j, k);
                    for l in k..active_p {
                        if x_row[l] != 0.0 {
                            third_tail[l - k] += direction * second * x_row[l];
                        }
                    }
                }
            }
        }
        return;
    }

    for j in 0..p {
        let first = x_row[j] * row_risk;
        first_moment[j] += direction * first;
        for k in j..p {
            let second = first * x_row[k];
            second_moment.add(j, k, direction * second);
            if ifirth == 1 {
                let third_tail = third_moment.tail_mut(j, k);
                for l in k..p {
                    third_tail[l - k] += direction * second * x_row[l];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_aggregated_event(
    data: &NativeData,
    event_index: usize,
    event_row: usize,
    coefficients: &[f64],
    ngv: i32,
    ifirth: i32,
    bresx_all: &[f64],
    risk_sum: f64,
    first_moment: &[f64],
    second_moment: &SymmetricMatrix,
    third_moment: &SymmetricCube,
    loglik: &mut f64,
    score: &mut [f64],
    hessian: &mut SymmetricMatrix,
    derivative_hessian: &mut SymmetricCube,
) {
    let p = data.p_total;
    let event_count = data.ibresc[event_index];
    let mut numerator = 0.0;
    for j in 0..p {
        numerator += bresx_all[j] * coefficients[j];
    }
    let contribution = numerator - f64::from(event_count) * risk_sum.max(LOWEST).ln();
    if ngv == p as i32 {
        *loglik += contribution * data.score_weight(event_row, 0);
    } else {
        *loglik += contribution;
    }

    let weights = data.score_weights(event_row);
    for j in 0..p {
        let weight_j = weights[j];
        score[j] += (bresx_all[j] - f64::from(event_count) * first_moment[j] / risk_sum) * weight_j;
        for k in j..p {
            let weight_k = weights[k];
            let centered_second =
                (second_moment.get(j, k) - first_moment[j] / risk_sum * first_moment[k]) / risk_sum;
            hessian.add(
                j,
                k,
                -f64::from(event_count) * centered_second * weight_j * weight_k,
            );

            if ifirth != 0 {
                let derivative_tail = derivative_hessian.tail_mut(j, k);
                for l in k..p {
                    let centered_third = centered_third_moment(
                        third_moment.get(j, k, l),
                        j,
                        k,
                        l,
                        risk_sum,
                        first_moment,
                        second_moment,
                    );
                    derivative_tail[l - k] +=
                        -f64::from(event_count) * centered_third * weight_j * weight_k * weights[l];
                }
            }
        }
    }
}

fn centered_third_moment(
    raw_third: f64,
    j: usize,
    k: usize,
    l: usize,
    risk_sum: f64,
    first_moment: &[f64],
    second_moment: &SymmetricMatrix,
) -> f64 {
    (raw_third
        - second_moment.get(k, l) * first_moment[j] / risk_sum
        - first_moment[l]
            * (second_moment.get(k, j) - first_moment[k] * first_moment[j] / risk_sum)
            / risk_sum
        - first_moment[k]
            * (second_moment.get(l, j) - first_moment[l] * first_moment[j] / risk_sum)
            / risk_sum)
        / risk_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_and_dense_moment_updates_are_identical() {
        let x = [1.5, -0.5, 0.0];
        let mut dense_risk = 0.0;
        let mut sparse_risk = 0.0;
        let mut dense_first = vec![0.0; 3];
        let mut sparse_first = vec![0.0; 3];
        let mut dense_second = SymmetricMatrix::zeros(3);
        let mut sparse_second = SymmetricMatrix::zeros(3);
        let mut dense_third = SymmetricCube::zeros(3);
        let mut sparse_third = SymmetricCube::zeros(3);
        let mut pattern_risk = 0.0;
        let mut pattern_first = vec![0.0; 3];
        let mut pattern_second = SymmetricMatrix::zeros(3);
        let mut pattern_third = SymmetricCube::zeros(3);

        update_risk_moments(
            &x,
            1.0,
            1.0,
            1,
            false,
            &mut dense_risk,
            &mut dense_first,
            &mut dense_second,
            &mut dense_third,
        );
        update_risk_moments(
            &x,
            1.0,
            1.0,
            1,
            true,
            &mut sparse_risk,
            &mut sparse_first,
            &mut sparse_second,
            &mut sparse_third,
        );

        let mut patterns = Matrix::zeros(1, 3);
        patterns.row_mut(0).copy_from_slice(&x);
        let mut prepared = Matrix::zeros(1, moment_count(3));
        let coefficients = [0.0; 3];
        prepare_pattern_moments(&patterns, &coefficients, 1, &mut prepared);
        update_precomputed_moments(
            prepared.row(0),
            1.0,
            1,
            &mut pattern_risk,
            &mut pattern_first,
            &mut pattern_second,
            &mut pattern_third,
        );

        assert_eq!(dense_risk, sparse_risk);
        assert_eq!(dense_risk, pattern_risk);
        assert_eq!(dense_first, sparse_first);
        assert_eq!(dense_first, pattern_first);
        for j in 0..3 {
            for k in j..3 {
                assert_eq!(dense_second.get(j, k), sparse_second.get(j, k));
                assert_eq!(dense_second.get(j, k), pattern_second.get(j, k));
                for l in k..3 {
                    assert_eq!(dense_third.get(j, k, l), sparse_third.get(j, k, l));
                    assert_eq!(dense_third.get(j, k, l), pattern_third.get(j, k, l));
                }
            }
        }
    }
}
