#![allow(clippy::needless_range_loop)]

use crate::matrix::{determinant_with_workspace, invert_into, Cube, Matrix};
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
    pub(crate) hessian: Matrix,
    derivative_hessian: Cube,
    x_all: Matrix,
    mask: Vec<bool>,
    bresx_all: Vec<f64>,
    linear_predictor: Vec<f64>,
    risk: Vec<f64>,
    first_moment: Vec<f64>,
    second_moment: Matrix,
    third_moment: Option<Cube>,
    information: Matrix,
    information_inverse: Matrix,
    determinant_workspace: Matrix,
    inverse_workspace: Vec<usize>,
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

        let mut x_all = Matrix::zeros(n, p);
        for row in 0..n {
            x_all.row_mut(row)[..data.p].copy_from_slice(data.x.row(row));
        }

        Self {
            mode,
            score: vec![0.0; p],
            hessian: Matrix::zeros(p, p),
            derivative_hessian: Cube::zeros(p),
            x_all,
            mask: vec![false; n],
            bresx_all: vec![0.0; p],
            linear_predictor: vec![0.0; n],
            risk: vec![0.0; n],
            first_moment: vec![0.0; p],
            second_moment: Matrix::zeros(p, p),
            third_moment: if data.ntde == 0 {
                Some(Cube::zeros(p))
            } else {
                None
            },
            information: Matrix::zeros(p, p),
            information_inverse: Matrix::zeros(p, p),
            determinant_workspace: Matrix::zeros(p, p),
            inverse_workspace: vec![0; p],
        }
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
    workspace.derivative_hessian.fill(0.0);

    let mode = workspace.mode;
    let LikeWorkspace {
        score,
        hessian,
        derivative_hessian,
        x_all,
        mask,
        bresx_all,
        linear_predictor,
        risk,
        first_moment,
        second_moment,
        third_moment,
        information,
        information_inverse,
        determinant_workspace,
        inverse_workspace,
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
            linear_predictor,
            risk,
            first_moment,
            second_moment,
        ),
        LikelihoodMode::IntervalFast => evaluate_interval_fast(
            data,
            coefficients,
            ifirth,
            ngv,
            x_all,
            &mut loglik,
            score,
            hessian,
            derivative_hessian,
            bresx_all,
            linear_predictor,
            risk,
            first_moment,
            second_moment,
            third_moment
                .as_mut()
                .expect("interval mode requires third-moment workspace"),
        ),
        LikelihoodMode::Fast => evaluate_fast(
            data,
            coefficients,
            ifirth,
            ngv,
            x_all,
            &mut loglik,
            score,
            hessian,
            derivative_hessian,
            linear_predictor,
            risk,
            first_moment,
            second_moment,
            third_moment
                .as_mut()
                .expect("fast mode requires third-moment workspace"),
        ),
    }

    let mut jcode = initial_jcode;
    if ifirth != 0 {
        information.copy_negated_from(hessian);
        invert_into(information, information_inverse, inverse_workspace);

        for j in 0..p {
            let mut trace = 0.0;
            for k in 0..p {
                let information_row = information_inverse.row(k);
                for l in 0..p {
                    trace -= information_row[l] * derivative_hessian.line(j, l)[k];
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
    hessian: &mut Matrix,
    derivative_hessian: &mut Cube,
    mask: &mut [bool],
    bresx_all: &mut [f64],
    linear_predictor: &mut [f64],
    risk: &mut [f64],
    first_moment: &mut [f64],
    second_moment: &mut Matrix,
) {
    let n = data.n;
    let p = data.p_total;

    for event_row in 0..n {
        if data.ibresc[event_row] == 0 {
            continue;
        }

        let event_time = data.t2[event_row] - 0.00001;
        for row in 0..n {
            mask[row] = data.t1[row] < event_time && data.t2[row] >= event_time;
        }

        for j in 0..data.p {
            bresx_all[j] = data.bresx.get(event_row, j);
        }
        for j in data.p..p {
            let time_column = j - data.p;
            let source_column = data.ftmap[time_column];
            let time_value = data.ft.get(event_row, time_column);
            for row in 0..n {
                x_all.set(row, j, data.x.get(row, source_column) * time_value);
            }
            bresx_all[j] = time_value * data.bresx.get(event_row, source_column);
        }

        for row in 0..n {
            linear_predictor[row] = x_all.row_dot(row, coefficients);
            risk[row] = linear_predictor[row].exp();
        }
        let mut risk_sum = 0.0;
        for row in 0..n {
            if mask[row] {
                risk_sum += risk[row];
            }
        }

        first_moment.fill(0.0);
        second_moment.fill(0.0);
        for row in 0..n {
            if mask[row] {
                let x_row = x_all.row(row);
                for j in 0..p {
                    let weighted_x = x_row[j] * risk[row];
                    first_moment[j] += weighted_x;
                    let second_row = second_moment.row_mut(j);
                    for k in 0..p {
                        second_row[k] += weighted_x * x_row[k];
                    }
                }
            }
        }

        let log_risk_sum = risk_sum.max(LOWEST).ln();
        let mut numerator = 0.0;
        for j in 0..p {
            numerator += bresx_all[j] * coefficients[j];
        }
        let contribution = numerator - f64::from(data.ibresc[event_row]) * log_risk_sum;
        if ngv == p as i32 {
            *loglik += contribution * data.score_weights.get(event_row, 0);
        } else {
            *loglik += contribution;
        }

        let weights = data.score_weights.row(event_row);
        for j in 0..p {
            let weight_j = weights[j];
            score[j] += (bresx_all[j]
                - f64::from(data.ibresc[event_row]) * first_moment[j] / risk_sum)
                * weight_j;

            let hessian_row = hessian.row_mut(j);
            for k in 0..p {
                let weight_k = weights[k];
                let centered_second = (second_moment.get(j, k)
                    - first_moment[j] / risk_sum * first_moment[k])
                    / risk_sum;
                hessian_row[k] +=
                    -f64::from(data.ibresc[event_row]) * centered_second * weight_j * weight_k;

                if ifirth != 0 {
                    let derivative_line = derivative_hessian.line_mut(j, k);
                    for l in 0..p {
                        let mut raw_third = 0.0;
                        for row in 0..n {
                            if mask[row] {
                                let x_row = x_all.row(row);
                                raw_third += x_row[l] * x_row[k] * x_row[j] * risk[row];
                            }
                        }
                        let centered_third = centered_third_moment(
                            raw_third,
                            j,
                            k,
                            l,
                            risk_sum,
                            first_moment,
                            second_moment,
                        );
                        derivative_line[l] += -f64::from(data.ibresc[event_row])
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
    hessian: &mut Matrix,
    derivative_hessian: &mut Cube,
    bresx_all: &mut [f64],
    linear_predictor: &mut [f64],
    risk: &mut [f64],
    first_moment: &mut [f64],
    second_moment: &mut Matrix,
    third_moment: &mut Cube,
) {
    let n = data.n;
    first_moment.fill(0.0);
    second_moment.fill(0.0);
    third_moment.fill(0.0);
    for row in 0..n {
        linear_predictor[row] = x_all.row_dot(row, coefficients);
        risk[row] = linear_predictor[row].exp();
    }

    let mut risk_sum = 0.0;
    let mut add_row = n as isize - 1;
    let mut start_position = 0usize;

    for event_row in (0..n).rev() {
        if data.ibresc[event_row] == 0 {
            continue;
        }
        let event_time = data.t2[event_row] - 0.00001;

        while add_row >= 0 {
            let row = add_row as usize;
            if data.t2[row] < event_time {
                break;
            }
            update_risk_moments(
                row,
                1.0,
                x_all,
                risk[row],
                ifirth,
                &mut risk_sum,
                first_moment,
                second_moment,
                third_moment,
            );
            add_row -= 1;
        }

        while start_position < n {
            let row = data.start_order[start_position];
            if data.t1[row] < event_time {
                break;
            }
            update_risk_moments(
                row,
                -1.0,
                x_all,
                risk[row],
                ifirth,
                &mut risk_sum,
                first_moment,
                second_moment,
                third_moment,
            );
            start_position += 1;
        }

        for j in 0..data.p {
            bresx_all[j] = data.bresx.get(event_row, j);
        }
        accumulate_aggregated_event(
            data,
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
    hessian: &mut Matrix,
    derivative_hessian: &mut Cube,
    linear_predictor: &mut [f64],
    risk: &mut [f64],
    first_moment: &mut [f64],
    second_moment: &mut Matrix,
    third_moment: &mut Cube,
) {
    let n = data.n;
    let p = data.p_total;
    first_moment.fill(0.0);
    second_moment.fill(0.0);
    third_moment.fill(0.0);
    for row in 0..n {
        linear_predictor[row] = x_all.row_dot(row, coefficients);
        risk[row] = linear_predictor[row].exp();
    }

    let mut risk_sum = 0.0;
    let mut tied_events = 0i32;
    for row in (0..n).rev() {
        let x_row = x_all.row(row);
        let weights = data.score_weights.row(row);
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

        update_risk_moments(
            row,
            1.0,
            x_all,
            risk[row],
            ifirth,
            &mut risk_sum,
            first_moment,
            second_moment,
            third_moment,
        );

        let contribution = linear_predictor[row] * f64::from(data.ic[row])
            - f64::from(current_events) * risk_sum.max(LOWEST).ln();
        if ngv == p as i32 {
            *loglik += contribution * data.score_weights.get(row, 0);
        } else {
            *loglik += contribution;
        }

        for j in 0..p {
            let weight_j = weights[j];
            score[j] += (x_row[j] * f64::from(data.ic[row])
                - f64::from(current_events) * first_moment[j] / risk_sum)
                * weight_j;
            let hessian_row = hessian.row_mut(j);
            for k in 0..p {
                let weight_k = weights[k];
                let centered_second = (second_moment.get(j, k)
                    - first_moment[j] / risk_sum * first_moment[k])
                    / risk_sum;
                hessian_row[k] +=
                    -f64::from(current_events) * centered_second * weight_j * weight_k;

                if ifirth != 0 {
                    let third_line = third_moment.line(j, k);
                    let derivative_line = derivative_hessian.line_mut(j, k);
                    for l in 0..p {
                        let centered_third = centered_third_moment(
                            third_line[l],
                            j,
                            k,
                            l,
                            risk_sum,
                            first_moment,
                            second_moment,
                        );
                        derivative_line[l] += -f64::from(current_events)
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
fn update_risk_moments(
    row: usize,
    direction: f64,
    x: &Matrix,
    row_risk: f64,
    ifirth: i32,
    risk_sum: &mut f64,
    first_moment: &mut [f64],
    second_moment: &mut Matrix,
    third_moment: &mut Cube,
) {
    let p = first_moment.len();
    *risk_sum += direction * row_risk;
    let x_row = x.row(row);
    for j in 0..p {
        let first = x_row[j] * row_risk;
        first_moment[j] += direction * first;
        let second_row = second_moment.row_mut(j);
        for k in 0..p {
            let second = first * x_row[k];
            second_row[k] += direction * second;
            if ifirth == 1 {
                let third_line = third_moment.line_mut(j, k);
                for l in 0..p {
                    third_line[l] += direction * second * x_row[l];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_aggregated_event(
    data: &NativeData,
    event_row: usize,
    coefficients: &[f64],
    ngv: i32,
    ifirth: i32,
    bresx_all: &[f64],
    risk_sum: f64,
    first_moment: &[f64],
    second_moment: &Matrix,
    third_moment: &Cube,
    loglik: &mut f64,
    score: &mut [f64],
    hessian: &mut Matrix,
    derivative_hessian: &mut Cube,
) {
    let p = data.p_total;
    let mut numerator = 0.0;
    for j in 0..p {
        numerator += bresx_all[j] * coefficients[j];
    }
    let contribution = numerator - f64::from(data.ibresc[event_row]) * risk_sum.max(LOWEST).ln();
    if ngv == p as i32 {
        *loglik += contribution * data.score_weights.get(event_row, 0);
    } else {
        *loglik += contribution;
    }

    let weights = data.score_weights.row(event_row);
    for j in 0..p {
        let weight_j = weights[j];
        score[j] += (bresx_all[j] - f64::from(data.ibresc[event_row]) * first_moment[j] / risk_sum)
            * weight_j;
        let hessian_row = hessian.row_mut(j);
        for k in 0..p {
            let weight_k = weights[k];
            let centered_second =
                (second_moment.get(j, k) - first_moment[j] / risk_sum * first_moment[k]) / risk_sum;
            hessian_row[k] +=
                -f64::from(data.ibresc[event_row]) * centered_second * weight_j * weight_k;

            if ifirth != 0 {
                let third_line = third_moment.line(j, k);
                let derivative_line = derivative_hessian.line_mut(j, k);
                for l in 0..p {
                    let centered_third = centered_third_moment(
                        third_line[l],
                        j,
                        k,
                        l,
                        risk_sum,
                        first_moment,
                        second_moment,
                    );
                    derivative_line[l] += -f64::from(data.ibresc[event_row])
                        * centered_third
                        * weight_j
                        * weight_k
                        * weights[l];
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
    second_moment: &Matrix,
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
