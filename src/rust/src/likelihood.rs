#![allow(clippy::needless_range_loop)]

use crate::matrix::{determinant, invert, Cube, Matrix};
use crate::native_data::NativeData;

const LOWEST: f64 = 1.0e-44;

pub(crate) struct LikeResult {
    pub(crate) loglik: f64,
    pub(crate) score: Vec<f64>,
    pub(crate) hessian: Matrix,
    pub(crate) jcode: i32,
}

pub(crate) fn evaluate(
    data: &NativeData,
    coefficients: &[f64],
    ifirth: i32,
    ngv: i32,
    penalty: f64,
    initial_jcode: i32,
) -> LikeResult {
    let n = data.n;
    let p = data.p_total;
    let mut loglik = 0.0;
    let mut score = vec![0.0; p];
    let mut hessian = Matrix::zeros(p, p);
    let mut derivative_hessian = Cube::zeros(p);
    let mut x_all = Matrix::zeros(n, p);
    for j in 0..data.p {
        for i in 0..n {
            x_all.set(i, j, data.x.get(i, j));
        }
    }

    let fast_mode = data.ntde == 0
        && data.t1.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            < data.t2.iter().copied().fold(f64::INFINITY, f64::min);
    let interval_fast = data.ntde == 0 && !fast_mode;

    if !fast_mode && !interval_fast {
        evaluate_general(
            data,
            coefficients,
            ifirth,
            ngv,
            &mut x_all,
            &mut loglik,
            &mut score,
            &mut hessian,
            &mut derivative_hessian,
        );
    }

    if interval_fast {
        evaluate_interval_fast(
            data,
            coefficients,
            ifirth,
            ngv,
            &x_all,
            &mut loglik,
            &mut score,
            &mut hessian,
            &mut derivative_hessian,
        );
    }

    if fast_mode {
        evaluate_fast(
            data,
            coefficients,
            ifirth,
            ngv,
            &x_all,
            &mut loglik,
            &mut score,
            &mut hessian,
            &mut derivative_hessian,
        );
    }

    let mut jcode = initial_jcode;
    if ifirth != 0 {
        let mut information = Matrix::zeros(p, p);
        for j in 0..p {
            for k in 0..p {
                information.set(j, k, -hessian.get(j, k));
            }
        }
        let information_inverse = invert(&information);

        for j in 0..p {
            let mut trace = 0.0;
            for k in 0..p {
                for l in 0..p {
                    trace -= information_inverse.get(k, l) * derivative_hessian.get(j, l, k);
                }
            }
            score[j] += trace * penalty;
        }

        let mut det = if p == 1 {
            information.get(0, 0)
        } else {
            determinant(&information)
        };
        if det < LOWEST {
            det = LOWEST;
            jcode = 2;
        }
        loglik += penalty * det.ln();
    }

    LikeResult {
        loglik,
        score,
        hessian,
        jcode,
    }
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
) {
    let n = data.n;
    let p = data.p_total;
    let mut mask = vec![false; n];
    let mut bresx_all = vec![0.0; p];
    let mut linear_predictor = vec![0.0; n];
    let mut risk = vec![0.0; n];
    let mut first_moment = vec![0.0; p];
    let mut second_moment = Matrix::zeros(p, p);

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

        for j in 0..p {
            let mut first = 0.0;
            for row in 0..n {
                if mask[row] {
                    first += x_all.get(row, j) * risk[row];
                }
            }
            first_moment[j] = first;

            for k in 0..p {
                let mut second = 0.0;
                for row in 0..n {
                    if mask[row] {
                        second += x_all.get(row, j) * x_all.get(row, k) * risk[row];
                    }
                }
                second_moment.set(j, k, second);
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

        for j in 0..p {
            let weight_j = data.score_weights.get(event_row, j);
            score[j] += (bresx_all[j]
                - f64::from(data.ibresc[event_row]) * first_moment[j] / risk_sum)
                * weight_j;

            for k in 0..p {
                let weight_k = data.score_weights.get(event_row, k);
                let centered_second = (second_moment.get(j, k)
                    - first_moment[j] / risk_sum * first_moment[k])
                    / risk_sum;
                hessian.add(
                    j,
                    k,
                    -f64::from(data.ibresc[event_row]) * centered_second * weight_j * weight_k,
                );

                if ifirth != 0 {
                    for l in 0..p {
                        let mut raw_third = 0.0;
                        for row in 0..n {
                            if mask[row] {
                                raw_third += x_all.get(row, l)
                                    * x_all.get(row, k)
                                    * x_all.get(row, j)
                                    * risk[row];
                            }
                        }
                        let centered_third = centered_third_moment(
                            raw_third,
                            j,
                            k,
                            l,
                            risk_sum,
                            &first_moment,
                            &second_moment,
                        );
                        derivative_hessian.add(
                            j,
                            k,
                            l,
                            -f64::from(data.ibresc[event_row])
                                * centered_third
                                * weight_j
                                * weight_k
                                * data.score_weights.get(event_row, l),
                        );
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
) {
    let n = data.n;
    let p = data.p_total;
    let mut first_moment = vec![0.0; p];
    let mut second_moment = Matrix::zeros(p, p);
    let mut third_moment = Cube::zeros(p);
    let mut linear_predictor = vec![0.0; n];
    let mut risk = vec![0.0; n];
    for row in 0..n {
        linear_predictor[row] = x_all.row_dot(row, coefficients);
        risk[row] = linear_predictor[row].exp();
    }

    let mut risk_sum = 0.0;
    let mut add_row = n as isize - 1;
    let mut start_position = 0usize;
    let mut bresx_all = vec![0.0; p];

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
                &mut first_moment,
                &mut second_moment,
                &mut third_moment,
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
                &mut first_moment,
                &mut second_moment,
                &mut third_moment,
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
            &bresx_all,
            risk_sum,
            &first_moment,
            &second_moment,
            &third_moment,
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
) {
    let n = data.n;
    let p = data.p_total;
    let mut first_moment = vec![0.0; p];
    let mut second_moment = Matrix::zeros(p, p);
    let mut third_moment = Cube::zeros(p);
    let mut linear_predictor = vec![0.0; n];
    let mut risk = vec![0.0; n];
    for row in 0..n {
        linear_predictor[row] = x_all.row_dot(row, coefficients);
        risk[row] = linear_predictor[row].exp();
    }

    let mut risk_sum = 0.0;
    let mut tied_events = 0i32;
    for row in (0..n).rev() {
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
            &mut first_moment,
            &mut second_moment,
            &mut third_moment,
        );

        let contribution = linear_predictor[row] * f64::from(data.ic[row])
            - f64::from(current_events) * risk_sum.max(LOWEST).ln();
        if ngv == p as i32 {
            *loglik += contribution * data.score_weights.get(row, 0);
        } else {
            *loglik += contribution;
        }

        for j in 0..p {
            let weight_j = data.score_weights.get(row, j);
            score[j] += (x_all.get(row, j) * f64::from(data.ic[row])
                - f64::from(current_events) * first_moment[j] / risk_sum)
                * weight_j;
            for k in 0..p {
                let weight_k = data.score_weights.get(row, k);
                let centered_second = (second_moment.get(j, k)
                    - first_moment[j] / risk_sum * first_moment[k])
                    / risk_sum;
                hessian.add(
                    j,
                    k,
                    -f64::from(current_events) * centered_second * weight_j * weight_k,
                );

                if ifirth != 0 {
                    for l in 0..p {
                        let centered_third = centered_third_moment(
                            third_moment.get(j, k, l),
                            j,
                            k,
                            l,
                            risk_sum,
                            &first_moment,
                            &second_moment,
                        );
                        derivative_hessian.add(
                            j,
                            k,
                            l,
                            -f64::from(current_events)
                                * centered_third
                                * weight_j
                                * weight_k
                                * data.score_weights.get(row, l),
                        );
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
    for j in 0..p {
        let first = x.get(row, j) * row_risk;
        first_moment[j] += direction * first;
        for k in 0..p {
            let second = first * x.get(row, k);
            second_moment.add(j, k, direction * second);
            if ifirth == 1 {
                for l in 0..p {
                    third_moment.add(j, k, l, direction * second * x.get(row, l));
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

    for j in 0..p {
        let weight_j = data.score_weights.get(event_row, j);
        score[j] += (bresx_all[j] - f64::from(data.ibresc[event_row]) * first_moment[j] / risk_sum)
            * weight_j;
        for k in 0..p {
            let weight_k = data.score_weights.get(event_row, k);
            let centered_second =
                (second_moment.get(j, k) - first_moment[j] / risk_sum * first_moment[k]) / risk_sum;
            hessian.add(
                j,
                k,
                -f64::from(data.ibresc[event_row]) * centered_second * weight_j * weight_k,
            );

            if ifirth != 0 {
                for l in 0..p {
                    let centered_third = centered_third_moment(
                        third_moment.get(j, k, l),
                        j,
                        k,
                        l,
                        risk_sum,
                        first_moment,
                        second_moment,
                    );
                    derivative_hessian.add(
                        j,
                        k,
                        l,
                        -f64::from(data.ibresc[event_row])
                            * centered_third
                            * weight_j
                            * weight_k
                            * data.score_weights.get(event_row, l),
                    );
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
