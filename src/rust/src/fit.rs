use crate::likelihood::{evaluate, LikeResult, LikeWorkspace};
use crate::matrix::{invert_into, Matrix};
use crate::native_data::NativeData;

pub(crate) fn fit(data: &NativeData, parms: &mut [f64], ioarray: &mut [f64]) {
    let mut like_workspace = LikeWorkspace::new(data);
    fit_with_workspace(data, parms, ioarray, &mut like_workspace);
}

pub(crate) fn fit_with_workspace(
    data: &NativeData,
    parms: &mut [f64],
    ioarray: &mut [f64],
    like_workspace: &mut LikeWorkspace,
) {
    let p = data.p_total;
    let io_nrow = 3 + p;
    debug_assert_eq!(ioarray.len(), io_nrow * p);
    like_workspace.reset_derivatives();

    let ifirth = parms[2] as i32;
    let maxit = parms[3] as i32;
    let maxhs = parms[4] as i32;
    let step = parms[5];
    let convergence = parms[6];
    let mut relative_ridge = parms[7];
    let gradient_convergence = parms[8];
    let ngv = parms[12] as i32;
    let penalty = parms[14];
    if relative_ridge < 1.0 {
        relative_ridge = 1.0;
    }

    parms[9] = -11.0;
    let mut offset = vec![0.0; p];
    let mut flags = vec![0i32; p];
    for j in 0..p {
        flags[j] = ioarray[io_nrow * j] as i32;
        offset[j] = ioarray[1 + io_nrow * j];
    }

    let mut coefficients = vec![0.0; p];
    for j in 0..p {
        if flags[j] == 0 {
            coefficients[j] = offset[j];
        } else if flags[j] == 2 {
            coefficients[j] = offset[j];
            flags[j] = 1;
        }
    }
    let mut previous_coefficients = coefficients.clone();
    let flag_sum: i32 = flags.iter().sum();

    let mut iteration = 0i32;
    let mut converged = false;
    let mut jcode = 0i32;
    let mut separation = 0i32;
    let mut loglik = 0.0;
    let mut previous_loglik;
    let mut working = Matrix::zeros(p, p);
    let mut variance = Matrix::zeros(p, p);
    let mut inverse_workspace = vec![0usize; p];
    let mut likelihood_calls = 0i32;

    while !converged && iteration < maxit {
        iteration += 1;
        previous_coefficients.clone_from(&coefficients);
        previous_loglik = loglik;
        parms[9] = -10.0;

        if iteration == 1 {
            let result = evaluate(
                data,
                &coefficients,
                ifirth,
                ngv,
                penalty,
                jcode,
                like_workspace,
            );
            assign_like_result(result, &mut loglik, &mut jcode);
            likelihood_calls += 1;
        }

        parms[9] = -9.0;
        parms[7] = f64::from(jcode);
        if jcode >= 1 {
            return;
        }

        working.copy_negated_from_symmetric(&like_workspace.hessian);
        invert_into(&working, &mut variance, &mut inverse_workspace);
        if iteration == 1 {
            parms[11] = loglik;
            parms[6] = quadratic_form(&variance, &like_workspace.score, None);
        }
        parms[10] = loglik;
        parms[9] = f64::from(iteration);
        parms[8] = f64::from(separation);

        if flag_sum != 0 {
            for i in 0..p {
                if flags[i] == 1 {
                    let mut change = 0.0;
                    let variance_row = variance.row(i);
                    for (j, &flag) in flags.iter().enumerate() {
                        change += variance_row[j] * like_workspace.score[j] * f64::from(flag);
                    }
                    if change.abs() > step {
                        change = change / change.abs() * step;
                    }
                    coefficients[i] += change;
                }
            }

            let result = evaluate(
                data,
                &coefficients,
                ifirth,
                ngv,
                penalty,
                jcode,
                like_workspace,
            );
            assign_like_result(result, &mut loglik, &mut jcode);
            likelihood_calls += 1;
            working.copy_negated_from_symmetric(&like_workspace.hessian);

            let mut half_steps = 0i32;
            while loglik <= previous_loglik
                && iteration != 1
                && half_steps <= maxhs
                && (ngv == p as i32 || ngv == 0)
            {
                half_steps += 1;
                if (relative_ridge - 1.0).abs() < 0.0001 {
                    for i in 0..p {
                        if flags[i] == 1 {
                            coefficients[i] = (coefficients[i] + previous_coefficients[i]) / 2.0;
                        }
                    }
                } else {
                    coefficients.clone_from(&previous_coefficients);
                    for i in 0..p {
                        working.set(i, i, working.get(i, i) * relative_ridge);
                    }
                    invert_into(&working, &mut variance, &mut inverse_workspace);
                    for i in 0..p {
                        if flags[i] == 1 {
                            let mut change = 0.0;
                            let variance_row = variance.row(i);
                            for (j, &flag) in flags.iter().enumerate() {
                                change +=
                                    variance_row[j] * like_workspace.score[j] * f64::from(flag);
                            }
                            if change.abs() > step {
                                change = change / change.abs() * step;
                            }
                            coefficients[i] += change;
                        }
                    }
                }

                let result = evaluate(
                    data,
                    &coefficients,
                    ifirth,
                    ngv,
                    penalty,
                    jcode,
                    like_workspace,
                );
                assign_like_result(result, &mut loglik, &mut jcode);
                likelihood_calls += 1;
            }
        }

        converged = true;
        if flag_sum > 0 {
            for i in 0..p {
                if (coefficients[i] - previous_coefficients[i]).abs() > convergence {
                    converged = false;
                }
                if like_workspace.score[i].abs() * f64::from(flags[i]) > gradient_convergence {
                    converged = false;
                }
            }
        }
    }

    working.copy_negated_from_symmetric(&like_workspace.hessian);
    invert_into(&working, &mut variance, &mut inverse_workspace);
    for j in 0..p {
        ioarray[2 + io_nrow * j] = coefficients[j];
        for i in 0..p {
            ioarray[3 + i + io_nrow * j] = variance.get(i, j);
        }
    }

    for i in 0..p {
        if like_workspace.hessian.get(i, i).abs() < 0.0001 {
            separation = 1;
        }
    }
    let _final_separation = separation;

    parms[8] = quadratic_form(&variance, &like_workspace.score, Some(&flags));
    parms[6] = like_workspace
        .score
        .iter()
        .zip(&flags)
        .map(|(&value, &flag)| value.abs() * f64::from(flag))
        .fold(0.0, f64::max);
    parms[7] = f64::from(jcode);
    parms[10] = loglik;
    parms[9] = f64::from(iteration);
    parms[3] = f64::from(likelihood_calls);
}

fn assign_like_result(result: LikeResult, loglik: &mut f64, jcode: &mut i32) {
    *loglik = result.loglik;
    *jcode = result.jcode;
}

pub(crate) fn quadratic_form(matrix: &Matrix, vector: &[f64], flags: Option<&[i32]>) -> f64 {
    let mut value = 0.0;
    for i in 0..vector.len() {
        let left = vector[i] * flags.map_or(1.0, |items| f64::from(items[i]));
        for j in 0..vector.len() {
            let right = vector[j] * flags.map_or(1.0, |items| f64::from(items[j]));
            value += left * matrix.get(i, j) * right;
        }
    }
    value
}
