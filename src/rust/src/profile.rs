#![allow(clippy::needless_range_loop)]

use crate::fit::fit;
use crate::likelihood::{evaluate, LikeWorkspace};
use crate::matrix::{invert_into, Matrix};
use crate::native_data::NativeData;

pub(crate) fn profile(data: &NativeData, parms: &mut [f64], ioarray: &mut [f64]) {
    let p = data.p_total;
    let io_nrow = 9usize;
    debug_assert_eq!(ioarray.len(), io_nrow * p);

    let ifirth = parms[2] as i32;
    let maxit = parms[3] as i32;
    let maxhs = parms[4] as i32;
    let step = parms[5];
    let convergence = parms[6];
    let chi = parms[7];
    let gradient_convergence = parms[8];
    let ngv = parms[12] as i32;
    let penalty = parms[14];
    parms[8] = 0.0;

    let mut flags = vec![0i32; p];
    let mut coefficients = vec![0.0; p];
    for j in 0..p {
        flags[j] = ioarray[io_nrow * j] as i32;
        coefficients[j] = ioarray[2 + io_nrow * j];
    }

    let mut like_workspace = LikeWorkspace::new(data);
    let initial = evaluate(
        data,
        &coefficients,
        ifirth,
        ngv,
        penalty,
        0,
        &mut like_workspace,
    );
    let maximum_loglik = initial.loglik;
    let target_loglik = maximum_loglik - 0.5 * chi;
    let initial_score = like_workspace.score.clone();
    let initial_hessian = like_workspace.hessian.clone();
    let mut factor_input = Matrix::zeros(p, p);
    let mut initial_variance = Matrix::zeros(p, p);
    let mut variance = Matrix::zeros(p, p);
    let mut inverse_workspace = vec![0usize; p];
    factor_input.copy_negated_from_symmetric(&initial_hessian);
    invert_into(&factor_input, &mut initial_variance, &mut inverse_workspace);
    let saved_coefficients = coefficients.clone();

    let mut confidence_limits = Matrix::zeros(p, 2);
    let mut unit = vec![0.0; p];
    let mut score = vec![0.0; p];
    for coefficient_index in 0..p {
        if flags[coefficient_index] != 1 {
            continue;
        }

        for direction in 0..2 {
            unit.fill(0.0);
            unit[coefficient_index] = 1.0;
            let direction_sign = if direction == 0 { -1.0 } else { 1.0 };
            let mut lambda = 0.0;
            variance.copy_from(&initial_variance);
            for j in 0..p {
                score[j] = initial_score[j] * f64::from(flags[j]);
            }
            coefficients.clone_from(&saved_coefficients);
            let mut converged = false;
            let mut iteration = 0i32;

            while !converged {
                iteration += 1;
                for row in 0..p {
                    let mut change = 0.0;
                    for col in 0..p {
                        change += (score[col] + lambda * unit[col]) * variance.get(row, col);
                    }
                    if change.abs() > step {
                        change = step.copysign(change);
                    }
                    coefficients[row] += change;
                }

                let result = evaluate(
                    data,
                    &coefficients,
                    ifirth,
                    ngv,
                    penalty,
                    0,
                    &mut like_workspace,
                );
                let loglik = result.loglik;
                for j in 0..p {
                    score[j] = like_workspace.score[j] * f64::from(flags[j]);
                }
                factor_input.copy_negated_from_symmetric(&like_workspace.hessian);
                invert_into(&factor_input, &mut variance, &mut inverse_workspace);

                let mut gradient_variance_gradient = 0.0;
                for row in 0..p {
                    for col in 0..p {
                        gradient_variance_gradient += score[row]
                            * score[col]
                            * variance.get(row, col)
                            * f64::from(flags[row])
                            * f64::from(flags[col]);
                    }
                }
                lambda = direction_sign
                    * (-(2.0 * (target_loglik - loglik - 0.5 * gradient_variance_gradient)
                        / variance.get(coefficient_index, coefficient_index)))
                    .sqrt();

                if (target_loglik - loglik).abs() <= convergence {
                    converged = true;
                }

                let mut constrained_gradient = 0.0;
                for row in 0..p {
                    for col in 0..p {
                        constrained_gradient += (score[row] + lambda * unit[row])
                            * variance.get(row, col)
                            * (score[col] + lambda * unit[col])
                            * f64::from(flags[row])
                            * f64::from(flags[col]);
                    }
                }
                if constrained_gradient > convergence {
                    converged = false;
                }
                if iteration == maxit {
                    converged = true;
                }
            }

            confidence_limits.set(
                coefficient_index,
                direction,
                coefficients[coefficient_index],
            );
            ioarray[6 + direction + io_nrow * coefficient_index] = f64::from(iteration);
        }
    }

    let mut likelihood_ratio = vec![0.0; p];
    for coefficient_index in 0..p {
        if flags[coefficient_index] == 1 {
            let mut restricted_parms = [0.0; 15];
            restricted_parms[0] = data.n as f64;
            restricted_parms[1] = data.p as f64;
            restricted_parms[2] = f64::from(ifirth);
            restricted_parms[3] = f64::from(maxit);
            restricted_parms[4] = f64::from(maxhs);
            restricted_parms[5] = step;
            restricted_parms[6] = convergence;
            restricted_parms[7] = 1.0;
            restricted_parms[8] = gradient_convergence;
            restricted_parms[12] = f64::from(ngv);
            restricted_parms[13] = data.ntde as f64;
            restricted_parms[14] = penalty;

            let restricted_nrow = 3 + p;
            let mut restricted_io = vec![0.0; restricted_nrow * p];
            for j in 0..p {
                restricted_io[restricted_nrow * j] = f64::from(flags[j]);
                restricted_io[1 + restricted_nrow * j] =
                    saved_coefficients[j] * f64::from(flags[j]);
            }
            restricted_io[restricted_nrow * coefficient_index] = 0.0;
            restricted_io[1 + restricted_nrow * coefficient_index] = 0.0;
            for j in 0..data.ntde {
                restricted_io[3 + restricted_nrow * (data.p + j)] = (data.ftmap[j] + 1) as f64;
            }

            fit(data, &mut restricted_parms, &mut restricted_io);
            if restricted_parms[7] >= 1.0 {
                parms[8] = 2.0;
            }
            likelihood_ratio[coefficient_index] = 2.0 * (maximum_loglik - restricted_parms[10]);
            ioarray[8 + io_nrow * coefficient_index] = restricted_parms[9];
        } else {
            likelihood_ratio[coefficient_index] = 0.0;
            ioarray[8 + io_nrow * coefficient_index] = 0.0;
        }
    }

    for j in 0..p {
        ioarray[3 + io_nrow * j] = confidence_limits.get(j, 0);
        ioarray[4 + io_nrow * j] = confidence_limits.get(j, 1);
        ioarray[5 + io_nrow * j] = likelihood_ratio[j];
    }
}
