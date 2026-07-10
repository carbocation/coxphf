library(coxphf)
library(survival)

fit_with_backend <- function(backend, ...) {
  old <- options(coxphf.native_backend = backend)
  on.exit(options(old), add = TRUE)
  coxphf(...)
}

test_with_backend <- function(backend, ...) {
  old <- options(coxphf.native_backend = backend)
  on.exit(options(old), add = TRUE)
  coxphftest(...)
}

fit_values <- function(fit) {
  unlist(c(
    coefficients = fit$coefficients,
    variance = fit$var,
    loglik = fit$loglik,
    iter = fit$iter,
    lower = fit$ci.lower,
    upper = fit$ci.upper,
    probability = fit$prob,
    iter_ci = fit$iter.ci
  ))
}

expect_backend_parity <- function(...) {
  fortran <- fit_with_backend("fortran", ...)
  rust <- fit_with_backend("rust", ...)
  stopifnot(isTRUE(all.equal(
    fit_values(fortran),
    fit_values(rust),
    tolerance = 1e-10,
    check.attributes = TRUE
  )))
}

standard <- data.frame(
  time = c(1, 2, 3),
  status = c(1, 1, 1),
  x = c(1, 1, 0)
)

interval <- data.frame(
  start = c(1, 2, 5, 2, 1, 7, 3, 4, 8, 8),
  stop = c(2, 3, 6, 7, 8, 9, 9, 9, 14, 17),
  event = c(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
  x = c(1, 0, 0, 1, 0, 1, 1, 1, 0, 0),
  z = c(0, 1, 1, 1, 0, 0, 1, 0, 1, 0)
)

ties <- data.frame(
  time = c(1, 1, 2, 2, 3, 4, 4, 5),
  status = c(1, 1, 1, 0, 1, 1, 0, 0),
  x = c(-1, 0.2, 0.4, 1, -0.3, 0.7, 1.1, -0.8),
  z = c(0, 1, 0, 1, 1, 0, 1, 0)
)

expect_backend_parity(
  Surv(time, status) ~ x,
  data = standard,
  pl = FALSE
)
expect_backend_parity(
  Surv(start, stop, event) ~ x + z,
  data = interval,
  pl = TRUE,
  maxit = 100
)

# Selective profiling must leave the full-model fit unchanged and reproduce
# the selected coefficient's result from a full profile. The unselected
# coefficient remains a freely estimated nuisance parameter during profiling.
full_profile <- fit_with_backend(
  "rust",
  Surv(start, stop, event) ~ x + z,
  data = interval,
  pl = TRUE,
  maxit = 100
)
selected_profile <- fit_with_backend(
  "rust",
  Surv(start, stop, event) ~ x + z,
  data = interval,
  pl = TRUE,
  pl.select = "x",
  maxit = 100
)
stopifnot(isTRUE(all.equal(
  unlist(full_profile[c("coefficients", "var", "loglik", "iter")]),
  unlist(selected_profile[c("coefficients", "var", "loglik", "iter")]),
  tolerance = 0,
  check.attributes = TRUE
)))
stopifnot(isTRUE(all.equal(
  c(
    lower = full_profile$ci.lower[["x"]],
    upper = full_profile$ci.upper[["x"]],
    probability = full_profile$prob[["x"]],
    iterations = full_profile$iter.ci["x", ]
  ),
  c(
    lower = selected_profile$ci.lower[["x"]],
    upper = selected_profile$ci.upper[["x"]],
    probability = selected_profile$prob[["x"]],
    iterations = selected_profile$iter.ci["x", ]
  ),
  tolerance = 0,
  check.attributes = TRUE
)))
stopifnot(
  identical(selected_profile$profiled, c(x = TRUE, z = FALSE)),
  is.na(selected_profile$ci.lower[["z"]]),
  is.na(selected_profile$ci.upper[["z"]]),
  is.na(selected_profile$prob[["z"]]),
  all(is.na(selected_profile$iter.ci["z", ]))
)

inference_profile <- fit_with_backend(
  "rust",
  Surv(start, stop, event) ~ x + z,
  data = interval,
  pl = TRUE,
  pl.select = "x",
  inference.only = TRUE,
  maxit = 100
)
stopifnot(
  isTRUE(all.equal(
    fit_values(inference_profile),
    fit_values(selected_profile),
    tolerance = 0,
    check.attributes = TRUE
  )),
  identical(inference_profile$means, selected_profile$means),
  identical(inference_profile$nevent, selected_profile$nevent),
  is.null(inference_profile$y),
  is.null(inference_profile$linear.predictors)
)
inference_glance <- generics::glance(inference_profile)
stopifnot(
  identical(inference_glance$n[[1]], inference_profile$n),
  identical(inference_glance$nevent[[1]], inference_profile$nevent)
)
inference_augment_error <- tryCatch(
  {
    coxphf::augment.coxphf(inference_profile)
    NA_character_
  },
  error = conditionMessage
)
stopifnot(grepl("inference.only=TRUE", inference_augment_error, fixed = TRUE))

selected_fortran <- fit_with_backend(
  "fortran",
  Surv(start, stop, event) ~ x + z,
  data = interval,
  pl = TRUE,
  pl.select = "x",
  maxit = 100
)
stopifnot(isTRUE(all.equal(
  fit_values(selected_fortran),
  fit_values(selected_profile),
  tolerance = 1e-10,
  check.attributes = TRUE
)))

selected_by_position <- fit_with_backend(
  "rust",
  Surv(start, stop, event) ~ x + z,
  data = interval,
  pl = TRUE,
  pl.select = 1L,
  maxit = 100
)
stopifnot(isTRUE(all.equal(
  fit_values(selected_by_position),
  fit_values(selected_profile),
  tolerance = 0,
  check.attributes = TRUE
)))

profile_error <- function(...) {
  tryCatch(
    {
      fit_with_backend("rust", ...)
      NA_character_
    },
    error = conditionMessage
  )
}
stopifnot(grepl(
  "Unknown coefficient in pl.select",
  profile_error(
    Surv(start, stop, event) ~ x + z,
    data = interval,
    pl.select = "missing"
  ),
  fixed = TRUE
))
stopifnot(grepl(
  "pl.select can only be used when pl=TRUE",
  profile_error(
    Surv(start, stop, event) ~ x + z,
    data = interval,
    pl = FALSE,
    pl.select = "x"
  ),
  fixed = TRUE
))
stopifnot(grepl(
  "fixed by adapt",
  profile_error(
    Surv(start, stop, event) ~ x + z,
    data = interval,
    pl.select = "z",
    adapt = c(1, 0)
  ),
  fixed = TRUE
))
stopifnot(grepl(
  "inference.only",
  profile_error(
    Surv(start, stop, event) ~ x + z,
    data = interval,
    pl = FALSE,
    inference.only = 1
  ),
  fixed = TRUE
))
expect_backend_parity(
  Surv(start, stop, event) ~ x + x:log(stop),
  data = interval,
  pl = FALSE,
  maxit = 100
)
expect_backend_parity(
  Surv(start, stop, event) ~ x + x:log(stop),
  data = interval,
  pl = TRUE,
  pl.select = "x",
  maxit = 100
)
expect_backend_parity(
  Surv(time, status) ~ x + z,
  data = ties,
  pl = FALSE
)
expect_backend_parity(
  Surv(start, stop, event) ~ x + z,
  data = interval,
  pl = FALSE,
  adapt = c(1, 0),
  penalty = 0.75
)

fortran_test <- test_with_backend(
  "fortran",
  Surv(start, stop, event) ~ x + z,
  data = interval,
  test = ~x
)
rust_test <- test_with_backend(
  "rust",
  Surv(start, stop, event) ~ x + z,
  data = interval,
  test = ~x
)
stopifnot(isTRUE(all.equal(
  unlist(fortran_test[c("testcov", "loglik", "df", "prob")]),
  unlist(rust_test[c("testcov", "loglik", "df", "prob")]),
  tolerance = 1e-10,
  check.attributes = TRUE
)))

old_backend <- options(coxphf.native_backend = NULL)
default_fit <- coxphf(Surv(time, status) ~ x, data = standard, pl = FALSE)
rust_fit <- fit_with_backend(
  "rust",
  Surv(time, status) ~ x,
  data = standard,
  pl = FALSE
)
options(old_backend)
stopifnot(isTRUE(all.equal(
  fit_values(default_fit),
  fit_values(rust_fit),
  tolerance = 0,
  check.attributes = TRUE
)))

# New arguments remain at the end of the public signature so historical
# positional calls still map their fourth argument to alpha.
positional_fit <- fit_with_backend(
  "rust",
  Surv(time, status) ~ x,
  standard,
  FALSE,
  0.1
)
stopifnot(
  identical(positional_fit$alpha, 0.1),
  identical(positional_fit$method.ci, "Wald")
)

# Covariate origins must not affect a Cox fit. A large calendar-year shift
# previously made risk-score exponentiation overflow even though the shifted
# model is mathematically identical.
origin_data <- interval
origin_data$calendar <- c(-4, -2, 0, 2, 4, -3, -1, 1, 3, 5)
shifted_data <- origin_data
shifted_data$calendar_shifted <- shifted_data$calendar + 1000000000

for (backend in c("fortran", "rust")) {
  origin_fit <- fit_with_backend(
    backend,
    Surv(start, stop, event) ~ calendar + x,
    data = origin_data,
    pl = TRUE,
    maxit = 100
  )
  shifted_fit <- fit_with_backend(
    backend,
    Surv(start, stop, event) ~ calendar_shifted + x,
    data = shifted_data,
    pl = TRUE,
    maxit = 100
  )
  stopifnot(isTRUE(all.equal(
    unname(fit_values(origin_fit)),
    unname(fit_values(shifted_fit)),
    tolerance = 1e-10,
    check.attributes = FALSE
  )))
  stopifnot(isTRUE(all.equal(
    unname(shifted_fit$means - origin_fit$means),
    c(1000000000, 0),
    tolerance = 0,
    check.attributes = FALSE
  )))
  stopifnot(isTRUE(all.equal(
    origin_fit$linear.predictors,
    shifted_fit$linear.predictors,
    tolerance = 1e-10,
    check.attributes = FALSE
  )))

  original_design <- model.matrix(~calendar + x, data = origin_data)[, -1, drop = FALSE]
  expected_lp <- as.vector(
    scale(original_design, origin_fit$means, scale = FALSE) %*%
      origin_fit$coefficients
  )
  stopifnot(isTRUE(all.equal(
    origin_fit$linear.predictors,
    expected_lp,
    tolerance = 1e-12,
    check.attributes = FALSE
  )))
}

# Centering the source covariate of a time-dependent effect changes only a
# time-specific common offset and therefore must not alter fitted parameters.
tde_origin <- interval
tde_origin$tde_x <- tde_origin$x
tde_shifted <- tde_origin
tde_shifted$tde_x_shifted <- tde_shifted$tde_x + 1000000
for (backend in c("fortran", "rust")) {
  origin_fit <- fit_with_backend(
    backend,
    Surv(start, stop, event) ~ z + tde_x + tde_x:log(stop),
    data = tde_origin,
    pl = FALSE,
    maxit = 100
  )
  shifted_fit <- fit_with_backend(
    backend,
    Surv(start, stop, event) ~ z + tde_x_shifted + tde_x_shifted:log(stop),
    data = tde_shifted,
    pl = FALSE,
    maxit = 100
  )
  stopifnot(isTRUE(all.equal(
    unname(fit_values(origin_fit)),
    unname(fit_values(shifted_fit)),
    tolerance = 1e-9,
    check.attributes = FALSE
  )))

  tde_design <- model.matrix(
    ~z + tde_x + tde_x:log(stop),
    data = tde_origin
  )[, -1, drop = FALSE]
  expected_tde_lp <- as.vector(
    scale(tde_design, origin_fit$means, scale = FALSE) %*%
      origin_fit$coefficients
  )
  stopifnot(isTRUE(all.equal(
    origin_fit$linear.predictors,
    expected_tde_lp,
    tolerance = 1e-10,
    check.attributes = FALSE
  )))

  if (backend == "rust") {
    tde_inference_fit <- fit_with_backend(
      backend,
      Surv(start, stop, event) ~ z + tde_x + tde_x:log(stop),
      data = tde_origin,
      pl = FALSE,
      inference.only = TRUE,
      maxit = 100
    )
    stopifnot(
      isTRUE(all.equal(
        fit_values(tde_inference_fit),
        fit_values(origin_fit),
        tolerance = 0,
        check.attributes = TRUE
      )),
      identical(tde_inference_fit$means, origin_fit$means)
    )
  }
}

constant_data <- transform(interval, constant = 1)
constant_error <- tryCatch(
  {
    fit_with_backend(
      "rust",
      Surv(start, stop, event) ~ x + constant,
      data = constant_data,
      pl = FALSE
    )
    NA_character_
  },
  error = conditionMessage
)
stopifnot(grepl("constant or numerically constant", constant_error, fixed = TRUE))

nonfinite_data <- interval
nonfinite_data$x[[1]] <- Inf
nonfinite_error <- tryCatch(
  {
    fit_with_backend(
      "rust",
      Surv(start, stop, event) ~ x + z,
      data = nonfinite_data,
      pl = FALSE
    )
    NA_character_
  },
  error = conditionMessage
)
stopifnot(grepl("contains a non-finite value", nonfinite_error, fixed = TRUE))

native_fit_failure <- numeric(15)
native_fit_failure[c(8, 10)] <- c(98, -98)
native_fit_failure_error <- tryCatch(
  {
    coxphf:::.coxphf_assert_finite_native(
      list(outpar = native_fit_failure, outtab = 0),
      "parameter estimation"
    )
    NA_character_
  },
  error = conditionMessage
)
stopifnot(grepl("Native parameter estimation failed", native_fit_failure_error, fixed = TRUE))

intercept_only <- fit_with_backend(
  "rust",
  Surv(time, status) ~ 1,
  data = ties,
  pl = FALSE
)
stopifnot(inherits(intercept_only, "coxph"))
intercept_inference_only <- fit_with_backend(
  "rust",
  Surv(time, status) ~ 1,
  data = ties,
  pl = FALSE,
  inference.only = TRUE
)
stopifnot(
  inherits(intercept_inference_only, "coxph"),
  is.null(intercept_inference_only$y),
  is.null(intercept_inference_only$linear.predictors)
)

missing_data <- ties
missing_data$x[[2]] <- NA_real_
missing_fit <- fit_with_backend(
  "rust",
  Surv(time, status) ~ x + z,
  data = missing_data,
  pl = FALSE
)
stopifnot(missing_fit$n == nrow(missing_data) - 1L)
stopifnot(length(missing_fit$linear.predictors) == nrow(missing_data))
stopifnot(is.na(missing_fit$linear.predictors[[2]]))

missing_last <- ties
missing_last$x[[nrow(missing_last)]] <- NA_real_
missing_last_fit <- fit_with_backend(
  "rust",
  Surv(time, status) ~ x + z,
  data = missing_last,
  pl = FALSE
)
stopifnot(
  length(missing_last_fit$linear.predictors) == nrow(missing_last),
  is.na(missing_last_fit$linear.predictors[[nrow(missing_last)]])
)

custom_rows <- ties
rownames(custom_rows) <- paste0("sample_", seq_len(nrow(custom_rows)))
custom_rows_fit <- fit_with_backend(
  "rust",
  Surv(time, status) ~ x + z,
  data = custom_rows,
  pl = FALSE
)
stopifnot(
  length(custom_rows_fit$linear.predictors) == nrow(custom_rows),
  all(is.finite(custom_rows_fit$linear.predictors))
)

# A profile calculation can fail even when the penalized coefficient and
# covariance fit is usable. Preserve that fit and mark the requested profile
# inference as incomplete instead of throwing away the effect estimate.
profile_failure_data <- data.frame(
  time = c(7, 7, 3, 7, 8, 2, 8, 2, 2, 8, 8, 2, 6, 6, 3, 8),
  status = c(0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
  z = c(
    0.156418, 0.472601, 0.573864, -0.765160,
    1.157448, 1.195263, 2.076093, -0.615839,
    -0.523722, 1.369940, 0.969076, -0.481999,
    0.150343, 0.664153, -0.933727, 1.138314
  ),
  w = c(1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1),
  exposure = c(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
)
profile_failure_effect <- fit_with_backend(
  "rust",
  Surv(time, status) ~ z + w + exposure,
  data = profile_failure_data,
  pl = FALSE,
  inference.only = TRUE
)
profile_failure_warnings <- character()
profile_failure_fit <- withCallingHandlers(
  fit_with_backend(
    "rust",
    Surv(time, status) ~ z + w + exposure,
    data = profile_failure_data,
    pl = TRUE,
    pl.select = "exposure",
    inference.only = TRUE
  ),
  warning = function(w) {
    profile_failure_warnings <<- c(
      profile_failure_warnings,
      conditionMessage(w)
    )
    invokeRestart("muffleWarning")
  }
)
stopifnot(
  isTRUE(all.equal(
    unlist(profile_failure_fit[c("coefficients", "var", "loglik", "iter")]),
    unlist(profile_failure_effect[c("coefficients", "var", "loglik", "iter")]),
    tolerance = 0,
    check.attributes = TRUE
  )),
  all(is.na(profile_failure_fit$ci.lower)),
  all(is.na(profile_failure_fit$ci.upper)),
  all(is.na(profile_failure_fit$prob)),
  all(is.na(profile_failure_fit$iter.ci)),
  identical(
    profile_failure_fit$profiled,
    c(z = FALSE, w = FALSE, exposure = TRUE)
  ),
  any(grepl(
    "returning coefficient and covariance estimates with profile fields set to NA",
    profile_failure_warnings,
    fixed = TRUE
  ))
)
