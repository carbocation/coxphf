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
expect_backend_parity(
  Surv(start, stop, event) ~ x + x:log(stop),
  data = interval,
  pl = FALSE,
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
