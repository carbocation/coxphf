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
