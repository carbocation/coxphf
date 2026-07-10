.coxphf_column_names <- function(x, prefix) {
  labels <- colnames(x)
  if (is.null(labels)) {
    labels <- paste0(prefix, seq_len(ncol(x)))
  }
  labels
}

.coxphf_validate_finite_matrix <- function(x, what) {
  if (ncol(x) == 0L) {
    return(invisible(NULL))
  }

  labels <- .coxphf_column_names(x, "V")
  for (j in seq_len(ncol(x))) {
    if (any(!is.finite(x[, j]))) {
      stop(
        what,
        " column '", labels[[j]],
        "' contains a non-finite value.",
        call. = FALSE
      )
    }
  }
  invisible(NULL)
}

.coxphf_column_scales <- function(x, what) {
  if (ncol(x) == 0L) {
    return(numeric())
  }
  if (nrow(x) < 2L) {
    stop("At least two complete observations are required.", call. = FALSE)
  }

  labels <- .coxphf_column_names(x, "V")
  scales <- numeric(ncol(x))
  for (j in seq_len(ncol(x))) {
    values <- x[, j]
    scale <- stats::sd(values)
    roundoff_floor <- 100 * .Machine$double.eps * max(1, max(abs(values)))
    if (!is.finite(scale) || scale <= roundoff_floor) {
      stop(
        what,
        " column '", labels[[j]],
        "' is constant or numerically constant.",
        call. = FALSE
      )
    }
    scales[[j]] <- scale
  }
  names(scales) <- labels
  scales
}

.coxphf_prepare_design <- function(obj, keep_original_design = TRUE) {
  if (nrow(obj$resp) == 0L) {
    stop("No complete observations are available for fitting.", call. = FALSE)
  }
  if (any(!is.finite(obj$resp))) {
    stop("The survival response contains a non-finite value.", call. = FALSE)
  }

  base <- as.matrix(obj$mm1)
  timedata <- as.matrix(obj$timedata)
  if (nrow(base) != nrow(obj$resp) || nrow(timedata) != nrow(obj$resp)) {
    stop("Internal model matrices have inconsistent row counts.", call. = FALSE)
  }
  .coxphf_validate_finite_matrix(base, "Model-matrix")
  .coxphf_validate_finite_matrix(timedata, "Time-function")

  base_scales <- .coxphf_column_scales(base, "Model-matrix")
  time_scales <- .coxphf_column_scales(timedata, "Time-function")
  base_centers <- colMeans(base)

  original_design <- if (keep_original_design) base else NULL
  original_means <- base_centers
  if (obj$NTDE > 0L) {
    if (keep_original_design) {
      time_dependent <- base[, obj$timeind, drop = FALSE] * timedata
      original_means <- c(original_means, colMeans(time_dependent))
      original_design <- cbind(base, time_dependent)
    } else {
      time_dependent_means <- vapply(
        seq_len(obj$NTDE),
        function(j) mean(base[, obj$timeind[[j]]] * timedata[, j]),
        numeric(1)
      )
      original_means <- c(original_means, time_dependent_means)
    }
  }
  if (keep_original_design) {
    colnames(original_design) <- obj$covnames
  }
  names(original_means) <- obj$covnames

  obj$mm1 <- sweep(sweep(base, 2L, base_centers, FUN = "-"), 2L, base_scales, FUN = "/")
  if (ncol(timedata) > 0L) {
    # Centering an ordinary Cox covariate changes only a risk-set-wide
    # constant. Time functions are scaled but not centered because centering
    # them reparameterizes the corresponding main-effect coefficient.
    obj$timedata <- sweep(timedata, 2L, time_scales, FUN = "/")
  } else {
    obj$timedata <- timedata
  }

  list(
    obj = obj,
    original_design = original_design,
    original_means = original_means,
    scale = c(
      unname(base_scales),
      unname(time_scales * base_scales[obj$timeind])
    ),
    center = unname(base_centers)
  )
}

.coxphf_assert_finite_native <- function(value, stage) {
  if (
    length(value$outpar) >= 10L &&
    isTRUE(value$outpar[8] == 98) &&
    isTRUE(value$outpar[10] == -98)
  ) {
    stop("Native ", stage, " failed.", call. = FALSE)
  }
  if (any(!is.finite(value$outpar)) || any(!is.finite(value$outtab))) {
    stop(
      "Native ", stage,
      " produced non-finite values after covariate stabilization. ",
      "This can indicate separation, collinearity, or failure to converge.",
      call. = FALSE
    )
  }
  invisible(value)
}
