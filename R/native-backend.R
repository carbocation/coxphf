.coxphf_native_backend <- function(
  backend = getOption("coxphf.native_backend", "rust")
) {
  match.arg(backend, c("fortran", "rust"))
}

.coxphf_native_payload <- function(obj, start_order, backend) {
  backend <- .coxphf_native_backend(backend)
  if (backend == "rust") {
    return(list(
      x = obj$mm1,
      response = obj$resp,
      start_order = as.integer(start_order),
      timedata = obj$timedata
    ))
  }

  n <- nrow(obj$resp)
  p_total <- ncol(obj$mm1) + obj$NTDE
  cards <- cbind(
    obj$mm1,
    obj$resp,
    start_order,
    matrix(1, n, p_total),
    obj$timedata
  )
  storage.mode(cards) <- "double"
  cards
}

.coxphf_native <- function(routine, payload, parms, ioarray,
                           backend = getOption("coxphf.native_backend", "rust")) {
  backend <- .coxphf_native_backend(backend)
  routine <- match.arg(routine, c("firthcox", "plcomp"))

  if (backend == "rust") {
    if (is.list(payload)) {
      return(.Call(
        paste0("coxphf_rust_", routine, "_call"),
        payload$x,
        payload$response,
        payload$start_order,
        payload$timedata,
        parms,
        ioarray,
        PACKAGE = "coxphf"
      ))
    }

    # Retain the original three-array ABI for direct internal callers and
    # numerical comparisons with historical builds.
    return(.C(
      paste0("rust_", routine),
      payload,
      outpar = parms,
      outtab = ioarray,
      PACKAGE = "coxphf"
    ))
  }

  if (is.list(payload)) {
    stop("The Fortran backend requires a full native payload.", call. = FALSE)
  }
  .Fortran(
    routine,
    payload,
    outpar = parms,
    outtab = ioarray,
    PACKAGE = "coxphf"
  )
}

.coxphf_native_fit_profile <- function(payload, fit_parms, fit_ioarray,
                                       profile_parms, profile_ioarray,
                                       profile_selection) {
  if (!is.list(payload)) {
    stop("Combined fit/profile execution requires the compact Rust payload.", call. = FALSE)
  }
  value <- .Call(
    "coxphf_rust_fit_profile_call",
    payload$x,
    payload$response,
    payload$start_order,
    payload$timedata,
    fit_parms,
    fit_ioarray,
    profile_parms,
    profile_ioarray,
    as.integer(profile_selection),
    PACKAGE = "coxphf"
  )
  list(
    fit = list(outpar = value$fit_outpar, outtab = value$fit_outtab),
    profile = list(
      outpar = value$profile_outpar,
      outtab = value$profile_outtab
    )
  )
}
