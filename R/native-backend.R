.coxphf_native <- function(routine, cards, parms, ioarray,
                           backend = getOption("coxphf.native_backend", "rust")) {
  backend <- match.arg(backend, c("fortran", "rust"))
  routine <- match.arg(routine, c("firthcox", "plcomp"))

  if (backend == "rust") {
    return(.C(
      paste0("rust_", routine),
      cards,
      outpar = parms,
      outtab = ioarray,
      PACKAGE = "coxphf"
    ))
  }

  .Fortran(
    routine,
    cards,
    outpar = parms,
    outtab = ioarray,
    PACKAGE = "coxphf"
  )
}
