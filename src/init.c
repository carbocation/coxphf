#include <R_ext/RS.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Fortran calls */
extern void F77_NAME(firthcox)(void *, void *, void *);
extern void F77_NAME(plcomp)(void *, void *, void *);

/* Rust routines use a C ABI and retain the three-array interface while the
   literal port is compared with the Fortran implementation. */
extern void rust_firthcox(void *, void *, void *);
extern void rust_plcomp(void *, void *, void *);
extern void rust_firthcox_compact(const double *, const double *, const int *,
                                  const double *, double *, double *);
extern void rust_plcomp_compact(const double *, const double *, const int *,
                               const double *, double *, double *);
extern void rust_fit_profile_compact(const double *, const double *, const int *,
                                     const double *, double *, double *,
                                     double *, double *, const int *);

typedef void (*RustCompactRoutine)(const double *, const double *, const int *,
                                   const double *, double *, double *);

static SEXP rust_compact_call(SEXP x, SEXP response, SEXP start_order,
                              SEXP timedata, SEXP parms, SEXP ioarray,
                              RustCompactRoutine routine)
{
    if (TYPEOF(x) != REALSXP || TYPEOF(response) != REALSXP ||
        TYPEOF(timedata) != REALSXP || TYPEOF(parms) != REALSXP ||
        TYPEOF(ioarray) != REALSXP || TYPEOF(start_order) != INTSXP) {
        Rf_error("Invalid compact Rust native payload types");
    }

    SEXP outpar = PROTECT(Rf_duplicate(parms));
    SEXP outtab = PROTECT(Rf_duplicate(ioarray));
    routine(REAL(x), REAL(response), INTEGER(start_order), REAL(timedata),
            REAL(outpar), REAL(outtab));

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_VECTOR_ELT(result, 0, outpar);
    SET_VECTOR_ELT(result, 1, outtab);
    SET_STRING_ELT(names, 0, Rf_mkChar("outpar"));
    SET_STRING_ELT(names, 1, Rf_mkChar("outtab"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
}

SEXP coxphf_rust_firthcox_call(SEXP x, SEXP response, SEXP start_order,
                                SEXP timedata, SEXP parms, SEXP ioarray)
{
    return rust_compact_call(x, response, start_order, timedata, parms, ioarray,
                             rust_firthcox_compact);
}

SEXP coxphf_rust_plcomp_call(SEXP x, SEXP response, SEXP start_order,
                             SEXP timedata, SEXP parms, SEXP ioarray)
{
    return rust_compact_call(x, response, start_order, timedata, parms, ioarray,
                             rust_plcomp_compact);
}

SEXP coxphf_rust_fit_profile_call(SEXP x, SEXP response, SEXP start_order,
                                  SEXP timedata, SEXP fit_parms,
                                  SEXP fit_ioarray, SEXP profile_parms,
                                  SEXP profile_ioarray, SEXP profile_selection)
{
    if (TYPEOF(x) != REALSXP || TYPEOF(response) != REALSXP ||
        TYPEOF(timedata) != REALSXP || TYPEOF(fit_parms) != REALSXP ||
        TYPEOF(fit_ioarray) != REALSXP || TYPEOF(profile_parms) != REALSXP ||
        TYPEOF(profile_ioarray) != REALSXP || TYPEOF(start_order) != INTSXP ||
        TYPEOF(profile_selection) != INTSXP) {
        Rf_error("Invalid combined Rust native payload types");
    }

    R_xlen_t p_total = (R_xlen_t) REAL(profile_parms)[1] +
                       (R_xlen_t) REAL(profile_parms)[13];
    if (XLENGTH(profile_selection) != p_total) {
        Rf_error("Profile selection length does not match the coefficient count");
    }

    SEXP fit_outpar = PROTECT(Rf_duplicate(fit_parms));
    SEXP fit_outtab = PROTECT(Rf_duplicate(fit_ioarray));
    SEXP profile_outpar = PROTECT(Rf_duplicate(profile_parms));
    SEXP profile_outtab = PROTECT(Rf_duplicate(profile_ioarray));
    rust_fit_profile_compact(
        REAL(x), REAL(response), INTEGER(start_order), REAL(timedata),
        REAL(fit_outpar), REAL(fit_outtab), REAL(profile_outpar),
        REAL(profile_outtab), INTEGER(profile_selection));

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 4));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_VECTOR_ELT(result, 0, fit_outpar);
    SET_VECTOR_ELT(result, 1, fit_outtab);
    SET_VECTOR_ELT(result, 2, profile_outpar);
    SET_VECTOR_ELT(result, 3, profile_outtab);
    SET_STRING_ELT(names, 0, Rf_mkChar("fit_outpar"));
    SET_STRING_ELT(names, 1, Rf_mkChar("fit_outtab"));
    SET_STRING_ELT(names, 2, Rf_mkChar("profile_outpar"));
    SET_STRING_ELT(names, 3, Rf_mkChar("profile_outtab"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(6);
    return result;
}

static const R_CMethodDef CEntries[] = {
    {"rust_firthcox", (DL_FUNC) &rust_firthcox, 3},
    {"rust_plcomp",   (DL_FUNC) &rust_plcomp,   3},
    {NULL, NULL, 0}
};

static const R_FortranMethodDef FortranEntries[] = {
    {"firthcox", (DL_FUNC) &F77_NAME(firthcox), 3},
    {"plcomp",   (DL_FUNC) &F77_NAME(plcomp),   3},
    {NULL, NULL, 0}
};

static const R_CallMethodDef CallEntries[] = {
    {"coxphf_rust_firthcox_call", (DL_FUNC) &coxphf_rust_firthcox_call, 6},
    {"coxphf_rust_plcomp_call",   (DL_FUNC) &coxphf_rust_plcomp_call,   6},
    {"coxphf_rust_fit_profile_call", (DL_FUNC) &coxphf_rust_fit_profile_call, 9},
    {NULL, NULL, 0}
};

void R_init_coxphf(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, CallEntries, FortranEntries, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
