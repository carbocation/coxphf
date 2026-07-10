# coxphf

## Overview

The package coxphf implements Firth's penalized maximum likelihood bias reduction method for Cox regression
which has been shown to provide a solution in case of monotone likelihood (nonconvergence of likelihood function).
The program fits profile penalized likelihood confidence intervals which were proved to outperform
Wald confidence intervals.

## Installation

The Rust backend requires a stable Rust toolchain with `cargo` and `rustc`.
Install one from [rustup](https://rustup.rs/) before installing this fork.

```r
# Install coxphf from CRAN
install.packages("coxphf")

# Install the Rust/fast-interval branch from GitHub
# install.packages("remotes")
remotes::install_github("carbocation/coxphf@feature/rust-backend")
```

The Rust backend is used by default. For numerical comparisons during
development, the retained reference backend can be selected with
`options(coxphf.native_backend = "fortran")`.

## Usage
The call of the main function of the library follows the structure of the standard functions requiring a data.frame and a formula for the model specification. 
The response must be a survival object as returned by the 'Surv' function (see its documentation in the survival package).
The resulting object belongs to the new class coxphf.

```r
library(survival)
data(breast)
fit.breast<-coxphf(data=breast, Surv(TIME,CENS)~T+N+G+CD)
summary(fit.breast)
```

## Acknowledgment

This work was supported by the Austrian Science Fund (FWF) (award I 2276).
