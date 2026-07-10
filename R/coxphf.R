#' Cox Regression with Firth's Penalized Likelihood
#' 
#' Implements Firth's penalized maximum likelihood bias reduction method  for Cox regression 
#' which has been shown to provide a solution in case of monotone likelihood (nonconvergence of likelihood function). 
#' The program fits profile penalized likelihood confidence intervals which were proved to outperform 
#' Wald confidence intervals.
#' 
#' The phenomenon of monotone likelihood in a sample causes parameter estimates of a Cox model to diverge, with 
#' infinite standard errors. Therefore, classical maximum likelihood analysis fails; the usual Wald confidence 
#' intervals cover the whole range of real numbers. Monotone likelihood appears if there is single covariate
#' or a linear combination of covariates such that at each event time, out of all individuals being at risk at 
#' that time, the individual with the highest (or at each event time the individual with the lowest) value for that covariate or linear combination experiences the event. It was shown that 
#' analysis by Firth's penalized likelihood method, particularly in conjunction with the computation
#' of profile likelihood confidence intervals and penalized likelihood ratio tests is superior to maximum
#' likelihood analysis. It completely removes the convergence problem mentioned in the paragraph on CONVERGENCE of
#' the description of the function \code{coxph}. The \code{formula} may involve time-dependent effects or 
#' time-dependent covariates. The response may
#' be given in counting process style, but it cannot be used for multivariate failure times, as the program has no option
#' to fit a robust covariance matrix. The user is responsible for the independency of observations within each risk set, i.e., 
#' the same individual should not appear twice within the same risk set.
#' 
#' @param formula a formula object, with the response on the left and the model terms on the right. 
#' The response must be a survival object as returned by the 'Surv' function (see its documentation in the survival package)
#' @param data a data.frame in which to interpret the variables named in the 'formula' argument.
#' @param pl specifies if confidence intervals and tests should be based on the profile penalized log likelihood (\code{pl=TRUE}, the default) or on the Wald method (\code{pl=FALSE}).
#' @param alpha the significance level (1-\eqn{\alpha} = the confidence level), 0.05 as default.
#' @param maxit maximum number of iterations (default value is 50)
#' @param maxhs maximum number of step-halvings per iterations (default value is 5). 
#' The increments of the parameter vector in one Newton-Rhaphson iteration step are halved,
#' unless the new likelihood is greater than the old one, maximally doing \code{maxhs} halvings.
#' @param epsilon specifies the maximum allowed change in standardized parameter estimates to declare convergence. Default value is 1e-6.
#' @param gconv specifies the maximum allowed absolute value of first derivative of likelihood to declare convergence. Default value is 0.0001.
#' @param maxstep specifies the maximum change of (standardized) parameter values allowed in one iteration. Default value is 0.5.
#' @param firth use of Firth's penalized maximum likelihood (\code{firth=TRUE}, default) or the standard maximum likelihood method (\code{firth=FALSE}) for fitting the Cox model.
#' @param adapt optional: specifies a vector of 1s and 0s, where 0 means that the corresponding parameter is fixed at 0, while 1 enables parameter estimation for that parameter. The length of adapt must be equal to the number of parameters to be estimated.
#' @param penalty strength of Firth-type penalty. Defaults to 0.5.
#' @param pl.select optional coefficient selection for profile likelihood
#' confidence intervals and tests. The default, \code{NULL}, profiles every
#' coefficient. Supply coefficient names, one-based positions, or a logical
#' vector with one element per coefficient to profile only a subset. Other
#' coefficients remain estimated as nuisance parameters, but their profile
#' confidence limits, tests, and iteration counts are returned as \code{NA}.
#' @param inference.only if \code{TRUE}, return coefficient-level inference
#' without retaining the observation-level response or constructing linear
#' predictors. This reduces result size and memory use in high-throughput analyses,
#' but methods that require those components, such as \code{augment()}, cannot
#' be used without refitting. Defaults to \code{FALSE}.
#' 
#' @return The object returned is of the class \code{coxphf} and has the following attributes:
#' \item{coefficients}{the parameter estimates}
#' \item{alpha}{the significance level = 1 - confidence level}
#' \item{var}{the estimated covariance matrix}
#' \item{df}{the degrees of freedom}
#' \item{loglik}{the null and maximimized (penalized) log likelihood}
#' \item{method.ties}{the ties handling method}
#' \item{iter}{the number of iterations needed to converge}
#' \item{n}{the number of observations}
#' \item{nevent}{the number of events}
#' \item{y}{the response, or \code{NULL} when \code{inference.only=TRUE}}
#' \item{formula}{the model formula}
#' \item{means}{the means of the covariates}
#' \item{linear.predictors}{the linear predictors, or \code{NULL} when
#' \code{inference.only=TRUE}}
#' \item{method}{the estimation method (Standard ML or Penalized ML)}
#' \item{method.ci}{the confidence interval estimation method (Profile Likelihood or Wald)}
#' \item{ci.lower}{the lower confidence limits}
#' \item{ci.upper}{the upper confidence limits}
#' \item{prob}{the p-values}
#' \item{call}{the function call}
#' \item{terms}{the terms object used}
#' \item{iter.ci}{the numbers of iterations needed for profile likelihood confidence interval estimation, and for maximizing the restricted likelihood for p-value computation.}
#' \item{profiled}{a named logical vector indicating which coefficients were
#' selected for profile likelihood confidence intervals and tests. Present only
#' when \code{pl=TRUE}.}
#' If parameter estimation succeeds but profile inference fails, the coefficient
#' and covariance estimates are retained and the profile fields are returned as
#' \code{NA} with a warning.
#' 
#' @export
#'
#' @encoding UTF-8
#' @examples
#' # fixed covariate and monotone likelihood
#' library(survival)
#' time<-c(1,2,3)
#' cens<-c(1,1,1)
#' x<-c(1,1,0)
#' sim<-cbind(time,cens,x)
#' sim<-data.frame(sim)
#' coxphf(sim, formula=Surv(time,cens)~x) #convergence attained!
#' #coxph(sim, formula=Surv(time,cens)~x)  #no convergence!
#' # time-dependent covariate
#' test2 <- data.frame(list(start=c(1, 2, 5, 2, 1, 7, 3, 4, 8, 8),
#'                          stop =c(2, 3, 6, 7, 8, 9, 9, 9,14,17),
#'                          event=c(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
#'                          x    =c(1, 0, 0, 1, 0, 1, 1, 1, 0, 0) ))
#' 
#' summary( coxphf( formula=Surv(start, stop, event) ~ x, pl=FALSE, data=test2))
#' 
#' 
#' # time-dependent effect
#' # the coxphf function can handle interactions of a (fixed or time-dependent)
#' # covariate with time
#' # such that the hazard ratio can be expressed as a function of time
#' 
#' summary(coxphf(formula=Surv(start, stop, event)~x+x:log(stop), data=test2, pl=FALSE, firth=TRUE))
#' 
#' # note that coxph would treat x:log(stop) as a fixed covariate
#' # (computed before the iteration process)
#' # coxphf treats x:log(stop) as a time-dependent covariate which changes (
#' # for the same individual!) over time
#' 
#' 
#' # time-dependent effect with monotone likelihood
#' 
#' test3 <- data.frame(list(start=c(1, 2, 5, 2, 1, 7, 3, 4, 8, 8),
#'                          stop =c(2, 3, 6, 7, 8, 9, 9, 9,14,17),
#'                          event=c(1, 0, 0, 1, 0, 1, 1, 0, 0, 0),
#'                          x    =c(1, 0, 0, 1, 0, 1, 1, 1, 0, 0) ))
#' 
#' summary( coxphf( formula=Surv(start, stop, event) ~ x+x:log(stop), pl=FALSE, maxit=400, data=test3))
#' 
#' 
#' # no convergence if option "firth" is turned off: 
#' # summary( coxphf(formula=Surv(start, stop, event) ~ x+x:log(stop), pl=F,
#' #                 data=test3, firth=FALSE)
#' 
#' 
#' data(breast)
#' fit.breast<-coxphf(data=breast, Surv(TIME,CENS)~T+N+G+CD)
#' summary(fit.breast)
#' 
#' @author Georg Heinze and Meinhard Ploner
#' @references Firth D (1993). Bias reduction of maximum likelihood estimates. \emph{Biometrika} 80:27--38.
#' 
#' Heinze G and Schemper M (2001). A Solution to the Problem of Monotone Likelihood in Cox Regression. \emph{Biometrics} 57(1):114--119. 
#' 
#' Heinze G (1999). Technical Report 10/1999: The application of Firth's procedure to Cox and logistic regression. Section of Clinical Biometrics, Department of Medical Computer Sciences, University of Vienna, Vienna.
#' @seealso [coxphfplot, coxphftest]
coxphf <-
function(
 formula,	# formula where .time. represents time for time-interactions
 data,
 pl=TRUE,
 alpha=0.05,
 maxit=50,
 maxhs=5,
 epsilon=1e-6,
 gconv=1e-4,
 maxstep=0.5,
 firth=TRUE,
 adapt=NULL,
 penalty=0.5,
 pl.select=NULL,
 inference.only=FALSE
){
### by MP und GH, 2006-2018
  if(!is.logical(firth)) stop("Please set option firth to TRUE or FALSE.\n")
  if(!is.logical(pl)) stop("Please set option pl to TRUE or FALSE.\n")
  if(!is.logical(inference.only) || length(inference.only) != 1L || is.na(inference.only)) {
    stop("Please set inference.only to TRUE or FALSE.")
  }
  if(!pl && !is.null(pl.select)) {
    stop("pl.select can only be used when pl=TRUE.")
  }

  mt <- stats::terms(formula, data=data)
  if(length(attr(mt, "term.labels")) == 0L){
    fit <- survival::coxph(formula, data)
    if(inference.only) {
      fit$y <- NULL
      fit$linear.predictors <- NULL
    }
    return(fit)
  }

  # Note that sorting is important because the native code below expects
  # the data to be sorted by time and status. decomposeSurv constructs and
  # reuses one model frame for the response and model matrix.
	obj <- decomposeSurv(formula, data, sort = TRUE)
  mt <- obj$terms

  # if only an intercept is included in the formula
  # fit a coxph model instead
  # since penalisation does not affect the baseline estimate
  if(ncol(obj$mm1) == 0){
    fit <- survival::coxph(formula, data)
    if(inference.only) {
      fit$y <- NULL
      fit$linear.predictors <- NULL
    }
    return(fit)
  }

	prepared <- .coxphf_prepare_design(
    obj,
    keep_original_design = !inference.only && obj$NTDE > 0L
  )
	obj <- prepared$obj
	n <- nrow(obj$resp)

  NTDE <- obj$NTDE
	mmm <- prepared$original_design
  Z.sd <- prepared$scale
        
	cov.name <- obj$covnames
  k <- ncol(obj$mm1)          # number of covariates
  backend <- .coxphf_native_backend()
        
  if(!is.null(adapt)){
      if(k+NTDE != length(adapt)) stop("length of adapt must match the number of parameters to be estimated.")
      if(any(adapt !=0 & adapt !=1)) stop("adapt must consist of 0s and 1s exclusively.")
  }

  profile.selection <- .coxphf_profile_selection(
    pl.select,
    cov.name,
    adapt
  )

	start.order <- order(obj$resp[, 1], decreasing = TRUE)
  NATIVE <- .coxphf_native_payload(obj, start.order, backend)
  PARMS <- c(n, k, firth, maxit, maxhs, maxstep, epsilon, 1, gconv, 0, 0, 0, 0, NTDE, penalty)
  IOARRAY <- rbind(rep(1, k+NTDE), matrix(0, 2+k+NTDE, k + NTDE))
  if(!is.null(adapt)) IOARRAY[1,]<-adapt
  if(NTDE>0)
  IOARRAY[4, (k+1):(k+NTDE)] <- obj$timeind
  storage.mode(PARMS) <- "double"
  storage.mode(IOARRAY) <- "double"

  ## --------------- Call native routine FIRTHCOX -----------------------------------
  profile.value <- NULL
  if(pl && backend == "rust") {
    PROFILE.PARMS <- c(PARMS[1:7], qchisq(1-alpha, 1), gconv, 0, 0, 0, 0, NTDE, penalty)
    PROFILE.IOARRAY <- rbind(
      rep(1, k+NTDE),
      rep(0, k+NTDE),
      rep(0, k+NTDE),
      matrix(0, 6, k+NTDE)
    )
    if(!is.null(adapt)) PROFILE.IOARRAY[1,]<-adapt
    if(NTDE>0) PROFILE.IOARRAY[4,(k+1):(k+NTDE)] <- obj$timeind
    storage.mode(PROFILE.PARMS) <- "double"
    storage.mode(PROFILE.IOARRAY) <- "double"
    combined.value <- .coxphf_native_fit_profile(
      NATIVE,
      PARMS,
      IOARRAY,
      PROFILE.PARMS,
      PROFILE.IOARRAY,
      profile.selection
    )
    value <- combined.value$fit
    profile.value <- combined.value$profile
  } else {
    value <- .coxphf_native("firthcox", NATIVE, PARMS, IOARRAY, backend)
  }
  .coxphf_assert_finite_native(value, "parameter estimation")
  if(value$outpar[8]) warning("Numerical problem in parameter estimation; check convergence.\n")
  outtab <- matrix(value$outtab, nrow=3+k+NTDE) #

  # --------------- create output object -------
  coef.orig <- outtab[3,  ]
  coefs <- coef.orig / Z.sd
  covs <- matrix(outtab[4:(k+3+NTDE), ], ncol=k+NTDE) / (Z.sd %*% t(Z.sd))

  dimnames(covs) <- list(cov.name, cov.name)
  vars <- diag(covs)
  names(coefs) <- cov.name
  df<-k+NTDE
  if(!is.null(adapt)) df<-sum(adapt>0)
  fit <- list(coefficients = coefs, alpha = alpha, var = covs, df = df,
              loglik = value$outpar[12:11], iter = value$outpar[10],
              method.ties = "breslow", n = n, ##terms = terms(formula),
              nevent = sum(obj$resp[, 3]),
              y = if(inference.only) NULL else obj$resp,
              linear.predictors = NULL,
              formula = formula, call = match.call())
  fit$means <- prepared$original_means
  if(!inference.only) {
    if(NTDE == 0L) {
      linear.predictors <- drop(obj$mm1 %*% coef.orig)
    } else {
      linear.predictors <- numeric(n)
      for(j in seq_len(k + NTDE)) {
        linear.predictors <- linear.predictors +
          (mmm[, j] - fit$means[[j]]) * coefs[[j]]
      }
    }
    fit$linear.predictors <- rep(NA_real_, obj$original_n)
    fit$linear.predictors[obj$row_index] <- linear.predictors
  }
	if(fit$iter>=maxit) warning("Convergence for parameter estimation not attained in ", maxit, " iterations.\n")
  if(firth) fit$method <- "Penalized ML"
  else fit$method <- "Standard ML"
  if(penalty != 0.5) fit$method<-paste(fit$method, " (penalty=",penalty,")",sep="")

  # --------------- Call native routine PLCOMP -------------------------------------
  if(pl) {
    profile.error <- NULL
    if(backend == "rust") {
      value <- profile.value
    } else {
      PROFILE.PARMS <- c(PARMS[1:7], qchisq(1-alpha, 1), gconv, 0, 0, 0, 0, NTDE, penalty)
      PROFILE.IOARRAY <- rbind(rep(1, k+NTDE), rep(0, k+NTDE), coef.orig, matrix(0, 6, k+NTDE))
      if(!is.null(adapt)) PROFILE.IOARRAY[1,]<-adapt
      if(NTDE>0) PROFILE.IOARRAY[4,(k+1):(k+NTDE)] <- obj$timeind
      storage.mode(PROFILE.PARMS) <- "double"
      storage.mode(PROFILE.IOARRAY) <- "double"
      value <- tryCatch(
        .coxphf_native("plcomp", NATIVE, PROFILE.PARMS, PROFILE.IOARRAY, backend),
        error = function(e) {
          profile.error <<- conditionMessage(e)
          NULL
        }
      )
    }

    fit$method.ci <- "Profile Likelihood"
    profile.failed <- is.null(value) ||
      isTRUE(value$outpar[9] == 98) ||
      any(!is.finite(value$outpar)) ||
      any(!is.finite(value$outtab))

    if(profile.failed) {
      detail <- if(is.null(profile.error)) "" else paste0(" ", profile.error)
      warning(
        "Profile-likelihood inference failed; returning coefficient and ",
        "covariance estimates with profile fields set to NA.",
        detail,
        call. = FALSE
      )
      fit$ci.lower <- rep(NA_real_, k + NTDE)
      fit$ci.upper <- rep(NA_real_, k + NTDE)
      fit$prob <- rep(NA_real_, k + NTDE)
      fit$iter.ci <- matrix(
        NA_real_,
        nrow = k + NTDE,
        ncol = 3L,
        dimnames = list(cov.name, c("Lower", "Upper", "P-value"))
      )
    } else {
      if(value$outpar[9]) warning("Numerical problem in estimating confidence intervals; check convergence.\n")
      fit$ci.lower <- exp(value$outtab[4,  ] / Z.sd)
      fit$ci.upper <- exp(value$outtab[5,  ] / Z.sd)
      fit$prob <- 1 - pchisq(value$outtab[6,  ], 1)
      fit$iter.ci<-t(value$outtab[7:9,])
      colnames(fit$iter.ci)<-c("Lower", "Upper", "P-value")
      rownames(fit$iter.ci)<-cov.name
      if(any(!is.finite(c(
        fit$ci.lower[profile.selection],
        fit$ci.upper[profile.selection],
        fit$prob[profile.selection]
      )))) {
        warning(
          "Profile-likelihood inference produced non-finite transformed values; ",
          "returning coefficient and covariance estimates with profile fields set to NA.",
          call. = FALSE
        )
        fit$ci.lower[profile.selection] <- NA_real_
        fit$ci.upper[profile.selection] <- NA_real_
        fit$prob[profile.selection] <- NA_real_
        fit$iter.ci[profile.selection, ] <- NA_real_
      } else {
        fit$ci.lower[profile.selection & fit$iter.ci[, 1] >= maxit] <- NA_real_
        fit$ci.upper[profile.selection & fit$iter.ci[, 2] >= maxit] <- NA_real_
        fit$prob[profile.selection & fit$iter.ci[, 3] >= maxit] <- NA_real_
        if(any(fit$iter.ci[profile.selection, , drop = FALSE] >= maxit)) {
          warning("Convergence in estimating profile likelihood CI or p-values not attained for all variables.\nConsider re-run with smaller maxstep and larger maxit.\n")
        }
        if(any(fit$iter.ci[profile.selection, 3] == -9)) {
          warning("Numerical error in computing penalized likelihood ratio test for some parameters.\n")
        }
        fit$prob[profile.selection & fit$iter.ci[, 3] == -9] <- NA_real_
      }

      fit$ci.lower[!profile.selection] <- NA_real_
      fit$ci.upper[!profile.selection] <- NA_real_
      fit$prob[!profile.selection] <- NA_real_
      fit$iter.ci[!profile.selection, ] <- NA_real_
    }
    fit$profiled <- stats::setNames(profile.selection, cov.name)
				
  } else {
    fit$method.ci <- "Wald"
    fit$ci.lower <- exp(coefs + qnorm(alpha/2) * vars^0.5)
    fit$ci.upper <- exp(coefs + qnorm(1 - alpha/2) * vars^0.5)
    fit$prob <- 1 - pchisq((coefs^2/vars), 1)
  }
  names(fit$prob) <- names(fit$ci.upper) <- names(fit$ci.lower) <- cov.name
  fit$terms <- mt
  attr(fit, "class") <- c("coxphf", "coxph")
  return(fit)
}

.coxphf_profile_selection <- function(selection, coefficient_names, adapt) {
  coefficient_count <- length(coefficient_names)
  if (is.null(selection)) {
    return(rep(TRUE, coefficient_count))
  }

  if (is.character(selection)) {
    unknown <- setdiff(selection, coefficient_names)
    if (length(unknown) > 0L) {
      stop(
        "Unknown coefficient in pl.select: ",
        paste(unknown, collapse = ", "),
        "."
      )
    }
    selected <- coefficient_names %in% selection
  } else if (is.logical(selection)) {
    if (length(selection) != coefficient_count || anyNA(selection)) {
      stop("A logical pl.select must have one non-missing value per coefficient.")
    }
    selected <- selection
  } else if (is.numeric(selection)) {
    if (
      anyNA(selection) || any(!is.finite(selection)) ||
      any(selection != floor(selection)) ||
      any(selection < 1L | selection > coefficient_count)
    ) {
      stop("A numeric pl.select must contain valid one-based coefficient positions.")
    }
    selected <- seq_len(coefficient_count) %in% as.integer(selection)
  } else {
    stop("pl.select must be NULL, coefficient names, positions, or a logical vector.")
  }

  if (!any(selected)) {
    stop("pl.select must select at least one coefficient.")
  }
  if (!is.null(adapt) && any(selected & adapt == 0)) {
    fixed <- coefficient_names[selected & adapt == 0]
    stop(
      "pl.select cannot include a coefficient fixed by adapt: ",
      paste(fixed, collapse = ", "),
      "."
    )
  }
  selected
}
