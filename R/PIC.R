#' Predictive Information Criteria
#'
#' @description \code{PIC} is a generic function for computing predictive information criteria (PIC).
#' Depending on the \code{\link[base]{class}} of the fitted model supplied to \code{object}, the function
#' invokes the appropriate method for computing PIC.
#'
#' @param object A fitted model.
#' @param newdata An optional data frame of validation data used to compute PIC. If omitted, the training data contained within \code{object} are used.
#' @param ... Further arguments passed to other methods.
#'
#' @details The PIC are model selection criteria that may be used to select from among predictive models in a candidate set.
#' The model with the minimum criterion value is preferred.
#'
#' The PIC asymptotically select the candidate model that minimizes the mean squared error of prediction (MSEP),
#' thus behaving similarly to the the Akaike Information Criterion (AIC). However in contrast to the AIC, the PIC
#' do not assume a panel of validation data that are independent and identically distributed to the set of training
#' data. The PIC are thus better able to accommodate training/validation data \emph{heterogeneity}, where training
#' and validation data may differ from one another in distribution.
#'
#' Data heterogeneity is arguably the more typical circumstance in practice, especially when one considers applications
#' where a set of covariates are used to model and predict some response. In these regression contexts, one more often predicts
#' values of the response for combinations of covariate values that were not necessarily used to train the predictive model.
#'
#'
#' @return The form of the value returned by \code{PIC} depends on the fitted model class and its method-specific arguments.
#' Details may be found in the documentation of each method.
#'
#' @seealso
#' \code{\link[picR]{PIC.lm}}
#'
#' @export
#'
PIC <- function(object, newdata, ...){
  UseMethod("PIC", object)
}


#' PIC method for Linear Models
#'
#' @description Computation of predictive information criteria for linear models.
#'
#' @param object A fitted model object of \code{\link[base]{class}} "lm".
#' @param newdata An optional data frame of validation data used to compute PIC. If omitted, the training data contained within \code{object} are used.
#' @param group_sizes An optional scalar or numeric vector indicating the sizes of \code{newdata} partitions. If omitted, \code{newdata} is not partitioned. See 'Details'.
#' @param ... Further arguments passed to or from other methods.
#'
#' @details \code{PIC.lm} computes PIC values based on the supplied model. Candidate models with relatively smaller criterion values are preferred.
#' Depending on the value(s) supplied to \code{group_sizes}, one of three implementations of PIC are computed:
#'
#' \itemize{
#'   \item \strong{iPIC}: The individualized predictive information criterion (iPIC) is computed when \code{group_sizes = 1}. A value
#'   of iPIC is determined for each \emph{individual} observation in \code{newdata}. Using iPIC, one may thus select optimal predictive
#'   models specific to each particular validation datapoint.
#'
#'   \item \strong{gPIC}: The group predictive information criterion (gPIC) is computed when \code{group_sizes > 1} or
#'   \code{is.vector(group_sizes) == TRUE}. A value of gPIC is determined for each cohort or \emph{group} of observations
#'   defined by the partitions of \code{newdata}. Using gPIC, one may thus select optimal predictive models specific to each
#'   group of validation datapoints. For the class of regression models, the gPIC value of a group of validation observations
#'   is equivalent to the sum of their individual iPIC values.
#'
#'   \item \strong{tPIC}: The total predictive information criterion (tPIC) is computed when \code{group_sizes = NULL}. Computation of
#'   the tPIC is the default, and one may use the tPIC to select the optimal predictive model for the entire set of validation
#'   datapoints. The tPIC and gPIC are equivalent when \code{group_sizes = m}, where \code{m} is the number of observations in
#'   \code{newdata}.
#' }
#'
#' The general formula used to compute the PIC for linear regression models is...
#' provide formula, define terms, share criterion insights re: penalty and goodness of fit. Give
#' pros/cons of using iPIC vs gPIC vs tPIC. Essentially, give an abridged discussion from pages 43-48
#' of thesis.
#'
#' @return If \code{group_sizes = NULL}, a scalar is returned. Otherwise, \code{newdata} is
#' returned with an appended column labeled 'PIC' containing either iPIC or gPIC values,
#' depending on what was provided to \code{group_sizes}.
#'
#' @seealso
#' \code{\link[picR]{PIC}}, \code{\link[stats]{lm}}
#'
#' @export
#'
PIC.lm <- function(object, newdata, group_sizes = NULL, ...){

  X.obs <- stats::model.matrix(object)

  if (missing(newdata) || is.null(newdata)) {
    X.pred  <- X.obs
    newdata <- object$model
  } else {
    tt <- stats::terms(object)
    Terms <- stats::delete.response(tt)
    X.pred <- stats::model.matrix(Terms, newdata)
  }

  sigma.mle <- sqrt(crossprod(object$residuals)/nrow(X.obs))
  nn        <- nrow(X.obs)

  gof.pic <- rep(log(2*pi*sigma.mle^2) + 1, nn)

  RS.dist <- diag(X.pred %*% invmat(crossprod(X.obs)) %*% t(X.pred))
  pen.pic <- 2*(RS.dist + 1/nn)

  pic.vec <- gof.pic + pen.pic

  if(is.null(group_sizes)){
    return(sum(pic.vec))
  } else{
    if(length(group_sizes) == 1){
      gpic    <- split(pic.vec, ceiling(seq_along(pic.vec)/group_sizes))
      gpic    <- lapply(gpic, sum)

      idx     <- 1:nrow(newdata)
      gidx    <- split(idx, ceiling(seq_along(idx)/group_sizes))

    } else{
      if(sum(group_sizes) != nrow(X.pred)){
        warning("Sum of supplied group sizes does not match total predicted observations.")
      } else{
        gpic    <- split(pic.vec, rep(1:length(group_sizes), group_sizes))
        gpic    <- lapply(gpic, sum)

        idx     <- 1:nrow(newdata)
        gidx    <- split(idx, rep(1:length(group_sizes), group_sizes))
      }
    }
    gdat    <- lapply(gidx, function(idx){newdata[idx,,drop = FALSE]})
    gdat    <- mapply(function(dat, val){cbind.data.frame(dat, PIC = val)},
                      dat = gdat, val = gpic, SIMPLIFY = FALSE)
    gdat    <- Reduce("rbind", gdat)
    return(gdat)
  }
}
