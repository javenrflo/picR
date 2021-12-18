#' Predictive Information Criteria
#'
#' @description \code{PIC} is a generic function for computing predictive information criteria.
#' Depending on the fitted model object \code{\link[base]{class}}, the function invokes the appropriate method for
#' determining PIC value(s) for the supplied model.
#'
#' @param object A fitted model object on which computed PIC value(s) are partially based.
#' @param newdata An optional data frame of validation data which may also inform the computation of PIC. If omitted, the training data contained within the fitted model \code{object} are used.
#' @param ... Further arguments passed to other methods.
#'
#' @details The PIC are criteria developed for model selection within predictive contexts. Given a
#' set of predictive models, the model with the minimum criterion value is preferred.
#'
#' The PIC are developed to asymptotically select the candidate model that minimizes the mean
#' squared error of prediction (MSEP), thus behaving similarly to the the Akaike Information
#' Criterion (AIC). Contrasting with AIC, however, the PIC do not require the assumption of
#' validation data that are independent and identically distributed to the set of training data.
#' The PIC thus offer greater flexibility in predictive model selection applications.
#'
#' @return The form of the value returned by \code{PIC} depends on the fitted model class and its method-specific arguments. Details may be found in the documentation of each method.
#'
#' @seealso
#' \code{\link{PIC.lm}}
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
#' @param newdata An optional data frame of validation data which may also inform the computation of PIC. If omitted, the training data contained within the fitted model \code{object} are used.
#' @param group_sizes A scalar or numeric vector indicating the sizes of \code{newdata} partitions. See 'Details'.
#' @param ... Further arguments passed to or from other methods.
#'
#' @details PIC.lm computes PIC values based on the supplied model.
#'
#' @return If \code{group_sizes = NULL}, a scalar is returned. Otherwise, \code{newdata} is
#' returned with an appended column labeled 'PIC' containing group-specific PIC values.
#'
#' @seealso
#' \code{\link{PIC}}, \code{\link[stats]{lm}}
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
