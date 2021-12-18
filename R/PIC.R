#' Compute Predictive Information Criteria (PIC)
#'
#' @description PIC is a generic function for computing PIC from the results of various model fitting functions. The function invokes particular \emph{methods} which depend on the class of the first argument.
#'
#' @param object a model object to evaluate via predictive information criteria (PIC)
#' @param ... additional arguments affecting the PIC computed
#'
#' @details Currently, only a single method exists for this generic function. This method is for the class "lm" of linear models. Additional methods are in development.
#'
#' @return The form of the value returned by PIC depends on the class of its argument and the additional arguments supplied. See the documentation of the particular method(s) for details of what is produced by that method.
#'
#' @export
#'
PIC <- function(object, ...){
  UseMethod("PIC", object)
}


#' PIC method for Linear Model Fits
#'
#' @param object Object of class inheriting from "lm"
#' @param newdata An optional data frame in which to look for variables with which to compute PIC. If omitted, the fitted values are used.
#' @param group_sizes A scalar or vector indicating the sizes of data partitions. See details.
#' @param ... further arguments passed to or from other methods.
#'
#' @details PIC.lm computes PIC values based on the supplied model.
#' @return value
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
