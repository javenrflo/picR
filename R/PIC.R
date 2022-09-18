#' Predictive Information Criteria
#'
#' @description \code{PIC} is the S3 generic function for computing predictive information criteria (PIC).
#' Depending on the \code{\link[base]{class}} of the fitted model supplied to **\code{object}**, the function
#' invokes the appropriate method for computing PIC.
#'
#' @param object A fitted model object.
#' @param newdata An optional dataframe to be used as validation data in computing PIC. If omitted, the training data contained within **\code{object}** are used.
#' @param ... Further arguments passed to other methods.
#'
#' @details The PIC are model selection criteria that may be used to select from among predictive models in a candidate set.
#' The model with the minimum criterion value is preferred.
#'
#' The PIC asymptotically select the candidate model that minimizes the mean squared error of prediction (MSEP),
#' thus behaving similarly to the the Akaike Information Criterion (AIC). However in contrast to the AIC, the PIC
#' do not assume a panel of validation data that are independent and identically distributed to the set of training
#' data. This effectively enables the PIC to accommodate training/validation data \emph{heterogeneity}, where training
#' and validation data may differ from one another in distribution.
#'
#' Data heterogeneity is arguably the more typical circumstance in practice, especially when one considers applications
#' where a set of covariates are used to model and predict some response. In these regression contexts, one often predicts
#' values of the response at combinations of covariate values not necessarily used in training the predictive model.
#'
#'
#' @return The form of the value returned by \code{PIC} depends on the fitted model class and its method-specific arguments.
#' Details may be found in the documentation of each method.
#'
#' @examples
#' data(iris)
#'
#' # Fit a regression model
#' mod <- lm(Sepal.Length ~ Sepal.Width + Species, data = iris)
#' PIC(object  = mod,
#'     newdata = data.frame(Sepal.Width = c(0.25, 1.74, 2.99),
#'                          Species = factor(c("setosa", "virginica", "virginica"),
#'                                           levels = c("setosa", "versicolor", "virginica"))))
#'
#' # Fit a bivariable regression model
#' mod <- lm(cbind(Sepal.Length, Sepal.Width) ~ Species + Petal.Length, data = iris)
#' # Note: For multivariable models, response variable columns must be included if
#' #       newdata is specified. If the values of the validation response(s) are
#' #       unknown, specify NA. If partially observed, specify NA only where unknown.
#' PIC(object  = mod,
#'     newdata = data.frame(Sepal.Length = c(4.1, NA, NA),
#'                          Sepal.Width  = c(NA,NA,3.2),
#'                          Petal.Length = c(1.2, 3.5, 7),
#'                          Species = factor(c("setosa", "virginica", "virginica"),
#'                                           levels = c("setosa", "versicolor", "virginica"))))
#'
#' @seealso
#' \code{\link[picR]{PIC.lm}}, \code{\link[picR]{PIC.mlm}}
#'
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
#' @param newdata An optional dataframe to be used as validation data in computing PIC. If omitted, the training data contained within \code{object} are used.
#' @param group_sizes An optional scalar or numeric vector indicating the sizes of \code{newdata} partitions. If omitted, \code{newdata} is not partitioned. See 'Details'.
#' @param bootstraps An optional numeric value indicating the number of bootstrap samples to use for a bootstrapped PIC. See 'Details'.
#' @param ... Further arguments passed to or from other methods.
#'
#' @details \code{PIC.lm} computes PIC values based on the supplied regression model. Candidate models with relatively smaller criterion values are preferred.
#' Depending on the value(s) supplied to **\code{group_sizes}**, one of three implementations of PIC are computed:
#'
#' \itemize{
#'   \item \strong{iPIC}: The individualized predictive information criterion (iPIC) is computed when **\code{group_sizes = 1}**. A value
#'   of iPIC is determined for each \emph{individual} observation in **\code{newdata}**. Using iPIC, one may thus select optimal predictive
#'   models specific to each particular validation datapoint.
#'
#'   \item \strong{gPIC}: The group predictive information criterion (gPIC) is computed when **\code{group_sizes > 1}** or
#'   **\code{is.vector(group_sizes) == TRUE}**. A value of gPIC is determined for each cohort or \emph{group} of observations
#'   defined by the partitions of **\code{newdata}**. Using gPIC, one may thus select optimal predictive models specific to each
#'   group of validation datapoints. For the class of regression models, the gPIC value of a group of validation observations
#'   is equivalent to the sum of their individual iPIC values.
#'
#'   \item \strong{tPIC}: The total predictive information criterion (tPIC) is computed when **\code{group_sizes = NULL}**. Computation of
#'   the tPIC is the default, and one may use the tPIC to select the optimal predictive model for the entire set of validation
#'   datapoints. The tPIC and gPIC are equivalent when **\code{group_sizes = m}**, where \code{m} is the number of observations in
#'   **\code{newdata}**. When **\code{newdata}** is not supplied, tPIC is exactly equivalent to the Akaike Information Criterion (\link[stats]{AIC}).
#' }
#'
#' If a numeric value is supplied to **\code{bootstraps}** the total Predictive information criterion (tPIC) is computed **\code{bootstraps}** times, where
#' generated bootstrap samples are each used as sets of validation data in computing the tPIC. The resulting tPIC values are then averaged to generate a single,
#' bootstrapped tPIC value. Model selection based on this bootstrapped tPIC value may lead to the selection of a more generally applicable predictive model whose
#' predictive accuracy is not strictly optimized to a particular set of validation data.
#'
#' For further details, see \href{https://iro.uiowa.edu/esploro/outputs/doctoral/A-new-class-of-information-criteria/9984097169902771?institution=01IOWA_INST}{*A new class of information criteria for improved prediction in the presence of training/validation data heterogeneity*}.
#'
#' @return If \code{group_sizes = NULL} or \code{bootstraps > 0}, a scalar is returned. Otherwise, \code{newdata} is
#' returned with an appended column labeled 'PIC' containing either iPIC or gPIC values,
#' depending on the value provided to \code{group_sizes}.
#'
#' @seealso
#' \code{\link[picR]{PIC}}, \code{\link[picR]{PIC.mlm}}, \code{\link[stats]{lm}}
#'
#' @references
#' Flores, J.E. (2021), *A new class of information criteria for improved prediction in the presence of training/validation data heterogeneity* \[Unpublished PhD dissertation\]. University of Iowa.
#'
#' @examples
#' data(iris)
#'
#' # Fit a regression model
#' mod <- lm(Sepal.Length ~ ., data = iris)
#' class(mod)
#'
#' # Hypothetical validation data
#' set.seed(1)
#' vdat <- iris[sample(1:nrow(iris), 10),]
#'
#' # tPIC, newdata not supplied
#' PIC(object = mod)
#' AIC(mod) # equivalent to PIC since training and validation data are the same above
#'
#' # tPIC, newdata supplied
#' PIC(object = mod, newdata = vdat)
#' AIC(mod) # not equivalent to PIC since training and validation data differ above
#'
#' # gPIC
#' PIC(object = mod, newdata = vdat, group_sizes = c(5,3,2))
#' PIC(object = mod, newdata = vdat, group_sizes = 5)
#'
#' # iPIC
#' PIC(object = mod, newdata = vdat, group_sizes = rep(1, 10))
#' PIC(object = mod, newdata = vdat, group_sizes = 1)
#'
#' # bootstrapped tPIC (based on 10 bootstrap samples)
#' set.seed(1)
#' PIC(object = mod, bootstraps = 10)
#'
#' @export
#'
PIC.lm <- function(object, newdata, group_sizes = NULL, bootstraps = NULL, ...){
  if(!inherits(object, "lm")){
    stop('object must be of class "lm"')
  }

  if(!missing(newdata)){
    if(!inherits(newdata, "data.frame")){
      stop('newdata must be a data.frame')
    }
  }

  if(!is.null(bootstraps)){

    if(length(bootstraps) != 1){
      stop("bootstraps must be a scalar")
    }

    if(bootstraps <= 0){
      stop("bootstraps must be greater than 0")
    }

    btPICs <- lapply(1:bootstraps,
                     function(x, object, ...){
                       vars <- as.character(attr(object$terms, "variables"))[-1] ## [1] is the list call
                       yidx <- attr(object$terms, "response") # index of response var

                       btdata <- object$model[, vars[-yidx], drop = FALSE] # select all but the response
                       bidx   <- sample(1:dim(btdata)[1], size = dim(btdata)[1], replace = TRUE)
                       btdata <- btdata[bidx,,drop = FALSE]
                       PIC(object = object, newdata = btdata,
                           group_sizes = NULL, bootstraps = NULL)
                     }, object = object)

    return(mean(Reduce("c", btPICs)))
  }

  X.obs <- stats::model.matrix(object)

  if (missing(newdata) || is.null(newdata)) {
    X.pred  <- X.obs
    newdata <- object$model
  } else {
    Terms <- stats::delete.response(object$terms)
    X.pred <- stats::model.matrix(Terms, newdata)
  }

  sigma.mle <- sqrt(crossprod(object$residuals)/nrow(X.obs))
  nn        <- nrow(X.obs)

  gof.pic <- rep(log(2*pi*sigma.mle^2) + 1, nrow(X.pred))

  RS.dist <- diag(X.pred %*% invmat(crossprod(X.obs)) %*% t(X.pred))
  pen.pic <- 2*(RS.dist + 1/nn)

  pic.vec <- gof.pic + pen.pic

  if(is.null(group_sizes)){

    return(sum(pic.vec))

  } else{
    if(length(group_sizes) == 1){
      if(group_sizes > nrow(X.pred)){
        stop("The largest size of a single group must be less than or equal to the total number of predicted observations.")
      } else if(group_sizes == nrow(X.pred)){

        return(sum(pic.vec))

      } else{
        gpic    <- split(pic.vec, ceiling(seq_along(pic.vec)/group_sizes))
        gpic    <- lapply(gpic, sum)

        idx     <- 1:nrow(newdata)
        gidx    <- split(idx, ceiling(seq_along(idx)/group_sizes))
      }

    } else{
      if(sum(group_sizes) != nrow(X.pred)){
        stop("Sum of supplied group sizes does not match the total number of predicted observations.")
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

#' PIC method for Multivariable Linear Models
#'
#' @description Computation of predictive information criteria for multivariable linear models. Currently, computations are
#' supported for only bivariable linear models.
#'
#' @param object A fitted model object of \code{\link[base]{class}} "mlm".
#' @param newdata An optional dataframe to be used as validation data in computing PIC. If omitted, the training data contained within \code{object} are used. If specified, **\code{newdata}** must contain columns for each model response. See 'Details'.
#' @param group_sizes An optional scalar or numeric vector indicating the sizes of \code{newdata} partitions. If omitted, \code{newdata} is not partitioned. See 'Details'.
#' @param bootstraps An optional numeric value indicating the number of bootstrap samples to use for a bootstrapped PIC. See 'Details'.
#' @param ... Further arguments passed to or from other methods.
#'
#' @details \code{PIC.mlm} computes PIC values based on the supplied multivariable regression model. Candidate models with relatively smaller criterion values are preferred.
#' Depending on the value(s) supplied to **\code{group_sizes}**, one of three implementations of PIC are computed:
#'
#' \itemize{
#'   \item \strong{iPIC}: The individualized predictive information criterion (iPIC) is computed when **\code{group_sizes = 1}**. A value
#'   of iPIC is determined for each \emph{individual} observation in **\code{newdata}**. Using iPIC, one may thus select optimal predictive
#'   models specific to each particular validation datapoint.
#'
#'   \item \strong{gPIC}: The group predictive information criterion (gPIC) is computed when **\code{group_sizes > 1}** or
#'   **\code{is.vector(group_sizes) == TRUE}**. A value of gPIC is determined for each cohort or \emph{group} of observations
#'   defined by the partitions of **\code{newdata}**. Using gPIC, one may thus select optimal predictive models specific to each
#'   group of validation datapoints. For the class of regression models, the gPIC value of a group of validation observations
#'   is equivalent to the sum of their individual iPIC values.
#'
#'   \item \strong{tPIC}: The total predictive information criterion (tPIC) is computed when **\code{group_sizes = NULL}**. Computation of
#'   the tPIC is the default, and one may use the tPIC to select the optimal predictive model for the entire set of validation
#'   datapoints. The tPIC and gPIC are equivalent when **\code{group_sizes = m}**, where \code{m} is the number of observations in
#'   **\code{newdata}**. When **\code{newdata}** is not supplied, tPIC is exactly equivalent to the Akaike Information Criterion (\link[stats]{AIC}).
#' }
#'
#' Distinct from the computation for the class of "lm" models (\link[picR]{PIC.lm}), the PIC computation for multivariable regression models differs
#' depending on the whether validation data are partially or completely unobserved. If partially unobserved, where only some values of the multivariable response vector
#' are unknown/unobserved, any remaining observed values are used in the PIC computation.
#'
#' If a numeric value is supplied to **\code{bootstraps}** the total Predictive information criterion (tPIC) is computed **\code{bootstraps}** times, where
#' generated bootstrap samples are each used as sets of validation data in computing the tPIC. It is assumed that the multivariable response vectors are each
#' completely unobserved. The resulting tPIC values are then averaged to generate a single,
#' bootstrapped tPIC value. Model selection based on this bootstrapped tPIC value may lead to the selection of a more generally applicable predictive model whose
#' predictive accuracy is not strictly optimized to a particular set of validation data.
#'
#' For further details, see \href{https://iro.uiowa.edu/esploro/outputs/doctoral/A-new-class-of-information-criteria/9984097169902771?institution=01IOWA_INST}{*A new class of information criteria for improved prediction in the presence of training/validation data heterogeneity*}.
#'
#' @return If \code{group_sizes = NULL} or \code{bootstraps > 0}, a scalar is returned. Otherwise, \code{newdata} is
#' returned with an appended column labeled 'PIC' containing either iPIC or gPIC values,
#' depending on the value provided to \code{group_sizes}.
#'
#' @seealso
#' \code{\link[picR]{PIC}}, \code{\link[picR]{PIC.lm}}, \code{\link[stats]{lm}}
#'
#' @references
#' Flores, J.E. (2021), *A new class of information criteria for improved prediction in the presence of training/validation data heterogeneity* \[Unpublished PhD dissertation\]. University of Iowa.
#'
#' @examples
#' require(dplyr, quietly = TRUE)
#' data(iris)
#'
#' # Fit a bivariable regression model
#' mod <- lm(cbind(Sepal.Length, Sepal.Width) ~ ., data = iris)
#' class(mod)
#'
#' # Hypothetical validation data
#' set.seed(1)
#' vdat <- iris[sample(1:nrow(iris), 10),]
#'
#' # tPIC, completely unobserved response data
#' PIC(object = mod, newdata = vdat %>% dplyr::mutate(Sepal.Length = NA, Sepal.Width = NA))
#'
#' # tPIC, partially unobserved response data
#' PIC(object = mod, newdata = vdat %>% dplyr::mutate(Sepal.Length = NA))
#'
#' # tPIC, mix of completely and partially unobserved cases.
#' PIC(object = mod, newdata = vdat %>%
#' dplyr::mutate(Sepal.Length = ifelse(Sepal.Length < 6, NA, Sepal.Length),
#' Sepal.Width = ifelse(Sepal.Width < 3.3, NA, Sepal.Width)))
#'
#' # tPIC, newdata not supplied
#' PIC(object = mod)
#'
#' # gPIC
#' PIC(object = mod, newdata = vdat, group_sizes = c(5,3,2))
#' PIC(object = mod, newdata = vdat, group_sizes = 5)
#'
#' # iPIC
#' PIC(object = mod, newdata = vdat, group_sizes = rep(1, 10))
#' PIC(object = mod, newdata = vdat, group_sizes = 1)
#'
#' # bootstrapped tPIC (based on 10 bootstrap samples)
#' set.seed(1)
#' PIC(object = mod, bootstraps = 10)
#'
#' @export
#'
PIC.mlm <- function(object, newdata, group_sizes = NULL, bootstraps = NULL, ...){

  if(!inherits(object, "mlm")){
    stop('object must be of class "mlm"')
  }

  vars <- as.character(attr(object$terms, "variables"))[-1] ## [1] is the list call
  yidx <- attr(object$terms, "response") # index of response var
  if(length(setdiff(all.vars(object$terms), vars[-yidx])) > 2){
    stop('Only bivariable response models are currently supported')
  }

  if(!missing(newdata)){
    if(!inherits(newdata, "data.frame")){
      stop('newdata must be a data.frame')
    }
    if(any(!(setdiff(all.vars(object$terms), vars[-yidx]) %in% names(newdata)))){
      stop('newdata must contain columns corresponding to each of the response variables. Values in these columns may be set to NA if the multivariate response is completely unobserved and should otherwise contain observed response values for cases where the multivariate response is partially observed.')
    }
  }

  if(!is.null(bootstraps)){

    if(length(bootstraps) != 1){
      stop("bootstraps must be a scalar")
    }

    if(bootstraps <= 0){
      stop("bootstraps must be greater than 0")
    }

    btPICs <- lapply(1:bootstraps,
                     function(x, object, vars, yidx, ...){
                       btdata <- object$model[, vars[-yidx], drop = FALSE] # select all but the response
                       bidx   <- sample(1:dim(btdata)[1], size = dim(btdata)[1], replace = TRUE)
                       btdata <- btdata[bidx,,drop = FALSE]
                       btdata[, setdiff(all.vars(object$terms), vars[-yidx])] <- NA
                       PIC(object = object, newdata = btdata,
                           group_sizes = NULL, bootstraps = NULL)
                     }, object = object, vars = vars, yidx = yidx)

    return(mean(Reduce("c", btPICs)))
  }

  X.obs  <- stats::model.matrix(object)

  if (missing(newdata) || is.null(newdata)) {
    X.pred         <- X.obs

    newdata        <- object$model
    newdata.rsp    <- NULL
    nomiss.rsp.idx <- vector("list", length = nrow(X.pred))

  } else {
    Terms <- stats::delete.response(object$terms)
    X.pred <- stats::model.matrix(Terms, newdata)

    newdata.rsp <- newdata[setdiff(all.vars(object$terms), vars[-yidx])]
    nomiss.rsp.idx <- apply(newdata.rsp, 1, function(x){which(!is.na(x))}, simplify = FALSE)
  }

  # Extract parameter estimates and compute universal elements for the criterion
  b.h  <- object$coefficients
  s.h  <- crossprod(object$residuals)/nrow(object$residuals)
  d.h  <- det(s.h)

  # The following makes use of the cyclical property of the trace.
  # Note that
  # trace(t(X.pred[i,,drop = FALSE]) %*% X.pred[i,,drop = FALSE] %*% solve(crossprod(X.obs)))
  # is equivalent to the ith element of
  # diag(X.pred %*% solve(crossprod(X.obs)) %*% t(X.pred))
  RS.dist  <- diag(X.pred %*% solve(crossprod(X.obs)) %*% t(X.pred))

  pic.vec <- Reduce("c",
                    lapply(1:nrow(newdata), function(idx, newdata.rsp, nomiss.rsp.idx, RS.dist, X.pred, X.obs, d.h){
                      if(length(nomiss.rsp.idx[[idx]]) %in% c(0, ncol(newdata.rsp))){
                        gof <- 2*log(2*pi) + log(d.h) + 2
                        pen <- 2*2*RS.dist[idx] + 2*(3/nrow(X.obs))
                        return(gof + pen)
                      } else{
                        nomissidx <- nomiss.rsp.idx[[idx]]
                        rii <- (newdata.rsp[idx, nomissidx] - crossprod(t(X.pred[idx, , drop = FALSE]),
                                                                       b.h[, nomissidx, drop = FALSE]))^2
                        gof <- 2*log(2*pi) + log(d.h) + 1 + rii/diag(s.h)[nomissidx]
                        pen <- 2*2*RS.dist[idx] + 2*(3/nrow(X.obs))*rii/diag(s.h)[nomissidx]
                        return(gof + pen)
                      }
                    },
                    newdata.rsp    = newdata.rsp,
                    nomiss.rsp.idx = nomiss.rsp.idx,
                    RS.dist        = RS.dist,
                    X.pred         = X.pred,
                    X.obs          = X.obs,
                    d.h            = d.h)
  )

  if(is.null(group_sizes)){

    return(sum(pic.vec))

  } else{
    if(length(group_sizes) == 1){
      if(group_sizes > nrow(X.pred)){
        stop("The largest size of a single group must be less than or equal to the total number of predicted observations.")
      } else if(group_sizes == nrow(X.pred)){

        return(sum(pic.vec))

      } else{
        gpic    <- split(pic.vec, ceiling(seq_along(pic.vec)/group_sizes))
        gpic    <- lapply(gpic, sum)

        idx     <- 1:nrow(newdata)
        gidx    <- split(idx, ceiling(seq_along(idx)/group_sizes))
      }

    } else{
      if(sum(group_sizes) != nrow(X.pred)){
        stop("Sum of supplied group sizes does not match the total number of predicted observations.")
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
