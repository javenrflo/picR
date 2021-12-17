PIC <- function(object, newdata, group_sizes = NULL){

  X.obs <- model.matrix(object)

  if (missing(newdata) || is.null(newdata)) {
    X.pred  <- X.obs
    newdata <- object$model
  } else {
    tt <- terms(object)
    Terms <- delete.response(tt)
    X.pred <- model.matrix(Terms, newdata)
  }

  sigma.mle <- sqrt(crossprod(object$residuals)/nrow(X.obs))
  nn        <- nrow(X.obs)

  gof.pic <- rep(log(2*pi*sigma.mle^2) + 1, nn)

  RS.dist <- diag(X.pred %*% inv.mat(crossprod(X.obs)) %*% t(X.pred))
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

inv.mat <- function(x){chol2inv(chol(x))}
