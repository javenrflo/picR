test_that("PIC.lm yields the same value as AIC() when newdata is not provided", {
  mod <- lm(Sepal.Length~., data = iris)

  expect_equal(PIC.lm(mod), AIC(mod))
  expect_equal(sum(unique(PIC.lm(mod, group_sizes = c(100,25,5,10,1,2,2,3,2))$PIC)), AIC(mod))
  expect_equal(sum(unique(PIC.lm(mod, group_sizes = 5)$PIC)), AIC(mod))
  expect_equal(sum(PIC.lm(mod, group_sizes = 1)$PIC), AIC(mod))
  expect_equal(sum(PIC.lm(mod, group_sizes = rep(1, 150))$PIC), AIC(mod))
})

test_that("PIC.lm yields a scalar when group_sizes is not provided, or when group_sizes = nrow(newdata)", {
  mod <- lm(Sepal.Length~., data = iris)

  expect_equal(length(PIC.lm(mod)), 1)
  expect_equal(length(PIC.lm(mod, group_sizes = nrow(iris))), 1)
})

test_that("PIC.lm yields a scalar when bootstraps is > 0", {
  set.seed(1)
  sdat <- data.frame(replicate(10, rnorm(20)))
  sdat$y1 <- sdat$X1 + sdat$X2 + rnorm(20)
  mod <- lm(y1 ~ X1, data = sdat)

  expect_equal(length(PIC.lm(mod, bootstraps = 2)), 1)
  expect_equal(length(PIC.lm(mod, group_sizes = nrow(sdat), bootstraps = 2)), 1)
  expect_equal(length(PIC.lm(mod, newdata = sdat, group_sizes = nrow(sdat), bootstraps = 2)), 1)
})

test_that("PIC.lm outputs correct error messages for incorrect inputs", {
  set.seed(1)
  sdat <- data.frame(replicate(10, rnorm(20)))
  sdatmat <- matrix(replicate(10, rnorm(20)), nrow = 20)
  sdat$y1 <- sdat$X1 + sdat$X2 + rnorm(20)
  mod <- lm(y1 ~ X1, data = sdat)

  expect_error(PIC.lm(sdat), 'object must be of class "lm"')
  expect_error(PIC.lm(mod, newdata = sdatmat), 'newdata must be a data.frame')
  expect_error(PIC.lm(mod, bootstraps = 0), 'bootstraps must be greater than 0')
  expect_error(PIC.lm(mod, bootstraps = c(1,2,5)), 'bootstraps must be a scalar')
  expect_error(PIC.lm(mod, group_sizes = c(1,2,5)), 'Sum of supplied group sizes does not match total predicted observations.')
  expect_error(PIC.lm(mod, newdata = sdat[1:10,], group_sizes = c(1,2,5)), 'Sum of supplied group sizes does not match total predicted observations.')
})

test_that("PIC.mlm yields the same value as the multivariable AIC computation when newdata is not provided", {
  set.seed(1)
  sdat <- data.frame(replicate(10, rnorm(20)))
  sdat$y1 <- sdat$X1 + sdat$X2 + rnorm(20)
  sdat$y2 <- sdat$X1 + sdat$X2 + rnorm(20)
  mod <- lm(cbind(y1, y2) ~ X1, data = sdat)

  mlmAIC <- nrow(sdat)*(2*log(2*pi) + log(det(crossprod(mod$residuals)/nrow(mod$residuals))) + 2) +
    2*(2*(nrow(mod$coefficients)) + (2*(2+1))/2)

  expect_equal(PIC.mlm(mod), mlmAIC)
  expect_equal(sum(unique(PIC.mlm(mod, group_sizes = c(5,4,1,10))$PIC)), mlmAIC)
  expect_equal(sum(unique(PIC.mlm(mod, group_sizes = 5)$PIC)), mlmAIC)
  expect_equal(sum(PIC.mlm(mod, group_sizes = 1)$PIC), mlmAIC)
  expect_equal(sum(PIC.mlm(mod, group_sizes = rep(1, 20))$PIC), mlmAIC)
})


