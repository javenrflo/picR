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
  mod <- lm(Sepal.Length~., data = iris)

  expect_equal(length(PIC.lm(mod, bootstraps = 2)), 1)
  expect_equal(length(PIC.lm(mod, group_sizes = nrow(iris), bootstraps = 2)), 1)
  expect_equal(length(PIC.lm(mod, newdata = iris[1:10,], group_sizes = 10, bootstraps = 2)), 1)
})

test_that("PIC.lm outputs correct error messages for incorrect inputs", {
  mod <- lm(Sepal.Length~., data = iris)

  expect_error(PIC.lm(iris), 'object must be of class "lm"')
  expect_error(PIC.lm(mod, newdata = as.matrix(iris)), 'newdata must be a data.frame')
  expect_error(PIC.lm(mod, bootstraps = 0), 'bootstraps must be greater than 0')
  expect_error(PIC.lm(mod, bootstraps = c(1,2,5)), 'bootstraps must be a scalar')
  expect_error(PIC.lm(mod, group_sizes = nrow(iris) + 1), 'The largest size of a single group must be less than or equal to the total number of predicted observations.')
  expect_error(PIC.lm(object = mod, newdata = iris[1:10,], group_sizes = 11), 'The largest size of a single group must be less than or equal to the total number of predicted observations.')
  expect_error(PIC.lm(mod, group_sizes = c(1,2,5)), 'Sum of supplied group sizes does not match the total number of predicted observations.')
  expect_error(PIC.lm(mod, newdata = iris[1:10,], group_sizes = c(1,2,5)), 'Sum of supplied group sizes does not match the total number of predicted observations.')
})

test_that("PIC.mlm yields the same value as the multivariable AIC computation when newdata is not provided", {
  mod <- lm(cbind(Sepal.Length, Sepal.Width)~., data = iris)

  mlmAIC <- nrow(iris)*(2*log(2*pi) + log(det(crossprod(mod$residuals)/nrow(mod$residuals))) + 2) +
    2*(2*(nrow(mod$coefficients)) + (2*(2+1))/2)

  expect_equal(PIC.mlm(mod), mlmAIC)
  expect_equal(sum(unique(PIC.mlm(mod, group_sizes = c(50,50,25,10,5,4,3,2,1))$PIC)), mlmAIC)
  expect_equal(sum(unique(PIC.mlm(mod, group_sizes = 5)$PIC)), mlmAIC)
  expect_equal(sum(PIC.mlm(mod, group_sizes = 1)$PIC), mlmAIC)
  expect_equal(sum(PIC.mlm(mod, group_sizes = rep(1, 150))$PIC), mlmAIC)
})

test_that("PIC.mlm yields a scalar when group_sizes is not provided, or when group_sizes = nrow(newdata)", {
  mod <- lm(cbind(Sepal.Length, Sepal.Width)~., data = iris)

  expect_equal(length(PIC.mlm(mod)), 1)
  expect_equal(length(PIC.mlm(mod, group_sizes = nrow(iris))), 1)
})

test_that("PIC.mlm yields a scalar when bootstraps is > 0", {
  mod <- lm(cbind(Sepal.Length, Sepal.Width)~., data = iris)

  expect_equal(length(PIC.mlm(mod, bootstraps = 2)), 1)
  expect_equal(length(PIC.mlm(mod, group_sizes = nrow(iris), bootstraps = 2)), 1)
  expect_equal(length(PIC.mlm(mod, newdata = iris[1:10,], group_sizes = 10, bootstraps = 2)), 1)
})

test_that("PIC.mlm outputs correct error messages for incorrect inputs", {
  mod <- lm(cbind(Sepal.Length, Sepal.Width)~., data = iris)
  mod2 <- lm(cbind(Sepal.Length, Sepal.Width, Petal.Length)~., data = iris)

  expect_error(PIC.mlm(iris), 'object must be of class "mlm"')
  expect_error(PIC.mlm(mod2), 'Only bivariable response models are currently supported')
  expect_error(PIC.mlm(mod, newdata = as.matrix(iris)), 'newdata must be a data.frame')
  expect_error(PIC.mlm(mod, bootstraps = 0), 'bootstraps must be greater than 0')
  expect_error(PIC.mlm(mod, bootstraps = c(1,2,5)), 'bootstraps must be a scalar')
  expect_error(PIC.mlm(mod, group_sizes = nrow(iris) + 1), 'The largest size of a single group must be less than or equal to the total number of predicted observations.')
  expect_error(PIC.mlm(mod, newdata = iris[1:10,], group_sizes = 11), 'The largest size of a single group must be less than or equal to the total number of predicted observations.')
  expect_error(PIC.mlm(mod, group_sizes = c(1,2,5)), 'Sum of supplied group sizes does not match the total number of predicted observations.')
  expect_error(PIC.mlm(mod, newdata = iris[1:10,], group_sizes = c(1,2,5)), 'Sum of supplied group sizes does not match the total number of predicted observations.')
  expect_error(PIC.mlm(mod, newdata = iris[1:10, c(3:5)]), 'newdata must contain columns corresponding to each of the response variables. Values in these columns may be set to NA if the multivariate response is completely unobserved and should otherwise contain observed response values for cases where the multivariate response is partially observed.')
})
