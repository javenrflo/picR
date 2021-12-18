test_that("PIC.lm yields the same value as AIC() when newdata is not provided", {
  mod <- lm(Sepal.Length~., data = iris)

  expect_equal(PIC.lm(mod), AIC(mod))
  expect_equal(sum(unique(PIC.lm(mod, group_sizes = 5)$PIC)), AIC(mod))
  expect_equal(sum(PIC.lm(mod, group_sizes = 1)$PIC), AIC(mod))
})
