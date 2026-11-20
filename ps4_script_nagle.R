library(tidyverse)
library(fpp3)
library(BVAR)
# Part 2 ---------------------------------------------------------------
liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)

liquor |>
  autoplot(sales) +
  labs(title = "U.S. Beer, Wine, and Liquor Store Sales",
       x = "Month", y = "Millions of dollars")

liquor |>
  gg_season(log(sales)) +
  labs(title = "Seasonal Plot: Log Liquor Store Sales",
       x = "Month", y = "log(sales)")

# Liquor store sales trend steadily upward, rising from roughly $1,500–2,000 million per 
# month in 1992 to roughly $4,000–5,000 million by 2019. The seasonal pattern is strong since December
# is by far the highest month, likely due to holiday purchases, while January and February are the lowest. 
# The December spikes get noticeably larger as the level of sales rises, so the seasonal swings are 
# proportional to the level (multiplicative). Taking logs converts this into roughly constant-sized swings, 
# which makes the seasonality easier to model.

liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets = ETS(log(sales)))

liquor_fit |> select(arima) |> report()
liquor_fit |> select(ets) |> report()

# fable chose ARIMA(1,0,2)(2,1,1)[12] with drift and ETS(A,A,A). The ARIMA applies no regular 
# difference (d = 0) but one seasonal difference (D = 1), so it models the 12-month change in 
# log sales; the drift term captures the upward trend. The ETS model has additive error, additive 
# (undamped) trend, and additive seasonality. Because it is fit to log sales, additive 
# seasonality in logs implies multiplicative seasonality in levels.

liquor_fit |> select(arima) |> gg_tsresiduals()

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)

# The ARIMA residuals look roughly centered at zero with stable variance and a roughly 
# normal histogram, but the ACF shows several significant spikes (notably near lags 9–10 
# and 23–24). The Ljung–Box test at lag 24 (dof = 6) gives Q = 120 with p ≈ 0, so we reject 
# that the residuals are white noise: there is remaining structure the model hasn’t captured. 
# This doesn’t mean the model will forecast poorly, which the horse race will test.

liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets = ETS(log(sales)),
        arima = ARIMA(log(sales)))

saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")

liquor_cv_fit <- readRDS("results/liquor_cv_fit.rds")

# The sample has 336 months (Jan 1992–Dec 2019). With a first training sample 
# of 120 months, stretch_tsibble creates 336 − 120 + 1 = 217 training samples. 
# We drop the last one (ending Dec 2019) because there’s no actual value to 
# compare its forecast to, leaving 216 forecast origins.

liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

saveRDS(liquor_cv_fc, "results/liquor_cv_fc.rds")

liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE)) |>
  knitr::kable(digits = 2)

# First forecast: Jan 2002; last forecast: Dec 2019 (216 one-step-ahead 
# forecasts). ARIMA is most accurate (RMSE 81.43, MAE 65.65), followed by ETS 
# (94.35, 72.31), with the seasonal naive benchmark far behind (161.26, 136.63).
# RMSE and MAE agree on the ranking. Both ARIMA and ETS cut the benchmark’s 
# errors roughly in half.

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$ets, h = 1, power = 2)
forecast::dm.test(liquor_errs$arima, liquor_errs$snaive, h = 1, power = 2)

# DM test (squared error loss), ARIMA vs ETS: DM = −2.91, p = 0.004. ARIMA vs
# seasonal random walk: DM = −10.76, p < 2.2e−16. The negative statistics mean 
# ARIMA has lower squared errors in both cases, and both differences are 
# statistically significant at the 1% level. ARIMA’s advantage over ETS is real 
# rather than noise, so the liquor chain should use the ARIMA model.

liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)
saveRDS(liquor_combo_fc, "results/liquor_combo_fc.rds")

liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE)) |>
  knitr::kable(digits = 2)

cor(liquor_errs$arima, liquor_errs$ets)

# 3a: The equal-weight combination (RMSE 85.19, MAE 66.77) beats ETS but is 
# worse than ARIMA alone, on both metrics. 

# 3b: The correlation between ARIMA 
# and ETS forecast errors is 0.88, so the two models tend to make the same 
# mistakes and averaging cancels out little error. Combined with the DM test 
# showing ARIMA is significantly more accurate than ETS, equal weights drag the 
# forecast toward the weaker model. Combining helps most when models are 
# similarly accurate and their errors are less correlated; neither condition 
# holds here.

# Part 4 ---------------------------------------------------------------
qd_raw <- read_csv("data/fredqd_2026-09.csv", show_col_types = FALSE) |>
  slice(-(1:2))

growth <- function(x) 100 * difference(log(x))

qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(date,
            gdp    = growth(GDPC1),
            cons   = growth(PCECC96),
            inv    = growth(GPDIC1),
            emp    = growth(PAYEMS),
            infl   = growth(GDPCTPI),
            unrate = difference(UNRATE),
            ffr    = difference(FEDFUNDS),
            gs10   = difference(GS10)) |>
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

y <- qd |> select(-date) |> as.matrix()
nrow(y)
head(y)

N <- nrow(y)
origins <- 120:(N - 1)
actual <- y[121:N, ]

# origins = 120:239, so we produce 120 one-quarter-ahead forecasts per variable.
# The first training sample ends in 1989 Q4, so the first forecast is for 1990 
# Q1; the last forecast is for 2019 Q4.

var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)             # columns: y_t, then its lags 1 to p
  M <- ncol(z)                     # number of variables
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))   # constant, latest p lags
  x %*% B
}

fc_var1 <- NULL
fc_var4 <- NULL
for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

rmse_by_var <- function(f) sqrt(colMeans((actual - f)^2))
rmse_by_var(fc_var1)
rmse_by_var(fc_var4)

# 4.2: The VAR(1) forecasts better for 7 of the 8 variables where only inflation 
# is (marginally) better with the VAR(4), 0.207 vs 0.208. Although the VAR(4) 
# nests the VAR(1), it estimates 33 coefficients per equation (264 total) from 
# as few as 120 observations. Most of the extra lag coefficients are probably 
# close to zero, but OLS estimates them with noise and that estimation
# error feeds directly into the forecasts. The smaller model accepts a little 
# bias in exchange for much lower variance, so it forecasts better which is the 
# bias–variance tradeoff.

bvar_cv <- function(lambda) {
  pr <- bv_priors(hyper = "lambda",
                  mn = bv_mn(lambda = bv_lambda(mode = lambda, sd = 0.4,
                                                min = 0.999 * lambda,
                                                max = 1.001 * lambda),
                             b = 0))
  fc <- NULL
  for (T in origins) {
    fit_T <- bvar(y[1:T, ], lags = 4, priors = pr,
                  n_draw = 2000, n_burn = 1000, verbose = FALSE)
    x <- c(1, t(y[T:(T - 3), ]))   # constant and 4 lags of each variable
    fc <- rbind(fc, x %*% coef(fit_T))   # forecast of y_{T+1}
  }
  fc
}

set.seed(422)
fc_bvar_03 <- bvar_cv(0.3)
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")

set.seed(422)
fc_bvar_005 <- bvar_cv(0.05)
saveRDS(fc_bvar_005, "results/fc_bvar_005.rds")

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)
saveRDS(var_results, "results/var_results.rds")
var_results |> knitr::kable(digits = 3)

# BVAR(4), λ = 0.3 vs OLS VAR(4): the BVAR is more accurate for all 8 variables.
# Both models have the same variables and lags, so the difference comes
# from the Minnesota prior shrinking the 264 coefficients toward zero, which
# reduces estimation noise without dropping any lags.

# vs VAR(1): mixed. The VAR(1) wins for 5 variables (gdp, cons, emp, unrate, 
# ffr) and the BVAR wins for 3 (inv, infl, gs10). The VAR(1) is itself a crude 
# form of shrinkage, setting lags 2–4 to exactly zero, and here it is about as 
# effective as the smoother Bayesian shrinkage.

# λ = 0.05: much worse than λ = 0.3 for 7 of 8 variables (e.g., emp 0.315 vs 
# 0.211, infl 0.330 vs 0.197) and only ffr improves. This prior is so tight that 
# it pushes forecasts toward the historical mean and throws away real dynamics,
# so there is such a thing as too much shrinkage.
# Not a fair estimate. If we chose the winner from this table, we’d be picking 
# it because it did best on this exact evaluation sample, after comparing 
# several models and λ values. Some of that edge is luck, so the winner’s 
# RMSE overstates how well it will forecast in the future (selection 
# bias/“winner’s curse”). A fair estimate would require a fresh holdout sample 
# that wasn’t used to choose the model.

e_var4 <- actual[, "gdp"] - fc_var4[, 1]
e_bvar <- actual[, "gdp"] - fc_bvar_03[, 1]
dm_gdp <- forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)
dm_gdp

e_var4_ffr <- actual[, "ffr"] - fc_var4[, 7]
e_bvar_ffr <- actual[, "ffr"] - fc_bvar_03[, 7]
dm_ffr <- forecast::dm.test(e_var4_ffr, e_bvar_ffr, h = 1, power = 2)
dm_ffr

saveRDS(list(gdp = dm_gdp, ffr = dm_ffr), "results/dm_var.rds")

# 4.3c: DM tests (squared error loss), VAR(4) vs BVAR(4) λ = 0.3. 
# GDP: DM = 2.14, p = 0.035. FFR: DM = 2.00, p = 0.048. The positive statistics 
# mean the VAR(4) has larger squared errors, so the BVAR is significantly more 
# accurate for both variables at the 5% level (though not at 1%). This confirms 
# that shrinkage genuinely improves on the unrestricted OLS VAR(4). However, 
# the evidence is only moderate, and I chose ffr because it had the largest 
# RMSE gap, so it is a favorable case. Many of the smaller differences in the 
# 4.3b table (e.g., BVAR vs VAR(1)) are likely not statistically significant, 
# so the rankings there should be read with caution.


