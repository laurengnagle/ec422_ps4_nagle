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





