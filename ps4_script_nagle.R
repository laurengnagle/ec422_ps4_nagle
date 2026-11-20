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