# 034 PRODUCTION MODEL PERFORMANCE

This institutional monitor uses only forecasts captured from actual production runs. It is not a backtest, redesign, or validation framework. Historical replay artifacts are explicitly excluded.

## Overall Health

**WATCH**

WATCH: 1 production signals have completed their 10-session outcome; 20 are required before institutional health classification. No backfilled replay observations are used.

- Completed signals: 1
- Pending signals: 10
- Latest captured signal: 2026-07-28

## Prediction Quality

| Metric | Latest |
|---|---:|
| Rolling Rank IC (20) | INSUFFICIENT_DATA |
| Rolling Rank IC (60) | INSUFFICIENT_DATA |
| Directional Accuracy | INSUFFICIENT_DATA |
| MAE | INSUFFICIENT_DATA |
| RMSE | INSUFFICIENT_DATA |
| Mean Prediction Error | INSUFFICIENT_DATA |
| Calibration Error | INSUFFICIENT_DATA |

## Portfolio and Allocation Quality

| Metric | Latest |
|---|---:|
| Rolling Portfolio Return | INSUFFICIENT_DATA |
| Rolling Sharpe | INSUFFICIENT_DATA |
| Rolling Sortino | INSUFFICIENT_DATA |
| Rolling Maximum Drawdown | INSUFFICIENT_DATA |
| Rolling Alpha vs SPY | INSUFFICIENT_DATA |
| Win Rate | INSUFFICIENT_DATA |
| Top ETF Hit Rate | INSUFFICIENT_DATA |
| Top 3 ETF Hit Rate | INSUFFICIENT_DATA |
| Allocation Turnover | 0.252 |
| Allocation Persistence | +74.8% |
| Average Holding Period | 2.3 sessions |
| Concentration (HHI) | 0.535 |

## Drift Detection

- Rank IC: **INSUFFICIENT_DATA**
- Directional Accuracy: **INSUFFICIENT_DATA**
- Prediction Bias: **INSUFFICIENT_DATA**

## ETF Error Analysis

| ETF | completed_predictions | average_predicted_return | average_realized_return | mean_error | directional_accuracy | overpredicted | underpredicted |
|---|---|---|---|---|---|---|---|
| XSOE | 1 | 0.014000 | -0.022107 | 0.036106 | 0.000000 | True | False |
| QQQM | 1 | 0.006612 | -0.041630 | 0.048242 | 0.000000 | True | False |
| SOXX | 1 | 0.023252 | -0.067520 | 0.090773 | 0.000000 | True | False |
| XLE | 1 | 0.006186 | 0.028551 | -0.022365 | 1.000000 | False | True |
| XLB | 1 | 0.006106 | 0.016014 | -0.009908 | 1.000000 | False | True |
| XLI | 1 | 0.006278 | 0.015690 | -0.009412 | 1.000000 | False | True |

## Recent Completed Predictions

| signal_date | ETF | expected_etf_return | realized_etf_return | prediction_error |
|---|---|---|---|---|
| 2026-07-13 00:00:00 | FEZ | 0.006793 | 0.010375 | -0.003582 |
| 2026-07-13 00:00:00 | GLD | -0.002020 | 0.020429 | -0.022449 |
| 2026-07-13 00:00:00 | IEF | 0.006751 | -0.000107 | 0.006859 |
| 2026-07-13 00:00:00 | IWM | 0.005915 | -0.001942 | 0.007857 |
| 2026-07-13 00:00:00 | QQQM | 0.006612 | -0.041630 | 0.048242 |
| 2026-07-13 00:00:00 | SOXX | 0.023252 | -0.067520 | 0.090773 |
| 2026-07-13 00:00:00 | TLT | 0.006819 | -0.002620 | 0.009439 |
| 2026-07-13 00:00:00 | XLB | 0.006106 | 0.016014 | -0.009908 |
| 2026-07-13 00:00:00 | XLE | 0.006186 | 0.028551 | -0.022365 |
| 2026-07-13 00:00:00 | XLF | 0.007046 | 0.014446 | -0.007401 |
| 2026-07-13 00:00:00 | XLI | 0.006278 | 0.015690 | -0.009412 |
| 2026-07-13 00:00:00 | XLP | 0.006793 | 0.009103 | -0.002310 |
| 2026-07-13 00:00:00 | XLRE | 0.006793 | 0.023714 | -0.016921 |
| 2026-07-13 00:00:00 | XLU | 0.006706 | -0.000875 | 0.007581 |
| 2026-07-13 00:00:00 | XLV | 0.006793 | 0.012329 | -0.005536 |
| 2026-07-13 00:00:00 | XSOE | 0.014000 | -0.022107 | 0.036106 |

## Regime Performance

| regime | completed_signals | average_portfolio_return | average_alpha | win_rate | average_rank_ic |
|---|---|---|---|---|---|
| Strong USD | 1 | -0.006917 | 0.008824 | 0.000000 | -0.447420 |
| Weak USD | 0 |  |  |  |  |
| High VIX | 0 |  |  |  |  |
| Low VIX | 0 |  |  |  |  |
| Strong Growth | 0 |  |  |  |  |
| Weak Growth | 1 | -0.006917 | 0.008824 | 0.000000 | -0.447420 |
| High Risk-Off | 0 |  |  |  |  |
| Low Risk-Off | 1 | -0.006917 | 0.008824 | 0.000000 | -0.447420 |

## Institutional Controls

- A prediction is captured before its outcome is known and keyed by signal date, ETF, and source commit.
- Outcomes mature only after ten later market sessions are available.
- ETF outcomes use the production target convention (signal close to maturity close).
- Portfolio and SPY outcomes use next-session adjusted open through maturity close.
- Health remains WATCH until at least 20 production signals have completed.
- No historical replay, current fitted model, or validation output is used to backfill a past production prediction.
