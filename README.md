# Projects-Temp-
Temp Place for Mostly Finished Projects
**Overview**

This repository contains Python implementations of core quantitative finance concepts, combining numerical methods, data analysis, and financial modeling. The code emphasizes translating theory into working models using clean program structure, vectorized computation, and reproducible analysis.

**Scope of Work**

**Core Python and Program Design**
Input validation, control flow, modular functions, and structured main() execution for repeatable numerical workflows.

**Numerical Computing**
Vectorized calculations with NumPy, matrix operations, and aggregation for portfolio valuation and optimization tasks.

**Portfolio and Asset Modeling**
Holdings valuation, portfolio aggregation, rebalancing logic, and return analysis across time horizons.

**Time-Value-of-Money and Cash Flow Models**
Compound growth, annuities, inflation adjustments, loan amortization, and depreciation schedules.

**Options Pricing**
Black–Scholes pricing for European and American options, including payoff construction, parameter sensitivity, and numerical approximation techniques.

**Time-Series and Statistical Analysis**
Return distributions, correlation analysis, ARIMA modeling, residual diagnostics, and out-of-sample forecast evaluation.

**Data Analysis with pandas**
Filtering, grouping, conditional slicing, and summary statistics on real financial datasets (e.g., loan-level and market data).

**Visualization**
Time-series plots, payoff diagrams, histograms, QQ plots, and diagnostic charts to support model interpretation.

**Tools and Methods**

Python

NumPy (linear algebra, vectorization)

pandas (data wrangling and time-series handling)

matplotlib (financial and statistical visualization)

Quantitative finance models (Black–Scholes, ARIMA, portfolio math)

# Portfolio Construction Backtester

A small Python script for testing basic quantitative portfolio construction methods on a user-defined set of tickers. It downloads historical adjusted close prices from Yahoo Finance, estimates expected returns and covariance using a rolling lookback window, rebalances monthly, and compares strategy performance to a benchmark (default: SPY). Includes transaction costs and optional sector caps.

## What it does
- **Data**: pulls adjusted close prices via `yfinance`
- **Returns**: computes daily simple returns
- **Rebalance**: monthly (end-of-month trading day)
- **Estimation**: trailing lookback window (default 252 trading days)
- **Strategies**
  - **EW**: Equal weight
  - **MINVAR**: Minimum variance (SLSQP optimizer)
  - **MAXSHARPE**: Maximum Sharpe ratio (SLSQP optimizer)
  - **RISKPAR**: Approximate risk parity (iterative scaling)
- **Constraints**
  - Fully invested: sum(weights) = 1
  - Long-only by default, optional shorting
  - Max position size per asset (default 30%)
  - Optional sector caps using Yahoo “sector” labels (best-effort)
- **Costs**: transaction costs applied on rebalance dates using turnover
  - Turnover = 0.5 * sum(|Δw|)
  - Cost hit = (transaction_cost_per_$) * turnover
- **Outputs**
  - Summary table: CAGR, Vol, Sharpe, Tracking Error, Information Ratio, Max Drawdown, Avg Turnover, Total TC
  - Last rebalance weights per strategy
  - Plots: cumulative growth of $1, drawdowns, rolling 12-month vol, turnover, last weights

## How to run
From terminal:
```bash


