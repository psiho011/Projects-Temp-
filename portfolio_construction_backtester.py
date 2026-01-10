# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:06:21 2026

@author: mpsih
"""

# -*- coding: utf-8 -*-
"""
Portfolio Backtester + Optimizers (EW / MinVar / MaxSharpe / RiskParity)
- Monthly rebalance
- Optional sector caps
- Optional shorting + max weight constraints
- Transaction costs applied on rebalance days
- Plots: equity curves, drawdowns, rolling vol, turnover, last weights
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def parse_tickers(raw: str):
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def download_adjclose(tickers, start="2018-01-01", end=None):
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError("No data returned. Check tickers or dates.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all").dropna()
    return prices


def returns_from_prices(prices):
    return prices.pct_change().dropna()


def annualize_mu(mu_daily, periods=252):
    return mu_daily * periods


def annualize_cov(cov_daily, periods=252):
    return cov_daily * periods


def portfolio_return(w, mu_ann):
    return float(np.dot(w, mu_ann))


def portfolio_vol(w, cov_ann):
    return float(np.sqrt(w.T @ cov_ann @ w))


def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())


def sector_map_yahoo(tickers):
    # Best effort (Yahoo sometimes returns None). If missing, label "Unknown".
    sectors = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            sec = info.get("sector", None)
            sectors[t] = sec if sec else "Unknown"
        except Exception:
            sectors[t] = "Unknown"
    return sectors


def parse_sector_caps(raw: str):
    """
    Input example:
      Technology:0.30,Financial Services:0.25,Unknown:0.10
    Returns dict {sector: cap}
    """
    caps = {}
    raw = raw.strip()
    if not raw:
        return caps
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            caps[k] = float(v)
        except ValueError:
            pass
    return caps


# ----------------------------
# Optimizers
# ----------------------------

def min_variance_weights(cov_ann, bounds, sector_A=None, sector_b=None):
    n = cov_ann.shape[0]
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if sector_A is not None and sector_b is not None:
        constraints.append({"type": "ineq", "fun": lambda w, A=sector_A, b=sector_b: b - (A @ w)})

    def obj(w):
        return w.T @ cov_ann @ w

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def max_sharpe_weights(mu_ann, cov_ann, rf, bounds, sector_A=None, sector_b=None):
    n = cov_ann.shape[0]
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if sector_A is not None and sector_b is not None:
        constraints.append({"type": "ineq", "fun": lambda w, A=sector_A, b=sector_b: b - (A @ w)})

    def neg_sharpe(w):
        pret = portfolio_return(w, mu_ann)
        pvol = portfolio_vol(w, cov_ann)
        if pvol == 0:
            return 1e9
        return -(pret - rf) / pvol

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def risk_parity_weights(cov_ann, bounds):
    """
    Simple, approximate risk parity. Enforces bounds by clipping then re-normalizing.
    """
    n = cov_ann.shape[0]
    w = np.ones(n) / n

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    def risk_contrib_frac(w):
        pvol = portfolio_vol(w, cov_ann)
        if pvol == 0:
            return np.ones_like(w) / len(w)
        mrc = (cov_ann @ w) / pvol
        rc = w * mrc
        s = rc.sum()
        return rc / s if s != 0 else np.ones_like(w) / len(w)

    target = np.ones(n) / n
    step = 0.05

    for _ in range(600):
        frac = risk_contrib_frac(w)
        w = w * (1 + step * (target - frac))

        w = np.clip(w, lb + 1e-10, ub)
        w = w / w.sum()

    return w


# ----------------------------
# Constraints builder
# ----------------------------

def build_bounds(n, allow_short=False, max_weight=1.0):
    if allow_short:
        return [(-max_weight, max_weight)] * n
    else:
        return [(0.0, max_weight)] * n


def build_sector_constraints(tickers, sector_caps):
    """
    Builds inequality constraints A @ w <= b, where each row corresponds to a sector cap.
    If sector cap dict is empty, returns (None, None).
    """
    if not sector_caps:
        return None, None

    sectors = sector_map_yahoo(tickers)

    rows = []
    b = []

    for sec, cap in sector_caps.items():
        row = np.zeros(len(tickers))
        for i, t in enumerate(tickers):
            if sectors[t] == sec:
                row[i] = 1.0
        if row.sum() > 0:
            rows.append(row)
            b.append(cap)

    if not rows:
        return None, None

    A = np.vstack(rows)
    b = np.array(b, dtype=float)
    return A, b


# ----------------------------
# Backtest engine (monthly rebalance)
# ----------------------------

def turnover(w_old, w_new):
    return 0.5 * float(np.sum(np.abs(w_new - w_old)))


def backtest_strategies(
    prices,
    benchmark_prices,
    rf=0.03,
    lookback_days=252,
    rebalance_freq="M",
    allow_short=False,
    max_weight=0.30,
    sector_caps=None,
    tc_per_dollar=0.0005
):
    tickers = list(prices.columns)
    n = len(tickers)

    rets = returns_from_prices(prices)
    bmk_rets = returns_from_prices(benchmark_prices).rename(columns={benchmark_prices.columns[0]: "BMK"})

    rets, bmk_rets = rets.align(bmk_rets, join="inner", axis=0)

    month_ends = rets.index.to_series().groupby(rets.index.to_period(rebalance_freq)).max().values
    month_ends = pd.DatetimeIndex(month_ends)

    bounds = build_bounds(n, allow_short=allow_short, max_weight=max_weight)
    sector_A, sector_b = build_sector_constraints(tickers, sector_caps or {})

    strategies = ["EW", "MINVAR", "MAXSHARPE", "RISKPAR"]
    weights_hist = {s: {} for s in strategies}
    daily_port_rets = {s: pd.Series(index=rets.index, dtype=float) for s in strategies}
    turnover_hist = {s: [] for s in strategies}
    cost_hist = {s: [] for s in strategies}

    w_prev = {s: np.ones(n) / n for s in strategies}

    bh_w = np.ones(n) / n
    bh_daily = (rets @ bh_w).rename("BUYHOLD_EW")

    for dt in rets.index:
        if dt in month_ends:
            end_loc = rets.index.get_loc(dt)
            start_loc = max(0, end_loc - lookback_days)
            window = rets.iloc[start_loc:end_loc]

            if len(window) < 30:
                w_new = {s: w_prev[s] for s in strategies}
            else:
                mu_ann = annualize_mu(window.mean()).values
                cov_ann = annualize_cov(window.cov()).values

                w_new = {}
                w_new["EW"] = np.ones(n) / n
                w_new["MINVAR"] = min_variance_weights(cov_ann, bounds=bounds, sector_A=sector_A, sector_b=sector_b)
                w_new["MAXSHARPE"] = max_sharpe_weights(mu_ann, cov_ann, rf=rf, bounds=bounds, sector_A=sector_A, sector_b=sector_b)
                w_new["RISKPAR"] = risk_parity_weights(cov_ann, bounds=bounds)

            for s in strategies:
                to = turnover(w_prev[s], w_new[s])
                cost = tc_per_dollar * to
                turnover_hist[s].append(to)
                cost_hist[s].append(cost)
                weights_hist[s][dt] = w_new[s]
                w_prev[s] = w_new[s]

        r_vec = rets.loc[dt].values
        for s in strategies:
            daily_port_rets[s].loc[dt] = float(np.dot(w_prev[s], r_vec))
            if dt in month_ends and len(cost_hist[s]) > 0:
                daily_port_rets[s].loc[dt] -= cost_hist[s][-1]

    perf = []
    bm = bmk_rets["BMK"]

    for s in strategies:
        pr = daily_port_rets[s].dropna()
        eq = (1 + pr).cumprod()

        ann_ret = (eq.iloc[-1]) ** (252 / len(pr)) - 1
        ann_vol = pr.std() * np.sqrt(252)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

        active = (pr - bm).dropna()
        te = active.std() * np.sqrt(252)
        ir = (active.mean() * 252) / te if te > 0 else np.nan

        mdd = max_drawdown(eq)
        avg_turn = float(np.mean(turnover_hist[s])) if turnover_hist[s] else 0.0
        total_cost = float(np.sum(cost_hist[s])) if cost_hist[s] else 0.0

        perf.append([s, ann_ret, ann_vol, sharpe, te, ir, mdd, avg_turn, total_cost])

    bm_eq = (1 + bm).cumprod()
    bm_ann_ret = (bm_eq.iloc[-1]) ** (252 / len(bm)) - 1
    bm_ann_vol = bm.std() * np.sqrt(252)
    bm_sharpe = (bm_ann_ret - rf) / bm_ann_vol if bm_ann_vol > 0 else np.nan
    bm_mdd = max_drawdown(bm_eq)

    perf_df = pd.DataFrame(
        perf,
        columns=["Strategy", "CAGR", "Vol", "Sharpe", "TrackingErr", "InfoRatio", "MaxDD", "AvgTurnover", "TotalTC"]
    ).set_index("Strategy")

    bh = bh_daily.dropna()
    bh_eq = (1 + bh).cumprod()
    bh_ann_ret = (bh_eq.iloc[-1]) ** (252 / len(bh)) - 1
    bh_ann_vol = bh.std() * np.sqrt(252)
    bh_sharpe = (bh_ann_ret - rf) / bh_ann_vol if bh_ann_vol > 0 else np.nan
    bh_mdd = max_drawdown(bh_eq)

    # ---- NEW: turnover + tc time series (indexed by rebalance dates) ----
    rebalance_index = pd.DatetimeIndex(list(weights_hist["EW"].keys())) if weights_hist["EW"] else pd.DatetimeIndex([])

    turnover_df = pd.DataFrame({s: pd.Series(turnover_hist[s], index=rebalance_index) for s in strategies}).sort_index()
    tc_df = pd.DataFrame({s: pd.Series(cost_hist[s], index=rebalance_index) for s in strategies}).sort_index()

    extras = {
        "tickers": tickers,
        "strategies": strategies,
        "benchmark": {"CAGR": bm_ann_ret, "Vol": bm_ann_vol, "Sharpe": bm_sharpe, "MaxDD": bm_mdd},
        "buyhold_ew": {"CAGR": bh_ann_ret, "Vol": bh_ann_vol, "Sharpe": bh_sharpe, "MaxDD": bh_mdd},
        "daily_returns": daily_port_rets,   # dict of Series
        "benchmark_returns": bm,            # Series
        "buyhold_returns": bh,              # Series
        "weights_hist": weights_hist,       # dict of {date: weight_vector}
        "turnover": turnover_df,            # DataFrame
        "tc": tc_df                          # DataFrame
    }

    return perf_df, extras


# ----------------------------
# Plotting
# ----------------------------

def plot_results(perf_df, extras, benchmark_ticker="SPY"):
    # Build equity curves
    strat_rets = pd.DataFrame(extras["daily_returns"]).dropna(how="all")
    bm = extras["benchmark_returns"].rename(benchmark_ticker)

    strat_rets, bm = strat_rets.align(bm, join="inner", axis=0)

    equity = (1 + strat_rets).cumprod()
    bm_eq = (1 + bm).cumprod()

    # 1) Equity curves
    plt.figure()
    for c in equity.columns:
        plt.plot(equity.index, equity[c], label=c)
    plt.plot(bm_eq.index, bm_eq.values, label=benchmark_ticker, linewidth=2)
    plt.title("Cumulative Growth of $1")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Drawdowns
    plt.figure()
    for c in equity.columns:
        dd = equity[c] / equity[c].cummax() - 1.0
        plt.plot(dd.index, dd.values, label=c)
    bm_dd = bm_eq / bm_eq.cummax() - 1.0
    plt.plot(bm_dd.index, bm_dd.values, label=benchmark_ticker, linewidth=2)
    plt.title("Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Rolling 12-month vol (252 trading days)
    plt.figure()
    roll = strat_rets.rolling(252).std() * np.sqrt(252)
    for c in roll.columns:
        plt.plot(roll.index, roll[c], label=c)
    bm_roll = bm.rolling(252).std() * np.sqrt(252)
    plt.plot(bm_roll.index, bm_roll.values, label=benchmark_ticker, linewidth=2)
    plt.title("Rolling 12-Month Volatility (Ann.)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Turnover at rebalance dates
    if "turnover" in extras and not extras["turnover"].empty:
        plt.figure()
        for c in extras["turnover"].columns:
            plt.plot(extras["turnover"].index, extras["turnover"][c], marker="o", label=c)
        plt.title("Turnover per Rebalance (0.5 * sum(|Δw|))")
        plt.xlabel("Rebalance Date")
        plt.ylabel("Turnover")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 5) Last rebalance weights (bar charts)
    tickers = extras["tickers"]
    weights_hist = extras["weights_hist"]

    for strat, wh in weights_hist.items():
        if not wh:
            continue
        last_dt = sorted(wh.keys())[-1]
        w = np.array(wh[last_dt])
        w_ser = pd.Series(w, index=tickers).sort_values(ascending=False)

        plt.figure()
        plt.bar(w_ser.index, w_ser.values)
        plt.title(f"Last Rebalance Weights: {strat} ({last_dt.date()})")
        plt.xlabel("Ticker")
        plt.ylabel("Weight")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.show()


# ----------------------------
# Main
# ----------------------------

def main():
    raw = input("Enter tickers (comma separated), e.g. AAPL,MSFT,AMZN: ").strip()
    tickers = parse_tickers(raw)
    if len(tickers) < 2:
        print("Need at least 2 tickers.")
        return

    start = input("Start date YYYY-MM-DD (default 2018-01-01): ").strip() or "2018-01-01"
    bmk = input("Benchmark ticker (default SPY): ").strip().upper() or "SPY"

    rf_raw = input("Annual risk-free rate (default 0.03): ").strip()
    rf = float(rf_raw) if rf_raw else 0.03

    lookback_raw = input("Lookback days for estimates (default 252): ").strip()
    lookback = int(lookback_raw) if lookback_raw else 252

    allow_short = (input("Allow shorting? (y/n, default n): ").strip().lower() == "y")

    maxw_raw = input("Max weight per asset (default 0.30): ").strip()
    maxw = float(maxw_raw) if maxw_raw else 0.30

    tc_raw = input("Transaction cost per $ traded (default 0.0005 = 5 bps): ").strip()
    tc = float(tc_raw) if tc_raw else 0.0005

    sector_raw = input("Optional sector caps (Yahoo sectors). Format: Technology:0.30,Financial Services:0.25 (or blank): ")
    sector_caps = parse_sector_caps(sector_raw)

    print("\nDownloading prices...")
    prices = download_adjclose(tickers, start=start)
    bmk_prices = download_adjclose([bmk], start=start)

    good = prices.columns[prices.notna().sum() > 50].tolist()
    prices = prices[good].dropna()
    if len(good) < 2:
        print("Not enough usable tickers after cleaning.")
        return
    if good != tickers:
        print("Using filtered tickers:", ", ".join(good))

    perf, extras = backtest_strategies(
        prices=prices,
        benchmark_prices=bmk_prices,
        rf=rf,
        lookback_days=lookback,
        rebalance_freq="M",
        allow_short=allow_short,
        max_weight=maxw,
        sector_caps=sector_caps,
        tc_per_dollar=tc
    )

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n" + "=" * 70)
    print("Backtest Summary (Monthly Rebalance)")
    print("=" * 70)
    print(perf)

    print(f"\nBenchmark ({bmk}) stats:")
    print(extras["benchmark"])

    print("\nBuy-and-hold Equal Weight (no rebalance, no costs) stats:")
    print(extras["buyhold_ew"])

    print("\n" + "=" * 70)
    print("Last Rebalance Weights")
    print("=" * 70)
    for s, wh in extras["weights_hist"].items():
        if not wh:
            continue
        last_dt = sorted(wh.keys())[-1]
        w = wh[last_dt]
        w_ser = pd.Series(w, index=prices.columns).sort_values(ascending=False)
        print(f"\n{s} @ {last_dt.date()}")
        print(w_ser.to_string())

    # ---- PLOTS ----
    plot_results(perf, extras, benchmark_ticker=bmk)


if __name__ == "__main__":
    main()
