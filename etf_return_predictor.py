# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:21:17 2026

@author: mpsih
"""

# -*- coding: utf-8 -*-
"""
ETF Return Direction Predictor (Research Tool)

- Downloads ETF prices (Yahoo Finance)
- Builds simple technical features
- Predicts next-day direction (up/down) using logistic regression
- Walk-forward evaluation (more realistic than a single train/test split)
- Plots model diagnostics + optional "signal equity curve" for visualization
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler


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
        px = data["Adj Close"].copy()
    else:
        px = data[["Adj Close"]].copy()
        px.columns = tickers

    px = px.dropna(how="all").dropna()
    return px


def rsi_like(series, window=14):
    # simple RSI-like oscillator (kept readable)
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(price_series: pd.Series) -> pd.DataFrame:
    """
    Predict next-day direction using simple, explainable features.
    Target: y_t = 1 if next-day return > 0 else 0
    """
    px = price_series.dropna().copy()
    ret1 = px.pct_change()

    df = pd.DataFrame(index=px.index)
    df["ret_1d"] = ret1
    df["ret_5d"] = px.pct_change(5)
    df["ret_21d"] = px.pct_change(21)

    # rolling vol (annualized-ish)
    df["vol_21d"] = ret1.rolling(21).std() * np.sqrt(252)

    # simple trend measures
    ma20 = px.rolling(20).mean()
    ma60 = px.rolling(60).mean()
    df["ma20_over_ma60"] = (ma20 / ma60) - 1

    # rsi-like oscillator
    df["rsi_14"] = rsi_like(px, 14)

    # target: next-day direction
    df["fwd_ret_1d"] = ret1.shift(-1)
    df["y_up"] = (df["fwd_ret_1d"] > 0).astype(int)

    # drop rows with NaNs from rolling calcs
    df = df.dropna()
    return df


def walk_forward_predict_proba(X, y, dates, min_train=504, step=21):
    """
    Walk-forward:
      - Start with min_train observations
      - Refit every 'step' days
      - Predict probabilities for the next block
    Returns a Series of predicted probabilities aligned to dates.
    """
    proba = pd.Series(index=dates, dtype=float)

    scaler = StandardScaler()
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000
    )

    n = len(X)
    start = min_train

    while start < n:
        train_end = start
        test_end = min(start + step, n)

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[train_end:test_end]

        # fit scaler on train only
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        p = model.predict_proba(X_test_s)[:, 1]

        proba.iloc[train_end:test_end] = p
        start = test_end

    return proba.dropna()


def plot_confusion(cm, title):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Down", "Up"])
    plt.yticks([0, 1], ["Down", "Up"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------

def main():
    raw = input("Enter ETF tickers (comma separated), e.g. SPY,QQQ,GLD: ").strip()
    tickers = parse_tickers(raw)
    if len(tickers) < 1:
        print("Need at least 1 ticker.")
        return

    start = input("Start date YYYY-MM-DD (default 2018-01-01): ").strip() or "2018-01-01"

    # optional: probability threshold for a "signal" plot
    thr_raw = input("Optional signal threshold for 'Up' (default 0.55): ").strip()
    thr = float(thr_raw) if thr_raw else 0.55

    print("\nDownloading prices...")
    prices = download_adjclose(tickers, start=start)

    # run model per ETF
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < 700:
            print(f"\n{t}: not enough history after cleaning (need ~700+ rows). Skipping.")
            continue

        df = build_features(s)

        feature_cols = ["ret_1d", "ret_5d", "ret_21d", "vol_21d", "ma20_over_ma60", "rsi_14"]
        X = df[feature_cols].values
        y = df["y_up"].values
        dates = df.index

        # walk-forward predicted probabilities
        p_up = walk_forward_predict_proba(X, y, dates, min_train=504, step=21)

        # align y to p_up dates
        y_eval = df.loc[p_up.index, "y_up"]
        fwd_ret = df.loc[p_up.index, "fwd_ret_1d"]

        y_hat = (p_up >= 0.5).astype(int)

        acc = accuracy_score(y_eval, y_hat)
        cm = confusion_matrix(y_eval, y_hat)

        print("\n" + "=" * 70)
        print(f"{t} — Walk-Forward Next-Day Direction Model")
        print("=" * 70)
        print(f"Samples evaluated: {len(p_up)}")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_eval, y_hat, target_names=["Down", "Up"]))

        # --- Plots ---
        # 1) Probability histogram
        plt.figure()
        plt.hist(p_up.values, bins=30, edgecolor="black")
        plt.title(f"{t}: Predicted P(Up) Distribution")
        plt.xlabel("Predicted probability of Up day")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # 2) Confusion matrix
        plot_confusion(cm, title=f"{t}: Confusion Matrix")

        # 3) Optional diagnostic "signal equity curve"
        # long when p_up >= thr, else cash (0 return)
        signal = (p_up >= thr).astype(int)
        strat_ret = signal.values * fwd_ret.values
        eq = (1 + pd.Series(strat_ret, index=p_up.index)).cumprod()
        buyhold = (1 + fwd_ret).cumprod()

        plt.figure()
        plt.plot(eq.index, eq.values, label=f"Signal (thr={thr})")
        plt.plot(buyhold.index, buyhold.values, label="Buy & Hold (same window)")
        plt.title(f"{t}: Diagnostic Equity Curve (Not Live Trading)")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
