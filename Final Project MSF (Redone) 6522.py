"""
Created on Wed Dec  3 15:57:43 2025

@author: mpsih
"""

# -*- coding: utf-8 -*-
"""
American option pricing via Monte Carlo with Merton jump-diffusion
and antithetic variates. Uses Yahoo data for S0 and sigma.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# 1. Merton jump-diffusion simulators

def simulate_merton_jump_diffusion(
    S0,
    alpha,
    sigma,
    lam,
    alpha_J,
    sigma_J,
    T=1.0,
    steps=252,
    n_paths=1,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    k = np.exp(alpha_J) - 1.0
    alpha_hat = alpha - lam * k

    paths = np.zeros((steps + 1, n_paths))
    paths[0, :] = S0

    for t in range(1, steps + 1):
        Z = np.random.randn(n_paths)
        m = np.random.poisson(lam * dt, size=n_paths)
        Z_J = np.random.randn(n_paths)

        jump_term = (
            m * (alpha_J - 0.5 * sigma_J**2)
            + sigma_J * np.sqrt(m) * Z_J
        )

        diffusion_term = (
            (alpha_hat - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

        paths[t, :] = paths[t - 1, :] * np.exp(diffusion_term + jump_term)

    return paths


def simulate_merton_jump_diffusion_antithetic(
    S0,
    alpha,
    sigma,
    lam,
    alpha_J,
    sigma_J,
    T=1.0,
    steps=252,
    n_paths=1000,
    seed=123
):
    """
    Antithetic Merton jump–diffusion simulator.
    """

    if seed is not None:
        np.random.seed(seed)

    # ensure even number of paths
    if n_paths % 2 == 1:
        n_paths += 1

    n_half = n_paths // 2
    dt = T / steps

    k = np.exp(alpha_J) - 1.0
    alpha_hat = alpha - lam * k

    paths = np.zeros((steps + 1, n_paths))
    paths[0, :] = S0

    for t in range(1, steps + 1):
        Z_half  = np.random.randn(n_half)
        ZJ_half = np.random.randn(n_half)
        m_half  = np.random.poisson(lam * dt, size=n_half)

        Z   = np.concatenate([Z_half, -Z_half])
        Z_J = np.concatenate([ZJ_half, -ZJ_half])
        m   = np.concatenate([m_half,  m_half])

        jump_term = (
            m * (alpha_J - 0.5 * sigma_J**2)
            + sigma_J * np.sqrt(m) * Z_J
        )

        diffusion_term = (
            (alpha_hat - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

        paths[t, :] = paths[t - 1, :] * np.exp(diffusion_term + jump_term)

    return paths

# 2. Naive American pricer

def price_american_naive(paths, K, r, T, option_type="call"):
    """
    Naïve Monte Carlo valuation of an American option (put or call).

    • paths: (steps+1, n_paths)
    • K: strike
    • r: risk-free (cont. comp.)
    • T: maturity in years
    • option_type: "call" or "put"
    """
    steps = paths.shape[0] - 1
    times = np.linspace(0, T, steps + 1)
    discounts = np.exp(-r * times)[:, None]

    if option_type.lower() == "call":
        intrinsic = np.maximum(paths - K, 0.0)
    elif option_type.lower() == "put":
        intrinsic = np.maximum(K - paths, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = intrinsic * discounts
    path_values = discounted.max(axis=0)
    return path_values.mean()

# 3. Efficient (antithetic) American pricer

def price_american_antithetic(
    S0,
    K,
    r,
    sigma,
    lam,
    alpha_J,
    sigma_J,
    T=1.0,
    steps=252,
    n_paths=10000,
    option_type="call",
    seed=123
):
    """
    Efficient American option pricing:
      - Merton jump–diffusion with antithetic variates
      - Same naive early-exercise rule
    """
    paths = simulate_merton_jump_diffusion_antithetic(
        S0=S0,
        alpha=r,   # intending to make risk neutral drift
        sigma=sigma,
        lam=lam,
        alpha_J=alpha_J,
        sigma_J=sigma_J,
        T=T,
        steps=steps,
        n_paths=n_paths,
        seed=seed
    )

    price = price_american_naive(paths, K=K, r=r, T=T, option_type=option_type)
    return price

# 4. Yahoo Finace gets S0 and sigma 
def get_yahoo_s0_and_sigma(ticker, years=1):
    """
    Get last close price (S0) and annualized vol (sigma) from Yahoo.

    Uses Close prices (auto_adjust=True by default) and converts to NumPy
    immediately to avoid any pandas alignment headaches.
    """
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)

    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}.")

    # If MultiIndex, pick 'Close' level
    if isinstance(data.columns, pd.MultiIndex):
        close_cols = [c for c in data.columns if c[0] == "Close"]
        if not close_cols:
            raise KeyError(f"'Close' not found in columns for {ticker}: {data.columns}")
        close = data[close_cols[0]].dropna().to_numpy()
    else:
        if "Close" not in data.columns:
            raise KeyError(f"'Close' not found in columns for {ticker}: {data.columns}")
        close = data["Close"].dropna().to_numpy()

    if close.shape[0] < 2:
        raise ValueError(f"Not enough price data for {ticker}: len={close.shape[0]}")

    # NumPy from here on
    log_rets = np.diff(np.log(close))
    sigma_daily = log_rets.std()
    sigma_annual = sigma_daily * np.sqrt(252)

    S0 = float(close[-1])

    return S0, sigma_annual

# 5. price American options for ticker

def price_american_for_ticker(
    ticker,
    r=0.04088,      # 10-year treasury yield
    T=0.5,          # 6-month maturity
    lam=1.0,
    alpha_J=-0.02,
    sigma_J=0.10,
    steps=252,
    n_paths_naive=2000,
    n_paths_eff=10000
):
    """
    For a given ticker:
      • Pull S0 and sigma from Yahoo
      • Set K = S0 (ATM option)
      • Price American call/put:
          - naive MC
          - antithetic MC
    """
    S0, sigma = get_yahoo_s0_and_sigma(ticker, years=1)
    K = S0  # ATM

    # naive paths under inded drift of risk-neutral
    paths = simulate_merton_jump_diffusion(
        S0=S0,
        alpha=r,
        sigma=sigma,
        lam=lam,
        alpha_J=alpha_J,
        sigma_J=sigma_J,
        T=T,
        steps=steps,
        n_paths=n_paths_naive,
        seed=123
    )

    call_naive = price_american_naive(paths, K=K, r=r, T=T, option_type="call")
    put_naive  = price_american_naive(paths, K=K, r=r, T=T, option_type="put")

    call_eff = price_american_antithetic(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        lam=lam,
        alpha_J=alpha_J,
        sigma_J=sigma_J,
        T=T,
        steps=steps,
        n_paths=n_paths_eff,
        option_type="call",
        seed=123
    )

    put_eff = price_american_antithetic(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        lam=lam,
        alpha_J=alpha_J,
        sigma_J=sigma_J,
        T=T,
        steps=steps,
        n_paths=n_paths_eff,
        option_type="put",
        seed=123
    )

    return {
        "ticker": ticker,
        "S0": S0,
        "sigma": sigma,
        "K": K,
        "r": r,
        "T": T,
        "call_naive": call_naive,
        "put_naive": put_naive,
        "call_eff": call_eff,
        "put_eff": put_eff,
    }


# 6. Run for your five tickers

if __name__ == "__main__":
    tickers = ["DUK", "TSM", "AMZN", "INTC", "TGT"]

    results = []
    for t in tickers:
        try:
            res = price_american_for_ticker(t)
        except Exception as e:
            print(f"\n*** Error for {t}: {e}")
            continue

        results.append(res)
        print(f"\n=== {t} ===")
        print(f"S0       : {res['S0']:.2f}")
        print(f"sigma    : {res['sigma']:.4f}")
        print(f"K (ATM)  : {res['K']:.2f}")
        print(f"T        : {res['T']:.2f} years")
        print(f"Naive    CALL: {res['call_naive']:.4f}")
        print(f"Efficient CALL: {res['call_eff']:.4f}")
        print(f"Naive    PUT : {res['put_naive']:.4f}")
        print(f"Efficient PUT: {res['put_eff']:.4f}")
#%%
    # 7. VISUALS FOR ALL TICKERS
    # Helper function for pathwise payoffs
    def american_pathwise_payoffs(paths, K, r, T, option_type="call"):
        steps = paths.shape[0] - 1
        times = np.linspace(0, T, steps + 1)
        discounts = np.exp(-r * times)[:, None]

        if option_type.lower() == "call":
            intrinsic = np.maximum(paths - K, 0.0)
        else:
            intrinsic = np.maximum(K - paths, 0.0)

        discounted = intrinsic * discounts
        return discounted.max(axis=0)  # shape (n_paths,)

    # Loop through each successfully priced ticker
    for res in results:
        ticker_vis = res["ticker"]
        S0_vis = res["S0"]
        sigma_vis = res["sigma"]
        K_vis = res["K"]
        r_vis = res["r"]
        T_vis = res["T"]

        lam_vis = 1.0
        alpha_J_vis = -0.02
        sigma_J_vis = 0.10
        steps_vis = 252
        n_paths_vis = 200   # fewer paths for to be able to see and not just brown overlapping

        print(f"\n\n=== Visualizing {ticker_vis} ===")

        #  1) Simulate example jump–diffusion paths 
        paths_vis = simulate_merton_jump_diffusion(
            S0=S0_vis,
            alpha=r_vis,
            sigma=sigma_vis,
            lam=lam_vis,
            alpha_J=alpha_J_vis,
            sigma_J=sigma_J_vis,
            T=T_vis,
            steps=steps_vis,
            n_paths=n_paths_vis,
            seed=123
        )

        #  2) PATH PLOT 
        plt.figure(figsize=(10, 6))
        plt.plot(paths_vis)
        plt.title(f"Simulated Jump–Diffusion Paths: {ticker_vis}")
        plt.xlabel("Time Step")
        plt.ylabel("Stock Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #3) HISTOGRAM: Terminal Prices S_T 
        ST = paths_vis[-1, :]
        plt.figure(figsize=(8, 5))
        plt.hist(ST, bins=30, edgecolor='black')
        plt.title(f"Distribution of Terminal Prices $S_T$: {ticker_vis}")
        plt.xlabel("Terminal Price $S_T$")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        #4) HISTOGRAM: American CALL Payoffs 
        call_vals = american_pathwise_payoffs(paths_vis, K_vis, r_vis, T_vis, "call")
        plt.figure(figsize=(8, 5))
        plt.hist(call_vals, bins=30, edgecolor='black')
        plt.title(f"Discounted American CALL Payoffs: {ticker_vis}")
        plt.xlabel("Payoff")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # 5) HISTOGRAM: American PUT Payoffs
        put_vals = american_pathwise_payoffs(paths_vis, K_vis, r_vis, T_vis, "put")
        plt.figure(figsize=(8, 5))
        plt.hist(put_vals, bins=30, edgecolor='black')
        plt.title(f"Discounted American PUT Payoffs: {ticker_vis}")
        plt.xlabel("Payoff")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # 6) HISTOGRAM: Efficient (Antithetic) American PUT Payoffs
        paths_vis_eff = simulate_merton_jump_diffusion_antithetic(
            S0=S0_vis,
            alpha=r_vis,
            sigma=sigma_vis,
            lam=lam_vis,
            alpha_J=alpha_J_vis,
            sigma_J=sigma_J_vis,
            T=T_vis,
            steps=steps_vis,
            n_paths=n_paths_vis,
            seed=123
        )
        
        put_vals_eff = american_pathwise_payoffs(paths_vis_eff, K_vis, r_vis, T_vis, "put")
        
        plt.figure(figsize=(8, 5))
        plt.hist(put_vals_eff, bins=30, edgecolor='black')
        plt.title(f"Efficient (Antithetic) Discounted American PUT Payoffs: {ticker_vis}")
        plt.xlabel("Payoff")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

