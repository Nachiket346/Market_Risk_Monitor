"""Historical and EWMA VaR / Expected Shortfall.

Convention: VaR and ES are returned as positive numbers, as a fraction of
portfolio value (0.02 = a 2% one-day loss).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def portfolio_returns(returns, weights):
    """Daily portfolio return, assuming weights are reset to target each day."""
    w = pd.Series(weights, dtype=float).reindex(returns.columns)
    if w.isna().any():
        raise ValueError("Weights missing for: " + ", ".join(w[w.isna()].index))
    if abs(w.sum() - 1.0) > 1e-6:
        raise ValueError(f"Weights sum to {w.sum():.4f}, expected 1.0")
    return returns.mul(w, axis=1).sum(axis=1)


def historical_var(r, conf=0.99):
    return float(-np.quantile(r, 1 - conf))


def historical_es(r, conf=0.99):
    cutoff = np.quantile(r, 1 - conf)
    tail = r[r <= cutoff]
    return float(-tail.mean())


def ewma_vol(r, lam=0.94):
    """EWMA volatility series (zero-mean, RiskMetrics).

    The value on date t uses returns up to and including t, so the last
    value is the forecast for the next day. The first observation seeds
    the recursion; with 3 years of data its effect is negligible.
    """
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var)


def ewma_var_es(r, conf=0.99, lam=0.94):
    """Next-day VaR and ES from EWMA volatility, assuming normal returns."""
    sigma = float(ewma_vol(r, lam).iloc[-1])
    z = norm.ppf(conf)
    var = z * sigma
    es = sigma * norm.pdf(z) / (1 - conf)
    return var, es


def risk_summary(r, confs=(0.95, 0.99), lam=0.94):
    rows = []
    for c in confs:
        e_var, e_es = ewma_var_es(r, c, lam)
        rows.append({
            "confidence": c,
            "hist_VaR": historical_var(r, c),
            "hist_ES": historical_es(r, c),
            "ewma_VaR": e_var,
            "ewma_ES": e_es,
        })
    return pd.DataFrame(rows).set_index("confidence")


def rolling_historical_var(r, conf=0.99, window=250):
    """Historical VaR recomputed on each date from the last `window` returns."""
    return -r.rolling(window).quantile(1 - conf)


def ewma_var_series(r, conf=0.99, lam=0.94):
    """EWMA (normal) VaR on each date. The value on date t is the forecast for t+1."""
    return norm.ppf(conf) * ewma_vol(r, lam)
