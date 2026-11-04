"""Marginal and component VaR (parametric, normal), showing what drives portfolio risk."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def component_var(returns, weights, conf=0.99):
    """Split portfolio VaR into each position's contribution.

    Uses the covariance of `returns` and a normal assumption, so the total is
    parametric VaR and will not equal historical VaR. The shares of risk are
    the useful output. Contributions add up exactly to the total.

    Returns (table, total_var, undiversified_var). VaR figures are fractions.
    """
    w = pd.Series(weights, dtype=float).reindex(returns.columns)
    if w.isna().any():
        raise ValueError("Weights missing for: " + ", ".join(w[w.isna()].index))
    cov = returns.cov()
    sigma_p = float(np.sqrt(w @ cov @ w))
    z = norm.ppf(conf)

    marginal = z * (cov @ w) / sigma_p
    component = w * marginal
    standalone = z * w * np.sqrt(np.diag(cov))

    table = pd.DataFrame({
        "Weight": w,
        "Marginal VaR": marginal,
        "Component VaR": component,
        "Share of risk": component / component.sum(),
    })
    return table, float(z * sigma_p), float(standalone.sum())

