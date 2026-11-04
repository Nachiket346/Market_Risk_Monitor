"""Volatility regime: is current volatility low or high versus its own history?"""
import pandas as pd


def realised_vol(r, window=20):
    """Rolling standard deviation of daily returns."""
    return r.rolling(window).std()


def vol_regime(r, window=20, cuts=(0.25, 0.75, 0.90)):
    """Classify current volatility by its percentile in the portfolio's own history.

    Calm: below the 25th percentile. Normal: 25th to 75th. Elevated: 75th to 90th.
    Stressed: 90th and above. The cut-offs are illustrative, not a standard.
    """
    calm_cut, elevated_cut, stressed_cut = cuts
    vol = realised_vol(r, window).dropna()
    current = float(vol.iloc[-1])
    percentile = float((vol <= current).mean())
    if percentile >= stressed_cut:
        regime = "Stressed"
    elif percentile >= elevated_cut:
        regime = "Elevated"
    elif percentile >= calm_cut:
        regime = "Normal"
    else:
        regime = "Calm"
    levels = [
        ("Calm / Normal", float(vol.quantile(calm_cut))),
        ("Normal / Elevated", float(vol.quantile(elevated_cut))),
        ("Elevated / Stressed", float(vol.quantile(stressed_cut))),
    ]
    return {
        "vol": current,
        "percentile": percentile,
        "regime": regime,
        "levels": levels,
        "series": vol,
    }

