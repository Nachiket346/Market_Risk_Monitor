"""Load config, download prices, convert to base currency, clean."""
import pandas as pd
import yaml
import yfinance as yf


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    total = sum(p["weight"] for p in cfg["portfolio"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights sum to {total:.4f}, expected 1.0")
    return cfg


def _download(tickers, start):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    return raw


def clean_prices(prices, max_ffill=3):
    """Sort, drop non-positive prices, fill short gaps, drop remaining gaps.

    US and India have different holidays. Short gaps (up to max_ffill days)
    are forward-filled, which gives a 0% return on the closed market's
    holiday. Dates that are still missing after that are dropped.
    """
    prices = prices.sort_index()
    prices = prices.where(prices > 0)
    prices = prices.ffill(limit=max_ffill)
    return prices.dropna()


def get_prices(cfg):
    """Prices for every position, converted to the base currency."""
    base = cfg["base_currency"]
    start = (pd.Timestamp.today() - pd.DateOffset(years=cfg["lookback_years"])).strftime("%Y-%m-%d")
    tickers = [p["ticker"] for p in cfg["portfolio"]]
    prices = _download(tickers, start)

    fx_pairs = {
        p["currency"]: f"{p['currency']}{base}=X"
        for p in cfg["portfolio"] if p["currency"] != base
    }
    if fx_pairs:
        fx = _download(list(fx_pairs.values()), start)
        for ccy, pair in fx_pairs.items():
            cols = [p["ticker"] for p in cfg["portfolio"] if p["currency"] == ccy]
            fx_rate = fx[pair].reindex(prices.index).ffill(limit=3)
            prices[cols] = prices[cols].mul(fx_rate, axis=0)

    return clean_prices(prices[tickers])


def simple_returns(prices):
    return prices.pct_change().dropna()


def find_outliers(returns, threshold=0.25):
    """Daily moves larger than threshold: check these are real, not bad data."""
    mask = returns.abs() > threshold
    return returns[mask].stack()

def find_outliers(returns, threshold=0.25):
    """Daily moves larger than threshold: check these are real, not bad data."""
    s = returns.stack()
    return s[s.abs() > threshold]

