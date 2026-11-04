"""Phase 1 check: run the core engine from the command line."""
from risk.data import load_config, get_prices, simple_returns, find_outliers
from risk.var_es import portfolio_returns, risk_summary, ewma_vol

cfg = load_config()
weights = {p["ticker"]: p["weight"] for p in cfg["portfolio"]}

prices = get_prices(cfg)
rets = simple_returns(prices)
print(f"{len(rets)} daily returns, {rets.index[0].date()} to {rets.index[-1].date()}")

out = find_outliers(rets)
if len(out):
    print("\nDaily moves above 25% (check for bad data):")
    print(out)

port = portfolio_returns(rets, weights)
print(f"\nLatest 1-day portfolio return: {port.iloc[-1]:.2%}")
print(f"Latest EWMA daily volatility:  {ewma_vol(port, cfg['ewma_lambda']).iloc[-1]:.2%}")
summary = risk_summary(port, cfg["confidence_levels"], cfg["ewma_lambda"])
print(f"\n1-day VaR / ES, % of portfolio (base currency {cfg['base_currency']}):")
print((summary * 100).round(2).astype(str) + "%")
