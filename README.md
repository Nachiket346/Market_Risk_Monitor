# Market Risk Monitor

A Streamlit dashboard that monitors the market risk of a multi-currency equity
portfolio (4 US and 4 Indian stocks, measured in INR). It shows VaR and Expected
Shortfall, checks them against limits, flags the volatility regime, and shows
which positions drive the risk.

**Live app:** https://marketriskmonitor-ahtmaedftpriid4q682u5b.streamlit.app/
Note:If the app is asleep, click the wake button and wait 30 seconds.
<img width="953" height="338" alt="Screenshot 2026-11-04 211142" src="https://github.com/user-attachments/assets/1ae3162c-6b2c-44d9-8fe2-e978791fbced" />


<img width="691" height="244" alt="Screenshot 2026-11-04 211200" src="https://github.com/user-attachments/assets/9471b5b4-79ab-4cd4-a166-5d38dcf9179d" />

## What it does

| Section | What it shows |
|---|---|
| Headline cards | Latest return, EWMA volatility, historical and EWMA VaR/ES, with rupee amounts |
| Limit monitor | 95% and 99% VaR, largest position and daily loss against limits, with Green / Amber / Red status |
| Volatility regime | 20-day realised volatility ranked against its own history: Calm, Normal, Elevated, Stressed |
| What drives the risk | Marginal and component VaR: each position's share of risk against its weight |
| VaR over time | Rolling historical VaR and EWMA VaR against the limit |
| Daily summary | One-page text summary, downloadable as a markdown file |

## Method

- **Data:** adjusted close prices from Yahoo Finance (via `yfinance`), 3 years. US prices are
  converted to INR with USD/INR, so the VaR includes currency risk.
- **Returns:** daily simple returns. Portfolio weights are assumed reset to target each day.
- **Historical VaR / ES:** empirical quantile and tail average over a chosen window.
- **EWMA VaR / ES:** RiskMetrics volatility (lambda 0.94) with a normal assumption.
- **Limit check:** uses the higher of historical (500-day window) and EWMA VaR. Amber from 80%
  of a limit, Red at or above 100%.
- **Volatility regime:** percentile of current 20-day volatility within its own history.
  Cut-offs at the 25th, 75th and 90th percentiles.
- **Risk contribution:** parametric (covariance) component VaR, which adds up exactly to total VaR.
- **Data cleaning:** non-positive prices removed, gaps of up to 3 days forward-filled (US and
  India have different holidays), moves above 25% flagged for review.

## Example finding

On data to early November 2026, at 99% confidence and a 500-day window, historical ES was about
2.7% against EWMA ES of about 1.7%. Returns have heavier tails than the normal assumption behind
EWMA, which therefore understates tail losses. Across the portfolio, AAPL contributed about 19% of
risk from 12.5% of capital. _These figures change with the data._

## Limitations

- Prototype on delayed public data. Not a production risk system.
- Limits and regime cut-offs are illustrative, not calibrated to any institution.
- EWMA VaR assumes normal returns. Historical VaR reacts slowly to new volatility.
- Weights are target weights, so no position drift is modelled.
- Contribution analysis is parametric, so its total differs from historical VaR.
- Equity only. No options, fixed income or credit risk.

## Run it locally

```
pip install -r requirements.txt
python -m pytest
streamlit run app.py
```

Change the portfolio and limits in `config.yaml`. Weights must sum to 1.

## Project structure

```
app.py                # Streamlit dashboard
config.yaml           # portfolio, limits, parameters
risk/
  data.py             # download, currency conversion, cleaning
  var_es.py           # historical and EWMA VaR / ES
  limits.py           # limit checks
  regime.py           # volatility regime
  contribution.py     # marginal and component VaR
  summary.py          # daily summary text
tests/                # unit tests
```
