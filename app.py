"""Daily Market Risk Monitor: Streamlit dashboard (Phase 2)."""
import plotly.graph_objects as go
import streamlit as st

from risk.data import load_config, get_prices, simple_returns, find_outliers
from risk.var_es import (
    portfolio_returns, historical_var, historical_es, ewma_var_es,
    ewma_vol, rolling_historical_var, ewma_var_series,
)
from risk.limits import check_limits
from risk.regime import vol_regime
from risk.contribution import component_var
from risk.summary import build_summary

st.set_page_config(page_title="Market Risk Monitor", layout="wide")

LIMIT_WINDOW = 500   # fixed window for the limit check, independent of the sidebar
STATUS_ICON = {"Green": "🟢 Green", "Amber": "🟡 Amber", "Red": "🔴 Red"}
REGIME_ICON = {"Calm": "🔵 Calm", "Normal": "🟢 Normal", "Elevated": "🟠 Elevated", "Stressed": "🔴 Stressed"}


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_returns():
    cfg = load_config()
    prices = get_prices(cfg)
    return cfg, simple_returns(prices)


st.title("Market Risk Monitor")

try:
    cfg, rets = load_returns()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if len(rets) < 120:
    st.error("Not enough price history to compute risk. Check tickers and internet connection.")
    st.stop()

base = cfg["base_currency"]
lam = cfg["ewma_lambda"]
weights = {p["ticker"]: p["weight"] for p in cfg["portfolio"]}
port = portfolio_returns(rets, weights)

# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    confs = cfg["confidence_levels"]
    conf = st.radio(
        "Confidence level", confs, index=len(confs) - 1,
        format_func=lambda c: f"{c:.0%}", horizontal=True,
    )
    window = st.slider(
        "Historical window (trading days)",
        min_value=60, max_value=len(port), value=min(500, len(port)), step=10,
    )
    value = st.number_input(
        f"Portfolio value ({base})", min_value=1_000, value=10_000_000, step=100_000,
    )
    st.caption("The window affects historical VaR/ES only. EWMA uses the full history.")

# ---- calculations ------------------------------------------------------------
recent = port.iloc[-window:]
h_var, h_es = historical_var(recent, conf), historical_es(recent, conf)
e_var, e_es = ewma_var_es(port, conf, lam)
vol = float(ewma_vol(port, lam).iloc[-1])
last_ret = float(port.iloc[-1])


def money(frac):
    return f"{base} {frac * value:,.0f}"


st.caption(
    f"Data as of {port.index[-1].date()}  |  base currency {base}  |  "
    f"1-day horizon, {conf:.0%} confidence"
)

# ---- headline cards ----------------------------------------------------------
def card(col, label, value, sub):
    col.metric(label, value, sub, delta_color="off", delta_arrow="off")


c1, c2 = st.columns(2)
card(c1, "Latest 1-day portfolio return", f"{last_ret:.2%}", money(last_ret))
card(c2, "Daily volatility (EWMA)", f"{vol:.2%}", f"annualised {vol * 252 ** 0.5:.1%}")

c3, c4, c5, c6 = st.columns(4)
card(c3, "Historical VaR", f"{h_var:.2%}", money(h_var))
card(c4, "Historical ES", f"{h_es:.2%}", money(h_es))
card(c5, "EWMA VaR", f"{e_var:.2%}", money(e_var))
card(c6, "EWMA ES", f"{e_es:.2%}", money(e_es))

if h_es > e_es * 1.15:
    st.info(
        "Historical ES is noticeably higher than EWMA ES. Returns have heavier "
        "tails than the normal assumption in EWMA, so EWMA understates tail losses."
    )

# ---- limit monitor -------------------------------------------------------------
st.subheader("Limit monitor")
lim_recent = port.iloc[-LIMIT_WINDOW:]
var_now = {c: max(historical_var(lim_recent, c), ewma_var_es(port, c, lam)[0]) for c in (0.95, 0.99)}
top_name, top_weight = max(weights.items(), key=lambda kv: kv[1])
limit_df = check_limits(
    var_now[0.95], var_now[0.99], top_weight, top_name,
    max(0.0, -last_ret), cfg.get("limits", {}),
)

red = limit_df.loc[limit_df["Status"] == "Red", "Limit"].tolist()
amber = limit_df.loc[limit_df["Status"] == "Amber", "Limit"].tolist()
if red:
    st.error(f"{len(red)} limit(s) breached: " + ", ".join(red))
elif amber:
    st.warning(f"{len(amber)} limit(s) near breach: " + ", ".join(amber))
else:
    st.success("All limits within bounds.")

shown = limit_df.copy()
shown["Status"] = shown["Status"].map(STATUS_ICON)
st.dataframe(
    shown, hide_index=True, width="stretch",
    column_config={
        "Current (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Limit (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Usage (%)": st.column_config.NumberColumn(format="%.0f%%"),
    },
)
st.caption(
    f"VaR rows use the higher of historical ({LIMIT_WINDOW}-day window) and EWMA VaR. "
    "Position size uses target weights. Amber starts at "
    f"{cfg.get('limits', {}).get('amber_threshold', 0.8):.0%} of the limit. Limits are illustrative."
)

# ---- volatility regime -----------------------------------------------------------
st.subheader("Volatility regime")
reg = vol_regime(port)
ann = 252 ** 0.5
r1, r2 = st.columns(2)
card(r1, "Regime", REGIME_ICON[reg["regime"]], f"higher than {reg['percentile']:.0%} of days in its own history")
card(r2, "20-day realised volatility (annualised)", f"{reg['vol'] * ann:.1%}", "Calm below 25th, Elevated from 75th, Stressed from 90th percentile")

fig0 = go.Figure()
fig0.add_scatter(x=reg["series"].index, y=reg["series"] * ann * 100, line_color="#7a9cc6", name="20-day vol")
for (label, level), colour in zip(reg["levels"], ["#1f77b4", "#ff7f0e", "#d62728"]):
    fig0.add_hline(y=level * ann * 100, line_dash="dot", line_color=colour,
                   annotation_text=label, annotation_position="top left")
fig0.update_layout(yaxis_title="Annualised volatility (%)", showlegend=False, margin=dict(t=30, b=10))
st.plotly_chart(fig0, width="stretch")

# ---- risk contribution ---------------------------------------------------------
st.subheader("What drives the risk")
contrib, total_par_var, undiversified = component_var(rets.iloc[-window:], weights, conf)
top = contrib["Share of risk"].idxmax()
st.markdown(
    f"**{top}** is the largest risk contributor: {contrib.loc[top, 'Share of risk']:.1%} of portfolio risk "
    f"from {contrib.loc[top, 'Weight']:.1%} of capital. "
    f"Diversification reduces VaR by {1 - total_par_var / undiversified:.1%} versus holding each position on its own."
)
fig3 = go.Figure()
fig3.add_bar(x=contrib.index, y=contrib["Weight"] * 100, name="Weight", marker_color="#7a9cc6")
fig3.add_bar(x=contrib.index, y=contrib["Share of risk"] * 100, name="Share of risk", marker_color="#d62728")
fig3.update_layout(barmode="group", yaxis_title="% of portfolio", margin=dict(t=30, b=10),
                   legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig3, width="stretch")
with st.expander("Contribution table"):
    st.dataframe(
        contrib.reset_index(names="Ticker"), hide_index=True, width="stretch",
        column_config={
            "Weight": st.column_config.NumberColumn(format="percent"),
            "Marginal VaR": st.column_config.NumberColumn(format="percent", help="Change in portfolio VaR per unit of that position"),
            "Component VaR": st.column_config.NumberColumn(format="percent"),
            "Share of risk": st.column_config.NumberColumn(format="percent"),
        },
    )
st.caption(
    "Parametric (normal) VaR from the covariance of returns in the selected window and confidence level. "
    "Its total will not match historical VaR. Read the shares, not the level."
)

# ---- chart 1: return distribution ---------------------------------------------
st.subheader("Distribution of daily returns")
fig = go.Figure()
fig.add_histogram(x=recent * 100, nbinsx=50, marker_color="#7a9cc6", name="Daily returns")
fig.add_vline(x=-h_var * 100, line_dash="dash", line_color="#d62728",
              annotation_text="Hist VaR", annotation_position="top left")
fig.add_vline(x=-e_var * 100, line_dash="dot", line_color="#ff7f0e",
              annotation_text="EWMA VaR", annotation_position="top right")
fig.update_layout(xaxis_title="Daily return (%)", yaxis_title="Days",
                  showlegend=False, margin=dict(t=30, b=10))
st.plotly_chart(fig, width="stretch")

# ---- chart 2: VaR over time ---------------------------------------------------
st.subheader("VaR over time")
roll_h = rolling_historical_var(port, conf, window).dropna() * 100
roll_e = ewma_var_series(port, conf, lam).loc[roll_h.index] * 100
fig2 = go.Figure()
fig2.add_scatter(x=roll_h.index, y=roll_h, name=f"Historical ({window}d window)", line_color="#d62728")
fig2.add_scatter(x=roll_e.index, y=roll_e, name="EWMA", line_color="#ff7f0e")
limit = cfg.get("limits", {}).get(f"var_{round(conf * 100)}_1d")
if limit:
    fig2.add_hline(y=limit * 100, line_dash="dash", line_color="grey",
                   annotation_text=f"Limit {limit:.1%}", annotation_position="top left")
fig2.update_layout(yaxis_title="1-day VaR (% of portfolio)", margin=dict(t=30, b=10),
                   legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig2, width="stretch")

# ---- daily summary -------------------------------------------------------------
st.subheader("Daily summary")
summary = build_summary(
    as_of=port.index[-1].date(), base=base, value=value, conf=conf, window=window,
    limit_df=limit_df, h_var=h_var, h_es=h_es, e_var=e_var, e_es=e_es, regime=reg,
    top_name=top, top_share=float(contrib.loc[top, "Share of risk"]),
    top_weight=float(contrib.loc[top, "Weight"]),
)
st.markdown(summary)
st.download_button(
    "Download summary (.md)", summary,
    file_name=f"risk_summary_{port.index[-1].date()}.md", mime="text/markdown",
)

# ---- details -----------------------------------------------------------------
with st.expander("Portfolio and data checks"):
    st.write("Positions (equal weights, converted to base currency):")
    st.dataframe(
        [{"Ticker": p["ticker"], "Weight": f"{p['weight']:.1%}", "Currency": p["currency"]}
         for p in cfg["portfolio"]],
        hide_index=True,
    )
    out = find_outliers(rets)
    if len(out):
        st.warning("Daily moves above 25%. Check these are real, not bad data:")
        st.write(out)
    else:
        st.success("No daily moves above 25%.")

with st.expander("How to read this"):
    st.markdown(
        "- **VaR**: the loss that should be exceeded on only 1 day in 20 (95%) or 1 in 100 (99%).\n"
        "- **ES**: the average loss on the days when VaR is exceeded. It is always larger than VaR.\n"
        "- **Historical** uses actual past returns in the chosen window. It reacts slowly to new volatility.\n"
        "- **EWMA** weights recent days more, so it reacts faster, but assumes normal returns "
        "and understates tail risk.\n"
        "- **Limit monitor**: Green below 80% of a limit, Amber from 80%, Red at or above 100%.\n"
        "- **Volatility regime**: compares 20-day volatility with the portfolio's own 3-year history "
        "(Calm, Normal, Elevated, Stressed). The cut-offs are illustrative. Overlapping windows make "
        "the history repetitive, so treat the percentile as a rough guide.\n"
        "- **What drives the risk**: each position's share of portfolio risk against its share of capital. "
        "A position with a bigger risk share than weight adds more risk than its size suggests.\n"
        "- The dashed grey line in the VaR chart is an illustrative limit, not a real one."
    )

st.caption(
    "Prototype on delayed public data (Yahoo Finance). Not a production risk system. "
    "Weights are assumed reset to target each day. Limits are illustrative."
)
