from risk.limits import check_limits
from risk.summary import build_summary

LIMITS = {"var_95_1d": 0.015, "var_99_1d": 0.022, "single_name_weight": 0.20,
          "daily_loss": 0.020, "amber_threshold": 0.80}
REGIME = {"regime": "Normal", "vol": 0.006, "percentile": 0.51}


def _summary(var95, var99, loss=0.0):
    df = check_limits(var95, var99, 0.125, "AAPL", loss, LIMITS)
    return build_summary(
        as_of="2026-11-03", base="INR", value=10_000_000, conf=0.99, window=500,
        limit_df=df, h_var=0.0165, h_es=0.0272, e_var=0.0148, e_es=0.0170,
        regime=REGIME, top_name="AAPL", top_share=0.191, top_weight=0.125,
    )


def test_green_summary():
    text = _summary(0.005, 0.008)
    assert "GREEN" in text and "2026-11-03" in text and "AAPL" in text


def test_amber_summary():
    assert "AMBER" in _summary(0.0125, 0.008)


def test_red_summary_names_the_breach():
    text = _summary(0.005, 0.030)
    assert "RED" in text and "99% 1-day VaR" in text


def test_summary_includes_rupee_amounts():
    assert "INR 165,000" in _summary(0.005, 0.008)
