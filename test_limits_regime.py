import numpy as np
import pandas as pd
import pytest

from risk.limits import usage_status, check_limits
from risk.regime import realised_vol, vol_regime

rng = np.random.default_rng(7)
LIMITS = {"var_95_1d": 0.015, "var_99_1d": 0.022, "single_name_weight": 0.20,
          "daily_loss": 0.020, "amber_threshold": 0.80}


def test_usage_status_boundaries():
    assert usage_status(0.79) == "Green"
    assert usage_status(0.80) == "Amber"
    assert usage_status(0.99) == "Amber"
    assert usage_status(1.00) == "Red"
    assert usage_status(1.50) == "Red"


def test_check_limits_statuses_and_usage():
    df = check_limits(0.010, 0.023, 0.125, "AAPL", 0.0, LIMITS)
    status = dict(zip(df["Limit"], df["Status"]))
    assert status["95% 1-day VaR"] == "Green"
    assert status["99% 1-day VaR"] == "Red"
    assert status["Latest 1-day loss"] == "Green"
    row = df[df["Limit"] == "95% 1-day VaR"].iloc[0]
    assert row["Usage (%)"] == pytest.approx(10 / 15 * 100)


def test_check_limits_skips_missing_limits():
    df = check_limits(0.01, 0.02, 0.125, "AAPL", 0.0, {"var_95_1d": 0.015})
    assert len(df) == 1


def test_realised_vol_matches_std_of_last_window():
    r = pd.Series(rng.normal(0, 0.01, 100))
    assert realised_vol(r, 20).iloc[-1] == pytest.approx(r.iloc[-20:].std())


def test_regime_stressed_after_vol_spike():
    r = pd.Series(np.concatenate([rng.normal(0, 0.007, 500), rng.normal(0, 0.03, 25)]))
    out = vol_regime(r)
    assert out["regime"] == "Stressed"
    assert 0 <= out["percentile"] <= 1


def test_regime_calm_after_quiet_period():
    r = pd.Series(np.concatenate([rng.normal(0, 0.015, 500), rng.normal(0, 0.003, 25)]))
    assert vol_regime(r)["regime"] == "Calm"


def test_regime_normal_in_the_middle():
    blocks = [0.005, 0.008, 0.012, 0.016, 0.020, 0.012]
    r = pd.Series(np.concatenate([rng.normal(0, v, 100) for v in blocks]))
    assert vol_regime(r)["regime"] == "Normal"


def test_regime_levels_are_increasing():
    r = pd.Series(rng.normal(0, 0.01, 600))
    levels = [lvl for _, lvl in vol_regime(r)["levels"]]
    assert levels == sorted(levels)
