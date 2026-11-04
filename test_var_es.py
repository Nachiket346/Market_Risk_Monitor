import numpy as np
import pandas as pd
import pytest
import yaml

from risk.data import load_config
from risk.var_es import (
    portfolio_returns, historical_var, historical_es,
    ewma_vol, ewma_var_es,
)

rng = np.random.default_rng(42)


def test_historical_var_matches_normal_theory():
    r = pd.Series(rng.normal(0, 0.01, 200_000))
    assert historical_var(r, 0.99) == pytest.approx(2.326 * 0.01, rel=0.03)


def test_es_is_at_least_var():
    r = pd.Series(rng.normal(0, 0.01, 5_000))
    for c in (0.95, 0.99):
        assert historical_es(r, c) >= historical_var(r, c)


def test_ewma_close_to_true_vol_when_vol_is_constant():
    r = pd.Series(rng.normal(0, 0.012, 3_000))
    assert ewma_vol(r).iloc[-1] == pytest.approx(0.012, rel=0.25)


def test_ewma_reacts_to_vol_shock_faster_than_history():
    calm = rng.normal(0, 0.008, 700)
    shock = rng.normal(0, 0.03, 20)
    r = pd.Series(np.concatenate([calm, shock]))
    ewma_var, _ = ewma_var_es(r, 0.99)
    assert ewma_var > historical_var(r, 0.99)


def test_ewma_es_exceeds_var():
    r = pd.Series(rng.normal(0, 0.01, 1_000))
    var, es = ewma_var_es(r, 0.99)
    assert es > var


def test_portfolio_returns_matches_manual_calc():
    df = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.00]})
    out = portfolio_returns(df, {"A": 0.4, "B": 0.6})
    assert out.tolist() == pytest.approx([0.022, -0.008])


def test_bad_weights_rejected():
    df = pd.DataFrame({"A": [0.01], "B": [0.02]})
    with pytest.raises(ValueError):
        portfolio_returns(df, {"A": 0.5, "B": 0.6})


def test_config_weights_must_sum_to_one(tmp_path):
    bad = {"portfolio": [{"ticker": "X", "weight": 0.5, "currency": "USD"}]}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        load_config(str(p))


def test_default_config_loads():
    cfg = load_config("config.yaml")
    assert sum(p["weight"] for p in cfg["portfolio"]) == pytest.approx(1.0)


def test_rolling_var_last_value_matches_window_var():
    from risk.var_es import rolling_historical_var
    r = pd.Series(rng.normal(0, 0.01, 600))
    roll = rolling_historical_var(r, 0.99, window=250)
    assert roll.iloc[-1] == pytest.approx(historical_var(r.iloc[-250:], 0.99))
    assert roll.iloc[:249].isna().all()


def test_ewma_var_series_last_value_matches_point_estimate():
    from risk.var_es import ewma_var_series
    r = pd.Series(rng.normal(0, 0.01, 800))
    series = ewma_var_series(r, 0.99)
    point, _ = ewma_var_es(r, 0.99)
    assert series.iloc[-1] == pytest.approx(point)
