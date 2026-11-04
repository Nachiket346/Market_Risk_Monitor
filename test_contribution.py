import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk.contribution import component_var

rng = np.random.default_rng(11)


def _returns(vols, n=3000):
    return pd.DataFrame({f"S{i}": rng.normal(0, v, n) for i, v in enumerate(vols)})


def test_components_sum_to_total_var():
    rets = _returns([0.01, 0.015, 0.02])
    rets["S1"] += 0.5 * rets["S0"]          # add correlation
    w = {"S0": 0.5, "S1": 0.3, "S2": 0.2}
    table, total, _ = component_var(rets, w, 0.99)
    assert table["Component VaR"].sum() == pytest.approx(total)
    assert table["Share of risk"].sum() == pytest.approx(1.0)


def test_total_matches_direct_formula():
    rets = _returns([0.01, 0.02])
    w = pd.Series({"S0": 0.6, "S1": 0.4})
    _, total, _ = component_var(rets, w.to_dict(), 0.95)
    sigma = np.sqrt(w.values @ rets.cov().values @ w.values)
    assert total == pytest.approx(norm.ppf(0.95) * sigma)


def test_single_asset_has_all_the_risk():
    rets = _returns([0.01])
    table, total, undiv = component_var(rets, {"S0": 1.0})
    assert table["Share of risk"].iloc[0] == pytest.approx(1.0)
    assert undiv == pytest.approx(total)


def test_riskier_asset_has_larger_share_than_weight():
    rets = _returns([0.01, 0.01, 0.03])
    w = {"S0": 1 / 3, "S1": 1 / 3, "S2": 1 / 3}
    table, _, _ = component_var(rets, w)
    assert table.loc["S2", "Share of risk"] > table.loc["S2", "Weight"]


def test_diversification_lowers_var():
    rets = _returns([0.01, 0.01, 0.01])
    w = {"S0": 1 / 3, "S1": 1 / 3, "S2": 1 / 3}
    _, total, undiversified = component_var(rets, w)
    assert total < undiversified


def test_missing_weight_rejected():
    rets = _returns([0.01, 0.02])
    with pytest.raises(ValueError):
        component_var(rets, {"S0": 1.0})
