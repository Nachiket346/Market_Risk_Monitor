"""Limit checks: compare risk measures to limits and assign Green / Amber / Red."""
import pandas as pd

COLUMNS = ["Limit", "Current (%)", "Limit (%)", "Usage (%)", "Status"]


def usage_status(usage, amber=0.80):
    """Red at or above 100% of the limit, Amber from the amber threshold, else Green."""
    if usage >= 1.0:
        return "Red"
    if usage >= amber:
        return "Amber"
    return "Green"


def check_limits(var_95, var_99, top_weight, top_name, daily_loss, limits):
    """Build the limit table. All inputs are fractions (0.02 = 2%).

    A limit missing from `limits` is skipped.
    """
    amber = limits.get("amber_threshold", 0.80)
    items = [
        ("95% 1-day VaR", var_95, limits.get("var_95_1d")),
        ("99% 1-day VaR", var_99, limits.get("var_99_1d")),
        (f"Largest position ({top_name})", top_weight, limits.get("single_name_weight")),
        ("Latest 1-day loss", daily_loss, limits.get("daily_loss")),
    ]
    rows = []
    for name, current, limit in items:
        if not limit:
            continue
        usage = current / limit
        rows.append({
            "Limit": name,
            "Current (%)": current * 100,
            "Limit (%)": limit * 100,
            "Usage (%)": usage * 100,
            "Status": usage_status(usage, amber),
        })
    return pd.DataFrame(rows, columns=COLUMNS)
