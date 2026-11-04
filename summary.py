"""Plain-text daily risk summary (markdown), for display and download."""


def build_summary(*, as_of, base, value, conf, window, limit_df,
                  h_var, h_es, e_var, e_es, regime, top_name, top_share, top_weight):
    """All VaR/ES/share/weight inputs are fractions (0.02 = 2%)."""
    def money(frac):
        return f"{base} {frac * value:,.0f}"

    red = limit_df.loc[limit_df["Status"] == "Red", "Limit"].tolist()
    amber = limit_df.loc[limit_df["Status"] == "Amber", "Limit"].tolist()
    if red:
        headline = f"RED, {len(red)} limit(s) breached ({', '.join(red)})"
    elif amber:
        headline = f"AMBER, {len(amber)} limit(s) near breach ({', '.join(amber)})"
    else:
        headline = "GREEN, all limits within bounds"

    lines = [
        f"**Market risk daily summary: {as_of}**",
        "",
        f"Base currency {base}, portfolio value {base} {value:,.0f}.",
        "",
        f"**Status:** {headline}",
        "",
        "**Limits**",
    ]
    for _, row in limit_df.iterrows():
        lines.append(
            f"- {row['Limit']}: {row['Current (%)']:.2f}% against a limit of "
            f"{row['Limit (%)']:.2f}% ({row['Usage (%)']:.0f}% used, {row['Status']})"
        )
    lines += [
        "",
        f"**1-day VaR and ES at {conf:.0%} confidence** (historical on a {window}-day window)",
        f"- Historical: VaR {h_var:.2%} ({money(h_var)}), ES {h_es:.2%} ({money(h_es)})",
        f"- EWMA: VaR {e_var:.2%} ({money(e_var)}), ES {e_es:.2%} ({money(e_es)})",
        "",
        "**Volatility regime**",
        f"- {regime['regime']}: 20-day volatility of {regime['vol'] * 252 ** 0.5:.1%} annualised, "
        f"higher than {regime['percentile']:.0%} of days in its own history",
        "",
        "**Largest risk driver**",
        f"- {top_name}: {top_share:.1%} of portfolio risk from {top_weight:.1%} of capital",
        "",
        "_Prototype on delayed public data. Limits are illustrative. Not a production risk system._",
    ]
    return "\n".join(lines)
