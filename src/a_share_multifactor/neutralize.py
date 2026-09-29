"""Cross-sectional factor neutralization."""

from __future__ import annotations

import pandas as pd


def _market_cap_bins(series: pd.Series, bins: int = 5) -> pd.Series:
    valid = series.dropna()
    if valid.empty or valid.nunique() < 2:
        return pd.Series(index=series.index, dtype="object")
    ranked = pd.qcut(valid, q=min(bins, valid.nunique()), duplicates="drop")
    result = pd.Series(index=series.index, dtype="object")
    result.loc[valid.index] = ranked.astype(str)
    return result


def neutralize_cross_section(
    df: pd.DataFrame,
    cols: list[str],
    by: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Demean factor columns within industry / market-cap groups per date."""
    if not by:
        return df.copy()
    missing = [field for field in by if field not in df.columns]
    if missing:
        raise ValueError(
            "Neutralization columns are missing: "
            + ", ".join(missing)
            + ". Refusing to leave the original factor values in place."
        )
    result = df.copy()
    group_cols = [date_col]

    if "industry" in by:
        group_cols.append("industry")
    if "market_cap" in by:
        result["_mcap_bin"] = result.groupby(date_col)["market_cap"].transform(
            lambda s: _market_cap_bins(s)
        )
        group_cols.append("_mcap_bin")

    if len(group_cols) == 1:
        raise ValueError(
            "Neutralization only supports industry and market_cap, got: " + ", ".join(by)
        )

    for col in cols:
        if col not in result.columns:
            continue
        group_mean = result.groupby(group_cols)[col].transform("mean")
        result[col] = result[col] - group_mean

    if "_mcap_bin" in result.columns:
        result = result.drop(columns=["_mcap_bin"])

    return result
