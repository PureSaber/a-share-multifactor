"""Availability gates shared by training and research validation."""

from __future__ import annotations

import re

import pandas as pd


def label_available_at(panel: pd.DataFrame, target: str, date_col: str = "date") -> pd.Series:
    """Return actual label endpoints; never treat a feature date as label maturity.

    Legacy forward_return_Nd frames can derive their endpoints from their ordered
    observations. Other targets must carry explicit availability metadata.
    """
    column = f"{target}_label_available_at"
    if column in panel:
        available = pd.to_datetime(panel[column], utc=True)
        endpoint_column = f"{target}_label_end_at"
        endpoint = pd.to_datetime(
            panel[endpoint_column if endpoint_column in panel else date_col], utc=True
        )
        if (available < endpoint).any():
            raise ValueError("Label availability cannot precede its endpoint")
        return available
    match = re.fullmatch(r"forward_return_([1-9][0-9]*)d", target)
    if not match:
        raise ValueError(f"Missing label availability column: {column}")
    horizon = int(match.group(1))
    if "symbol" in panel:
        ordered = panel.sort_values(["symbol", date_col])
        endpoint = ordered.groupby("symbol")[date_col].shift(-horizon).reindex(panel.index)
    else:
        dates = pd.Series(sorted(pd.to_datetime(panel[date_col]).unique()))
        endpoint = pd.to_datetime(panel[date_col]).map(dict(zip(dates, dates.shift(-horizon))))
    return pd.to_datetime(endpoint, utc=True)


def mature_labels(panel: pd.DataFrame, target: str, as_of: object) -> pd.Series:
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    return label_available_at(panel, target).lt(cutoff)
