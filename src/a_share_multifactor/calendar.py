"""Trading calendar helpers for rebalance scheduling."""

from __future__ import annotations

import pandas as pd


def trading_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """Return all unique trading dates sorted."""
    return pd.DatetimeIndex(pd.to_datetime(dates).drop_duplicates().sort_values())


def weekly_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """Return the last trading day of each ISO week."""
    series = pd.to_datetime(dates).drop_duplicates().sort_values()
    grouped = series.groupby(series.dt.to_period("W"))
    return pd.DatetimeIndex(grouped.max().values)


def rebalance_dates(dates: pd.Series, freq: str) -> pd.DatetimeIndex:
    """Return rebalance dates from available trading dates."""
    series = pd.to_datetime(dates).drop_duplicates().sort_values()
    if freq == "monthly":
        grouped = series.groupby(series.dt.to_period("M"))
        return pd.DatetimeIndex(grouped.max().values)
    if freq == "weekly":
        return weekly_dates(dates)
    raise ValueError(f"Unsupported rebalance frequency: {freq}")


def _retail_trade_dates(dates: pd.Series, trade_freq: str) -> pd.DatetimeIndex:
    if trade_freq == "daily":
        return trading_dates(dates)
    if trade_freq == "weekly":
        return weekly_dates(dates)
    if trade_freq == "monthly":
        return rebalance_dates(dates, "monthly")
    raise ValueError(f"Unsupported retail trade frequency: {trade_freq}")


def score_schedule_dates(
    dates: pd.Series,
    rebalance_freq: str,
    retail_mode: bool = False,
    trade_freq: str = "monthly",
) -> pd.DatetimeIndex:
    """Dates on which composite scores are computed."""
    if retail_mode and trade_freq in {"daily", "weekly", "monthly"}:
        return _retail_trade_dates(dates, trade_freq)
    return rebalance_dates(dates, rebalance_freq)


def trade_schedule_dates(
    dates: pd.Series,
    rebalance_freq: str,
    retail_mode: bool = False,
    trade_freq: str = "monthly",
) -> pd.DatetimeIndex:
    """Dates on which retail portfolio trades may occur."""
    if retail_mode and trade_freq in {"daily", "weekly", "monthly"}:
        return _retail_trade_dates(dates, trade_freq)
    return rebalance_dates(dates, rebalance_freq)
