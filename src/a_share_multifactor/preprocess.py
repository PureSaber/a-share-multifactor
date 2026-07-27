"""Factor preprocessing: winsorize, standardize, forward returns."""

from __future__ import annotations

import pandas as pd

from a_share_multifactor.config import AppConfig
from a_share_multifactor.factors import apply_factor_directions, compute_factors
from a_share_multifactor.neutralize import neutralize_cross_section
from a_share_multifactor.calendar import rebalance_dates as get_rebalance_dates


def winsorize_cross_section(
    df: pd.DataFrame,
    cols: list[str],
    quantiles: tuple[float, float],
    date_col: str = "date",
) -> pd.DataFrame:
    """Winsorize factor columns at cross-sectional quantiles per date."""
    result = df.copy()
    lower_q, upper_q = quantiles

    for col in cols:
        if col not in result.columns:
            continue
        lower = result.groupby(date_col)[col].transform(lambda s: s.quantile(lower_q))
        upper = result.groupby(date_col)[col].transform(lambda s: s.quantile(upper_q))
        result[col] = result[col].clip(lower=lower, upper=upper)

    return result


def standardize_cross_section(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = "date",
    method: str = "zscore",
) -> pd.DataFrame:
    """Standardize factor columns cross-sectionally per date."""
    if method != "zscore":
        raise ValueError(f"Unsupported standardize method: {method}")

    result = df.copy()

    for col in cols:
        if col not in result.columns:
            continue
        mean = result.groupby(date_col)[col].transform("mean")
        std = result.groupby(date_col)[col].transform(lambda s: s.std(ddof=0))
        result[col] = (result[col] - mean) / std.replace(0, pd.NA)
        result[col] = result[col].fillna(0.0)

    return result


def add_forward_return(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Add forward return column over *window* trading days."""
    result = df.copy()
    col_name = f"forward_return_{window}d"
    result[col_name] = result.groupby(symbol_col)[price_col].transform(
        lambda s: s.shift(-window) / s - 1
    )
    return result


def add_period_return(
    df: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    price_col: str = "close",
    symbol_col: str = "symbol",
    date_col: str = "date",
    col_name: str = "period_return",
) -> pd.DataFrame:
    """Add return from each rebalance date to the next rebalance date."""
    result = df.copy()
    result[col_name] = pd.NA
    rebalance_list = list(pd.to_datetime(rebalance_dates))

    for idx, start_date in enumerate(rebalance_list[:-1]):
        end_date = rebalance_list[idx + 1]
        start_rows = (
            result[result[date_col] == start_date]
            .drop_duplicates(subset=[symbol_col])
            .set_index(symbol_col)[price_col]
        )
        end_rows = (
            result[result[date_col] == end_date]
            .drop_duplicates(subset=[symbol_col])
            .set_index(symbol_col)[price_col]
        )
        common = start_rows.index.intersection(end_rows.index)
        period_ret = (end_rows.loc[common] / start_rows.loc[common]) - 1
        mask = (result[date_col] == start_date) & (result[symbol_col].isin(common))
        result.loc[mask, col_name] = result.loc[mask, symbol_col].map(period_ret)

    return result


def prepare_factor_panel(config: AppConfig, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute factors, apply directions, preprocess, and add return columns."""
    panel = compute_factors(raw_df, factor_names=config.factors)
    factor_cols = [col for col in config.factors if col in panel.columns]

    panel = apply_factor_directions(panel, factor_cols, config.factor_directions)
    panel = winsorize_cross_section(panel, factor_cols, config.preprocess.winsorize)

    if config.preprocess.neutralize:
        panel = neutralize_cross_section(
            panel,
            factor_cols,
            config.preprocess.neutralize_by,
        )

    panel = standardize_cross_section(
        panel,
        factor_cols,
        method=config.preprocess.standardize,
    )
    panel = add_forward_return(panel, config.forward_return_days)

    if config.holding_period == "rebalance":
        rebalance_idx = get_rebalance_dates(panel["date"], config.rebalance_freq)
        panel = add_period_return(panel, rebalance_idx)

    return panel
