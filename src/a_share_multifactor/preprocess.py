"""Factor preprocessing: winsorize, standardize, forward returns."""

from __future__ import annotations

import pandas as pd

from a_share_multifactor.calendar import rebalance_dates as get_rebalance_dates
from a_share_multifactor.config import AppConfig
from a_share_multifactor.factors import apply_factor_directions, compute_factors
from a_share_multifactor.neutralize import neutralize_cross_section
from a_share_multifactor.tradability import assign_executable_period_returns


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
        raw = result[col]
        mean = raw.groupby(result[date_col]).transform("mean")
        std = raw.groupby(result[date_col]).transform(lambda s: s.std(ddof=0))
        scaled = (raw - mean) / std.replace(0, pd.NA)
        zero_dispersion = std.eq(0) & raw.notna()
        result[col] = scaled.mask(zero_dispersion, 0.0)

    return result


def add_forward_return(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
    symbol_col: str = "symbol",
    *,
    basis: str = "same_close",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add a gross price label and its availability; not a simulated fill.

    ``next_open`` uses open(t+1) to open(t+1+window) on the panel's common
    session grid. Missing endpoint bars stay missing. ``same_close`` preserves
    the historical diagnostic convention for callers that explicitly need it.
    """
    if type(window) is not int or window < 1:
        raise ValueError("forward return window must be positive")
    if basis not in {"same_close", "next_open"}:
        raise ValueError("forward return basis must be same_close or next_open")
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    if result.duplicated([symbol_col, date_col]).any():
        raise ValueError("forward labels require unique symbol/date rows")
    result = result.sort_values([symbol_col, date_col])
    col_name = f"forward_return_{window}d"
    result[f"{col_name}_basis"] = basis
    if basis == "next_open":
        if "open" not in result:
            raise ValueError("next_open labels require open prices")
        dates = pd.Series(sorted(result[date_col].dropna().unique()))
        entry_dates = result[date_col].map(dict(zip(dates, dates.shift(-1))))
        exit_dates = result[date_col].map(dict(zip(dates, dates.shift(-window - 1))))
        indexed = result.set_index([symbol_col, date_col])

        def at(column, endpoints):
            keys = pd.MultiIndex.from_arrays([result[symbol_col], endpoints])
            return pd.Series(indexed[column].reindex(keys).array, index=result.index)

        entry = pd.to_numeric(at("open", entry_dates), errors="coerce")
        exit_price = pd.to_numeric(at("open", exit_dates), errors="coerce")
        valid = (
            entry.gt(0) & entry.lt(float("inf")) & exit_price.gt(0) & exit_price.lt(float("inf"))
        )
        result[col_name] = (exit_price / entry - 1).where(valid)
        result[f"{col_name}_label_start_at"] = entry_dates
        result[f"{col_name}_label_end_at"] = exit_dates
        known = pd.DataFrame({"exit": pd.to_datetime(exit_dates, utc=True)})
        if "available_at" in result:
            known["entry_known"] = pd.to_datetime(at("available_at", entry_dates), utc=True)
            known["exit_known"] = pd.to_datetime(at("available_at", exit_dates), utc=True)
        result[f"{col_name}_label_available_at"] = known.max(axis=1).where(
            valid & known.notna().all(axis=1)
        )
        return result
    result[col_name] = result.groupby(symbol_col)[price_col].transform(
        lambda s: s.shift(-window) / s - 1
    )
    result[f"{col_name}_label_end_at"] = result.groupby(symbol_col)[date_col].shift(-window)
    # Daily bars are usable after the endpoint close. Training uses strictly
    # earlier dates, so a same-day midnight cannot make the label available early.
    availability = "available_at" if "available_at" in result else date_col
    result[f"{col_name}_label_available_at"] = result.groupby(symbol_col)[availability].shift(
        -window
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
    """Add the next-open holding-period return on each rebalance date.

    ``price_col`` is retained for callers that still pass it. The fill price is
    the next session open, after limit-up, limit-down and halt checks.
    """
    del price_col
    return assign_executable_period_returns(
        df,
        rebalance_dates,
        symbol_col=symbol_col,
        date_col=date_col,
        col_name=col_name,
    )


def prepare_factor_inputs(config: AppConfig, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and validate factor inputs without scoring or future-return labels."""
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
    missing = [col for col in config.factors if col not in panel.columns]
    if missing:
        raise ValueError("Configured factors are missing from the panel: " + ", ".join(missing))
    empty = [col for col in factor_cols if panel[col].isna().all()]
    if empty:
        raise ValueError(
            "Configured factors are entirely missing: "
            + ", ".join(empty)
            + ". Missing factor values stay missing and are left out of the composite."
        )
    return panel


def prepare_factor_panel(config: AppConfig, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute factors, apply directions, preprocess, and add return columns."""
    panel = prepare_factor_inputs(config, raw_df)
    panel = add_forward_return(panel, config.forward_return_days, basis=config.ic_return_basis)

    if config.holding_period == "rebalance":
        rebalance_idx = get_rebalance_dates(panel["date"], config.rebalance_freq)
        panel = add_period_return(panel, rebalance_idx)

    return panel
