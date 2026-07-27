"""Factor computation for A-share multi-factor model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

FactorFn = Callable[[pd.DataFrame], pd.Series]


def compute_momentum(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling return over *window* trading days."""
    return close.pct_change(window)


def compute_reversal(close: pd.Series, window: int = 5) -> pd.Series:
    """Short-term reversal factor."""
    return -close.pct_change(window)


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling return volatility."""
    return close.pct_change().rolling(window).std()


def compute_turnover(volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling average volume as turnover proxy."""
    return volume.rolling(window).mean()


def _momentum_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["close"].transform(lambda s: compute_momentum(s, 20))


def _reversal_5d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["close"].transform(lambda s: compute_reversal(s, 5))


def _volatility_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["close"].transform(lambda s: compute_volatility(s, 20))


def _turnover_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["volume"].transform(lambda s: compute_turnover(s, 20))


FACTOR_REGISTRY: dict[str, FactorFn] = {
    "momentum_20d": _momentum_20d,
    "reversal_5d": _reversal_5d,
    "volatility_20d": _volatility_20d,
    "turnover_20d": _turnover_20d,
}

# Factors sourced directly from merged price/fundamental columns.
PASSTHROUGH_FACTORS = {"market_cap", "pe_ratio", "pb_ratio"}


def compute_factors(price_df: pd.DataFrame, factor_names: list[str] | None = None) -> pd.DataFrame:
    """
    Compute factor values from price data.

    Expected columns: symbol, date, close, volume, market_cap, pe_ratio, pb_ratio
    """
    factors = price_df.copy()
    names = factor_names or list(FACTOR_REGISTRY.keys()) + list(PASSTHROUGH_FACTORS)

    for name in names:
        if name in FACTOR_REGISTRY:
            factors[name] = FACTOR_REGISTRY[name](factors)
        elif name not in PASSTHROUGH_FACTORS and name not in factors.columns:
            factors[name] = np.nan

    return factors


def apply_factor_directions(
    df: pd.DataFrame,
    factor_cols: list[str],
    directions: dict[str, int],
) -> pd.DataFrame:
    """Multiply factors by configured direction (+1 long, -1 invert)."""
    result = df.copy()
    for col in factor_cols:
        if col in result.columns:
            result[col] = result[col] * directions.get(col, 1)
    return result
