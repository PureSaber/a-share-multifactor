"""Factor computation for A-share multi-factor model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from quant_factors.core import compute_factors as qf_compute

FactorFn = Callable[[pd.DataFrame], pd.Series]


def compute_momentum(close: pd.Series, window: int = 20) -> pd.Series:
    """Legacy helper: rolling pct change on a single close series."""
    return close.pct_change(window)


PASSTHROUGH_FACTORS = {
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "forecast_score",
    "industry_rs_20d",
}

# Legacy registry names mapped to quant-factors outputs
_QF_ALIASES = {
    "momentum_20d": "momentum_20d",
    "reversal_5d": "reversal_5d",
    "volatility_20d": "volatility_20d",
    "turnover_20d": "turnover_20d",
}


def _legacy_momentum_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["close"].transform(lambda s: s.pct_change(20))


def _legacy_reversal_5d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["close"].transform(lambda s: -s.pct_change(5))


def _legacy_volatility_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["close"].transform(lambda s: s.pct_change().rolling(20).std())


def _legacy_turnover_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["volume"].transform(lambda s: s.rolling(20).mean())


def _northbound_chg_5d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["northbound_hold_ratio"].transform(
        lambda s: s.pct_change(5, fill_method=None)
    )


FACTOR_REGISTRY: dict[str, FactorFn] = {
    "momentum_20d": _legacy_momentum_20d,
    "reversal_5d": _legacy_reversal_5d,
    "volatility_20d": _legacy_volatility_20d,
    "turnover_20d": _legacy_turnover_20d,
    "northbound_chg_5d": _northbound_chg_5d,
}


def compute_factors(price_df: pd.DataFrame, factor_names: list[str] | None = None) -> pd.DataFrame:
    """
    Compute factor values from price data.

    Uses quant-factors for shared OHLCV factors; keeps local registry for
    northbound and passthrough columns.
    """
    names = factor_names or list(FACTOR_REGISTRY.keys()) + list(PASSTHROUGH_FACTORS)
    qf_names = [n for n in names if n in _QF_ALIASES]
    local_names = [n for n in names if n not in qf_names or n in FACTOR_REGISTRY]

    base = price_df.copy()
    if qf_names:
        base = qf_compute(base, factors=qf_names)

    for name in local_names:
        if name in FACTOR_REGISTRY and name not in base.columns:
            base[name] = FACTOR_REGISTRY[name](base)
        elif name not in PASSTHROUGH_FACTORS and name not in base.columns:
            base[name] = np.nan

    return base


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
