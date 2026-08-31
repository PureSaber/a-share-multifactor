"""Factor computation for A-share multi-factor model.

Shared OHLCV factors come from ``quant_factors`` (catalog pin ``0.3.0``).
Local registry keeps multifactor-only columns (northbound, passthrough).
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from quant_factors.core import compute_factors as qf_compute

try:
    QUANT_FACTORS_VERSION = version("quant-factors")
except PackageNotFoundError:  # pragma: no cover
    QUANT_FACTORS_VERSION = "0.3.0"

FactorFn = Callable[[pd.DataFrame], pd.Series]

# Contract: these names must match quant_factors.core factor ids.
SHARED_QF_FACTORS = (
    "momentum_20d",
    "reversal_5d",
    "volatility_20d",
    "turnover_20d",
)

PASSTHROUGH_FACTORS = {
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "forecast_score",
    "industry_rs_20d",
}


def compute_momentum(close: pd.Series, window: int = 20) -> pd.Series:
    """Legacy helper: rolling pct change on a single close series."""
    return close.pct_change(window)


def _northbound_chg_5d(df: pd.DataFrame) -> pd.Series:
    return df.groupby("symbol")["northbound_hold_ratio"].transform(
        lambda s: s.pct_change(5, fill_method=None)
    )


FACTOR_REGISTRY: dict[str, FactorFn] = {
    "northbound_chg_5d": _northbound_chg_5d,
}


def quant_factors_version() -> str:
    """Version of the shared quant-factors package (Wave 3 contract)."""
    return QUANT_FACTORS_VERSION


def compute_factors(price_df: pd.DataFrame, factor_names: list[str] | None = None) -> pd.DataFrame:
    """
    Compute factor values from price data.

    Uses quant-factors for shared OHLCV factors; keeps local registry for
    northbound and passthrough columns.
    """
    names = factor_names or list(SHARED_QF_FACTORS) + list(FACTOR_REGISTRY) + list(
        PASSTHROUGH_FACTORS
    )
    qf_names = [n for n in names if n in SHARED_QF_FACTORS]
    local_names = [n for n in names if n not in qf_names]

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
