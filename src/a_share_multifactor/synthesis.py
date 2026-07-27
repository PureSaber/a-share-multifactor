"""Factor synthesis into composite score."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from a_share_multifactor.calendar import score_schedule_dates
from a_share_multifactor.config import AppConfig
from a_share_multifactor.ic_analysis import calc_ic_series, summarize_ic


def equal_weight_score(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Sum standardized factors with equal weights."""
    result = panel.copy()
    available = [col for col in factor_cols if col in result.columns]
    if not available:
        result["composite_score"] = float("nan")
        return result
    result["composite_score"] = result[available].mean(axis=1)
    return result


def ic_weight_score(
    panel: pd.DataFrame,
    factor_cols: list[str],
    ic_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Weight standardized factors by absolute IR (fallback to mean IC)."""
    result = panel.copy()
    available = [col for col in factor_cols if col in result.columns]
    if not available:
        result["composite_score"] = float("nan")
        return result

    weights: dict[str, float] = {}
    for factor in available:
        row = ic_summary.loc[ic_summary["factor"] == factor]
        if row.empty:
            weights[factor] = 0.0
            continue
        ir = row.iloc[0]["ir"]
        mean_ic = row.iloc[0]["mean_ic"]
        weight = abs(ir) if pd.notna(ir) else abs(mean_ic) if pd.notna(mean_ic) else 0.0
        weights[factor] = float(weight)

    total = sum(weights.values())
    if total == 0:
        return equal_weight_score(result, available)

    composite = sum(result[factor] * (weights[factor] / total) for factor in available)
    result["composite_score"] = composite
    return result


def _weights_from_ic_summary(ic_summary: pd.DataFrame, available: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for factor in available:
        row = ic_summary.loc[ic_summary["factor"] == factor]
        if row.empty:
            weights[factor] = 0.0
            continue
        ir = row.iloc[0]["ir"]
        mean_ic = row.iloc[0]["mean_ic"]
        weights[factor] = float(
            abs(ir) if pd.notna(ir) else abs(mean_ic) if pd.notna(mean_ic) else 0.0
        )
    total = sum(weights.values())
    if total == 0:
        return {factor: 1.0 / len(available) for factor in available}
    return {factor: weights[factor] / total for factor in available}


def rolling_ic_weight_score(
    panel: pd.DataFrame,
    factor_cols: list[str],
    return_col: str,
    lookback_months: int = 12,
    date_col: str = "date",
    score_dates: pd.DatetimeIndex | None = None,
    min_score_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute composite score using rolling IC/IR weights without lookahead."""
    result = panel.copy()
    available = [col for col in factor_cols if col in result.columns]
    result["composite_score"] = float("nan")
    if not available:
        return result

    dates = (
        score_dates
        if score_dates is not None
        else pd.DatetimeIndex(sorted(result[date_col].dropna().unique()))
    )
    if min_score_date is not None:
        dates = dates[dates >= pd.Timestamp(min_score_date)]
    for current_date in dates:
        lookback_start = pd.Timestamp(current_date) - pd.DateOffset(months=lookback_months)
        hist = result[(result[date_col] >= lookback_start) & (result[date_col] < current_date)]
        if hist.empty:
            continue

        ic_rows: list[dict[str, float | str]] = []
        for factor in available:
            ic_series = calc_ic_series(hist, factor, return_col, date_col=date_col)
            summary = summarize_ic(ic_series)
            ic_rows.append({"factor": factor, **summary})

        weights = _weights_from_ic_summary(pd.DataFrame(ic_rows), available)
        mask = result[date_col] == current_date
        result.loc[mask, "composite_score"] = sum(
            result.loc[mask, factor] * weights[factor] for factor in available
        )

    return result


def rolling_ml_score(
    panel: pd.DataFrame,
    factor_cols: list[str],
    return_col: str,
    method: str = "ridge",
    lookback_months: int = 12,
    ridge_alpha: float = 1.0,
    date_col: str = "date",
    score_dates: pd.DatetimeIndex | None = None,
    min_score_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rolling OLS/Ridge fit on past data to predict composite score without lookahead."""
    result = panel.copy()
    available = [col for col in factor_cols if col in result.columns]
    result["composite_score"] = float("nan")
    if not available:
        return result

    model_cls = Ridge if method == "ridge" else LinearRegression
    dates = (
        score_dates
        if score_dates is not None
        else pd.DatetimeIndex(sorted(result[date_col].dropna().unique()))
    )
    if min_score_date is not None:
        dates = dates[dates >= pd.Timestamp(min_score_date)]

    for current_date in dates:
        lookback_start = pd.Timestamp(current_date) - pd.DateOffset(months=lookback_months)
        hist = result[(result[date_col] >= lookback_start) & (result[date_col] < current_date)]
        train = hist[available + [return_col]].dropna()
        if len(train) < len(available) + 5:
            continue

        x_train = train[available].to_numpy()
        y_train = train[return_col].to_numpy()
        model = model_cls(alpha=ridge_alpha) if method == "ridge" else model_cls()
        model.fit(x_train, y_train)

        mask = result[date_col] == current_date
        x_pred = result.loc[mask, available].dropna()
        if x_pred.empty:
            continue
        preds = model.predict(x_pred.to_numpy())
        result.loc[x_pred.index, "composite_score"] = preds

    return result


def synthesize(
    panel: pd.DataFrame,
    config: AppConfig,
    ic_summary: pd.DataFrame | None = None,
    min_score_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Apply configured synthesis method."""
    factor_cols = [col for col in config.factors if col in panel.columns]

    if config.synthesis.method == "ic_weight":
        if ic_summary is None:
            raise ValueError("ic_summary is required for ic_weight synthesis")
        return ic_weight_score(panel, factor_cols, ic_summary)

    if config.synthesis.method == "rolling_ic_weight":
        score_dates = score_schedule_dates(
            panel["date"],
            config.rebalance_freq,
            config.costs.retail_mode,
            config.costs.trade_freq,
        )
        return rolling_ic_weight_score(
            panel,
            factor_cols,
            config.forward_return_col,
            lookback_months=config.synthesis.lookback_months,
            score_dates=score_dates,
            min_score_date=min_score_date,
        )

    if config.synthesis.method in {"ridge", "ols"}:
        score_dates = score_schedule_dates(
            panel["date"],
            config.rebalance_freq,
            config.costs.retail_mode,
            config.costs.trade_freq,
        )
        return rolling_ml_score(
            panel,
            factor_cols,
            config.forward_return_col,
            method=config.synthesis.method,
            lookback_months=config.synthesis.lookback_months,
            ridge_alpha=config.synthesis.ridge_alpha,
            score_dates=score_dates,
            min_score_date=min_score_date,
        )

    return equal_weight_score(panel, factor_cols)
