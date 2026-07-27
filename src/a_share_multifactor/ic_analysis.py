"""Information Coefficient (IC) analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def calc_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    """Pearson IC between factor and forward return."""
    aligned = pd.concat([factor, forward_return], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def calc_rank_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    """Spearman rank IC between factor and forward return."""
    aligned = pd.concat([factor, forward_return], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def calc_ic_series(
    factors: pd.DataFrame,
    factor_col: str,
    return_col: str,
    date_col: str = "date",
    rank: bool = False,
) -> pd.Series:
    """Cross-sectional IC per rebalance date."""
    ic_fn = calc_rank_ic if rank else calc_ic
    ic_values = []
    dates = []
    for date, group in factors.groupby(date_col):
        ic_values.append(ic_fn(group[factor_col], group[return_col]))
        dates.append(date)
    suffix = "rank_" if rank else ""
    return pd.Series(ic_values, index=dates, name=f"ic_{suffix}{factor_col}")


def summarize_ic(ic_series: pd.Series) -> dict[str, float]:
    """Summarize IC series into mean, std, IR, and positive ratio."""
    clean = ic_series.dropna()
    if clean.empty:
        return {
            "mean_ic": float("nan"),
            "std_ic": float("nan"),
            "ir": float("nan"),
            "ic_positive_ratio": float("nan"),
        }

    mean_ic = float(clean.mean())
    std_ic = float(clean.std(ddof=0))
    ir = mean_ic / std_ic if std_ic not in (0.0, float("nan")) else float("nan")
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": ir,
        "ic_positive_ratio": float((clean > 0).mean()),
    }


def analyze_factors(
    panel: pd.DataFrame,
    factor_cols: list[str],
    return_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Batch IC summary for multiple factors."""
    rows: list[dict[str, float | str]] = []
    for factor_col in factor_cols:
        ic_series = calc_ic_series(panel, factor_col, return_col, date_col=date_col)
        rank_ic_series = calc_ic_series(panel, factor_col, return_col, date_col=date_col, rank=True)
        summary = summarize_ic(ic_series)
        rank_summary = summarize_ic(rank_ic_series)
        rows.append(
            {
                "factor": factor_col,
                "mean_ic": summary["mean_ic"],
                "std_ic": summary["std_ic"],
                "ir": summary["ir"],
                "ic_positive_ratio": summary["ic_positive_ratio"],
                "mean_rank_ic": rank_summary["mean_ic"],
                "ir_rank": rank_summary["ir"],
            }
        )
    return pd.DataFrame(rows)


def export_ic_series(
    panel: pd.DataFrame,
    factor_cols: list[str],
    return_col: str,
    output_dir: Path,
    date_col: str = "date",
) -> None:
    """Export per-factor IC time series CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for factor_col in factor_cols:
        ic_series = calc_ic_series(panel, factor_col, return_col, date_col=date_col)
        ic_series.to_csv(output_dir / f"ic_series_{factor_col}.csv", header=["ic"])


def analyze_ic_decay(
    panel: pd.DataFrame,
    factor_cols: list[str],
    horizons: list[int],
    date_col: str = "date",
    price_col: str = "close",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Compare IC across multiple forward return horizons."""
    rows: list[dict[str, float | str | int]] = []
    for horizon in horizons:
        return_col = f"forward_return_{horizon}d"
        temp = panel.copy()
        temp[return_col] = temp.groupby(symbol_col)[price_col].transform(
            lambda s, h=horizon: s.shift(-h) / s - 1
        )
        for factor_col in factor_cols:
            ic_series = calc_ic_series(temp, factor_col, return_col, date_col=date_col)
            summary = summarize_ic(ic_series)
            rows.append(
                {
                    "factor": factor_col,
                    "horizon_days": horizon,
                    "mean_ic": summary["mean_ic"],
                    "ir": summary["ir"],
                }
            )
    return pd.DataFrame(rows)
