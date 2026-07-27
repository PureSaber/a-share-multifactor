"""HTML report generation for backtest outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from a_share_multifactor.quantile_backtest import BacktestResult


def _table_html(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"<h2>{title}</h2><p>No data</p>"
    return f"<h2>{title}</h2>{df.to_html(classes='table', border=0)}"


def write_html_report(
    results: BacktestResult,
    ic_report: pd.DataFrame,
    output_path: Path,
    config_summary: str = "",
) -> None:
    """Write a simple HTML report with IC and backtest tables."""
    sections = [
        "<html><head><meta charset='utf-8'><title>Factor Research Report</title>",
        "<style>body{font-family:sans-serif;margin:2rem;} .table{border-collapse:collapse;}",
        "td,th{border:1px solid #ddd;padding:6px;}</style></head><body>",
        "<h1>A-Share Multifactor Report</h1>",
        f"<p>{config_summary}</p>",
        _table_html(ic_report, "IC Summary"),
        _table_html(results.stats, "Backtest Statistics"),
    ]

    if not results.turnover.empty:
        sections.append(_table_html(results.turnover.reset_index(), "Turnover"))

    if not results.excess_returns.empty:
        sections.append(_table_html(results.excess_returns.reset_index(), "Excess Returns"))

    sections.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
