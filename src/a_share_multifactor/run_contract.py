"""Adapter from the equity backtest to quant-lab run schema v1."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from quant_lab.contracts import RunManifest, write_standard_run

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import AppConfig
from a_share_multifactor.quantile_backtest import BacktestResult, _assign_quantiles


def _code_version(repo_root: Path) -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=repo_root, check=False).returncode != 0
    return f"{revision}+dirty" if dirty else revision


def _returns_frame(results: BacktestResult, config: AppConfig) -> pd.DataFrame:
    rows: list[dict] = []
    turnover = (
        results.turnover["turnover"]
        if not results.turnover.empty and "turnover" in results.turnover
        else pd.Series(dtype=float)
    )
    cost_rate = 2 * (config.costs.commission + config.costs.slippage)
    for strategy in results.quantile_returns.columns:
        net = results.quantile_returns[strategy].dropna()
        nav = results.cumulative_returns[strategy].reindex(net.index)
        benchmark = results.benchmark_returns.reindex(net.index)
        for date, net_return in net.items():
            estimated_cost = float(turnover.get(date, 0.0)) * cost_rate
            rows.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "gross_return": float(net_return) + estimated_cost,
                    "net_return": float(net_return),
                    "nav": float(nav.get(date, np.nan)),
                    "benchmark_return": float(benchmark.get(date, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _position_and_order_frames(
    panel: pd.DataFrame, config: AppConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict] = []
    orders: list[dict] = []
    exposures: list[dict] = []
    previous: dict[str, float] = {}
    strategy = f"Q{config.quantiles}"
    factor_cols = [factor for factor in config.factors if factor in panel.columns]
    for date in rebalance_dates(panel["date"], config.rebalance_freq):
        day = panel[panel["date"] == date].copy()
        if day.empty:
            continue
        day["quantile"] = _assign_quantiles(day["composite_score"], config.quantiles)
        selected = day[day["quantile"] == float(config.quantiles)].copy()
        if selected.empty:
            continue
        weight = 1.0 / len(selected)
        current = {str(symbol): weight for symbol in selected["symbol"]}
        for symbol, target_weight in current.items():
            positions.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "symbol": symbol,
                    "quantity": np.nan,
                    "market_value": np.nan,
                    "weight": target_weight,
                    "side": "long",
                }
            )
        for symbol in sorted(set(previous) | set(current)):
            delta = current.get(symbol, 0.0) - previous.get(symbol, 0.0)
            if abs(delta) <= 1e-12:
                continue
            orders.append(
                {
                    "timestamp": date,
                    "strategy": strategy,
                    "symbol": symbol,
                    "side": "buy" if delta > 0 else "sell",
                    "quantity": np.nan,
                    "target_weight": current.get(symbol, 0.0),
                    "order_type": "rebalance_target",
                    "status": "simulated_filled",
                }
            )
        for factor in factor_cols:
            exposures.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "exposure_type": "factor",
                    "name": factor,
                    "value": float(pd.to_numeric(selected[factor], errors="coerce").mean()),
                }
            )
        previous = current
    return pd.DataFrame(positions), pd.DataFrame(orders), pd.DataFrame(exposures)


def write_equity_standard_run(
    run_dir: Path,
    results: BacktestResult,
    scored_panel: pd.DataFrame,
    config: AppConfig,
    *,
    dataset_snapshots: dict[str, str] | None = None,
) -> RunManifest:
    positions, orders, exposures = _position_and_order_frames(scored_panel, config)
    turnover = (
        results.turnover.reset_index()
        if not results.turnover.empty
        else pd.DataFrame(columns=["date", "turnover"])
    )
    costs = pd.DataFrame(
        {
            "date": turnover.get("date", pd.Series(dtype="object")),
            "strategy": f"Q{config.quantiles}",
            "symbol": "__portfolio__",
            "commission": turnover.get("turnover", pd.Series(dtype=float))
            * 2
            * config.costs.commission,
            "slippage": turnover.get("turnover", pd.Series(dtype=float))
            * 2
            * config.costs.slippage,
            "market_impact": 0.0,
            "borrow_cost": 0.0,
        }
    )
    if not costs.empty:
        costs["total_cost"] = costs[["commission", "slippage"]].sum(axis=1)
    metrics = {
        "statistics": results.stats.to_dict(orient="records"),
        "periods": len(results.quantile_returns),
        "turnover_mean": float(turnover["turnover"].mean()) if not turnover.empty else 0.0,
    }
    return write_standard_run(
        run_dir,
        project="a-share-multifactor",
        run_id=run_dir.name,
        strategy=f"Q{config.quantiles}",
        frames={
            "returns": _returns_frame(results, config),
            "positions": positions,
            "orders": orders,
            "costs": costs,
            "exposures": exposures,
        },
        metrics=metrics,
        config=asdict(config),
        code_version=_code_version(Path(__file__).resolve().parents[2]),
        dataset_snapshots=dataset_snapshots,
        tags={"asset_class": "cn_equity", "research_type": "multifactor"},
    )
