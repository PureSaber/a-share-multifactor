"""Reproducible exploratory case: fixed strategy versus exposure-matched buy/hold."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from quant_lab.trials import TrialRegistry

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import _dict_to_config
from a_share_multifactor.corporate_actions import action_events
from a_share_multifactor.decision_contract import clean_json
from a_share_multifactor.decision_workflow import load_inputs, sha256, write_watchlist_catalog
from a_share_multifactor.performance import return_statistics
from a_share_multifactor.preprocess import prepare_factor_panel
from a_share_multifactor.research_validation import run_research_validation
from a_share_multifactor.run_contract import (
    _code_version,
    _installed_internal_dependencies,
    _replay,
    _target_schedule,
    replay_results,
)
from a_share_multifactor.synthesis import synthesize


def run_case(config_path: Path, inputs: Path, output: Path, *, sessions: int = 126) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest, frames = load_inputs(inputs)
    cfg = _dict_to_config(settings["app"])
    cfg = replace(
        cfg, filters=replace(cfg.filters, use_historical_universe=False), rebalance_freq="daily"
    )
    raw = frames["raw"].copy().sort_values(["symbol", "date"])
    raw["date"] = pd.to_datetime(raw.date)
    adjusted = frames["adjusted"][["symbol", "date", "open", "high", "low", "close"]]
    research = raw.drop(columns=["open", "high", "low", "close"]).merge(
        adjusted, on=["symbol", "date"], validate="one_to_one"
    )
    panel = prepare_factor_panel(cfg, research)
    scored = (
        synthesize(panel, cfg)
        .drop(columns=["open", "high", "low", "close"])
        .merge(
            raw[["symbol", "date", "open", "high", "low", "close"]],
            on=["symbol", "date"],
            validate="one_to_one",
        )
    )
    dates = pd.DatetimeIndex(sorted(scored.date.unique()))
    if sessions < 2 or sessions > len(dates):
        raise ValueError("Invalid research case window")
    start, end = dates[-sessions], dates[-1]
    identity = {
        "code_version": _code_version(Path(__file__).resolve().parents[2]),
        "dependencies": _installed_internal_dependencies(),
    }
    parameters = {
        "config_sha256": sha256(config_path),
        "sessions": sessions,
        "inputs": sha256(inputs / "manifest.json"),
    }
    registry = TrialRegistry(output.parent / "case-experiments.db")
    study = "exploratory-" + output.name
    registry.register(
        study,
        {
            "hypothesis": "20-day momentum and low volatility outperform a 50%-invested equal-weight basket after costs",
            "parameters": [parameters],
            "code_identity": identity,
            "selection_rule": "fixed single specification; retrospective case, no untouched holdout claim",
        },
    )
    attempt = registry.start(study, parameters)
    try:
        simulation = scored[scored.date >= start].copy()
        schedule = rebalance_dates(
            pd.Series(pd.to_datetime(frames["calendar"].date)), settings["frequency"]
        )
        simulation.loc[~simulation.date.isin(schedule), "composite_score"] = np.nan
        catalog = write_watchlist_catalog(
            output, settings, str(start.date()), str((end + pd.Timedelta(days=30)).date())
        )
        actions = action_events(frames.get("actions"), raw, adjusted, start, end)
        result = _replay(
            simulation,
            cfg,
            "strategy",
            catalog_path=catalog,
            corporate_actions=actions,
            risk_limits=settings["risk"],
        )
        strategy = replay_results(result, cfg)
        # Same maximum invested capital, same costs and next-bar matching; no look-ahead entry.
        invested_fraction = min(
            1 - cfg.costs.cash_buffer, cfg.costs.max_position_weight * cfg.costs.max_holdings
        )
        baseline_cfg = replace(
            cfg,
            quantiles=1,
            costs=replace(
                cfg.costs,
                max_holdings=len(settings["watchlist"]),
                max_position_weight=invested_fraction / len(settings["watchlist"]),
                cash_buffer=1 - invested_fraction,
            ),
        )
        baseline_panel = simulation.copy()
        baseline_panel["composite_score"] = 1.0
        first_schedule = _target_schedule(
            baseline_panel[baseline_panel.date == start], baseline_cfg, catalog_path=catalog
        )
        baseline_replay = _replay(
            baseline_panel,
            baseline_cfg,
            "buy-hold",
            catalog_path=catalog,
            corporate_actions=actions,
            target_schedule=first_schedule,
        )
        baseline = replay_results(baseline_replay, baseline_cfg)
        navs = pd.concat(
            [
                strategy.quantile_returns.iloc[:, 0].rename("strategy_net_return"),
                baseline.quantile_returns.iloc[:, 0].rename("equal_weight_buy_hold_net_return"),
            ],
            axis=1,
        )
        benchmark = frames["benchmark"].set_index("date").benchmark_return.reindex(navs.index)
        if benchmark.isna().any():
            raise ValueError("Benchmark coverage does not match research interval")
        benchmark.iloc[0] = 0
        navs["hs300_price_return_before_costs"] = benchmark
        navs.to_csv(output / "comparison_returns.csv")
        validation = run_research_validation(panel, cfg.factors, cfg.forward_return_col, cfg)
        validation.fold_metrics.to_csv(output / "folds.csv", index=False)
        validation.multiple_testing.to_csv(output / "fdr.csv", index=False)
        summary = {
            "scope": "retrospective exploratory fixed-watchlist case",
            "start": str(start.date()),
            "end": str(end.date()),
            "sessions": sessions,
            "hypothesis": registry.definition(study)["definition"]["hypothesis"],
            "study_id": study,
            "identity": identity,
            "inputs": manifest,
            "invested_fraction_limit": invested_fraction,
            "corporate_action_count": len(actions),
            "dividend_tax": "gross distributions; personal holding-period tax excluded",
            "factor_validation": validation.summary,
            "metrics": {name: return_statistics(navs[name].iloc[1:], 252) for name in navs},
            "strategy_fills": len(result.frames["fills"]),
            "baseline_fills": len(baseline_replay.frames["fills"]),
            "limitations": [
                "watchlist selection bias",
                "current adjusted-price vintage",
                "daily-bar execution approximation",
                "not independent holdout",
                "no alpha claim",
            ],
        }
        summary = clean_json(summary)
        (output / "case.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        for name, replay in (("strategy", result), ("baseline", baseline_replay)):
            for table in ("fills", "costs", "cash_ledger", "portfolio_snapshots"):
                replay.frames[table].to_parquet(output / f"{name}-{table}.parquet", index=False)
        registry.finish(
            attempt,
            "completed",
            {"case": str(output / "case.json"), "sha256": sha256(output / "case.json")},
        )
        return summary
    except Exception as exc:
        registry.finish(attempt, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=126)
    args = parser.parse_args()
    result = run_case(args.config, args.inputs, args.output, sessions=args.sessions)
    print(json.dumps({"case": str(args.output), "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
