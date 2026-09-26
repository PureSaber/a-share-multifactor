"""Equity/ETF research adapter: explicit recipes into one QExec ledger per candidate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from quant_data_kit.research_coverage import (
    asof_history,
    attach_history,
    load_history,
    preflight,
)
from quant_factors.core import compute_factors
from quant_factors.research import factor_report, factor_requirements
from quant_lab.research import canonical, file_hash

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import _dict_to_config
from a_share_multifactor.corporate_actions import action_events
from a_share_multifactor.decision_workflow import load_inputs, validate_inputs
from a_share_multifactor.performance import return_statistics
from a_share_multifactor.run_contract import (
    _replay,
    _target_schedule,
    _write_certified_v2,
    load_fixture_catalog,
    replay_results,
)


class EquityResearchExecutor:
    """One immutable-input cache shared by the preregistered candidate batch."""

    def __init__(self):
        self._inputs = {}
        self._features = {}

    def _load(self, recipe):
        root = Path(recipe["inputs"]["bundle"])
        key = file_hash(root / "manifest.json")
        # Revalidate source bytes even on a cache hit; a stale cache must not hide mutation.
        manifest, frames = load_inputs(root)
        identity = {name: entry["sha256"] for name, entry in manifest["files"].items()}
        cache_key = canonical({"manifest": key, "files": identity})
        if cache_key not in self._inputs:
            self._inputs[cache_key] = (manifest, frames)
        _, frames = self._inputs[cache_key]
        return cache_key, frames

    def __call__(self, recipe: dict, candidate: dict, output: Path) -> dict:
        if recipe["backend"] != "equity":
            raise ValueError("Equity adapter received a different backend")
        strategy = candidate["strategy"]
        if strategy["family"] not in {"rank", "etf_trend", "buy_hold"}:
            raise ValueError("Unsupported equity strategy family")
        cache_key, frames = self._load(recipe)
        start, end = recipe["interval"]["start"], recipe["interval"]["end"]
        raw = frames["raw"].copy()
        raw["date"] = pd.to_datetime(raw.date)
        raw = raw[raw.date <= pd.Timestamp(end)].sort_values(["symbol", "date"])
        symbols = sorted(raw.symbol.unique())
        adjusted = frames["adjusted"].copy()
        adjusted["date"] = pd.to_datetime(adjusted.date)
        adjusted = adjusted[adjusted.date <= pd.Timestamp(end)]
        prices = adjusted[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(prices).all().all() or not prices.gt(0).all().all():
            raise ValueError("Adjusted prices must be finite and positive")
        validation_frames = {**frames, "raw": raw}
        validate_inputs(validation_frames, symbols, pd.Timestamp(end))
        if set(zip(raw.symbol, raw.date)) != set(zip(adjusted.symbol, adjusted.date)):
            raise ValueError("Raw/adjusted bar keys differ")
        research = raw.drop(columns=["open", "high", "low", "close"]).merge(
            adjusted[["symbol", "date", "open", "high", "low", "close"]],
            on=["symbol", "date"],
            validate="one_to_one",
        )
        history = None
        if "history" in recipe["inputs"]:
            _, history = load_history(Path(recipe["inputs"]["history"]))
            fields = {
                k: v
                for k, v in recipe.get("required_history", {}).items()
                if v in {"fundamentals", "classification"}
            }
            if fields:
                research = attach_history(research, history, fields)
        names = list(candidate["factors"])
        requirements = factor_requirements(names)
        fundamental_columns = {
            column
            for requirement in requirements.values()
            if requirement.get("pit_required")
            for column in requirement["columns"]
        }
        missing_mappings = sorted(
            column
            for column in fundamental_columns
            if recipe.get("required_history", {}).get(column) != "fundamentals"
        )
        if missing_mappings:
            raise ValueError(
                f"Fundamental factors require a publication-time mapping: {missing_mappings}"
            )
        if strategy["family"] == "etf_trend":
            requirements["trend_filter"] = {
                "columns": ["close"],
                "warmup_bars": strategy["trend_window"],
            }
        if any(req.get("pit_required") for req in requirements.values()) and history is None:
            raise ValueError("Fundamental factors require an imported publication-time history")
        check = preflight(
            research,
            frames["calendar"],
            symbols=symbols,
            start=start,
            end=end,
            requirements=requirements,
            history=history,
            required_history=recipe.get("required_history", {}),
        )
        (output / "preflight.json").write_text(canonical(check), encoding="utf-8")
        if not check["passed"]:
            raise ValueError("Data preflight failed; inspect preflight.json")
        if history is not None:
            for day in sorted(research.loc[research.date.between(start, end), "date"].unique()):
                timestamp = pd.Timestamp(day).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
                for field, domain in recipe.get("required_history", {}).items():
                    if domain not in {"status", "universe"}:
                        continue
                    rows = asof_history(history, as_of=timestamp, domain=domain, field=field)
                    relevant = rows[rows.symbol.isin(symbols)]
                    if relevant.value.eq("false").any():
                        raise ValueError(
                            "Changing universe or restricted historical status requires an execution-status model; fixed-universe replay rejected"
                        )
        # Feature values use only past prices, cached independently of costs/allocation.
        features_key = canonical(
            {
                "input": cache_key,
                "factors": names,
                "end": end,
                "history": file_hash(Path(recipe["inputs"]["history"]) / "manifest.json")
                if history is not None
                else None,
            }
        )
        if features_key not in self._features:
            self._features[features_key] = compute_factors(research, names)
        features = self._features[features_key].copy()
        valid_features = features[names].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(valid_features).all(axis=1)
        ranks = pd.concat(
            [
                features.groupby("date")[name].rank(pct=True) * direction
                for name, direction in candidate["factors"].items()
            ],
            axis=1,
        )
        features["composite_score"] = ranks.mean(axis=1).where(finite)
        features["eligible"] = finite
        if strategy["family"] == "etf_trend":
            ma = features.groupby("symbol").close.transform(
                lambda series: series.rolling(strategy["trend_window"]).mean()
            )
            features["eligible"] &= features.close.gt(ma)
        delay = candidate["signal_delay"]
        if delay:
            features["composite_score"] = features.groupby("symbol").composite_score.shift(delay)
            features["eligible"] = features.groupby("symbol").eligible.shift(delay).eq(True)
        scored = features.drop(columns=["open", "high", "low", "close"]).merge(
            raw[["symbol", "date", "open", "high", "low", "close"]],
            on=["symbol", "date"],
            validate="one_to_one",
        )
        scored = scored[(scored.date >= pd.Timestamp(start)) & (scored.date <= pd.Timestamp(end))]
        if (
            scored.empty
            or scored.date.min() != pd.Timestamp(start)
            or scored.date.max() != pd.Timestamp(end)
        ):
            raise ValueError("Evaluation endpoints must be explicit observed trading sessions")
        multiplier = candidate["cost_multiplier"]
        costs = dict(recipe["costs"])
        for field in ("commission", "min_commission", "stamp_tax", "slippage"):
            costs[field] *= multiplier
        cfg = _dict_to_config(
            {
                "factors": names,
                "factor_directions": candidate["factors"],
                "quantiles": 1,
                "rebalance_freq": "daily",
                "holding_period": "fixed",
                "costs": {
                    **costs,
                    "retail_mode": False,
                    "max_holdings": strategy["top_n"],
                    "max_position_weight": strategy["max_weight"],
                    "cash_buffer": strategy["cash_buffer"],
                },
            }
        )
        if "catalog" in recipe["inputs"]:
            catalog = Path(recipe["inputs"]["catalog"])
            catalog_frame = load_fixture_catalog(catalog)
        else:
            raise ValueError("Research requires an explicit versioned instrument catalog")
        if strategy["family"] == "etf_trend" and not catalog_frame.asset_class.eq("etf").all():
            raise ValueError("ETF template requires ETF instrument specifications")
        if "catalog" in recipe["inputs"]:
            (output / "instrument_catalog.csv").write_bytes(catalog.read_bytes())
            catalog = output / "instrument_catalog.csv"
        invested = min(1 - strategy["cash_buffer"], strategy["top_n"] * strategy["max_weight"])
        if strategy["family"] == "buy_hold":
            cfg = replace(
                cfg,
                costs=replace(
                    cfg.costs,
                    max_holdings=len(symbols),
                    max_position_weight=invested / len(symbols),
                    cash_buffer=1 - invested,
                ),
            )
            first = scored[scored.date == scored.date.min()].assign(composite_score=1.0)
            schedule = _target_schedule(first, cfg, catalog_path=catalog)
        else:
            schedule = {}
            for day in rebalance_dates(scored.date, strategy["frequency"]):
                current = scored[scored.date == day].copy()
                if not np.isfinite(current.composite_score).all():
                    raise ValueError(f"Missing signal in evaluation interval: {day.date()}")
                if history is not None:
                    timestamp = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
                    for field, domain in recipe.get("required_history", {}).items():
                        if domain not in {"status", "universe"}:
                            continue
                        rows = asof_history(history, as_of=timestamp, domain=domain, field=field)
                        rows = rows[rows.symbol.isin(symbols)]
                        if domain == "status" and rows.value.eq("false").any():
                            raise ValueError(
                                "Restricted historical status requires an execution-status model"
                            )
                        allowed = set(rows.loc[rows.value.eq("true"), "symbol"])
                        current["eligible"] &= current.symbol.isin(allowed)
                selected = current[current.eligible]
                targets = (
                    _target_schedule(selected, cfg, catalog_path=catalog) if len(selected) else {}
                )
                schedule[day.date()] = targets.get(day.date(), {})
        actions = action_events(
            frames.get("actions"), raw, adjusted, pd.Timestamp(start), pd.Timestamp(end)
        )
        replay = _replay(
            scored,
            cfg,
            candidate["candidate_id"],
            catalog_path=catalog,
            corporate_actions=actions,
            target_schedule=schedule,
            risk_limits=recipe.get("risk", {}),
            strategy_id="research-" + candidate["candidate_id"],
        )
        result = replay_results(replay, cfg)
        returns = result.quantile_returns.iloc[:, 0]
        returns.to_csv(output / "returns.csv", header=["net_return"])
        statistics = return_statistics(returns.iloc[1:], 252)
        factors = factor_report(
            research,
            names,
            cutoff=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
            start=start,
            end=end,
        )
        (output / "factors.json").write_text(canonical(factors), encoding="utf-8")
        metrics = {
            "total_return": statistics["total_return"],
            "max_drawdown": statistics["max_drawdown"],
            "sharpe": statistics["sharpe"],
            "fills": len(replay.frames["fills"]),
            "cost_total": float(
                sum(
                    replay.frames["costs"].amount_units.astype(float)
                    / 10.0 ** replay.frames["costs"].amount_scale.astype(float)
                )
            ),
        }
        metrics = {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in metrics.items()
        }
        _write_certified_v2(
            output,
            scored,
            cfg,
            {
                "research-inputs": "sha256:"
                + file_hash(Path(recipe["inputs"]["bundle"]) / "manifest.json")
            },
            replay=replay,
            catalog_path=catalog,
            research_metrics={"research_metrics": metrics},
        )
        segments = []
        for year, group in returns.iloc[1:].groupby(returns.iloc[1:].index.year):
            segments.append(
                {
                    "year": int(year),
                    "sessions": len(group),
                    "net_return": float((1 + group).prod() - 1),
                }
            )
        for offset in range(1, len(returns), 21):
            group = returns.iloc[offset : offset + 21]
            segments.append(
                {
                    "window": str(group.index[0].date()),
                    "sessions": len(group),
                    "net_return": float((1 + group).prod() - 1),
                }
            )
        return {
            "metrics": metrics,
            "segments": segments,
            "factor_evidence": factors,
            "comparison": {
                "start": start,
                "end": end,
                "currency": "CNY",
                "frequency": "daily",
                "universe": symbols,
                "costs": costs,
                "invested_limit": invested,
            },
            "scope": "synthetic-software-demonstration"
            if raw.source.astype(str).str.contains("synthetic").any()
            else "retrospective-fixed-specification-qexec-research",
            "limitations": [
                "not independent holdout performance",
                "current adjusted-price vintage",
                "daily-bar execution; no order-book queue",
                "initial-capital target sizing",
                "gross dividends; personal dividend tax excluded",
            ],
        }
