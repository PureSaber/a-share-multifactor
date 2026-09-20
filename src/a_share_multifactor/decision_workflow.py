"""Real public daily data -> reproducible research -> one paper decision card.

This bounded watchlist profile is deliberately distinct from historical index
research and from the L2 market-data release certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from quant_data_kit.exceptions import ValidationError
from quant_lab.trials import TrialRegistry

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import _dict_to_config
from a_share_multifactor.corporate_actions import action_events
from a_share_multifactor.decision_contract import SCHEMA_VERSION, clean_json, write_decision
from a_share_multifactor.ic_analysis import analyze_factors, analyze_ic_decay
from a_share_multifactor.label_timing import mature_labels
from a_share_multifactor.performance import return_statistics
from a_share_multifactor.preprocess import prepare_factor_panel
from a_share_multifactor.research_validation import run_research_validation
from a_share_multifactor.run_contract import (
    _code_version,
    _installed_internal_dependencies,
    _replay,
    _write_certified_v2,
    replay_results,
)
from a_share_multifactor.synthesis import synthesize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_json(path: Path, value: dict):
    path.write_text(
        json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def fetch_inputs(root: Path, settings: dict, start: str, end: str, captured_at: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "origin": "live_public_api",
        "captured_at": captured_at,
        "requested_start": start,
        "requested_end": end,
        "files": {},
    }
    kinds = ["calendar", "raw", "adjusted", "benchmark"]
    if settings.get("corporate_actions", True):
        kinds.append("actions")
    if settings.get("trading_status", "off") != "off":
        kinds.append("status")
    for kind in kinds:
        path = root / f"{kind}.parquet"
        command = [
            sys.executable,
            "-m",
            "a_share_multifactor.market_data",
            kind,
            "--start",
            start,
            "--end",
            end,
            "--output",
            str(path.resolve()),
            "--symbols",
            *[item["symbol"] for item in settings["watchlist"]],
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.get("provider_timeout_seconds", 90),
            )
        except subprocess.TimeoutExpired:
            if kind != "status" or settings.get("trading_status") != "advisory":
                raise
            result = subprocess.CompletedProcess(command, 124, "", "Current ST/halt feed timed out")
        if result.returncode:
            (root / f"{kind}-error.txt").write_text(result.stderr, encoding="utf-8")
            if kind == "status" and settings.get("trading_status") == "advisory":
                manifest.setdefault("warnings", []).append("Current ST/halt feed unavailable")
                _save_json(root / "manifest.json", manifest)
                continue
            raise RuntimeError(f"{kind} provider failed; see inputs/{kind}-error.txt")
        manifest["files"][kind] = {"file": path.name, "sha256": sha256(path)}
        _save_json(root / "manifest.json", manifest)
    return manifest


def load_inputs(root: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for name in (
        "calendar",
        "raw",
        "adjusted",
        "benchmark",
        *[key for key in ("actions", "status") if key in manifest["files"]],
    ):
        entry = manifest["files"][name]
        path = (root / entry["file"]).resolve()
        if path.parent != root.resolve() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Input snapshot integrity failed: {name}")
        frames[name] = pd.read_parquet(path)
    return manifest, frames


def validate_inputs(frames: dict, symbols: list[str], as_of: pd.Timestamp) -> dict:
    raw = frames["raw"]
    calendar = pd.DatetimeIndex(pd.to_datetime(frames["calendar"]["date"]))
    if calendar.empty or calendar.max() <= as_of:
        raise ValueError("Trading calendar does not cover the next session")
    expected = calendar[calendar <= as_of].max()
    if raw.empty or set(raw.symbol) != set(symbols):
        raise ValueError("Missing watchlist prices")
    if raw.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate symbol/date bars")
    if not raw.adjustment.eq("none").all() or not raw.volume_unit.eq("share").all():
        raise ValueError("Simulation requires raw prices and volume in shares")
    numbers = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
    if not np.isfinite(numbers).all().all() or (numbers < 0).any().any():
        raise ValueError("Invalid OHLCV values")
    if (
        (numbers[["open", "low", "close"]] <= 0).any().any()
        or (numbers.high < numbers[["open", "low", "close"]].max(axis=1)).any()
        or (numbers.low > numbers[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("Inconsistent OHLC bars")
    if pd.to_datetime(raw.date).max() > as_of:
        raise ValueError("Input contains bars after the decision cutoff")
    for symbol, group in raw.groupby("symbol"):
        dates = pd.DatetimeIndex(pd.to_datetime(group.date))
        expected_dates = calendar[(calendar >= dates.min()) & (calendar <= expected)]
        if dates.max() != expected or set(dates) != set(expected_dates):
            raise ValueError(f"Stale or missing sessions for {symbol}; no forward-filled prices")
    benchmark = frames["benchmark"]
    if benchmark.empty or pd.to_datetime(benchmark.date).max() < expected:
        raise ValueError("Benchmark is stale or missing")
    return {
        "passed": True,
        "as_of": expected.date().isoformat(),
        "price_rows": len(raw),
        "symbols": len(symbols),
        "volume_unit": "share",
        "price_adjustment": "none",
        "sources": sorted(raw.source.unique()),
        "historical_publication_times_verified": False,
        "universe_scope": "fixed demonstration watchlist, not historical HS300",
        "revision_history": "snapshot captured now; historical vintages unavailable",
    }


def validate_trading_status(
    rows: pd.DataFrame | None,
    *,
    symbols: list[str],
    as_of: pd.Timestamp,
    now: pd.Timestamp,
    required: bool,
    max_age_hours: float,
) -> str | list[dict]:
    """Admit only a complete, same-session, recently captured status snapshot."""

    if max_age_hours <= 0:
        raise ValueError("trading_status_max_age_hours must be positive")
    if rows is None:
        if required:
            raise ValueError("A current ST/halt snapshot is required")
        return "unverified"
    if set(rows.symbol) != set(symbols) or rows.symbol.duplicated().any():
        raise ValueError("Incomplete/duplicate trading-status snapshot")
    captured = pd.to_datetime(rows.captured_at, utc=True)
    now_utc = pd.Timestamp(now).tz_convert("UTC")
    sessions = pd.to_datetime(rows.session).dt.normalize()
    expected = pd.Timestamp(as_of).normalize()
    if (captured > now_utc).any() or not sessions.eq(expected).all():
        raise ValueError("Future or wrong-session trading-status snapshot")
    if ((now_utc - captured) > pd.Timedelta(hours=max_age_hours)).any():
        raise ValueError("Stale trading-status snapshot")
    if not rows.status.eq("no_reported_restriction").all():
        raise ValueError("Current ST/halt restriction or unknown tradability requires review")
    return rows[["symbol", "status", "source", "captured_at"]].to_dict("records")


def write_watchlist_catalog(root: Path, settings: dict, start: str, end: str) -> Path:
    rows = []
    for item in settings["watchlist"]:
        rows.append(
            {
                "symbol": item["symbol"],
                "asset_class": "equity",
                "product_type": "stock",
                "venue": item["venue"],
                "price_scale": 2,
                "price_tick": "0.01",
                "quantity_step": 1,
                "lot_size": 100,
                "commission_rate": 0,
                "stamp_duty_rate": 0,
                "effective_from": f"{start}T00:00:00Z",
                "effective_to": f"{end}T00:00:00Z",
                "available_at": f"{start}T00:00:00Z",
            }
        )
    path = root / "watchlist_catalog.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _dependency_revisions() -> dict[str, str]:
    return _installed_internal_dependencies()


def _paper_proposal(replay, scored, config, catalog_path, as_of):
    ticks = (
        pd.read_csv(catalog_path, dtype={"symbol": str, "price_tick": str})
        .set_index("symbol")
        .price_tick.to_dict()
    )
    snapshots = replay.frames["portfolio_snapshots"]
    last = snapshots.iloc[-1]
    nav = float(last.nav_units / 10**last.nav_scale)
    quantities = {
        symbol: float(value.units / 10**value.scale)
        for symbol, value in replay.ledger.snapshot(
            replay.events[-1].available_at
        ).positions.items()
        if value.units
    }
    prices = scored.loc[scored.date == as_of].set_index("symbol").close.to_dict()
    # Show the very orders accepted by the recorded replay, never a second
    # independently sized portfolio that differs from the execution artifacts.
    orders = replay.frames["orders"]
    pending = orders[orders.status.isin(["accepted", "partially_filled"])]
    targets = dict(quantities)
    order_ids = {}
    for order in pending.itertuples():
        remaining = order.quantity_units / 10**order.quantity_scale - (
            order.filled_quantity_units / 10**order.filled_quantity_scale
        )
        targets[order.instrument_id] = targets.get(order.instrument_id, 0) + (
            remaining if order.side == "buy" else -remaining
        )
        order_ids[order.instrument_id] = order.order_id
    targets = {symbol: amount for symbol, amount in targets.items() if amount > 0}
    current = [
        {
            "symbol": symbol,
            "quantity": quantity,
            "close": prices[symbol],
            "weight": quantity * prices[symbol] / nav,
        }
        for symbol, quantity in sorted(quantities.items())
    ]
    target_rows = [
        {"symbol": symbol, "quantity": quantity, "weight": quantity * prices[symbol] / nav}
        for symbol, quantity in sorted(targets.items())
    ]
    trades = []
    fees = 0.0
    slippage = 0.0
    for symbol in sorted(set(targets) | set(quantities)):
        delta = targets.get(symbol, 0) - quantities.get(symbol, 0)
        if not delta:
            continue
        reference = Decimal(str(prices[symbol]))
        tick = Decimal(ticks[symbol])
        direction = Decimal(1 if delta > 0 else -1)
        execution_price = reference * (1 + direction * Decimal(str(config.costs.slippage)))
        rounding = ROUND_CEILING if delta > 0 else ROUND_FLOOR
        execution_price = (execution_price / tick).to_integral_value(rounding=rounding) * tick
        slip = float(abs(execution_price - reference) * Decimal(str(abs(delta))))
        execution_amount = float(execution_price * Decimal(str(abs(delta))))
        fee = max(execution_amount * config.costs.commission, config.costs.min_commission)
        fee += execution_amount * config.costs.stamp_tax if delta < 0 else 0
        fees += fee
        slippage += slip
        trades.append(
            {
                "symbol": symbol,
                "order_id": order_ids[symbol],
                "side": "buy" if delta > 0 else "sell",
                "quantity": abs(delta),
                "reference_close": prices[symbol],
                "estimated_execution_price": float(execution_price),
                "estimated_fee": fee,
                "estimated_slippage": slip,
                "execution": "next-session simulation; price and fill not guaranteed",
            }
        )
    return (
        nav,
        current,
        target_rows,
        trades,
        {"currency": "CNY", "fees": fees, "slippage": slippage, "total": fees + slippage},
    )


def _run_decision(
    config_path: Path,
    output_root: Path,
    *,
    as_of: str | None = None,
    inputs: Path | None = None,
    now: pd.Timestamp | None = None,
) -> Path:
    now = now or pd.Timestamp(datetime.now(timezone.utc))
    local = now.tz_convert("Asia/Shanghai")
    cutoff = pd.Timestamp(as_of).normalize() if as_of else local.tz_localize(None).normalize()
    if not as_of and local.hour < 16:
        cutoff -= pd.Timedelta(days=1)
    if cutoff > local.tz_localize(None).normalize():
        raise ValueError("Decision cutoff cannot be in the future")
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = config_path.resolve()
    card = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "blocked",
        "as_of": cutoff.date().isoformat(),
        "generated_at": now.isoformat(),
        "valid_until": None,
        "scope": "paper_simulation_only",
        "data_quality": {"passed": False},
        "validation": {"passed": False},
        "current_positions": [],
        "targets": [],
        "proposed_trades": [],
        "estimated_cost": {},
        "risk": {},
        "evidence": {},
        "reasons": [],
    }
    registry = TrialRegistry(output_root / "experiments.db")
    attempt = None
    try:
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict) or not isinstance(settings.get("app"), dict):
            raise TypeError("Decision configuration requires an app mapping")
        if not isinstance(settings.get("risk"), dict):
            raise TypeError("Decision configuration requires a risk mapping")
        watchlist = settings.get("watchlist")
        if not isinstance(watchlist, list) or not watchlist:
            raise TypeError("Decision configuration requires a non-empty watchlist")
        symbols = [str(item["symbol"]) for item in watchlist]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Decision watchlist contains duplicate symbols")
        classifications = {
            str(item["symbol"]): str(item["industry"]).strip()
            for item in watchlist
            if str(item.get("industry", "")).strip()
        }
        if settings["risk"].get("max_industry_weight") is not None and len(classifications) != len(
            watchlist
        ):
            raise ValueError(
                "Every watchlist item requires an industry when industry risk is enabled"
            )
        settings["risk"] = {**settings["risk"], "classifications": classifications}
        config = _dict_to_config(settings["app"])
        config = replace(
            config,
            universe="explicit_watchlist",
            filters=replace(config.filters, use_historical_universe=False),
        )
        card["risk"] = settings["risk"]
        config_digest = sha256(config_path)
        _save_json(
            run_dir / "experiment.json",
            {
                "run_id": run_id,
                "config_sha256": config_digest,
                "settings": settings,
                "selection": "predeclared; no parameter search or return ranking",
            },
        )
        if config.costs.retail_mode or config.synthesis.method != "equal_weight":
            raise ValueError(
                "The first decision profile uses fixed equal weights and QExec simulation"
            )
        if settings["frequency"] not in {"daily", "weekly"} or not config.validation.enabled:
            raise ValueError("Decision profile requires daily/weekly frequency and validation")
        identity = {
            "code_version": _code_version(Path(__file__).resolve().parents[2]),
            "internal_dependencies": _dependency_revisions(),
        }
        card["evidence"].update(identity)
        study = settings.get("study", {})
        parameters = {"config_sha256": config_digest}
        study_id = (
            study.get("id")
            or "decision-"
            + hashlib.sha256(
                (config_digest + json.dumps(identity, sort_keys=True)).encode()
            ).hexdigest()[:20]
        )
        definition = {
            "hypothesis": study.get(
                "hypothesis", "Predeclared watchlist momentum/volatility paper trial"
            ),
            "parameters": [parameters],
            "code_identity": identity,
            "selection_rule": "retain all attempts; no retrospective winner selection",
        }
        if study.get("holdout_start"):
            definition["holdout_start"] = study["holdout_start"]
            definition["holdout_end"] = study["holdout_end"]
        registry.register(study_id, definition, now=now.to_pydatetime())
        attempt = registry.start(study_id, parameters, context={"run_id": run_id})
        card["evidence"].update(study_id=study_id, attempt_id=attempt)
        state_path = output_root.resolve() / "paper_state.json"
        prior = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        if prior and prior["config_sha256"] != config_digest:
            raise ValueError("Paper configuration changed; start a separate output directory/trial")
        if prior and prior.get("execution_identity") != identity:
            raise ValueError(
                "Paper code/dependencies changed or are unrecorded; start a separate trial"
            )
        source_root = inputs.resolve() if inputs else run_dir / "inputs"
        start = (cutoff - pd.Timedelta(days=settings.get("history_days", 730))).date().isoformat()
        if not inputs:
            fetch_inputs(source_root, settings, start, cutoff.date().isoformat(), now.isoformat())
        manifest, frames = load_inputs(source_root)
        card["evidence"]["inputs"] = str(source_root)
        card["evidence"]["input_manifest_sha256"] = sha256(source_root / "manifest.json")
        card["evidence"]["input_origin"] = manifest["origin"]
        quality = validate_inputs(frames, symbols, cutoff)
        card["data_quality"] = quality
        asof = pd.Timestamp(quality["as_of"])
        if asof == local.tz_localize(None).normalize() and local.hour < 16:
            raise ValueError("Today's daily bar is not admitted before 16:00 Asia/Shanghai")
        card["as_of"] = quality["as_of"]
        quality["current_trading_status"] = validate_trading_status(
            frames.get("status"),
            symbols=symbols,
            as_of=asof,
            now=now,
            required=settings.get("trading_status") == "required",
            max_age_hours=float(settings.get("trading_status_max_age_hours", 24)),
        )
        calendar = pd.DatetimeIndex(pd.to_datetime(frames["calendar"].date)).sort_values()
        next_session = calendar[calendar > asof].min()
        valid_until = next_session.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        card["valid_until"] = valid_until.isoformat()
        raw = frames["raw"].copy().sort_values(["symbol", "date"])
        raw["date"] = pd.to_datetime(raw.date)
        config = replace(config, start_date=start, end_date=card["as_of"], rebalance_freq="daily")
        adjusted = frames["adjusted"][["symbol", "date", "open", "high", "low", "close"]]
        if adjusted.duplicated(["symbol", "date"]).any():
            raise ValueError("Duplicate adjusted-price observations")
        # Research price ratios use the current adjusted snapshot, while every
        # execution and position mark uses actual unadjusted traded prices.
        research = raw.drop(columns=["open", "high", "low", "close"]).merge(
            adjusted, on=["symbol", "date"], how="left", validate="one_to_one"
        )
        if research.close.isna().any():
            raise ValueError("Adjusted research prices do not cover raw observations")
        panel = prepare_factor_panel(config, research)
        scored = synthesize(panel, config)
        scored = scored.drop(columns=["open", "high", "low", "close"]).merge(
            raw[["symbol", "date", "open", "high", "low", "close"]],
            on=["symbol", "date"],
            validate="one_to_one",
        )
        # Validate factor evidence separately from portfolio performance. All labels
        # used to determine a fold's signs must have matured before that fold.
        diagnostic_panel = panel
        if study.get("holdout_start"):
            holdout_start = pd.Timestamp(study["holdout_start"])
            diagnostic_panel = panel[
                (panel.date < holdout_start)
                & mature_labels(panel, config.forward_return_col, holdout_start)
            ]
        validation = run_research_validation(
            diagnostic_panel, config.factors, config.forward_return_col, config
        )
        validation.fold_metrics.to_csv(run_dir / "validation_folds.csv", index=False)
        validation.multiple_testing.to_csv(run_dir / "validation_fdr.csv", index=False)
        card["validation"] = {
            "passed": True,
            "factor_diagnostics": validation.summary,
            "investment_effectiveness": "unproven",
            "historical_scope": "retrospective fixed watchlist, current adjusted-price snapshot",
            "selection_bias": "current example watchlist; not a historical index",
            "corporate_actions": "historical factor diagnostics are descriptive only",
            "prospective_holdout": {
                "start": study.get("holdout_start"),
                "end": study.get("holdout_end"),
                "status": "collecting_not_evaluated"
                if study.get("holdout_start")
                else "not_registered",
            },
        }
        sessions = calendar[(calendar >= raw.date.min()) & (calendar <= asof)]
        # Default forward account starts at the first observed close. Optional
        # retrospective replay is explicit and keeps the corporate-action guard.
        count = int(settings.get("simulation_sessions", 1))
        if not 1 <= count <= len(sessions):
            raise ValueError("simulation_sessions must be positive and within available history")
        simulation_start = pd.Timestamp(prior["start"]) if prior else sessions[-count]
        simulation = scored[scored.date >= simulation_start].copy()
        simulation["decision_allowed"] = True
        if now >= valid_until or manifest["origin"] != "live_public_api":
            simulation.loc[simulation.date == asof, "decision_allowed"] = False
        actions = action_events(frames.get("actions"), raw, adjusted, simulation_start, asof)
        action_identity = {event.event_id: repr(event) for event in actions}
        if prior and any(
            action_identity.get(key) != value
            for key, value in prior.get("applied_actions", {}).items()
        ):
            raise ValueError("Previously applied corporate actions changed or disappeared")
        card["validation"]["corporate_actions"] = {
            "events": len(actions),
            "cash_policy": "gross dividend; personal dividend tax not modeled",
            "supported": "previous-session entitlement, same-day cash/share delivery",
            "historical_availability": "captured source snapshot; not verified historical vintages",
        }
        schedule = rebalance_dates(pd.Series(calendar), settings["frequency"])
        # Full exchange calendar prevents a truncated mid-week run from inventing
        # a Friday rebalance. Warm-up and non-schedule rows cannot emit orders.
        simulation.loc[~simulation.date.isin(schedule), "composite_score"] = np.nan
        if prior:
            prior_panel_path = Path(prior["scored_panel"])
            if sha256(prior_panel_path) != prior["scored_panel_sha256"]:
                raise ValueError("Saved paper signal snapshot was mutated")
            frozen = pd.read_parquet(prior_panel_path)
            old_asof = pd.Timestamp(prior["as_of"])
            if asof < old_asof:
                raise ValueError("Cannot move the paper account backward in time")
            previous_prices = frozen[["symbol", "date", "open", "high", "low", "close", "volume"]]
            observed = simulation.merge(
                previous_prices, on=["symbol", "date"], suffixes=("", "_old")
            )
            for column in ["open", "high", "low", "close", "volume"]:
                if not np.allclose(observed[column], observed[f"{column}_old"], rtol=0, atol=1e-8):
                    raise ValueError(
                        f"Historical raw data revision detected: {column}; reconcile before advancing"
                    )
            if len(observed) != len(frozen):
                raise ValueError("New download does not cover frozen paper history")
            fresh = simulation[simulation.date > old_asof].copy()
            # If a daily invocation was missed, mark the holdings through the gap
            # without manufacturing signals supposedly generated on those days.
            fresh.loc[fresh.date < asof, "composite_score"] = np.nan
            fresh.loc[fresh.date < asof, "decision_allowed"] = False
            simulation = pd.concat([frozen, fresh], ignore_index=True)
        catalog = write_watchlist_catalog(
            run_dir,
            settings,
            simulation_start.date().isoformat(),
            (asof + pd.Timedelta(days=30)).date().isoformat(),
        )
        replay = _replay(
            simulation,
            config,
            run_id,
            catalog_path=catalog,
            risk_limits=settings["risk"],
            corporate_actions=actions,
            account_id=str(settings.get("account_id", "a-share-multifactor-account")),
            strategy_id=str(settings.get("strategy_id", "a-share-multifactor-qexec")),
        )
        results = replay_results(replay, config)
        returns = results.quantile_returns.iloc[:, 0]
        benchmark = frames["benchmark"].set_index("date").benchmark_return.reindex(returns.index)
        if benchmark.isna().any():
            raise ValueError("Benchmark does not cover every simulated NAV date")
        benchmark.iloc[0] = 0.0  # Virtual account starts at the first observed close.
        stats = results.stats.iloc[0].to_dict()
        card["validation"].update(
            {
                "simulation_start": str(simulation_start.date()),
                "net_performance": stats,
                "hs300_price_index": return_statistics(benchmark.iloc[1:], 252),
                "forward_observation_days": int(
                    (
                        returns.index
                        > pd.Timestamp(prior["first_observed_asof"] if prior else asof)
                    ).sum()
                ),
                "forward_signal_observations": (
                    prior.get("observations", 1) + int(asof > pd.Timestamp(prior["as_of"]))
                    if prior
                    else 1
                ),
            }
        )
        ic = analyze_factors(diagnostic_panel, config.factors, config.forward_return_col)
        decay = analyze_ic_decay(diagnostic_panel, config.factors, [1, 5, 20])
        snapshots = {name: f"sha256:{entry['sha256']}" for name, entry in manifest["files"].items()}
        _write_certified_v2(
            run_dir,
            simulation,
            config,
            snapshots,
            replay=replay,
            catalog_path=catalog,
            research_metrics={
                "ic_summary": json.loads(ic.to_json(orient="records")),
                "ic_decay": json.loads(decay.to_json(orient="records")),
                "research_validation": clean_json(card["validation"]),
                "portfolio_risk_checks": clean_json(list(replay.risk_checks)),
            },
            internal_dependencies=identity["internal_dependencies"],
        )
        standard = run_dir / "standard" / "v2" / "run_manifest.json"
        card["evidence"].update(
            {
                "standard_manifest": str(standard),
                "standard_manifest_sha256": sha256(standard),
                "scored_panel": "standard/v2/config.json dataset lineage",
                "code_version": identity["code_version"],
            }
        )
        results.quantile_returns.to_csv(run_dir / "net_returns.csv")
        results.stats.to_csv(run_dir / "backtest_stats.csv", index=False)
        simulation.to_parquet(run_dir / "scored_panel.parquet", index=False)
        nav, current, targets, trades, costs = _paper_proposal(
            replay, simulation, config, catalog, asof
        )
        card["current_positions"] = current
        card["risk"] = {
            **settings["risk"],
            "portfolio_checks": clean_json(list(replay.risk_checks)),
            "nav": nav,
            "currency": "CNY",
            "account_type": "virtual, no user brokerage holdings",
            "account_id": replay.account_id,
            "strategy_id": replay.strategy_id,
            "rebalance_frequency": settings["frequency"],
            "allocation": {
                key: value
                for key, value in asdict(config.costs).items()
                if key
                in {
                    "commission",
                    "min_commission",
                    "stamp_tax",
                    "slippage",
                    "lot_size",
                    "initial_capital",
                    "max_holdings",
                    "participation_rate",
                    "cash_buffer",
                    "max_position_weight",
                }
            },
            "limits_of_daily_bars": "queue position and intraday tradability not verified",
        }
        reasons = []
        if now >= valid_until:
            reasons.append("decision_expired_refresh_required")
        if asof not in schedule and not trades:
            reasons.append("not_a_scheduled_rebalance_session")
        if not targets:
            reasons.append("no_valid_or_affordable_targets")
        if abs(stats["max_drawdown"]) > settings["risk"]["max_drawdown"]:
            reasons.append("simulation_drawdown_limit")
        if any(row["weight"] > settings["risk"]["max_single_weight"] for row in targets):
            reasons.append("target_concentration_limit")
        critical_rules = sorted(
            {
                alert["rule_id"]
                for check in replay.risk_checks
                if check.get("has_critical")
                for alert in check.get("alerts", [])
            }
        )
        if critical_rules:
            reasons.append("portfolio_risk_limit:" + ",".join(critical_rules))
        if manifest["origin"] != "live_public_api":
            reasons.append("non_live_input_snapshot")
        card["status"] = "observe" if reasons else "paper_ready"
        card["reasons"] = reasons or [
            "fresh_prices_validated; next-session paper trial only",
            "historical returns do not establish investable alpha",
        ]
        if not reasons:
            card.update(targets=targets, proposed_trades=trades, estimated_cost=costs)
        if manifest["origin"] == "live_public_api" and now < valid_until:
            state_temp = state_path.with_suffix(".tmp")
            _save_json(
                state_temp,
                {
                    "start": str(simulation_start.date()),
                    "as_of": card["as_of"],
                    "first_observed_asof": prior["first_observed_asof"] if prior else card["as_of"],
                    "config_sha256": config_digest,
                    "execution_identity": identity,
                    "scored_panel": str(run_dir / "scored_panel.parquet"),
                    "scored_panel_sha256": sha256(run_dir / "scored_panel.parquet"),
                    "observations": card["validation"]["forward_signal_observations"],
                    "applied_actions": action_identity,
                },
            )
            state_temp.replace(state_path)
    except (
        ValidationError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        OSError,
        yaml.YAMLError,
        subprocess.TimeoutExpired,
    ) as exc:
        card["status"] = "blocked"
        card["targets"] = []
        card["proposed_trades"] = []
        card["reasons"] = [f"{type(exc).__name__}: {exc}"]
    if attempt is None:
        # Even malformed configurations and version failures remain in the search history.
        parameters = {"run_id": run_id}
        registry.register(
            run_id,
            {
                "hypothesis": "invalid/unstarted decision attempt",
                "parameters": [parameters],
                "code_identity": card["evidence"].get("code_version", "unverified"),
                "selection_rule": "retain failed attempts",
            },
        )
        attempt = registry.start(run_id, parameters)
    registry.finish(
        attempt,
        "failed" if card["status"] == "blocked" else "completed",
        {
            "run_id": run_id,
            "status": card["status"],
            "reasons": card["reasons"],
            "run_path": str(run_dir),
        },
    )
    write_decision(card, run_dir)
    # The pointer is updated even after failure: a stale previous BUY card must
    # never remain the apparent latest output when today's refresh fails.
    pointer = output_root.resolve() / "latest.json"
    temporary = pointer.with_suffix(".tmp")
    _save_json(
        temporary,
        {"run_id": run_id, "status": card["status"], "decision": str(run_dir / "decision.json")},
    )
    temporary.replace(pointer)
    return run_dir


def run_decision(config_path: Path, output_root: Path, **kwargs) -> Path:
    """Serialize writers to one virtual account; never race its state/pointer."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ".decision.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("A decision run already owns this output account lock") from exc
    try:
        with handle:
            handle.write(
                "Single-writer account lock; remove only after confirming the process ended.\n"
            )
            handle.flush()
            return _run_decision(config_path, output_root, **kwargs)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description="Real-data A-share research and paper decision")
    parser.add_argument("--config", type=Path, default=Path("configs/decision_watchlist.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/decisions"))
    parser.add_argument(
        "--as-of", help="Optional historical cutoff YYYY-MM-DD; stale cards cannot act"
    )
    parser.add_argument(
        "--inputs", type=Path, help="Replay a hash-verified captured inputs directory"
    )
    args = parser.parse_args()
    result = run_decision(args.config, args.output, as_of=args.as_of, inputs=args.inputs)
    card = json.loads((result / "decision.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": card["status"],
                "decision": str(result / "decision.html"),
                "reasons": card["reasons"],
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(2 if card["status"] == "blocked" else 0)


if __name__ == "__main__":
    main()
