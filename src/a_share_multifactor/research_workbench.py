"""Equity/ETF research adapter: explicit recipes into one QExec ledger per candidate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from quant_data_kit import StatusEvent
from quant_data_kit.research_coverage import (
    asof_history,
    attach_history,
    load_history,
    preflight,
)
from quant_execution import resolve_a_share_replay_status
from quant_factors.expressions import (
    compute_research_factors,
    expression_requirements,
    validate_expressions,
)
from quant_factors.research import factor_report
from quant_lab.research import canonical, file_hash
from quant_portfolio import validate_research_allocation

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

_EXECUTION_FIELDS = {"mode", "universe_field", "status_fields", "max_retry_sessions"}
_STATUS_FIELDS = {"listed", "delisted", "tradable", "limit_up", "limit_down"}


def validate_research_execution(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the closed historical-execution recipe fragment."""

    if not isinstance(value, Mapping):
        raise TypeError("execution must be a mapping")
    unknown = set(value) - _EXECUTION_FIELDS
    if unknown:
        raise ValueError(f"execution contains unknown fields: {sorted(unknown)}")
    mode = value.get("mode")
    if mode not in {"fixed", "dynamic"}:
        raise ValueError("execution.mode must be fixed or dynamic")
    universe_field = value.get("universe_field")
    if mode == "dynamic":
        if not isinstance(universe_field, str) or not universe_field.strip():
            raise ValueError("dynamic execution requires a non-empty universe_field")
        universe_field = universe_field.strip()
    elif universe_field is not None:
        raise ValueError("fixed execution cannot declare universe_field")
    status_fields = value.get("status_fields")
    if not isinstance(status_fields, Mapping):
        raise TypeError("execution.status_fields must be a mapping")
    if set(status_fields) != _STATUS_FIELDS:
        raise ValueError(
            "execution.status_fields must contain exactly listed, delisted, tradable, "
            "limit_up and limit_down"
        )
    normalized_status = {}
    for name in sorted(_STATUS_FIELDS):
        field = status_fields[name]
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"execution.status_fields.{name} must be a non-empty string")
        normalized_status[name] = field.strip()
    if len(set(normalized_status.values())) != len(normalized_status):
        raise ValueError("execution.status_fields must map to distinct history fields")
    retries = value.get("max_retry_sessions", 5)
    if isinstance(retries, bool) or not isinstance(retries, int):
        raise TypeError("execution.max_retry_sessions must be an integer")
    if not 0 <= retries <= 252:
        raise ValueError("execution.max_retry_sessions must be in [0, 252]")
    return {
        "mode": mode,
        "universe_field": universe_field,
        "status_fields": normalized_status,
        "max_retry_sessions": retries,
    }


def _as_bool(rows: pd.DataFrame, symbols: list[str], *, field: str, cutoff: pd.Timestamp) -> dict:
    indexed = rows.set_index("symbol")
    missing = sorted(set(symbols) - set(indexed.index))
    if missing:
        raise ValueError(
            f"Historical field {field!r} is missing at {cutoff.isoformat()}: {missing}"
        )
    values = indexed.loc[symbols, "value"].astype(str)
    invalid = sorted(values[~values.isin(["true", "false"])].index.astype(str))
    if invalid:
        raise ValueError(
            f"Historical field {field!r} is not boolean at {cutoff.isoformat()}: {invalid}"
        )
    return {symbol: indexed.loc[symbol, "value"] == "true" for symbol in symbols}


def _execution_history(
    history: pd.DataFrame,
    execution: dict[str, object],
    *,
    symbols: list[str],
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, tuple[StatusEvent, ...]]:
    rows: list[dict] = []
    events: list[StatusEvent] = []
    status_fields = execution["status_fields"]
    for day in sessions:
        local_day = pd.Timestamp(day).tz_localize("Asia/Shanghai")
        market_open = local_day + pd.Timedelta(hours=9, minutes=25)
        decision_close = local_day + pd.Timedelta(hours=15)
        status_values: dict[str, dict[str, bool]] = {}
        for semantic, field in status_fields.items():
            at_open = asof_history(history, as_of=market_open, domain="status", field=field)
            at_close = asof_history(history, as_of=decision_close, domain="status", field=field)
            open_values = _as_bool(at_open, symbols, field=field, cutoff=market_open)
            close_values = _as_bool(at_close, symbols, field=field, cutoff=decision_close)
            changed = sorted(
                symbol for symbol in symbols if open_values[symbol] != close_values[symbol]
            )
            if changed:
                raise ValueError(
                    "Daily-bar execution cannot causally replay status first available after "
                    f"the opening match: field={field!r}, session={day.date()}, symbols={changed}"
                )
            status_values[semantic] = open_values
        if execution["mode"] == "dynamic":
            universe_field = str(execution["universe_field"])
            universe_rows = asof_history(
                history,
                as_of=decision_close,
                domain="universe",
                field=universe_field,
            )
            universe = _as_bool(
                universe_rows,
                symbols,
                field=universe_field,
                cutoff=decision_close,
            )
        else:
            universe = dict.fromkeys(symbols, True)
        event_at = market_open.tz_convert("UTC").to_pydatetime()
        for sequence, symbol in enumerate(symbols):
            flags = {name: status_values[name][symbol] for name in _STATUS_FIELDS}
            status = resolve_a_share_replay_status(**flags)
            reason = "delisted" if flags["delisted"] else "historical-trading-status"
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "symbol": symbol,
                    "in_universe": universe[symbol],
                    "status": status,
                    **flags,
                }
            )
            events.append(
                StatusEvent(
                    event_id=f"research-status:{day.date()}:{symbol}",
                    instrument_id=symbol,
                    event_time=event_at,
                    received_at=event_at,
                    available_at=event_at,
                    source="research-history",
                    trading_day=day.date(),
                    session_id=f"CN-A-SHARE:{day.date()}",
                    sequence=sequence,
                    status=status,
                    reason=reason,
                )
            )
    return pd.DataFrame(rows), tuple(events)


def _dynamic_market_issues(
    research: pd.DataFrame,
    status: pd.DataFrame,
    requirements: dict[str, dict],
    *,
    start: str,
    end: str,
) -> tuple[list[dict], list[dict]]:
    bars = research.copy()
    bars["date"] = pd.to_datetime(bars.date)
    keys = set(zip(bars.symbol.astype(str), bars.date))
    issues: list[dict] = []
    coverage: list[dict] = []
    warmup = max((item["warmup_bars"] for item in requirements.values()), default=1)
    required_columns = sorted(
        {column for requirement in requirements.values() for column in requirement["columns"]}
    )
    interval = status[status.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    for symbol, states in interval.groupby("symbol", sort=True):
        # Listed names keep requiring prices after a universe exit because an
        # existing holding may still need an evidenced sale.
        expected = states[states.listed & states.status.ne("suspended")]
        forbidden = states[~states.listed]
        missing_sessions = [
            day.date().isoformat()
            for day in expected.date
            if (str(symbol), pd.Timestamp(day)) not in keys
        ]
        forbidden_sessions = [
            day.date().isoformat()
            for day in forbidden.date
            if (str(symbol), pd.Timestamp(day)) in keys
        ]
        group = bars[bars.symbol.astype(str).eq(str(symbol))].sort_values("date")
        if expected.empty:
            detail = {
                "symbol": str(symbol),
                "missing_tradable_sessions": [],
                "bars_while_unlisted": forbidden_sessions,
                "warmup_available": 0,
                "warmup_required": 0,
                "missing_columns": [],
                "missing_values": {},
            }
            coverage.append(detail)
            if forbidden_sessions:
                issues.append({"code": "DYNAMIC_PRICE_OR_FEATURE_COVERAGE", "detail": detail})
            continue
        prior = group[group.date < pd.Timestamp(start)].tail(warmup)
        missing_columns = [column for column in required_columns if column not in group]
        relevant = group[group.date <= pd.Timestamp(end)].tail(len(expected) + warmup)
        missing_values = {
            column: int((~np.isfinite(pd.to_numeric(relevant[column], errors="coerce"))).sum())
            for column in required_columns
            if column in relevant
        }
        detail = {
            "symbol": str(symbol),
            "missing_tradable_sessions": missing_sessions,
            "bars_while_unlisted": forbidden_sessions,
            "warmup_available": len(prior),
            "warmup_required": warmup,
            "missing_columns": missing_columns,
            "missing_values": missing_values,
        }
        coverage.append(detail)
        if (
            missing_sessions
            or forbidden_sessions
            or len(prior) < warmup
            or missing_columns
            or any(missing_values.values())
        ):
            issues.append({"code": "DYNAMIC_PRICE_OR_FEATURE_COVERAGE", "detail": detail})
    return issues, coverage


def _instrument_master_issues(
    catalog: pd.DataFrame,
    status: pd.DataFrame | None,
    sessions: pd.DatetimeIndex,
) -> tuple[list[dict], list[dict]]:
    """Check master availability only where a symbol can enter an order target."""

    issues: list[dict] = []
    coverage: list[dict] = []
    for row in catalog.sort_values("symbol").itertuples(index=False):
        symbol = str(row.symbol)
        if status is None:
            relevant = sessions
        else:
            states = status[status.symbol.astype(str).eq(symbol)]
            mask = states.listed.astype(bool)
            relevant = pd.DatetimeIndex(states.loc[mask, "date"])
        available_at = pd.Timestamp(row.available_at)
        available_at = (
            available_at.tz_localize("UTC")
            if available_at.tzinfo is None
            else available_at.tz_convert("UTC")
        )
        effective_from = pd.Timestamp(row.effective_from)
        effective_from = (
            effective_from.tz_localize("UTC")
            if effective_from.tzinfo is None
            else effective_from.tz_convert("UTC")
        )
        effective_to = pd.Timestamp(row.effective_to)
        effective_to = (
            effective_to.tz_localize("UTC")
            if effective_to.tzinfo is None
            else effective_to.tz_convert("UTC")
        )
        uncovered = []
        for day in relevant:
            market_open = (
                pd.Timestamp(day).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=25)
            ).tz_convert("UTC")
            if not (available_at <= market_open and effective_from <= market_open < effective_to):
                uncovered.append(pd.Timestamp(day).date().isoformat())
        detail = {
            "symbol": symbol,
            "required_sessions": len(relevant),
            "available_at": available_at.isoformat(),
            "effective_from": effective_from.isoformat(),
            "effective_to": effective_to.isoformat(),
            "uncovered_count": len(uncovered),
            "first_uncovered_session": uncovered[0] if uncovered else None,
            "last_uncovered_session": uncovered[-1] if uncovered else None,
        }
        coverage.append(detail)
        if uncovered:
            issues.append({"code": "INSTRUMENT_MASTER_PIT_COVERAGE", "detail": detail})
    return issues, coverage


def _default_candidate(recipe: dict) -> dict:
    return {
        "candidate_id": "preflight",
        "factors": recipe["factors"],
        "factor_expressions": recipe.get("factor_expressions", {}),
        "strategy": recipe["strategy"],
        "allocation": recipe.get("allocation"),
        "signal_delay": 0,
        "cost_multiplier": 1,
    }


def _prepare_research_inputs(
    recipe: dict,
    candidate: dict,
    *,
    frames: dict | None = None,
) -> dict:
    if recipe.get("backend") != "equity":
        raise ValueError("Equity adapter received a different backend")
    strategy = candidate["strategy"]
    if strategy["family"] not in {"rank", "etf_trend", "buy_hold"}:
        raise ValueError("Unsupported equity strategy family")
    if frames is None:
        _, frames = load_inputs(Path(recipe["inputs"]["bundle"]))
    if "catalog" not in recipe["inputs"]:
        raise ValueError("Research requires an explicit versioned instrument catalog")
    catalog = Path(recipe["inputs"]["catalog"])
    catalog_frame = load_fixture_catalog(catalog)
    symbols = sorted(catalog_frame.symbol.astype(str).unique())
    start, end = recipe["interval"]["start"], recipe["interval"]["end"]
    raw = frames["raw"].copy()
    raw["symbol"] = raw.symbol.astype(str)
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw[raw.date <= pd.Timestamp(end)].sort_values(["symbol", "date"])
    adjusted = frames["adjusted"].copy()
    adjusted["symbol"] = adjusted.symbol.astype(str)
    adjusted["date"] = pd.to_datetime(adjusted.date)
    adjusted = adjusted[adjusted.date <= pd.Timestamp(end)].sort_values(["symbol", "date"])
    if set(raw.symbol) - set(symbols):
        raise ValueError("Market input contains symbols absent from the instrument catalog")
    prices = adjusted[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(prices).all().all() or not prices.gt(0).all().all():
        raise ValueError("Adjusted prices must be finite and positive")
    if set(zip(raw.symbol, raw.date)) != set(zip(adjusted.symbol, adjusted.date)):
        raise ValueError("Raw/adjusted bar keys differ")

    execution = (
        validate_research_execution(recipe["execution"])
        if recipe.get("execution") is not None
        else None
    )
    allocation_value = candidate.get("allocation", recipe.get("allocation"))
    allocation = (
        validate_research_allocation(allocation_value) if allocation_value is not None else None
    )
    history = None
    if "history" in recipe["inputs"]:
        _, history = load_history(Path(recipe["inputs"]["history"]))
    if execution is not None and history is None:
        raise ValueError("Historical execution requires an imported point-in-time history")
    required_history = recipe.get("required_history", {})
    if execution is not None:
        for field in execution["status_fields"].values():
            if required_history.get(field) != "status":
                raise ValueError(
                    f"Execution status field {field!r} must map to required_history.status"
                )
        universe_field = execution["universe_field"]
        if universe_field is not None and required_history.get(universe_field) != "universe":
            raise ValueError(
                f"Dynamic universe field {universe_field!r} must map to required_history.universe"
            )

    calendar = pd.DatetimeIndex(pd.to_datetime(frames["calendar"].date)).sort_values()
    sessions = calendar[
        (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
    ].unique()
    if (
        len(sessions) == 0
        or sessions[0] != pd.Timestamp(start)
        or sessions[-1] != pd.Timestamp(end)
    ):
        raise ValueError("Evaluation endpoints must be explicit observed trading sessions")
    if execution is None and history is not None:
        for day in sessions:
            cutoff = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
            for field, domain in required_history.items():
                if domain not in {"status", "universe"}:
                    continue
                rows = asof_history(history, as_of=cutoff, domain=domain, field=field)
                relevant = rows[rows.symbol.isin(symbols)]
                if relevant.value.eq("false").any():
                    raise ValueError(
                        "Restricted historical status or changing universe requires an explicit "
                        "execution recipe"
                    )
    status_panel = None
    status_events: tuple[StatusEvent, ...] = ()
    if execution is not None:
        status_panel, status_events = _execution_history(
            history,
            execution,
            symbols=symbols,
            sessions=sessions,
        )

    if execution is None:
        validation_frames = {**frames, "raw": raw}
        validate_inputs(validation_frames, symbols, pd.Timestamp(end))
    else:
        if raw.empty or raw.duplicated(["symbol", "date"]).any():
            raise ValueError("Dynamic market input is empty or has duplicate symbol/date bars")
        if not raw.adjustment.eq("none").all() or not raw.volume_unit.eq("share").all():
            raise ValueError("Simulation requires raw prices and volume in shares")
        numbers = raw[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numbers).all().all() or (numbers < 0).any().any():
            raise ValueError("Invalid dynamic-universe OHLCV values")

    research = raw.drop(columns=["open", "high", "low", "close"]).merge(
        adjusted[["symbol", "date", "open", "high", "low", "close"]],
        on=["symbol", "date"],
        validate="one_to_one",
    )
    if history is not None:
        fields = {
            key: domain
            for key, domain in required_history.items()
            if domain in {"fundamentals", "classification"}
        }
        if fields:
            research = attach_history(research, history, fields)
    names = list(candidate["factors"])
    expressions = validate_expressions(
        candidate.get("factor_expressions", recipe.get("factor_expressions", {}))
    )
    requirements = expression_requirements(names, expressions)
    fundamental_columns = {
        column
        for requirement in requirements.values()
        if requirement.get("pit_required")
        for column in requirement["columns"]
    }
    missing_mappings = sorted(
        column for column in fundamental_columns if required_history.get(column) != "fundamentals"
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
    if any(item.get("pit_required") for item in requirements.values()) and history is None:
        raise ValueError("Fundamental factors require an imported publication-time history")

    check = preflight(
        research,
        frames["calendar"],
        symbols=symbols,
        start=start,
        end=end,
        requirements=requirements if execution is None else {},
        history=history,
        required_history=required_history,
    )
    issues = list(check["issues"])
    coverage = list(check["coverage"])
    if execution is not None:
        issues = [item for item in issues if item["code"] != "PRICE_OR_FEATURE_COVERAGE"]
        dynamic_issues, coverage = _dynamic_market_issues(
            research,
            status_panel,
            requirements,
            start=start,
            end=end,
        )
        issues.extend(dynamic_issues)
    master_issues, master_coverage = _instrument_master_issues(
        catalog_frame,
        status_panel,
        sessions,
    )
    issues.extend(master_issues)
    report = {
        **check,
        "schema_version": "asm.research-preflight/v1",
        "passed": not issues,
        "issues": issues,
        "coverage": coverage,
        "requirements": requirements,
        "allocation": allocation,
        "execution": execution,
        "instrument_master": master_coverage,
    }
    return {
        "frames": frames,
        "raw": raw,
        "adjusted": adjusted,
        "research": research,
        "history": history,
        "names": names,
        "expressions": expressions,
        "requirements": requirements,
        "symbols": symbols,
        "catalog": catalog,
        "catalog_frame": catalog_frame,
        "allocation": allocation,
        "execution": execution,
        "status_panel": status_panel,
        "status_events": status_events,
        "evaluation_sessions": sessions,
        "report": report,
    }


def preflight_recipe(recipe: dict, candidate: dict | None = None) -> dict:
    """Read and validate one recipe without replaying or writing execution artifacts."""

    try:
        prepared = _prepare_research_inputs(recipe, candidate or _default_candidate(recipe))
        return prepared["report"]
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": "asm.research-preflight/v1",
            "passed": False,
            "issues": [{"code": "RECIPE_OR_INPUT", "detail": str(exc)}],
            "requirements": {},
            "coverage": [],
        }


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
        cache_key, frames = self._load(recipe)
        prepared = _prepare_research_inputs(recipe, candidate, frames=frames)
        check = prepared["report"]
        (output / "preflight.json").write_text(canonical(check), encoding="utf-8")
        if not check["passed"]:
            raise ValueError("Data preflight failed; inspect preflight.json")
        strategy = candidate["strategy"]
        start, end = recipe["interval"]["start"], recipe["interval"]["end"]
        raw = prepared["raw"]
        adjusted = prepared["adjusted"]
        research = prepared["research"]
        history = prepared["history"]
        names = prepared["names"]
        expressions = prepared["expressions"]
        symbols = prepared["symbols"]
        allocation = prepared["allocation"]
        execution = prepared["execution"]
        status_panel = prepared["status_panel"]
        # Feature values use only past prices, cached independently of costs/allocation.
        features_key = canonical(
            {
                "input": cache_key,
                "factors": names,
                "expressions": expressions,
                "end": end,
                "history": file_hash(Path(recipe["inputs"]["history"]) / "manifest.json")
                if history is not None
                else None,
            }
        )
        if features_key not in self._features:
            self._features[features_key] = compute_research_factors(
                research,
                names,
                expressions,
            )
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
        if status_panel is not None:
            scored = scored.merge(
                status_panel,
                on=["symbol", "date"],
                validate="one_to_one",
            )
            scored["eligible"] &= scored["listed"].astype(bool)
            if execution["mode"] == "dynamic":
                scored["eligible"] &= scored["in_universe"].astype(bool)
        if scored.empty or (
            execution is None
            and (scored.date.min() != pd.Timestamp(start) or scored.date.max() != pd.Timestamp(end))
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
        catalog = prepared["catalog"]
        catalog_frame = prepared["catalog_frame"]
        if (
            strategy["family"] == "etf_trend"
            and not catalog_frame.product_type.str.lower().eq("etf").all()
        ):
            raise ValueError("ETF template requires ETF instrument specifications")
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
        allocation_schedule = None
        if allocation is None and strategy["family"] == "buy_hold":
            first = scored[scored.date == scored.date.min()].assign(composite_score=1.0)
            schedule = _target_schedule(first, cfg, catalog_path=catalog)
        elif allocation is None:
            schedule = {}
            for day in rebalance_dates(scored.date, strategy["frequency"]):
                current = scored[scored.date == day].copy()
                if not np.isfinite(current.composite_score).all():
                    raise ValueError(f"Missing signal in evaluation interval: {day.date()}")
                selected = current[current.eligible]
                targets = (
                    _target_schedule(selected, cfg, catalog_path=catalog) if len(selected) else {}
                )
                schedule[day.date()] = targets.get(day.date(), {})
        else:
            schedule = {}
            allocation_schedule = {}
            closes = (
                research.pivot(index="date", columns="symbol", values="close")
                .sort_index()
                .pct_change(fill_method=None)
            )
            allocation_days = (
                [pd.Timestamp(scored.date.min())]
                if strategy["family"] == "buy_hold"
                else list(rebalance_dates(scored.date, strategy["frequency"]))
            )
            for day in allocation_days:
                current = scored[scored.date == day].copy()
                eligible = current[current.eligible]
                if (
                    strategy["family"] != "buy_hold"
                    and not np.isfinite(eligible.composite_score).all()
                ):
                    raise ValueError(f"Missing signal in evaluation interval: {day.date()}")
                selected = (
                    eligible.sort_values(["composite_score", "symbol"], ascending=[False, True])
                    .head(cfg.costs.max_holdings)
                    .copy()
                )
                if strategy["family"] == "buy_hold":
                    selected["composite_score"] = 1.0
                day_invested = min(invested, len(selected) * cfg.costs.max_position_weight)
                allocation_schedule[day.date()] = {
                    "scores": selected.set_index("symbol").composite_score.astype(float),
                    "trailing_returns": closes.loc[closes.index <= day],
                    "allocation": allocation,
                    "invested_limit": day_invested,
                    "max_weight": cfg.costs.max_position_weight,
                    "linear_costs": pd.Series(
                        cfg.costs.commission + cfg.costs.stamp_tax + cfg.costs.slippage,
                        index=selected.symbol.astype(str),
                        dtype=float,
                    ),
                }
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
            allocation_schedule=allocation_schedule,
            status_events=prepared["status_events"],
            execution_policy=execution,
            risk_limits=recipe.get("risk", {}),
            strategy_id="research-" + candidate["candidate_id"],
        )
        result = replay_results(replay, cfg)
        evaluation_sessions = pd.DatetimeIndex(prepared["evaluation_sessions"])
        returns = result.quantile_returns.iloc[:, 0].reindex(evaluation_sessions)
        if returns.isna().any():
            missing = [day.date().isoformat() for day in returns.index[returns.isna()]]
            raise ValueError(f"Certified replay is missing evaluation sessions: {missing}")
        returns.to_csv(output / "returns.csv", header=["net_return"])
        statistics = return_statistics(returns.iloc[1:], 252)
        factors = factor_report(
            research,
            names,
            cutoff=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
            start=start,
            end=end,
            expressions=expressions,
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
        rejected = replay.frames["order_events"]
        rejected = rejected[rejected.to_status.isin(["rejected", "expired", "cancelled"])]
        execution_diagnostics = {
            "schema_version": "asm.research-execution-diagnostics/v1",
            "allocation_decisions": list(replay.allocation_decisions),
            "retry_diagnostics": list(replay.retry_diagnostics),
            "unfilled_orders": [
                {
                    "event_time": pd.Timestamp(row.event_time).isoformat(),
                    "order_id": row.order_id,
                    "to_status": row.to_status,
                    "reason": row.reason,
                }
                for row in rejected.itertuples()
            ],
        }
        (output / "execution_diagnostics.json").write_text(
            canonical(execution_diagnostics),
            encoding="utf-8",
        )
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
                "allocation": allocation,
                "execution": execution,
            },
            "scope": "synthetic-software-demonstration"
            if raw.source.astype(str).str.contains("synthetic").any()
            else (
                "retrospective-dynamic-universe-qexec-research"
                if execution is not None and execution["mode"] == "dynamic"
                else "retrospective-fixed-specification-qexec-research"
            ),
            "limitations": [
                "not independent holdout performance",
                "current adjusted-price vintage",
                "daily-bar execution; no order-book queue",
                (
                    "initial-capital target sizing"
                    if allocation is None
                    else "current-ledger-NAV target sizing"
                ),
                "gross dividends; personal dividend tax excluded",
                "held delistings fail without a supported disposition fact",
            ],
        }
