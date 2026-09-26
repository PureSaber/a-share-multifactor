"""One configured QExec replay for account facts and user-facing result views."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from quant_data_kit import (
    AssetClass,
    BarEvent,
    FixedPoint,
    InstrumentSpec,
    StatusEvent,
    SymbolMapping,
)
from quant_execution import (
    DeterministicBroker,
    DeterministicRunEngine,
    ExactAccountLedger,
    OrderIntent,
    OrderType,
    Side,
    StrategyContext,
    TimeInForce,
)
from quant_lab import load_and_validate_standard_run, write_standard_run_v2
from quant_lab.contracts import RunManifest
from quant_risk_monitor import DecisionPortfolioLimits, check_decision_portfolio

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import AppConfig
from a_share_multifactor.execution_models import (
    ConfiguredAShareRiskGate,
    ConfiguredBarMatchingModel,
)
from a_share_multifactor.performance import return_statistics
from a_share_multifactor.quantile_backtest import BacktestResult, _assign_quantiles

_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "fixtures" / "a_share_instrument_catalog_v1.csv"
)
_STRATEGY_ID = "a-share-multifactor-qexec"
_ACCOUNT_ID = "a-share-multifactor-account"
_MONEY_SCALE = 8
_DEPENDENCIES = {
    "quant-data-kit": "v0.8.1",
    "quant-execution": "v0.5.1",
    "quant-lab": "v0.3.1",
    "quant-factors": "v0.3.0",
    "quant-portfolio": "v0.4.2",
    "quant-risk-monitor": "v0.4.0",
}
_V2_COLUMNS = {
    "returns": [
        "event_time",
        "strategy_id",
        "gross_return",
        "net_return",
        "nav_units",
        "nav_scale",
        "base_currency",
    ],
    "positions": [
        "event_time",
        "account_id",
        "strategy_id",
        "instrument_id",
        "quantity_units",
        "quantity_scale",
        "mark_price_units",
        "mark_price_scale",
        "market_value_units",
        "market_value_scale",
        "currency",
        "fx_rate_units",
        "fx_rate_scale",
        "fx_snapshot_id",
        "base_market_value_units",
        "base_market_value_scale",
    ],
    "portfolio_snapshots": [
        "event_time",
        "account_id",
        "base_currency",
        "nav_units",
        "nav_scale",
        "cash_value_units",
        "cash_value_scale",
        "market_value_units",
        "market_value_scale",
        "unrealized_pnl_units",
        "unrealized_pnl_scale",
        "realized_pnl_units",
        "realized_pnl_scale",
        "margin_used_units",
        "margin_used_scale",
    ],
    "exposures": [
        "event_time",
        "account_id",
        "strategy_id",
        "exposure_type",
        "name",
        "value",
        "unit",
    ],
    "orders": [
        "event_time",
        "order_id",
        "idempotency_key",
        "account_id",
        "strategy_id",
        "instrument_id",
        "side",
        "quantity_units",
        "quantity_scale",
        "order_type",
        "limit_price_units",
        "limit_price_scale",
        "stop_price_units",
        "stop_price_scale",
        "time_in_force",
        "reduce_only",
        "status",
        "filled_quantity_units",
        "filled_quantity_scale",
        "version",
    ],
    "order_events": [
        "event_time",
        "event_id",
        "order_id",
        "event_sequence",
        "from_status",
        "to_status",
        "fill_quantity_units",
        "fill_quantity_scale",
        "reason",
    ],
    "fills": [
        "event_time",
        "fill_id",
        "order_id",
        "account_id",
        "strategy_id",
        "instrument_id",
        "side",
        "quantity_units",
        "quantity_scale",
        "price_units",
        "price_scale",
        "currency",
        "liquidity_role",
        "venue_trade_id",
    ],
    "costs": [
        "event_time",
        "cost_id",
        "account_id",
        "strategy_id",
        "instrument_id",
        "fill_id",
        "cost_type",
        "amount_units",
        "amount_scale",
        "currency",
    ],
    "cash_ledger": [
        "event_time",
        "transaction_id",
        "idempotency_key",
        "event_type",
        "reference_id",
        "posting_index",
        "ledger_account",
        "account_id",
        "currency",
        "amount_units",
        "amount_scale",
        "instrument_id",
        "quantity_delta_units",
        "quantity_delta_scale",
    ],
    "margin": [
        "event_time",
        "account_id",
        "instrument_id",
        "initial_margin_units",
        "maintenance_margin_units",
        "margin_scale",
        "currency",
    ],
}


def _code_version(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        distribution = importlib.metadata.distribution("a-share-multifactor")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        revision = direct.get("vcs_info", {}).get("commit_id", "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise RuntimeError("installed strategy requires immutable VCS provenance")
        return revision
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    if status.strip():
        raise RuntimeError("certified runs require a clean Git worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _installed_internal_dependencies() -> dict[str, str]:
    revisions = {}
    for name in _DEPENDENCIES:
        package = importlib.import_module(name.replace("-", "_"))
        root = Path(package.__file__).resolve().parents[2]
        if (root / ".git").exists():
            revisions[name] = _code_version(root)
        else:
            distribution = importlib.metadata.distribution(name)
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            revisions[name] = (
                direct.get("vcs_info", {}).get("commit_id") or f"v{distribution.version}"
            )
    return revisions


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return {"type": "datetime", "value": stamp.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, (np.floating, float)):
        return {"type": "float", "value": float(value).hex()}
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return {"type": "string", "value": str(value)}


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash frame values independently of row, column, index, and dtype ordering."""
    columns = sorted(str(column) for column in frame.columns)
    if len(columns) != len(set(columns)):
        raise ValueError("canonical frame hashing requires unique column names")
    canonical_rows = []
    for row in frame.rename(columns=str)[columns].itertuples(index=False, name=None):
        record = {column: _canonical_value(value) for column, value in zip(columns, row)}
        canonical_rows.append(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    payload = "[" + ",".join(sorted(canonical_rows)) + "]"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _fixed(value: object, scale: int) -> FixedPoint:
    amount = Decimal(str(value)).quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP)
    return FixedPoint(int(amount.scaleb(scale)), scale)


def _decimal(value: FixedPoint) -> Decimal:
    return Decimal(value.units).scaleb(-value.scale)


def load_fixture_catalog(path: Path = _CATALOG_PATH) -> pd.DataFrame:
    """Load the explicit, versioned fixture catalog; never infer by symbol pattern."""
    catalog = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "symbol",
        "asset_class",
        "product_type",
        "venue",
        "price_scale",
        "price_tick",
        "quantity_step",
        "lot_size",
        "commission_rate",
        "stamp_duty_rate",
        "effective_from",
        "effective_to",
        "available_at",
    }
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"fixture catalog is missing columns: {sorted(missing)}")
    if catalog["symbol"].duplicated().any():
        raise ValueError("fixture catalog contains duplicate symbols")
    return catalog


def build_instrument_master(
    panel: pd.DataFrame,
    *,
    catalog_path: Path = _CATALOG_PATH,
    symbols: list[str] | None = None,
) -> tuple[dict[str, InstrumentSpec], tuple[SymbolMapping, ...]]:
    """Build PIT InstrumentSpec and SymbolMapping objects from the catalog."""
    catalog = load_fixture_catalog(catalog_path).set_index("symbol")
    symbols = sorted(panel["symbol"].astype(str).unique()) if symbols is None else sorted(symbols)
    missing = sorted(set(symbols) - set(catalog.index))
    if missing:
        raise ValueError(
            f"certified replay requires explicit fixture catalog entries; missing={missing}"
        )
    specs: dict[str, InstrumentSpec] = {}
    mappings: list[SymbolMapping] = []
    for symbol in symbols:
        row = catalog.loc[symbol]
        effective_from = _utc(row["effective_from"]).to_pydatetime()
        effective_to = _utc(row["effective_to"]).to_pydatetime()
        symbol_dates = pd.to_datetime(panel.loc[panel["symbol"].astype(str).eq(symbol), "date"])
        if not symbol_dates.empty and (
            symbol_dates.min().date() < effective_from.date()
            or symbol_dates.max().date() >= effective_to.date()
        ):
            raise ValueError(f"panel dates for {symbol} fall outside fixture validity window")
        price_scale = int(row["price_scale"])
        specs[symbol] = InstrumentSpec(
            instrument_id=symbol,
            asset_class=AssetClass(row["asset_class"]),
            product_type=row["product_type"],
            venue=row["venue"],
            native_symbol=symbol,
            settlement_currency="CNY",
            price_tick=_fixed(row["price_tick"], price_scale),
            quantity_step=_fixed(row["quantity_step"], 0),
            contract_multiplier=_fixed("1", 0),
            calendar_id="CN-A-SHARE",
            effective_from=effective_from,
            effective_to=effective_to,
            available_at=_utc(row["available_at"]).to_pydatetime(),
            base_currency="CNY",
            quote_currency="CNY",
            metadata={
                "lot_size": row["lot_size"],
                "commission_rate": row["commission_rate"],
                "stamp_duty_rate": row["stamp_duty_rate"],
                "fee_fields_scope": row.get("fee_fields_scope", "catalog-declared"),
                "catalog_scope": (
                    "fixture-certified-not-listing-history"
                    if catalog_path == _CATALOG_PATH
                    else "declared-watchlist-rules-not-listing-history"
                ),
            },
        )
        mappings.append(
            SymbolMapping(
                source="fixture-certified"
                if catalog_path == _CATALOG_PATH
                else "declared-watchlist",
                provider_symbol=symbol,
                instrument_id=symbol,
                effective_from=effective_from,
                effective_to=effective_to,
                available_at=_utc(row["available_at"]).to_pydatetime(),
            )
        )
    return specs, tuple(mappings)


def _build_events(panel: pd.DataFrame, specs: dict[str, InstrumentSpec]) -> tuple[BarEvent, ...]:
    ordered = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    if ordered.duplicated(["date", "symbol"]).any():
        raise ValueError("certified replay requires one bar per symbol and trading day")
    events: list[BarEvent] = []
    for index, row in ordered.iterrows():
        symbol = str(row["symbol"])
        scale = specs[symbol].price_tick.scale
        day = pd.Timestamp(row["date"]).date()
        timestamp = _utc(day) + pd.Timedelta(hours=1, minutes=30)
        bar_end = _utc(day) + pd.Timedelta(hours=7, microseconds=int(index))
        volume = max(0, int(Decimal(str(row.get("volume", 0))).to_integral_value()))
        events.append(
            BarEvent(
                event_id=f"daily-bar:{day.isoformat()}:{symbol}",
                instrument_id=symbol,
                event_time=bar_end.to_pydatetime(),
                received_at=bar_end.to_pydatetime(),
                available_at=bar_end.to_pydatetime(),
                source=str(row.get("source", "fixture-certified")),
                trading_day=day,
                session_id=f"CN-A-SHARE:{day.isoformat()}",
                sequence=index,
                bar_start=timestamp.to_pydatetime(),
                bar_end=bar_end.to_pydatetime(),
                open_price=_fixed(row["open"], scale),
                high_price=_fixed(row["high"], scale),
                low_price=_fixed(row["low"], scale),
                close_price=_fixed(row["close"], scale),
                volume=FixedPoint(volume, 0),
                is_complete=True,
            )
        )
    return tuple(events)


def _target_schedule(
    panel: pd.DataFrame, config: AppConfig, *, catalog_path: Path = _CATALOG_PATH
) -> dict[date, dict[str, int]]:
    catalog = load_fixture_catalog(catalog_path).set_index("symbol")
    schedule: dict[date, dict[str, int]] = {}
    for rebalance_date in rebalance_dates(panel["date"], config.rebalance_freq):
        day = panel[pd.to_datetime(panel["date"]) == pd.Timestamp(rebalance_date)].copy()
        if day.empty:
            continue
        day["composite_score"] = pd.to_numeric(day["composite_score"], errors="coerce")
        day["quantile"] = _assign_quantiles(day["composite_score"], config.quantiles)
        selected = day[day["quantile"] == float(config.quantiles)]
        selected = selected.dropna(subset=["close", "composite_score"])
        if config.costs.max_holdings > 0:
            selected = selected.nlargest(config.costs.max_holdings, "composite_score")
        allocation = (
            Decimal(str(config.costs.initial_capital))
            * min(
                Decimal(str(1 - config.costs.cash_buffer)) / len(selected),
                Decimal(str(config.costs.max_position_weight)),
            )
            if len(selected)
            else Decimal(0)
        )
        target: dict[str, int] = {}
        for _, row in selected.sort_values("symbol").iterrows():
            symbol = str(row["symbol"])
            lot = int(catalog.loc[symbol, "lot_size"])
            budget = max(Decimal(0), allocation - Decimal(str(config.costs.min_commission)))
            unit_cost = Decimal(str(row["close"])) * Decimal(
                str((1 + config.costs.slippage) * (1 + config.costs.commission))
            )
            shares = int((budget / unit_cost) // lot) * lot
            if shares > 0:
                target[symbol] = shares
        if target:
            schedule[pd.Timestamp(rebalance_date).date()] = target
    return schedule


class _TargetWeightStrategy:
    """Emit only QExec OrderIntent objects; it never mutates positions."""

    sends_live_orders = False

    def __init__(
        self,
        schedule: dict[date, dict[str, int]],
        ledger=None,
        trigger_symbols=None,
        risk_limits=None,
        blocked_dates=None,
        initial_capital=0,
        allocation_schedule=None,
        catalog=None,
        costs=None,
        broker=None,
        execution_policy=None,
        risk_schedule=None,
        risk_gate=None,
    ) -> None:
        self.schedule = schedule
        self.ledger = ledger
        self.trigger_symbols = trigger_symbols or {}
        self.risk_limits = risk_limits or {}
        self.blocked_dates = blocked_dates or set()
        self.initial_capital = initial_capital
        self.allocation_schedule = allocation_schedule
        self.catalog = catalog
        self.costs = costs
        self.broker = broker
        self.execution_policy = execution_policy
        self.risk_schedule = risk_schedule or {}
        self.risk_gate = risk_gate
        self.reset()

    def reset(self) -> None:
        self._deferred_targets = {}
        self._target_active = False
        self._peak_nav = float(self.initial_capital)
        self._closing_prices = {}
        self._risk_halted = False
        self._risk_halt_reason = None
        self._risk_checks = []
        self._allocation_decisions = []
        self._retry_diagnostics = []
        self._attempts = {}
        self._target_revision = 0

    def capture_state(self):
        return {
            "deferred_targets": self._deferred_targets.copy(),
            "target_active": self._target_active,
            "peak_nav": self._peak_nav,
            "closing_prices": self._closing_prices.copy(),
            "risk_halted": self._risk_halted,
            "risk_halt_reason": self._risk_halt_reason,
            "risk_checks": list(self._risk_checks),
            "allocation_decisions": list(self._allocation_decisions),
            "retry_diagnostics": list(self._retry_diagnostics),
            "attempts": self._attempts.copy(),
            "target_revision": self._target_revision,
        }

    def restore_state(self, state):
        self._deferred_targets = state["deferred_targets"].copy()
        self._target_active = state["target_active"]
        self._peak_nav = state["peak_nav"]
        self._closing_prices = state["closing_prices"].copy()
        self._risk_halted = state["risk_halted"]
        self._risk_halt_reason = state["risk_halt_reason"]
        self._risk_checks = list(state["risk_checks"])
        self._allocation_decisions = list(state["allocation_decisions"])
        self._retry_diagnostics = list(state["retry_diagnostics"])
        self._attempts = state["attempts"].copy()
        self._target_revision = state["target_revision"]

    @property
    def risk_checks(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._risk_checks)

    @property
    def allocation_decisions(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._allocation_decisions)

    @property
    def retry_diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._retry_diagnostics)

    def _allocation_target(self, day: date, snapshot, nav: Decimal) -> dict[str, int]:
        from quant_portfolio import research_allocation_weights, validate_research_allocation

        plan = self.allocation_schedule[day]
        scores = plan["scores"]
        settings = validate_research_allocation(plan["allocation"])
        held = {
            str(symbol): _decimal(quantity)
            for symbol, quantity in snapshot.positions.items()
            if _decimal(quantity) != 0
        }
        missing_current_marks = sorted(set(held) - set(self._closing_prices))
        if missing_current_marks:
            raise ValueError(
                "Current-NAV allocation is missing a close for held positions: "
                f"{missing_current_marks}"
            )
        current_weights = pd.Series(
            {
                symbol: float(quantity * _decimal(self._closing_prices[symbol]) / nav)
                for symbol, quantity in held.items()
            },
            dtype=float,
        )
        if scores.empty or plan["invested_limit"] <= 0:
            if float(current_weights.abs().sum()) > float(settings["max_turnover"]) + 1e-10:
                raise ValueError("allocation.max_turnover is infeasible for liquidation")
            weights = pd.Series(dtype=float)
            target = {}
        else:
            weights = research_allocation_weights(
                scores,
                plan["trailing_returns"],
                current_weights,
                plan["linear_costs"],
                invested_limit=plan["invested_limit"],
                max_weight=plan["max_weight"],
                config=settings,
                **{
                    key: plan[key]
                    for key in ("factor_exposures", "factor_bounds", "covariance_override")
                    if key in plan
                },
            )
            target = {}
            catalog = self.catalog.set_index("symbol")
            for symbol, weight in weights.sort_index().items():
                if symbol not in self._closing_prices:
                    raise ValueError(f"Current-NAV allocation is missing a close for {symbol}")
                lot = int(catalog.loc[symbol, "lot_size"])
                budget = max(
                    Decimal(0),
                    nav * Decimal(str(weight)) - Decimal(str(self.costs.min_commission)),
                )
                unit_cost = _decimal(self._closing_prices[symbol]) * Decimal(
                    str((1 + self.costs.slippage) * (1 + self.costs.commission))
                )
                shares = int((budget / unit_cost) // lot) * lot
                if shares > 0:
                    target[str(symbol)] = shares
        self._allocation_decisions.append(
            {
                "session": day.isoformat(),
                "nav": float(nav),
                "mode": plan["allocation"]["mode"],
                "weights": {str(key): float(value) for key, value in weights.items()},
                "targets": target.copy(),
            }
        )
        return target

    def _portfolio_check(self, *, target, current, nav: Decimal, event, stage="target") -> bool:
        symbols = set(target) | set(current)
        missing_marks = sorted(symbol for symbol in symbols if symbol not in self._closing_prices)
        if missing_marks:
            payload = {
                "alerts": [
                    {
                        "rule_id": "portfolio.missing_mark",
                        "severity": "critical",
                        "message": "a current close is required for every portfolio position",
                        "details": {"symbols": missing_marks},
                    }
                ],
                "count": 1,
                "has_critical": True,
                "metrics": {},
            }
        else:
            target_weights = {
                symbol: Decimal(quantity) * _decimal(self._closing_prices[symbol]) / nav
                for symbol, quantity in target.items()
                if quantity != 0
            }
            current_weights = {
                symbol: Decimal(quantity) * _decimal(self._closing_prices[symbol]) / nav
                for symbol, quantity in current.items()
                if quantity > 0
            }
            turnover = sum(
                abs(
                    target_weights.get(symbol, Decimal(0)) - current_weights.get(symbol, Decimal(0))
                )
                for symbol in set(target_weights) | set(current_weights)
            )
            cost_per_turnover = self.risk_limits.get("estimated_cost_rate_per_turnover")
            result = check_decision_portfolio(
                target_weights=target_weights,
                current_weights=current_weights,
                classifications=self.risk_schedule.get(event.trading_day, {}).get(
                    "classifications", self.risk_limits.get("classifications", {})
                ),
                limits=DecisionPortfolioLimits.from_mapping(self.risk_limits),
                estimated_cost_rate=(
                    None
                    if cost_per_turnover is None
                    else turnover * Decimal(str(cost_per_turnover))
                ),
            )
            payload = result.to_dict()
            if self.risk_schedule:
                from a_share_multifactor.research_risk import check_model_target

                item = self.risk_schedule[event.trading_day]
                cash_policy_reason = None
                if not target_weights:
                    cash_policy_reason = (
                        "full_cash_exit"
                        if stage == "target" and current_weights
                        else "full_cash_state"
                    )
                report, alerts = check_model_target(
                    item,
                    target_weights,
                    cash_policy_reason=cash_policy_reason,
                )
                payload["factor_risk"] = report
                payload["alerts"].extend(alerts)
                payload["count"] = len(payload["alerts"])
                payload["has_critical"] = any(
                    alert.get("severity") == "critical" for alert in payload["alerts"]
                )
        self._risk_checks.append(
            {
                "session": event.trading_day.isoformat(),
                "checked_at": pd.Timestamp(event.available_at).isoformat(),
                "stage": stage,
                **payload,
            }
        )
        return not payload["has_critical"]

    def on_event(self, context: StrategyContext, event: BarEvent) -> tuple[OrderIntent, ...]:
        if self.ledger is None:
            return ()
        if isinstance(event, StatusEvent):
            if event.status.lower() == "closed":
                snapshot = self.ledger.snapshot(event.available_at)
                position = snapshot.positions.get(event.instrument_id)
                if position is not None and position.units:
                    raise ValueError(
                        "Held unlisted position requires explicit supported disposition evidence; "
                        "universe exit is not disposition evidence"
                    )
            return ()
        if not hasattr(event, "close_price"):
            return ()
        self._closing_prices[event.instrument_id] = event.close_price
        return self._after_close(context, event)

    def _after_close(self, context, event):
        # Wait until ALL symbols' bars and outstanding fills for this session
        # have reached the ledger. Per-symbol early snapshots double-counted
        # still-pending buys when daily rebalancing coincided with their fills.
        if event.instrument_id != self.trigger_symbols.get(event.trading_day):
            return ()
        snapshot = self.ledger.snapshot(event.available_at)
        nav_decimal = _decimal(snapshot.nav)
        nav = float(nav_decimal)
        self._peak_nav = max(self._peak_nav, nav)
        if nav <= 0:
            return ()
        current = {
            symbol: int(_decimal(quantity)) for symbol, quantity in snapshot.positions.items()
        }
        if self.risk_schedule or self.risk_limits:
            compliant = self._portfolio_check(
                target=current, current=current, nav=nav_decimal, event=event, stage="realized"
            )
            if (
                not compliant
                and any(current.values())
                and (
                    not self._risk_halted
                    or self.risk_limits.get("exposure_breach_action", "halt") == "liquidate"
                )
            ):
                self._risk_halted = True
                self._risk_halt_reason = "exposure_breach"
        if self._peak_nav and 1 - nav / self._peak_nav > self.risk_limits.get("max_drawdown", 1):
            if (
                not self._risk_halted
                or self.risk_limits.get("drawdown_action", "halt") == "liquidate"
            ):
                self._risk_halt_reason = "drawdown"
            self._risk_halted = True
        if self._risk_halted:
            self._deferred_targets = {}
            self._target_active = False
            if self.broker is not None:
                for order in tuple(self.broker.open_orders):
                    self.broker.cancel(
                        order.order_id,
                        idempotency_key=f"risk-halt:{order.order_id}",
                        created_at=event.available_at,
                    )
                    self.risk_gate.release_order(order)
            action_field = (
                "drawdown_action"
                if self._risk_halt_reason == "drawdown"
                else "exposure_breach_action"
            )
            action = self.risk_limits.get(action_field, "halt")
            self._risk_checks.append(
                {
                    "session": event.trading_day.isoformat(),
                    "checked_at": pd.Timestamp(event.available_at).isoformat(),
                    action_field: action,
                    "halt_reason": self._risk_halt_reason,
                    "drawdown": 1 - nav / self._peak_nav,
                    "has_critical": True,
                    "alerts": [
                        {
                            "rule_id": "portfolio." + self._risk_halt_reason,
                            "severity": "critical",
                            "message": "Risk kill switch is latched",
                        }
                    ],
                }
            )
            if action == "halt":
                return ()
            # A kill switch liquidates through the same broker and exact ledger;
            # ordinary turnover/cash/active-risk targets must not forbid exits.
            return tuple(
                OrderIntent(
                    idempotency_key=f"risk-halt:{event.trading_day}:{symbol}",
                    account_id=context.account_id,
                    strategy_id=context.strategy_id,
                    instrument_id=symbol,
                    side=Side.SELL,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.IOC,
                    reduce_only=True,
                    created_at=event.available_at,
                )
                for symbol, quantity in snapshot.positions.items()
                if quantity.units > 0
            )
        if event.trading_day in self.blocked_dates:
            return ()
        new_target = False
        if self.allocation_schedule is not None and event.trading_day in self.allocation_schedule:
            target = self._allocation_target(event.trading_day, snapshot, nav_decimal)
            new_target = True
        else:
            target = self.schedule.get(
                event.trading_day,
                (
                    self._deferred_targets
                    if self.execution_policy is None or self._target_active
                    else {}
                ),
            )
            new_target = event.trading_day in self.schedule
        if not target and not new_target and not self._target_active:
            return ()
        if not self._portfolio_check(target=target, current=current, nav=nav_decimal, event=event):
            self._deferred_targets = {}
            return ()
        if self.execution_policy is not None:
            if new_target:
                self._deferred_targets = dict(target)
                self._target_active = True
                self._target_revision += 1
                self._attempts = {}
            target = self._deferred_targets
            projected = current.copy()
            for order in self.broker.open_orders:
                remaining = order.intent.quantity.units - order.filled_quantity.units
                direction = 1 if order.intent.side is Side.BUY else -1
                projected[order.intent.instrument_id] = (
                    projected.get(order.intent.instrument_id, 0) + direction * remaining
                )
            delta = {
                symbol: target.get(symbol, 0) - projected.get(symbol, 0)
                for symbol in sorted(set(target) | set(projected))
            }
        else:
            delta = {
                symbol: target.get(symbol, 0) - current.get(symbol, 0)
                for symbol in sorted(set(target) | set(current))
            }
        sells = {symbol: quantity for symbol, quantity in delta.items() if quantity < 0}
        if self.execution_policy is None:
            self._deferred_targets = dict(target) if sells else {}
        elif sells:
            self._deferred_targets = dict(target)
        trades = sells or {symbol: quantity for symbol, quantity in delta.items() if quantity > 0}
        if self.execution_policy is not None:
            if not trades and not self.broker.open_orders:
                self._deferred_targets = {}
                self._target_active = False
                return ()
            admitted = {}
            max_attempts = 1 + int(self.execution_policy["max_retry_sessions"])
            for symbol, quantity in trades.items():
                side = "buy" if quantity > 0 else "sell"
                key = (self._target_revision, symbol, side)
                attempts = self._attempts.get(key, 0)
                if attempts >= max_attempts:
                    diagnostic = {
                        "target_revision": self._target_revision,
                        "session": event.trading_day.isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "attempts": attempts,
                        "reason": "max_retry_sessions_exhausted",
                    }
                    if not any(
                        item["target_revision"] == self._target_revision
                        and item["symbol"] == symbol
                        and item["side"] == side
                        for item in self._retry_diagnostics
                    ):
                        self._retry_diagnostics.append(diagnostic)
                    continue
                self._attempts[key] = attempts + 1
                admitted[symbol] = quantity
            trades = admitted
        return tuple(
            OrderIntent(
                idempotency_key=(
                    f"{context.strategy_id}:{event.trading_day}:{symbol}:"
                    f"{'buy' if quantity > 0 else 'sell'}"
                    + (
                        f":r{self._attempts.get((self._target_revision, symbol, 'buy' if quantity > 0 else 'sell'), 0)}"
                        if self.execution_policy is not None
                        else ""
                    )
                ),
                account_id=context.account_id,
                strategy_id=context.strategy_id,
                instrument_id=symbol,
                side=Side.BUY if quantity > 0 else Side.SELL,
                quantity=FixedPoint(abs(quantity), 0),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
                reduce_only=quantity < 0,
                created_at=event.available_at,
            )
            for symbol, quantity in trades.items()
        )


class _RecordingLedger(ExactAccountLedger):
    """Capture snapshots from the exact ledger instance used by QExec."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._recorded_snapshots: dict[datetime, Any] = {}
        super().__init__(*args, **kwargs)

    def reset(self, *, opened_at: datetime | None = None) -> None:
        super().reset(opened_at=opened_at)
        self._recorded_snapshots = {}

    def capture_state(self) -> dict[str, object]:
        state = super().capture_state()
        state["recorded_snapshots"] = self._recorded_snapshots.copy()
        return state

    def restore_state(self, state: dict[str, object]) -> None:
        base_state = dict(state)
        recorded_snapshots = base_state.pop("recorded_snapshots")
        super().restore_state(base_state)
        self._recorded_snapshots = recorded_snapshots.copy()

    def _record(self, event_time: datetime) -> None:
        snapshot = self.snapshot(event_time)
        self._recorded_snapshots[snapshot.event_time] = snapshot

    def observe_market(self, event: Any, **kwargs: Any) -> Any:
        result = super().observe_market(event, **kwargs)
        self._record(event.available_at)
        return result

    def apply(self, event: Any, **kwargs: Any) -> Any:
        result = super().apply(event, **kwargs)
        self._record(event.event_time)
        return result

    def apply_with_trading_day(self, event: Any, **kwargs: Any) -> Any:
        result = super().apply_with_trading_day(event, **kwargs)
        self._record(event.event_time)
        return result

    @property
    def recorded_snapshots(self) -> tuple[Any, ...]:
        return tuple(self._recorded_snapshots[key] for key in sorted(self._recorded_snapshots))


@dataclass(frozen=True)
class CertifiedReplay:
    result: Any
    events: tuple[BarEvent, ...]
    instruments: dict[str, InstrumentSpec]
    mappings: tuple[SymbolMapping, ...]
    ledger: _RecordingLedger
    frames: dict[str, pd.DataFrame]
    account_id: str
    strategy_id: str
    risk_checks: tuple[dict[str, Any], ...]
    allocation_decisions: tuple[dict[str, Any], ...] = ()
    retry_diagnostics: tuple[dict[str, Any], ...] = ()
    runtime_risk_events: tuple[str, ...] = ()


def _frame(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_V2_COLUMNS[name])


def _replay(
    scored_panel: pd.DataFrame,
    config: AppConfig,
    run_id: str,
    *,
    catalog_path: Path = _CATALOG_PATH,
    risk_limits: dict | None = None,
    risk_schedule: dict | None = None,
    corporate_actions: tuple = (),
    target_schedule: dict | None = None,
    allocation_schedule: dict | None = None,
    status_events: tuple = (),
    execution_policy: dict | None = None,
    account_id: str = _ACCOUNT_ID,
    strategy_id: str = _STRATEGY_ID,
) -> CertifiedReplay:
    if not account_id.strip() or not strategy_id.strip():
        raise ValueError("account_id and strategy_id must be non-empty")
    if config.costs.retail_mode:
        raise ValueError(
            "QExec replay does not support retail early-exit/min-holding rules; "
            "use the explicit daily/weekly decision profile (retail_mode=false)"
        )
    costs = config.costs
    if "adjustment" in scored_panel and not scored_panel["adjustment"].eq("none").all():
        raise ValueError("Execution requires unadjusted traded prices (adjustment=none)")
    if (
        min(costs.commission, costs.min_commission, costs.stamp_tax) < 0
        or not 0 <= costs.cash_buffer < 1
        or not 0 < costs.max_position_weight <= 1
        or costs.initial_capital <= 0
    ):
        raise ValueError("Invalid execution costs or allocation limits")
    replay_symbols = sorted(
        set(scored_panel["symbol"].astype(str)) | {event.instrument_id for event in status_events}
    )
    instruments, mappings = build_instrument_master(
        scored_panel,
        catalog_path=catalog_path,
        symbols=replay_symbols,
    )
    instruments = {
        symbol: replace(
            spec,
            metadata={
                **spec.metadata,
                "commission_rate": str(costs.commission),
                "stamp_duty_rate": str(
                    costs.stamp_tax if spec.asset_class is AssetClass.EQUITY else 0
                ),
                "min_commission": str(costs.min_commission),
                "execution_fee_source": "configured-strategy-cost-assumption",
            },
        )
        for symbol, spec in instruments.items()
    }
    bars = _build_events(scored_panel, instruments)
    events = tuple(
        sorted(
            (*bars, *status_events, *corporate_actions),
            key=lambda e: (e.available_at, e.instrument_id),
        )
    )
    ledger = _RecordingLedger(
        account_id=account_id,
        base_currency="CNY",
        instruments=instruments,
        initial_cash={"CNY": _fixed(config.costs.initial_capital, 2)},
        money_scale=_MONEY_SCALE,
    )
    broker = DeterministicBroker()
    from quant_risk_monitor.cross_asset import (
        CrossAssetRiskLimits,
        CrossAssetRiskPolicy,
        PITRiskInputs,
        PriceObservation,
    )

    limits = risk_limits or {}
    policy = CrossAssetRiskPolicy(
        instruments=instruments,
        limits=CrossAssetRiskLimits(
            max_gross_leverage=limits.get("max_gross_weight"),
            max_instrument_concentration=limits.get("max_single_weight"),
        ),
        inputs=PITRiskInputs(
            prices=tuple(
                PriceObservation(
                    instrument_id=event.instrument_id,
                    price=event.close_price,
                    observed_at=event.available_at,
                    available_at=event.available_at,
                )
                for event in bars
            )
        ),
    )
    risk_gate = ConfiguredAShareRiskGate(instruments=instruments, ledger=ledger, policies=(policy,))
    strategy = _TargetWeightStrategy(
        _target_schedule(scored_panel, config, catalog_path=catalog_path)
        if target_schedule is None
        else target_schedule,
        ledger=ledger,
        trigger_symbols={event.trading_day: event.instrument_id for event in bars},
        risk_limits=risk_limits,
        initial_capital=costs.initial_capital,
        allocation_schedule=allocation_schedule,
        catalog=load_fixture_catalog(catalog_path),
        costs=costs,
        broker=broker,
        execution_policy=execution_policy,
        risk_schedule=risk_schedule,
        risk_gate=risk_gate,
        blocked_dates=set(
            pd.to_datetime(
                scored_panel.loc[~scored_panel["decision_allowed"].astype(bool), "date"]
            ).dt.date
        )
        if "decision_allowed" in scored_panel
        else set(),
    )
    engine = DeterministicRunEngine(
        run_id=run_id,
        account_id=account_id,
        strategy_id=strategy_id,
        strategy=strategy,
        broker=broker,
        risk_gate=risk_gate,
        matching_model=ConfiguredBarMatchingModel(
            instruments, slippage=costs.slippage, participation_rate=costs.participation_rate
        ),
        ledger=ledger,
    )
    result = engine.replay(events, seed=0)
    artifacts = engine.artifacts
    if artifacts is None:
        raise RuntimeError("QExec replay did not produce artifacts")
    snapshots = ledger.recorded_snapshots
    mark_by_time: dict[datetime, dict[str, FixedPoint]] = {}
    current_marks: dict[str, FixedPoint] = {}
    for event in events:
        if isinstance(event, BarEvent):
            current_marks[event.instrument_id] = event.close_price
        elif getattr(event, "ratio", None) and event.instrument_id in current_marks:
            current_marks[event.instrument_id] = _fixed(
                _decimal(current_marks[event.instrument_id]) / _decimal(event.ratio), _MONEY_SCALE
            )
        mark_by_time[event.available_at] = current_marks.copy()
    snapshot_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    previous_nav = Decimal(str(config.costs.initial_capital))
    previous_gross_nav = previous_nav
    fee_by_time: dict[datetime, Decimal] = {}
    for fee in artifacts.fees:
        fee_by_time[fee.event_time] = fee_by_time.get(fee.event_time, Decimal(0)) + _decimal(
            fee.amount
        )
    cumulative_fees = Decimal(0)
    for snapshot in snapshots:
        event_time = snapshot.event_time
        cash_value = sum((_decimal(value) for value in snapshot.cash_balances.values()), Decimal(0))
        market_value = Decimal(0)
        unrealized = sum(
            (_decimal(value) for value in snapshot.unrealized_pnl.values()), Decimal(0)
        )
        realized = sum((_decimal(value) for value in snapshot.realized_pnl.values()), Decimal(0))
        for instrument_id, quantity in snapshot.positions.items():
            spec = instruments[instrument_id]
            mark = mark_by_time.get(event_time, {}).get(instrument_id)
            if mark is None:
                raise ValueError(f"Missing market mark for open position: {instrument_id}")
            notional = _decimal(quantity) * _decimal(mark) * _decimal(spec.contract_multiplier)
            market_value += notional
            base_value = _fixed(notional, _MONEY_SCALE)
            position_rows.append(
                {
                    "event_time": event_time,
                    "account_id": snapshot.account_id,
                    "strategy_id": strategy_id,
                    "instrument_id": instrument_id,
                    "quantity_units": quantity.units,
                    "quantity_scale": quantity.scale,
                    "mark_price_units": mark.units,
                    "mark_price_scale": mark.scale,
                    "market_value_units": base_value.units,
                    "market_value_scale": base_value.scale,
                    "currency": spec.settlement_currency,
                    "fx_rate_units": 1,
                    "fx_rate_scale": 0,
                    "fx_snapshot_id": "fx:CNY:1",
                    "base_market_value_units": base_value.units,
                    "base_market_value_scale": base_value.scale,
                }
            )
        nav = _decimal(snapshot.nav)
        net_return = float(nav / previous_nav - 1) if previous_nav else 0.0
        previous_nav = nav
        cumulative_fees += fee_by_time.get(event_time, Decimal(0))
        gross_nav = nav + cumulative_fees
        gross_return = float(gross_nav / previous_gross_nav - 1) if previous_gross_nav else 0.0
        previous_gross_nav = gross_nav
        snapshot_rows.append(
            {
                "event_time": event_time,
                "account_id": snapshot.account_id,
                "base_currency": snapshot.base_currency,
                "nav_units": snapshot.nav.units,
                "nav_scale": snapshot.nav.scale,
                "cash_value_units": _fixed(cash_value, _MONEY_SCALE).units,
                "cash_value_scale": _MONEY_SCALE,
                "market_value_units": _fixed(market_value, _MONEY_SCALE).units,
                "market_value_scale": _MONEY_SCALE,
                "unrealized_pnl_units": _fixed(unrealized, _MONEY_SCALE).units,
                "unrealized_pnl_scale": _MONEY_SCALE,
                "realized_pnl_units": _fixed(realized, _MONEY_SCALE).units,
                "realized_pnl_scale": _MONEY_SCALE,
                "margin_used_units": snapshot.initial_margin.units,
                "margin_used_scale": snapshot.initial_margin.scale,
            }
        )
        return_rows.append(
            {
                "event_time": event_time,
                "strategy_id": strategy_id,
                "gross_return": gross_return,
                "net_return": net_return,
                "nav_units": snapshot.nav.units,
                "nav_scale": snapshot.nav.scale,
                "base_currency": snapshot.base_currency,
            }
        )
        for instrument_id, spec in sorted(instruments.items()):
            if snapshot.initial_margin.units != 0 or snapshot.maintenance_margin.units != 0:
                raise ValueError("A-share/ETF certification requires zero margin")
            margin_rows.append(
                {
                    "event_time": event_time,
                    "account_id": snapshot.account_id,
                    "instrument_id": instrument_id,
                    "initial_margin_units": 0,
                    "maintenance_margin_units": 0,
                    "margin_scale": snapshot.initial_margin.scale,
                    "currency": spec.settlement_currency,
                }
            )
    order_rows = []
    for order in artifacts.orders:
        intent = order.intent
        order_rows.append(
            {
                "event_time": intent.created_at,
                "order_id": order.order_id,
                "idempotency_key": intent.idempotency_key,
                "account_id": intent.account_id,
                "strategy_id": intent.strategy_id,
                "instrument_id": intent.instrument_id,
                "side": intent.side.value,
                "quantity_units": intent.quantity.units,
                "quantity_scale": intent.quantity.scale,
                "order_type": intent.order_type.value,
                "limit_price_units": None,
                "limit_price_scale": None,
                "stop_price_units": None,
                "stop_price_scale": None,
                "time_in_force": intent.time_in_force.value,
                "reduce_only": intent.reduce_only,
                "status": order.status.value,
                "filled_quantity_units": order.filled_quantity.units,
                "filled_quantity_scale": order.filled_quantity.scale,
                "version": order.version,
            }
        )
    order_event_rows = [
        {
            "event_time": item.event_time,
            "event_id": item.event_id,
            "order_id": item.order_id,
            "event_sequence": item.sequence,
            "from_status": item.from_status.value,
            "to_status": item.to_status.value,
            "fill_quantity_units": item.fill_quantity.units if item.fill_quantity else None,
            "fill_quantity_scale": item.fill_quantity.scale if item.fill_quantity else None,
            "reason": item.reason,
        }
        for item in artifacts.order_events
    ]
    fill_rows = [
        {
            "event_time": item.event_time,
            "fill_id": item.fill_id,
            "order_id": item.order_id,
            "account_id": item.account_id,
            "strategy_id": item.strategy_id,
            "instrument_id": item.instrument_id,
            "side": item.side.value,
            "quantity_units": item.quantity.units,
            "quantity_scale": item.quantity.scale,
            "price_units": item.price.units,
            "price_scale": item.price.scale,
            "currency": instruments[item.instrument_id].settlement_currency,
            "liquidity_role": item.liquidity_role.value,
            "venue_trade_id": item.venue_trade_id,
        }
        for item in artifacts.fills
    ]
    fills_by_id = {fill.fill_id: fill for fill in artifacts.fills}
    # QExec v0.5.1 exposes one unified Fee per non-futures fill.  Preserve its
    # native maker/taker taxonomy; the adapter must not manufacture a second
    # commission/stamp-duty fee model or relabel the certified artifact.
    cost_rows = [
        {
            "event_time": fee.event_time,
            "cost_id": fee.fee_id,
            "account_id": fee.account_id,
            "strategy_id": strategy_id,
            "instrument_id": fills_by_id[fee.fill_id].instrument_id,
            "fill_id": fee.fill_id,
            "cost_type": fee.fee_type,
            "amount_units": fee.amount.units,
            "amount_scale": fee.amount.scale,
            "currency": fee.currency,
        }
        for fee in artifacts.fees
    ]
    cash_rows = []
    for transaction in artifacts.ledger_transactions:
        for posting_index, posting in enumerate(transaction.postings):
            cash_rows.append(
                {
                    "event_time": transaction.event_time,
                    "transaction_id": transaction.transaction_id,
                    "idempotency_key": transaction.idempotency_key,
                    "event_type": transaction.event_type.value,
                    "reference_id": transaction.reference_id,
                    "posting_index": posting_index,
                    "ledger_account": posting.ledger_account,
                    "account_id": account_id,
                    "currency": posting.currency,
                    "amount_units": posting.amount.units,
                    "amount_scale": posting.amount.scale,
                    "instrument_id": posting.instrument_id,
                    "quantity_delta_units": posting.quantity_delta.units
                    if posting.quantity_delta
                    else None,
                    "quantity_delta_scale": posting.quantity_delta.scale
                    if posting.quantity_delta
                    else None,
                }
            )
    exposure_rows: list[dict[str, Any]] = []
    factor_cols = [factor for factor in config.factors if factor in scored_panel.columns]
    for rebalance_date in sorted(_target_schedule(scored_panel, config, catalog_path=catalog_path)):
        day = scored_panel[pd.to_datetime(scored_panel["date"]).dt.date == rebalance_date]
        event_time = next(
            event.available_at for event in events if event.trading_day == rebalance_date
        )
        for factor in factor_cols:
            value = pd.to_numeric(day[factor], errors="coerce").mean()
            if pd.notna(value):
                exposure_rows.append(
                    {
                        "event_time": event_time,
                        "account_id": account_id,
                        "strategy_id": strategy_id,
                        "exposure_type": "factor",
                        "name": factor,
                        "value": float(value),
                        "unit": "score",
                    }
                )
    frames = {
        "returns": _frame("returns", sorted(return_rows, key=lambda row: row["event_time"])),
        "positions": _frame("positions", sorted(position_rows, key=lambda row: row["event_time"])),
        "portfolio_snapshots": _frame(
            "portfolio_snapshots", sorted(snapshot_rows, key=lambda row: row["event_time"])
        ),
        "exposures": _frame("exposures", sorted(exposure_rows, key=lambda row: row["event_time"])),
        "orders": _frame("orders", sorted(order_rows, key=lambda row: row["event_time"])),
        "order_events": _frame(
            "order_events", sorted(order_event_rows, key=lambda row: row["event_time"])
        ),
        "fills": _frame("fills", sorted(fill_rows, key=lambda row: row["event_time"])),
        "costs": _frame("costs", sorted(cost_rows, key=lambda row: row["event_time"])),
        "cash_ledger": _frame(
            "cash_ledger",
            sorted(
                cash_rows,
                key=lambda row: (row["event_time"], row["transaction_id"], row["posting_index"]),
            ),
        ),
        "margin": _frame("margin", sorted(margin_rows, key=lambda row: row["event_time"])),
    }
    # Portfolio snapshots retain every event; returns have one observation per
    # completed trading day so downstream consumers do not annualize symbols as
    # if they were separate days. Both come from the same recorded ledger.
    returns = frames["returns"]
    if not returns.empty:
        days = (
            pd.to_datetime(returns["event_time"], utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.normalize()
        )
        daily = returns.groupby(days, sort=True).tail(1).copy()
        for column in ("net_return", "gross_return"):
            compounded = (1 + returns[column]).groupby(days, sort=True).prod() - 1
            daily[column] = compounded.to_numpy()
        frames["returns"] = daily.reset_index(drop=True)
    return CertifiedReplay(
        result,
        events,
        instruments,
        mappings,
        ledger,
        frames,
        account_id,
        strategy_id,
        strategy.risk_checks,
        strategy.allocation_decisions,
        strategy.retry_diagnostics,
        artifacts.risk_events,
    )


def _write_certified_v2(
    run_dir: Path,
    scored_panel: pd.DataFrame,
    config: AppConfig,
    dataset_snapshots: dict[str, str] | None,
    *,
    replay: CertifiedReplay | None = None,
    catalog_path: Path = _CATALOG_PATH,
    research_metrics: dict | None = None,
    internal_dependencies: dict[str, str] | None = None,
) -> Any:
    replay = replay or _replay(scored_panel, config, run_dir.name, catalog_path=catalog_path)
    snapshots = dict(dataset_snapshots or {})
    catalog_sha256 = _file_sha256(catalog_path)
    catalog_key = (
        "fixture-catalog-v1" if catalog_path == _CATALOG_PATH else "watchlist-rule-assumptions-v1"
    )
    certified_snapshots = {
        catalog_key: f"sha256:{catalog_sha256}",
        "scored-panel-v1": f"sha256:{_canonical_frame_sha256(scored_panel)}",
    }
    for name, digest in certified_snapshots.items():
        existing = snapshots.get(name)
        if existing is not None and existing != digest:
            raise ValueError(f"dataset snapshot conflict for {name}")
        snapshots[name] = digest
    certified_inputs = [f"dataset:{catalog_key}", "dataset:scored-panel-v1"]
    lineage = {
        "config": certified_inputs,
        "metrics": certified_inputs,
        "returns": ["portfolio_snapshots", *certified_inputs],
        "positions": ["portfolio_snapshots", *certified_inputs],
        "portfolio_snapshots": ["cash_ledger", *certified_inputs],
        "exposures": certified_inputs,
        "orders": certified_inputs,
        "order_events": ["orders"],
        "fills": ["orders", *certified_inputs],
        "costs": ["fills"],
        "cash_ledger": ["fills", *certified_inputs],
        "margin": ["portfolio_snapshots"],
    }
    config_payload = asdict(config)
    config_payload["certification"] = {
        "path": "qexec-deterministic-replay",
        "legacy_modules": ["trading_costs", "trade_ledger"],
        "legacy_modules_are": "research-only",
        "margin_policy": "AShareRule cash account: zero initial and maintenance margin; no aggregate margin replication",
        "fee_classification": (
            f"QExec {_DEPENDENCIES['quant-execution']} unified maker/taker; "
            "commission/stamp classification unavailable"
        ),
        "cost_policy": "configured proportional commission plus per-order minimum, sell stamp tax, adverse slippage",
        "evidence_scope": "deterministic simulation; not historical market-data or investment certification",
    }
    write_standard_run_v2(
        run_dir,
        project="a-share-multifactor",
        run_id=run_dir.name,
        strategy_ids=[replay.strategy_id],
        profile="backtest-ledger",
        frames=replay.frames,
        metrics={
            "qexec_run": asdict(replay.result),
            "orders": len(replay.frames["orders"]),
            "fills": len(replay.frames["fills"]),
            "certification": "single DeterministicRunEngine -> RuleBookRiskGate -> ExactAccountLedger replay",
            "backtest_stats": json.loads(
                replay_results(replay, config).stats.to_json(orient="records")
            ),
            **(research_metrics or {}),
        },
        config=config_payload,
        code_version=_code_version(Path(__file__).resolve().parents[2]),
        internal_dependencies=internal_dependencies or _installed_internal_dependencies(),
        random_seed=0,
        dataset_snapshots=snapshots,
        instrument_master_version=(f"a-share-explicit-catalog@sha256:{catalog_sha256[:12]}"),
        execution_model_version="a-share-configured-qexec-next-bar-v2",
        base_currency="CNY",
        lineage=lineage,
        capabilities=["backtest", "deterministic-replay", "t-plus-one"],
        tags={
            "asset_class": "cn-a-share-and-etf",
            "certification": "qexec",
            "research_type": "multifactor",
        },
    )
    return load_and_validate_standard_run(run_dir)


def replay_results(replay: CertifiedReplay, config: AppConfig) -> BacktestResult:
    """Daily marks and every performance number come from the same exact ledger."""
    frame = replay.frames["returns"].sort_values("event_time").copy()
    frame["date"] = (
        pd.to_datetime(frame["event_time"], utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    closes = frame.groupby("date", sort=True).tail(1).set_index("date")
    nav = pd.Series(
        [
            float(Decimal(int(row.nav_units)).scaleb(-int(row.nav_scale)))
            for row in closes.itertuples()
        ],
        index=closes.index,
        dtype=float,
    )
    returns = nav.pct_change(fill_method=None)
    if len(nav):
        returns.iloc[0] = nav.iloc[0] / config.costs.initial_capital - 1
    name = f"Q{config.quantiles}"
    return BacktestResult(
        quantile_returns=pd.DataFrame({name: returns}),
        cumulative_returns=pd.DataFrame({name: nav / config.costs.initial_capital}),
        long_short=pd.Series(dtype=float),
        # The first observed close is the account's opening anchor, not a
        # completed return interval. Keep it in NAV exports but not annualization.
        stats=pd.DataFrame([{"portfolio": name, **return_statistics(returns.iloc[1:], 252)}]),
    )


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
        for item_date, net_return in net.items():
            estimated_cost = float(turnover.get(item_date, 0.0)) * cost_rate
            rows.append(
                {
                    "date": item_date,
                    "strategy": strategy,
                    "gross_return": float(net_return) + estimated_cost,
                    "net_return": float(net_return),
                    "nav": float(nav.get(item_date, np.nan)),
                    "benchmark_return": float(benchmark.get(item_date, np.nan)),
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
    for item_date in rebalance_dates(panel["date"], config.rebalance_freq):
        day = panel[panel["date"] == item_date].copy()
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
                    "date": item_date,
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
            if abs(delta) > 1e-12:
                orders.append(
                    {
                        "timestamp": item_date,
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
                    "date": item_date,
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
) -> RunManifest | Any:
    """Publish one execution result; all user-facing views use its daily NAV.

    Legacy multi-quantile calculations remain research helpers, but no longer
    produce a contradictory standard/v1 account beside the exact v2 ledger.
    """
    import shutil

    from a_share_multifactor.report import write_html_report

    replay = _replay(scored_panel, config, run_dir.name)
    research_metrics = {}
    for name in ("ic_summary", "ic_decay"):
        path = run_dir / f"{name}.csv"
        if path.exists():
            research_metrics[name] = json.loads(pd.read_csv(path).to_json(orient="records"))
    manifest = _write_certified_v2(
        run_dir,
        scored_panel,
        config,
        dataset_snapshots,
        replay=replay,
        research_metrics=research_metrics,
    )
    canonical = replay_results(replay, config)
    results.__dict__.update(canonical.__dict__)
    results.quantile_returns.to_csv(run_dir / "quantile_returns.csv")
    results.cumulative_returns.to_csv(run_dir / "cumulative_returns.csv")
    results.stats.to_csv(run_dir / "backtest_stats.csv", index=False)
    # Preserve legacy research views explicitly, not as the account's returns.
    for name in ("long_short.csv", "excess_returns.csv", "turnover.csv"):
        old = run_dir / name
        if old.exists():
            old.rename(run_dir / f"legacy_research_{name}")
    ic = pd.DataFrame(research_metrics.get("ic_summary", []))
    write_html_report(
        results,
        ic,
        run_dir / "report.html",
        "QExec configured next-bar simulation; daily ledger NAV; research only",
    )
    latest = run_dir.parent / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("long_short.csv", "excess_returns.csv", "turnover.csv"):
        legacy_view = latest / name
        if legacy_view.exists():
            legacy_view.unlink()  # Current run's research copy is preserved above.
    for name in (
        "quantile_returns.csv",
        "cumulative_returns.csv",
        "backtest_stats.csv",
        "report.html",
    ):
        shutil.copy2(run_dir / name, latest / name)
    return manifest
