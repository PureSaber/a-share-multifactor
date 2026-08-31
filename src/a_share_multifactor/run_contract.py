"""Research v1 compatibility and the certified QExec backtest-ledger path.

The legacy writer remains for historical standard/v1 readers. New certification
artifacts are produced only from one DeterministicRunEngine replay.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from quant_data_kit import AssetClass, BarEvent, FixedPoint, InstrumentSpec, SymbolMapping
from quant_execution import (
    BarMatchingModel,
    DeterministicBroker,
    DeterministicRunEngine,
    ExactAccountLedger,
    OrderIntent,
    OrderType,
    RuleBookRiskGate,
    Side,
    StrategyContext,
    TimeInForce,
)
from quant_lab import load_and_validate_standard_run, write_standard_run_v2
from quant_lab.contracts import RunManifest, write_standard_run

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import AppConfig
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
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    if status.strip():
        raise RuntimeError("certified runs require a clean Git worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


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
    catalog = pd.read_csv(path, dtype=str)
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
    panel: pd.DataFrame, *, catalog_path: Path = _CATALOG_PATH
) -> tuple[dict[str, InstrumentSpec], tuple[SymbolMapping, ...]]:
    """Build PIT InstrumentSpec and SymbolMapping objects from the catalog."""
    catalog = load_fixture_catalog(catalog_path).set_index("symbol")
    symbols = sorted(panel["symbol"].astype(str).unique())
    missing = sorted(set(symbols) - set(catalog.index))
    if missing:
        raise ValueError(
            f"certified replay requires explicit fixture catalog entries; missing={missing}"
        )
    first_date = pd.to_datetime(panel["date"]).min().date()
    last_date = pd.to_datetime(panel["date"]).max().date()
    specs: dict[str, InstrumentSpec] = {}
    mappings: list[SymbolMapping] = []
    for symbol in symbols:
        row = catalog.loc[symbol]
        effective_from = _utc(row["effective_from"]).to_pydatetime()
        effective_to = _utc(row["effective_to"]).to_pydatetime()
        if first_date < effective_from.date() or last_date >= effective_to.date():
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
                "catalog_scope": "fixture-certified-not-listing-history",
            },
        )
        mappings.append(
            SymbolMapping(
                source="fixture-certified",
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
        timestamp = _utc(day) + pd.Timedelta(hours=8, microseconds=int(index))
        bar_end = timestamp + pd.Timedelta(minutes=1)
        volume = max(1, int(Decimal(str(row.get("volume", 1))).to_integral_value()))
        events.append(
            BarEvent(
                event_id=f"fixture-bar:{day.isoformat()}:{symbol}",
                instrument_id=symbol,
                event_time=bar_end.to_pydatetime(),
                received_at=bar_end.to_pydatetime(),
                available_at=bar_end.to_pydatetime(),
                source="fixture-certified",
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


def _target_schedule(panel: pd.DataFrame, config: AppConfig) -> dict[date, dict[str, int]]:
    catalog = load_fixture_catalog().set_index("symbol")
    schedule: dict[date, dict[str, int]] = {}
    for rebalance_date in rebalance_dates(panel["date"], config.rebalance_freq):
        day = panel[pd.to_datetime(panel["date"]) == pd.Timestamp(rebalance_date)].copy()
        if day.empty:
            continue
        day["composite_score"] = pd.to_numeric(day["composite_score"], errors="coerce")
        day["quantile"] = _assign_quantiles(day["composite_score"], config.quantiles)
        selected = day[day["quantile"] == float(config.quantiles)]
        if selected.empty:
            selected = day.nlargest(1, "composite_score")
        selected = selected.dropna(subset=["close"])
        allocation = (
            Decimal(str(config.costs.initial_capital)) / len(selected)
            if len(selected)
            else Decimal(0)
        )
        target: dict[str, int] = {}
        for _, row in selected.sort_values("symbol").iterrows():
            symbol = str(row["symbol"])
            lot = int(catalog.loc[symbol, "lot_size"])
            shares = int((allocation / Decimal(str(row["close"]))) // lot) * lot
            if shares > 0:
                target[symbol] = shares
        if target:
            schedule[pd.Timestamp(rebalance_date).date()] = target
    return schedule


class _TargetWeightStrategy:
    """Emit only QExec OrderIntent objects; it never mutates positions."""

    sends_live_orders = False

    def __init__(self, schedule: dict[date, dict[str, int]]) -> None:
        self.schedule = schedule
        self.reset()

    def reset(self) -> None:
        self._planned: dict[str, int] = {}
        self._pending: dict[str, tuple[Side, int]] = {}
        self._deferred_buys: dict[str, int] = {}
        self._day: date | None = None

    def on_event(self, context: StrategyContext, event: BarEvent) -> tuple[OrderIntent, ...]:
        if event.trading_day != self._day:
            self._day = event.trading_day
            self._pending = {
                symbol: (Side.BUY, quantity) for symbol, quantity in self._deferred_buys.items()
            }
            self._deferred_buys = {}
            target = self.schedule.get(event.trading_day)
            if target is not None:
                sells: dict[str, int] = {}
                buys: dict[str, int] = {}
                for symbol in sorted(set(self._planned) | set(target)):
                    delta = target.get(symbol, 0) - self._planned.get(symbol, 0)
                    if delta:
                        if delta > 0:
                            buys[symbol] = delta
                        else:
                            sells[symbol] = abs(delta)
                self._pending.update(
                    {symbol: (Side.SELL, quantity) for symbol, quantity in sells.items()}
                )
                if sells:
                    self._deferred_buys.update(buys)
                else:
                    self._pending.update(
                        {symbol: (Side.BUY, quantity) for symbol, quantity in buys.items()}
                    )
                self._planned = dict(target)
        pending = self._pending.pop(event.instrument_id, None)
        if pending is None:
            return ()
        side, quantity = pending
        return (
            OrderIntent(
                idempotency_key=f"{context.strategy_id}:{event.trading_day.isoformat()}:{event.instrument_id}:{side.value}",
                account_id=context.account_id,
                strategy_id=context.strategy_id,
                instrument_id=event.instrument_id,
                side=side,
                quantity=FixedPoint(quantity, 0),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                created_at=event.available_at,
            ),
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


def _frame(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_V2_COLUMNS[name])


def _replay(scored_panel: pd.DataFrame, config: AppConfig, run_id: str) -> CertifiedReplay:
    instruments, mappings = build_instrument_master(scored_panel)
    events = _build_events(scored_panel, instruments)
    strategy = _TargetWeightStrategy(_target_schedule(scored_panel, config))
    ledger = _RecordingLedger(
        account_id=_ACCOUNT_ID,
        base_currency="CNY",
        instruments=instruments,
        initial_cash={"CNY": _fixed(config.costs.initial_capital, 2)},
        money_scale=_MONEY_SCALE,
    )
    engine = DeterministicRunEngine(
        run_id=run_id,
        account_id=_ACCOUNT_ID,
        strategy_id=_STRATEGY_ID,
        strategy=strategy,
        broker=DeterministicBroker(),
        risk_gate=RuleBookRiskGate(instruments=instruments, ledger=ledger),
        matching_model=BarMatchingModel(instruments, participation_rate="1"),
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
        current_marks[event.instrument_id] = event.close_price
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
                mark = _fixed(
                    snapshot.cost_basis.get(instrument_id, FixedPoint(1, spec.price_tick.scale)),
                    spec.price_tick.scale,
                )
            notional = _decimal(quantity) * _decimal(mark) * _decimal(spec.contract_multiplier)
            market_value += notional
            base_value = _fixed(notional, _MONEY_SCALE)
            position_rows.append(
                {
                    "event_time": event_time,
                    "account_id": snapshot.account_id,
                    "strategy_id": _STRATEGY_ID,
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
                "strategy_id": _STRATEGY_ID,
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
            "strategy_id": _STRATEGY_ID,
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
                    "account_id": _ACCOUNT_ID,
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
    for rebalance_date in sorted(_target_schedule(scored_panel, config)):
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
                        "account_id": _ACCOUNT_ID,
                        "strategy_id": _STRATEGY_ID,
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
    return CertifiedReplay(result, events, instruments, mappings, ledger, frames)


def _write_certified_v2(
    run_dir: Path,
    scored_panel: pd.DataFrame,
    config: AppConfig,
    dataset_snapshots: dict[str, str] | None,
) -> Any:
    replay = _replay(scored_panel, config, run_dir.name)
    snapshots = dict(dataset_snapshots or {})
    catalog_sha256 = _file_sha256(_CATALOG_PATH)
    certified_snapshots = {
        "fixture-catalog-v1": f"sha256:{catalog_sha256}",
        "scored-panel-v1": f"sha256:{_canonical_frame_sha256(scored_panel)}",
    }
    for name, digest in certified_snapshots.items():
        existing = snapshots.get(name)
        if existing is not None and existing != digest:
            raise ValueError(f"dataset snapshot conflict for {name}")
        snapshots[name] = digest
    certified_inputs = ["dataset:fixture-catalog-v1", "dataset:scored-panel-v1"]
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
    }
    write_standard_run_v2(
        run_dir,
        project="a-share-multifactor",
        run_id=run_dir.name,
        strategy_ids=[_STRATEGY_ID],
        profile="backtest-ledger",
        frames=replay.frames,
        metrics={
            "qexec_run": asdict(replay.result),
            "orders": len(replay.frames["orders"]),
            "fills": len(replay.frames["fills"]),
            "certification": "single DeterministicRunEngine -> RuleBookRiskGate -> ExactAccountLedger replay",
        },
        config=config_payload,
        code_version=_code_version(Path(__file__).resolve().parents[2]),
        internal_dependencies=_DEPENDENCIES,
        random_seed=0,
        dataset_snapshots=snapshots,
        instrument_master_version=(f"a-share-fixture-catalog-v1@sha256:{catalog_sha256[:12]}"),
        execution_model_version="quant-execution-v0.5.1-bar-replay-v1",
        base_currency="CNY",
        lineage=lineage,
        capabilities=["backtest", "deterministic-replay", "pit", "t-plus-one"],
        tags={
            "asset_class": "cn-a-share-and-etf",
            "certification": "qexec",
            "research_type": "multifactor",
        },
    )
    return load_and_validate_standard_run(run_dir)


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
    """Write legacy v1 plus certified v2, then validate the preferred v2 manifest."""
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
    write_standard_run(
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
        metrics={
            "statistics": results.stats.to_dict(orient="records"),
            "periods": len(results.quantile_returns),
        },
        config=asdict(config),
        code_version=_code_version(Path(__file__).resolve().parents[2]),
        dataset_snapshots=dataset_snapshots,
        tags={"asset_class": "cn_equity", "research_type": "legacy-research"},
    )
    return _write_certified_v2(run_dir, scored_panel, config, dataset_snapshots)
