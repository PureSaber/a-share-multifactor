import subprocess
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from a_share_multifactor import run_contract
from a_share_multifactor.config import AppConfig
from a_share_multifactor.run_contract import (
    _build_events,
    _canonical_frame_sha256,
    _canonical_value,
    _code_version,
    _position_and_order_frames,
    _replay,
    _returns_frame,
    _target_schedule,
    _TargetWeightStrategy,
    _write_certified_v2,
    build_instrument_master,
    load_fixture_catalog,
)


def _certified_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=45, freq="B")
    symbols = ["000001", "000002", "000003", "000004", "000005", "510300"]
    rows = []
    for day_index, timestamp in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            price = 10 + symbol_index + day_index * 0.01
            if symbol == "510300":
                price = 3.125 + day_index * 0.001
            rows.append(
                {
                    "date": timestamp,
                    "symbol": symbol,
                    "open": price,
                    "high": price + 0.02,
                    "low": price - 0.02,
                    "close": price,
                    "volume": 1_000_000,
                    "composite_score": float(
                        10
                        if symbol == "510300" and day_index >= 22
                        else -1
                        if symbol == "510300"
                        else symbol_index
                    ),
                    "market_cap": float(100 + symbol_index),
                }
            )
    return pd.DataFrame(rows)


def test_fixture_catalog_is_explicit_and_pit() -> None:
    catalog = load_fixture_catalog()
    assert {"510300", "159919"}.issubset(set(catalog["symbol"]))
    assert catalog["symbol"].is_unique
    assert (
        pd.to_datetime(catalog["effective_from"], utc=True)
        < pd.to_datetime(catalog["effective_to"], utc=True)
    ).all()


def test_instrument_master_uses_stock_and_etf_contracts() -> None:
    panel = _certified_panel()
    specs, mappings = build_instrument_master(panel)
    assert specs["000001"].asset_class.value == "equity"
    assert specs["510300"].asset_class.value == "etf"
    assert specs["510300"].price_tick.scale == 3
    assert all(mapping.source == "fixture-certified" for mapping in mappings)


def test_scored_panel_hash_is_canonical_and_value_sensitive() -> None:
    panel = _certified_panel()
    reordered = panel.sample(frac=1, random_state=7)[reversed(panel.columns)].reset_index(drop=True)
    assert _canonical_frame_sha256(panel) == _canonical_frame_sha256(reordered)
    changed = panel.copy()
    changed.loc[0, "composite_score"] += 0.0001
    assert _canonical_frame_sha256(panel) != _canonical_frame_sha256(changed)


def test_canonical_hash_supports_typed_values_and_rejects_duplicate_columns() -> None:
    values = {
        "none": None,
        "date": date(2025, 1, 2),
        "timestamp": pd.Timestamp("2025-01-02", tz="Asia/Shanghai"),
        "bool": np.bool_(True),
        "integer": np.int64(3),
        "float": np.float64(1.25),
        "decimal": Decimal("1.250"),
        "mapping": {2: "b", 1: "a"},
        "sequence": [1, "a"],
        "string": "value",
    }
    assert all(
        _canonical_value(value) is not None for key, value in values.items() if key != "none"
    )
    duplicate_columns = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(ValueError, match="unique column"):
        _canonical_frame_sha256(duplicate_columns)


def test_fixture_and_event_validation_fail_closed(tmp_path: Path) -> None:
    catalog = load_fixture_catalog()
    missing_column = tmp_path / "missing.csv"
    catalog.drop(columns="price_tick").to_csv(missing_column, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_fixture_catalog(missing_column)

    duplicate = tmp_path / "duplicate.csv"
    pd.concat([catalog, catalog.iloc[[0]]], ignore_index=True).to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="duplicate symbols"):
        load_fixture_catalog(duplicate)

    missing_symbol = _certified_panel().copy()
    missing_symbol.loc[0, "symbol"] = "NOT-CATALOGUED"
    with pytest.raises(ValueError, match="explicit fixture catalog"):
        build_instrument_master(missing_symbol)

    outside_window = _certified_panel().copy()
    outside_window["date"] = pd.to_datetime(outside_window["date"]) + pd.DateOffset(years=20)
    with pytest.raises(ValueError, match="outside fixture validity"):
        build_instrument_master(outside_window)

    panel = _certified_panel().head(1)
    specs, _ = build_instrument_master(panel)
    duplicated_bar = pd.concat([panel, panel], ignore_index=True)
    with pytest.raises(ValueError, match="one bar per symbol"):
        _build_events(duplicated_bar, specs)


def test_production_events_have_deterministic_sequences_and_are_order_normalized() -> None:
    panel = _certified_panel()
    specs, _ = build_instrument_master(panel)
    events = _build_events(panel.sample(frac=1, random_state=17), specs)
    repeated = _build_events(panel.sample(frac=1, random_state=91), specs)

    assert all(isinstance(event.sequence, int) and event.sequence >= 0 for event in events)
    assert [event.event_id for event in events] == [event.event_id for event in repeated]
    assert [event.sequence for event in events] == [event.sequence for event in repeated]
    assert list(events) == sorted(events, key=lambda event: (event.available_at, event.event_id))


def test_production_event_builder_rejects_duplicate_identity_after_reordering() -> None:
    panel = _certified_panel()
    specs, _ = build_instrument_master(panel)
    duplicate = pd.concat([panel.sample(frac=1, random_state=3), panel.iloc[[0]]])
    with pytest.raises(ValueError, match="one bar per symbol"):
        _build_events(duplicate, specs)


def test_certified_snapshot_conflict_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_contract, "_replay", lambda *_args, **_kw: object())
    with pytest.raises(ValueError, match="snapshot conflict"):
        _write_certified_v2(
            tmp_path,
            _certified_panel(),
            AppConfig(),
            {"fixture-catalog-v1": "sha256:wrong"},
        )


def test_legacy_v1_frame_builders_remain_read_compatible() -> None:
    dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
    results = SimpleNamespace(
        turnover=pd.DataFrame({"turnover": [0.5, 0.25]}, index=dates),
        quantile_returns=pd.DataFrame({"Q5": [0.01, 0.02]}, index=dates),
        cumulative_returns=pd.DataFrame({"Q5": [1.01, 1.0302]}, index=dates),
        benchmark_returns=pd.Series([0.001, 0.002], index=dates),
    )
    returns = _returns_frame(results, AppConfig())
    assert len(returns) == 2
    assert (returns["gross_return"] > returns["net_return"]).all()

    panel = _certified_panel()
    positions, orders, exposures = _position_and_order_frames(
        panel, AppConfig(factors=["market_cap"], rebalance_freq="monthly")
    )
    assert not positions.empty
    assert set(orders["side"]).issubset({"buy", "sell"})
    assert not exposures.empty


def test_empty_rebalance_days_and_zero_lot_targets_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _certified_panel()
    missing_day = pd.Timestamp("2019-12-31")
    monkeypatch.setattr(run_contract, "rebalance_dates", lambda *_args: [missing_day])
    assert _target_schedule(panel, AppConfig()) == {}
    positions, orders, exposures = _position_and_order_frames(panel, AppConfig())
    assert positions.empty
    assert orders.empty
    assert exposures.empty

    actual_day = pd.Timestamp(panel["date"].min())
    monkeypatch.setattr(run_contract, "rebalance_dates", lambda *_args: [actual_day])
    config = AppConfig(costs=replace(AppConfig().costs, initial_capital=1.0))
    assert _target_schedule(panel, config) == {}


def test_replay_fails_closed_when_qexec_omits_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingArtifactsEngine:
        def __init__(self, **_kwargs: object) -> None:
            self.artifacts = None

        def replay(self, _events: object, *, seed: int) -> object:
            assert seed == 0
            return object()

    monkeypatch.setattr(run_contract, "DeterministicRunEngine", MissingArtifactsEngine)
    with pytest.raises(RuntimeError, match="did not produce artifacts"):
        _replay(_certified_panel(), AppConfig(), "missing-artifacts")


def test_nan_factor_exposure_is_not_published() -> None:
    panel = _certified_panel()
    panel["market_cap"] = np.nan
    replay = _replay(
        panel,
        AppConfig(
            start_date="2020-01-02",
            end_date="2020-03-06",
            factors=["market_cap"],
            quantiles=5,
            rebalance_freq="monthly",
        ),
        "nan-exposure",
    )
    assert replay.frames["exposures"].empty


def test_empty_turnover_writes_an_empty_legacy_cost_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def capture_standard_run(_run_dir: Path, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(run_contract, "write_standard_run", capture_standard_run)
    monkeypatch.setattr(run_contract, "_write_certified_v2", lambda *_args: "validated-v2")
    monkeypatch.setattr(run_contract, "_code_version", lambda _root: "a" * 40)
    results = SimpleNamespace(
        turnover=pd.DataFrame(columns=["turnover"]),
        quantile_returns=pd.DataFrame(),
        cumulative_returns=pd.DataFrame(),
        benchmark_returns=pd.Series(dtype=float),
        stats=pd.DataFrame(),
    )

    result = run_contract.write_equity_standard_run(
        tmp_path / "empty-turnover", results, _certified_panel(), AppConfig()
    )

    assert result == "validated-v2"
    costs = captured["frames"]["costs"]
    assert costs.empty
    assert "total_cost" not in costs


def test_code_version_fails_closed_on_dirty_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "M4 Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "m4@example.invalid"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=tmp_path, check=True)
    assert len(_code_version(tmp_path)) == 40

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        _code_version(tmp_path)


def test_certified_replay_is_deterministic_and_emits_execution_facts() -> None:
    panel = _certified_panel()
    config = AppConfig(
        start_date="2020-01-02",
        end_date="2020-03-06",
        factors=["market_cap"],
        quantiles=5,
        rebalance_freq="monthly",
    )
    replays = [_replay(panel, config, "golden-m4") for _ in range(3)]
    state = replays[0].ledger.capture_state()
    replays[0].ledger.restore_state(state)
    assert [
        (replay.result.event_sha256, replay.result.fill_sha256, replay.result.ledger_sha256)
        for replay in replays
    ] == [
        (
            replays[0].result.event_sha256,
            replays[0].result.fill_sha256,
            replays[0].result.ledger_sha256,
        )
    ] * 3
    assert len(replays[0].frames["orders"]) > 0
    assert len(replays[0].frames["fills"]) > 0
    assert len(replays[0].frames["costs"]) > 0
    assert len(replays[0].frames["cash_ledger"]) > 0
    assert (replays[0].frames["fills"]["instrument_id"] == "510300").any()
    assert set(replays[0].frames["costs"]["cost_type"]) == {"taker"}
    assert set(replays[0].frames) == {
        "returns",
        "positions",
        "portfolio_snapshots",
        "exposures",
        "orders",
        "order_events",
        "fills",
        "costs",
        "cash_ledger",
        "margin",
    }
    assert (
        replays[0].frames["returns"]["gross_return"] != replays[0].frames["returns"]["net_return"]
    ).any()
    assert (replays[0].frames["margin"]["initial_margin_units"] == 0).all()
    assert (replays[0].frames["margin"]["maintenance_margin_units"] == 0).all()
    assert not _TargetWeightStrategy({}).sends_live_orders
    fills = replays[0].frames["fills"].sort_values("event_time")
    for sell in fills[fills["side"] == "sell"].itertuples():
        earlier_buys = fills[
            (fills["instrument_id"] == sell.instrument_id)
            & (fills["side"] == "buy")
            & (fills["event_time"] < sell.event_time)
        ]
        assert not earlier_buys.empty
        assert (
            pd.Timestamp(sell.event_time).date()
            > pd.Timestamp(earlier_buys.iloc[-1]["event_time"]).date()
        )


def test_qexec_fee_events_reconcile_to_unified_costs() -> None:
    replay = _replay(
        _certified_panel(),
        AppConfig(
            start_date="2020-01-02",
            end_date="2020-03-06",
            factors=["market_cap"],
            quantiles=5,
            rebalance_freq="monthly",
        ),
        "fee-reconciliation",
    )
    ledger_fee_total = Decimal(0)
    fee_transactions = []
    for transaction in replay.ledger.transactions:
        if transaction.event_type.value == "fee":
            fee_transactions.append(transaction)
            ledger_fee_total += sum(
                Decimal(posting.amount.units).scaleb(-posting.amount.scale)
                for posting in transaction.postings
                if posting.ledger_account == "expenses:fees"
            )
    costs = replay.frames["costs"]
    cost_total = sum(
        Decimal(row.amount_units).scaleb(-row.amount_scale) for row in costs.itertuples()
    )
    assert fee_transactions
    assert len(fee_transactions) == len(costs)
    assert ledger_fee_total == cost_total

    fills = replay.frames["fills"].set_index("fill_id")
    observed_sides = set()
    for fill_id, group in costs.groupby("fill_id"):
        fill = fills.loc[fill_id]
        observed_sides.add(fill.side)
        spec = replay.instruments[fill.instrument_id]
        rate = Decimal(spec.metadata["commission_rate"])
        if fill.side == "sell":
            rate += Decimal(spec.metadata["stamp_duty_rate"])
        expected = (
            Decimal(int(fill.quantity_units)).scaleb(-int(fill.quantity_scale))
            * Decimal(int(fill.price_units)).scaleb(-int(fill.price_scale))
            * rate
        ).quantize(Decimal("1e-8"))
        actual = sum(
            Decimal(row.amount_units).scaleb(-row.amount_scale) for row in group.itertuples()
        )
        assert actual == expected
    assert observed_sides == {"buy", "sell"}
    assert set(costs["cost_type"]) == {"taker"}
