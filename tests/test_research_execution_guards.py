from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from a_share_multifactor.config import AppConfig
from a_share_multifactor.corporate_actions import action_events
from a_share_multifactor.decision_workflow import write_watchlist_catalog
from a_share_multifactor.run_contract import ReplayInputError, _replay, _TargetWeightStrategy


def distribution_fixture(tmp_path, *, cash="1", ratio="1", reference_cash="0.8"):
    dates = pd.bdate_range("2026-07-08", periods=4)
    raw = pd.DataFrame(
        [
            {
                "symbol": "600036",
                "date": day,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000000,
                "composite_score": 1.0 if day == dates[0] else float("nan"),
            }
            for day, price in zip(dates, [10, 10, 9.2, 9.2])
        ]
    )
    adjusted = raw.copy()
    adjusted["close"] = 9.2
    actions = pd.DataFrame(
        [
            {
                "event_id": "distribution",
                "symbol": "600036",
                "announced_date": dates[0],
                "record_date": dates[1],
                "ex_date": dates[2],
                "pay_date": dates[2],
                "shares_available_date": dates[2],
                "cash_per_share": cash,
                "share_ratio": ratio,
                "source": "synthetic-entitlement",
                "adjustment_cash_per_share": reference_cash,
                "adjustment_share_ratio": "1",
                "adjustment_reference_source": "synthetic-exchange",
                "adjustment_reference_sha256": "a" * 64,
            }
        ]
    )
    config = AppConfig()
    config = replace(
        config,
        quantiles=1,
        rebalance_freq="daily",
        costs=replace(
            config.costs,
            retail_mode=False,
            initial_capital=10000,
            commission=0,
            min_commission=0,
            stamp_tax=0,
            slippage=0,
            max_position_weight=0.5,
            cash_buffer=0,
            max_holdings=1,
        ),
    )
    catalog = write_watchlist_catalog(
        tmp_path, {"watchlist": [{"symbol": "600036", "venue": "SSE"}]}, "2026-07-01", "2026-08-01"
    )
    return dates, raw, adjusted, actions, config, catalog


def test_exchange_reference_does_not_reduce_actual_entitlement(tmp_path):
    dates, raw, adjusted, actions, config, catalog = distribution_fixture(tmp_path)
    events = action_events(actions, raw, adjusted, dates[0], dates[-1])
    assert events[0].cash_amount.to_decimal() == 1
    result = _replay(
        raw, config, "different-reference", catalog_path=catalog, corporate_actions=events
    )
    final = result.ledger.snapshot()
    assert final.positions["600036"].to_decimal() == 500
    assert final.cash_balances["CNY"].to_decimal() == 5500
    assert final.nav.to_decimal() == 10100
    actions.loc[0, "adjustment_reference_sha256"] = ""
    with pytest.raises(ValueError, match="source evidence"):
        action_events(actions, raw, adjusted, dates[0], dates[-1])


def test_reference_only_event_creates_no_shareholder_cash_or_shares(tmp_path):
    dates, raw, adjusted, actions, _, _ = distribution_fixture(tmp_path, cash="0")
    actions.loc[0, "record_date"] = pd.NaT
    assert action_events(actions, raw, adjusted, dates[0], dates[-1]) == ()


@pytest.mark.parametrize("holding", [True, False])
def test_unverified_entitlement_guard_only_stops_an_exposed_account(tmp_path, holding):
    dates, raw, _, _, config, catalog = distribution_fixture(tmp_path)
    at = (dates[2].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)).isoformat()
    guards = (
        {
            "asset": "600036",
            "available_at": at,
            "reason": "unverified entitlement",
            "source": "synthetic-issuer-evidence",
        },
    )
    kwargs = {} if holding else {"target_schedule": {}}
    if holding:
        with pytest.raises(ReplayInputError, match="unverified entitlement") as caught:
            _replay(raw, config, "guard", catalog_path=catalog, input_guards=guards, **kwargs)
        assert caught.value.instrument_id == "600036"
        assert caught.value.occurred_at == pd.Timestamp(at)
    else:
        replay = _replay(raw, config, "guard", catalog_path=catalog, input_guards=guards, **kwargs)
        assert replay.ledger.snapshot().nav.to_decimal() == 10000
        assert replay.frames["fills"].empty


def test_fractional_entitlement_is_annotated_and_not_rounded(tmp_path):
    dates, raw, adjusted, actions, config, catalog = distribution_fixture(
        tmp_path, ratio="1.400595"
    )
    events = action_events(actions, raw, adjusted, dates[0], dates[-1])
    with pytest.raises(ReplayInputError, match="quantity_step") as caught:
        _replay(raw, config, "fractional", catalog_path=catalog, corporate_actions=events)
    assert caught.value.instrument_id == "600036"
    assert caught.value.occurred_at == events[0].available_at


@pytest.mark.parametrize("branch, added", [("sensitivity_floor", 200), ("sensitivity_ceil", 201)])
def test_share_rule_uses_record_date_account_and_can_liquidate_odd_remainder(
    tmp_path, branch, added
):
    dates, raw, adjusted, actions, config, catalog = distribution_fixture(
        tmp_path, ratio="1.400595"
    )
    events = action_events(actions, raw, adjusted, dates[0], dates[-1])
    extra = raw.iloc[[-1]].copy()
    extra["date"] = dates[-1] + pd.offsets.BDay(1)
    raw = pd.concat([raw, extra], ignore_index=True)
    rule = {
        "entitlement_id": "distribution",
        "record_date": str(dates[1].date()),
        "ordinary_additional_ratio": "0.400595",
        "known_at": dates[0].tz_localize("Asia/Shanghai").isoformat(),
        "credit_at": events[0].available_at.isoformat(),
        "sellable_date": str(dates[2].date()),
        "evidence_source": "synthetic-issuer",
        "evidence_sha256": "a" * 64,
        "fractional_rule": "largest_fraction",
        "evidence_status": branch,
        "reference_price": "6.50",
    }
    schedule = {dates[0].date(): {"600036": 500}, dates[3].date(): {}}
    kwargs = {
        "catalog_path": catalog,
        "corporate_actions": events,
        "target_schedule": schedule,
        "share_rules": {"distribution": rule},
        "snapshot_frequency": "event",
    }
    replay = _replay(raw, config, "share-branch", **kwargs)
    action_positions = replay.frames["positions"]
    action_positions = action_positions[
        pd.to_datetime(action_positions.event_time, utc=True).eq(events[0].available_at)
    ]
    assert len(action_positions) == 1
    assert Decimal(int(action_positions.iloc[0].mark_price_units)).scaleb(
        -int(action_positions.iloc[0].mark_price_scale)
    ) == Decimal("6.50")
    assert all(q.units == 0 for q in replay.ledger.snapshot().positions.values())
    sold = replay.frames["fills"].query('side == "sell"')
    assert int(sold.quantity_units.sum()) == 500 + added
    assert replay.ledger.snapshot().cash_balances["CNY"].to_decimal() == (
        Decimal(5500) + Decimal(500 + added) * Decimal("9.2")
    )
    repeated = _replay(raw, config, "share-branch", **kwargs)
    assert repeated.ledger.snapshot() == replay.ledger.snapshot()
    rule["evidence_status"] = "unresolved"
    with pytest.raises(ReplayInputError) as caught:
        _replay(raw, config, "strict-share", **kwargs)
    assert caught.value.diagnostics["reason"] == "entitlement_unresolved"


def test_initial_budget_retries_but_never_resets_after_publication(monkeypatch):
    from a_share_multifactor import allocation_execution

    day = pd.Timestamp("2026-01-05").date()
    plan = {
        "scores": pd.Series({"A": 1.0}),
        "initial_max_turnover": 2.0,
        "invested_limit": 0.95,
        "allocation": {
            "protocol_version": 2,
            "mode": "equal",
            "risk_objective": "absolute",
            "forecast_horizon_sessions": 5,
            "expected_return_model": "none",
            "max_turnover": 0.2,
        },
    }
    strategy = SimpleNamespace(
        allocation_schedule={day: plan}, _allocation_decisions=[], _closing_prices={}
    )
    snapshot = SimpleNamespace(positions={})
    budgets = []

    def allocate(strategy, selected, snapshot, nav):
        budgets.append(selected["allocation"]["max_turnover"])
        if len(budgets) == 1:
            raise ValueError("missing causal risk input")
        return pd.Series({"A": 0.95}), {"A": 100}, {}

    monkeypatch.setattr(allocation_execution, "executable_allocation", allocate)
    with pytest.raises(ValueError, match="missing causal"):
        _TargetWeightStrategy._allocation_target(strategy, day, snapshot, Decimal(1000))
    _TargetWeightStrategy._allocation_target(strategy, day, snapshot, Decimal(1000))
    # Even a later liquidated account must not receive another initial-build exemption.
    _TargetWeightStrategy._allocation_target(strategy, day, snapshot, Decimal(1000))
    assert budgets == [2.0, 2.0, 0.2]
    assert plan["allocation"]["max_turnover"] == 0.2


@pytest.mark.parametrize("budget", [True, "2", float("nan"), -1, 2.1])
def test_initial_build_budget_requires_explicit_bounded_number(budget):
    day = pd.Timestamp("2026-01-05").date()
    strategy = SimpleNamespace(allocation_schedule={day: {"initial_max_turnover": budget}})
    with pytest.raises(ValueError, match="initial_max_turnover"):
        _TargetWeightStrategy._allocation_target(
            strategy, day, SimpleNamespace(positions={}), Decimal(1000)
        )


@pytest.mark.parametrize(
    "field,value", [("asset", "missing"), ("reason", ""), ("available_at", "2026-07-10")]
)
def test_invalid_input_guard_cannot_enter_a_replay(tmp_path, field, value):
    _, raw, _, _, config, catalog = distribution_fixture(tmp_path)
    guard = {
        "asset": "600036",
        "reason": "missing entitlement",
        "source": "synthetic",
        "available_at": "2026-07-10T09:00:00+08:00",
        field: value,
    }
    with pytest.raises(ValueError, match="invalid evidenced"):
        _replay(raw, config, "bad-guard", catalog_path=catalog, input_guards=(guard,))
