from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from a_share_multifactor.config import AppConfig
from a_share_multifactor.performance import return_statistics
from a_share_multifactor.preprocess import add_forward_return
from a_share_multifactor.quantile_backtest import _periods_per_year
from a_share_multifactor.run_contract import _replay
from a_share_multifactor.synthesis import synthesize


@pytest.mark.parametrize("method", ["ols", "ridge", "rolling_ic_weight", "ic_weight"])
def test_future_prices_cannot_change_existing_scores(method):
    rng = np.random.default_rng(71)
    dates = pd.bdate_range("2025-01-01", periods=90)
    rows = []
    for symbol in ["A", "B", "C", "D", "E"]:
        prices = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        for day, price in zip(dates, prices):
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "close": price,
                    "f1": rng.normal(),
                    "f2": rng.normal(),
                }
            )
    raw = pd.DataFrame(rows)
    cutoff = dates[65]
    altered = raw.copy()
    altered.loc[altered.date > cutoff, "close"] *= 2
    config = AppConfig(factors=["f1", "f2"], rebalance_freq="daily")
    config.synthesis = replace(config.synthesis, method=method)
    original = synthesize(add_forward_return(raw, 20), config)
    changed = synthesize(add_forward_return(altered, 20), config)
    pd.testing.assert_series_equal(
        original.loc[original.date <= cutoff, "composite_score"],
        changed.loc[changed.date <= cutoff, "composite_score"],
    )
    assert original.loc[original.date == cutoff, "composite_score"].notna().all()


def test_label_endpoint_is_symbol_specific_and_sorted():
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "date": pd.to_datetime(["2025-01-06", "2025-01-02", "2025-01-10"]),
            "close": [110, 100, 121],
        }
    )
    labels = add_forward_return(frame, 1)
    assert labels.loc[1, "forward_return_1d"] == pytest.approx(0.1)
    assert labels.loc[1, "forward_return_1d_label_end_at"] == pd.Timestamp("2025-01-06")
    assert pd.isna(labels.loc[2, "forward_return_1d_label_available_at"])


def test_geometric_loss_and_arithmetic_sharpe_are_distinct():
    stats = return_statistics(pd.Series([0.1, -0.1] * 6), 12)
    assert stats["ann_return"] == pytest.approx(0.99**6 - 1)
    assert stats["total_return"] == pytest.approx(stats["ann_return"])
    assert stats["sharpe"] == pytest.approx(0)
    assert return_statistics(pd.Series([-0.1]), 252)["max_drawdown"] == pytest.approx(-0.1)
    assert _periods_per_year(AppConfig(rebalance_freq="weekly")) == 52


def test_qexec_cost_changes_reach_cash_nav_and_fills():
    from test_certified_execution import _certified_panel

    config = AppConfig()
    panel = _certified_panel()
    cheap = _replay(panel, config, "cost-check")
    costly = _replay(
        panel,
        replace(
            config,
            costs=replace(
                config.costs, commission=0.02, min_commission=50, stamp_tax=0.02, slippage=0.05
            ),
        ),
        "cost-check",
    )
    assert not cheap.frames["costs"].equals(costly.frames["costs"])
    assert not cheap.frames["returns"].equals(costly.frames["returns"])
    assert not cheap.frames["fills"].equals(costly.frames["fills"])
    assert costly.frames["portfolio_snapshots"].iloc[-1].nav_units < (
        cheap.frames["portfolio_snapshots"].iloc[-1].nav_units
    )


def test_unsupported_retail_rules_are_rejected_before_replay():
    with pytest.raises(ValueError, match="does not support retail"):
        _replay(pd.DataFrame(), AppConfig(costs=replace(AppConfig().costs, retail_mode=True)), "x")


@pytest.mark.parametrize("change", ["adjusted", "negative_fee", "no_cash", "excess_weight"])
def test_execution_rejects_invalid_economic_inputs(change):
    from test_certified_execution import _certified_panel

    panel = _certified_panel()
    config = AppConfig()
    if change == "adjusted":
        panel["adjustment"] = "qfq"
    elif change == "negative_fee":
        config.costs.commission = -0.01
    elif change == "no_cash":
        config.costs.initial_capital = 0
    else:
        config.costs.max_position_weight = 2
    with pytest.raises(ValueError):
        _replay(panel, config, "invalid-input")


def test_blocked_signal_days_and_risk_limits_stop_orders():
    from test_certified_execution import _certified_panel

    panel = _certified_panel()
    config = AppConfig(rebalance_freq="daily")
    blocked = _replay(panel.assign(decision_allowed=False), config, "blocked-days")
    assert blocked.frames["orders"].empty
    concentration = _replay(panel, config, "concentration", risk_limits={"max_single_weight": 0.01})
    assert concentration.frames["orders"].empty
    assert concentration.risk_checks[-1]["has_critical"]
    assert concentration.risk_checks[-1]["alerts"][0]["rule_id"] == ("portfolio.max_single_weight")
    turnover = _replay(panel, config, "turnover", risk_limits={"max_turnover": 0.01})
    assert turnover.frames["orders"].empty
    assert turnover.risk_checks[-1]["alerts"][0]["rule_id"] == "portfolio.max_turnover"
    # Trigger a portfolio loss after the first next-bar fill and ensure the
    # drawdown gate emits no more rebalance intents after that observed loss.
    dates = sorted(panel.date.unique())
    for col in ["open", "high", "low", "close"]:
        panel.loc[panel.date >= dates[2], col] *= 0.5
    drawdown = _replay(panel, config, "drawdown", risk_limits={"max_drawdown": 0.05})
    assert not drawdown.frames["fills"].empty
    times = pd.to_datetime(drawdown.frames["orders"].event_time, utc=True).dt.tz_localize(None)
    assert not (times.dt.normalize() >= dates[2]).any()


def test_strategy_without_account_and_insolvent_account_cannot_emit_orders():
    from types import SimpleNamespace

    from quant_data_kit import FixedPoint

    from a_share_multifactor.run_contract import _TargetWeightStrategy

    assert _TargetWeightStrategy({}).on_event(None, None) == ()
    day = pd.Timestamp("2025-01-02").date()
    ledger = SimpleNamespace(snapshot=lambda _: SimpleNamespace(nav=FixedPoint(0, 0), positions={}))
    strategy = _TargetWeightStrategy({day: {"A": 100}}, ledger=ledger, trigger_symbols={day: "A"})
    event = SimpleNamespace(
        trading_day=day, instrument_id="A", available_at=None, close_price=FixedPoint(10, 0)
    )
    assert strategy.on_event(None, event) == ()
    state = strategy.capture_state()
    strategy.reset()
    strategy.restore_state(state)
    assert strategy.capture_state() == state


def test_drawdown_halt_survives_recovery_and_checkpoint_restore():
    from types import SimpleNamespace

    from quant_data_kit import FixedPoint

    from a_share_multifactor.run_contract import _TargetWeightStrategy

    day = pd.Timestamp("2025-01-02").date()
    account = SimpleNamespace(nav=FixedPoint(90, 0), positions={})
    ledger = SimpleNamespace(snapshot=lambda _: account)
    strategy = _TargetWeightStrategy(
        {day: {"A": 1}},
        ledger=ledger,
        trigger_symbols={day: "A"},
        initial_capital=100,
        risk_limits={"max_drawdown": 0.05},
    )
    event = SimpleNamespace(
        trading_day=day, instrument_id="A", available_at=None, close_price=FixedPoint(10, 0)
    )
    context = SimpleNamespace(strategy_id="s", account_id="a")
    assert strategy.on_event(context, event) == ()
    state = strategy.capture_state()
    strategy.reset()
    strategy.restore_state(state)
    account.nav = FixedPoint(110, 0)
    assert strategy.on_event(context, event) == ()
    assert strategy.capture_state()["risk_halted"]


def test_empty_account_results_do_not_manufacture_performance():
    from types import SimpleNamespace

    from a_share_multifactor.run_contract import _V2_COLUMNS, replay_results

    result = replay_results(
        SimpleNamespace(frames={"returns": pd.DataFrame(columns=_V2_COLUMNS["returns"])}),
        AppConfig(),
    )
    assert result.quantile_returns.empty
    assert pd.isna(result.stats.iloc[0].ann_return)


def test_minimum_commission_is_charged_once_across_partial_fills_and_rolls_back():
    from types import SimpleNamespace

    from quant_data_kit import FixedPoint
    from quant_execution import LiquidityRole, Side
    from test_certified_execution import _certified_panel

    from a_share_multifactor.execution_models import ConfiguredAShareRiskGate, decimal

    replay = _replay(_certified_panel(), AppConfig(), "fee-model")
    gate = ConfiguredAShareRiskGate(instruments=replay.instruments, ledger=replay.ledger)
    event = replay.events[0]
    gate.observe(event)
    order = SimpleNamespace(order_id="one-order")

    def fill(identity, quantity):
        return SimpleNamespace(
            fill_id=identity,
            instrument_id=event.instrument_id,
            account_id="test",
            event_time=event.available_at,
            liquidity_role=LiquidityRole.TAKER,
            side=Side.BUY,
            quantity=FixedPoint(quantity, 0),
            price=FixedPoint(10, 0),
        )

    first = gate.fee_for(fill("first", 100), order)
    assert decimal(first.amount) == 5
    assert gate.fee_for(fill("second", 100), order) is None
    checkpoint = gate.capture_state()
    third = gate.fee_for(fill("third", 20000), order)
    assert float(decimal(first.amount) + decimal(third.amount)) == pytest.approx(60.6)
    gate.restore_state(checkpoint)
    assert gate.fee_for(fill("third", 20000), order) == third
