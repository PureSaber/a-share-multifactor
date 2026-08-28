from dataclasses import replace

import pandas as pd
import pytest

from a_share_multifactor.calendar import (
    _retail_trade_dates,
    rebalance_dates,
    score_schedule_dates,
    trade_schedule_dates,
    trading_dates,
    weekly_dates,
)
from a_share_multifactor.config import AppConfig, CostsConfig
from a_share_multifactor.trade_ledger import (
    build_trade_ledger,
    capital_curve_from_returns,
    period_start_capitals,
)
from a_share_multifactor.trading_costs import (
    HoldingMeta,
    _buy_symbol,
    _early_exit_triggered,
    _holding_trading_days,
    _trading_day_index,
    _update_streak,
    build_symbol_ranks,
    buy_trade_cost,
    compute_period_return,
    estimate_leg_rebalance_cost,
    portfolio_value,
    retail_rebalance,
    retail_turnover,
    round_to_lots,
    select_retail_targets,
    sell_rank_limit,
    sell_trade_cost,
    simulate_daily_retail_portfolio,
)


def _retail_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    for day_index, day in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            score = symbol_index if day_index < len(dates) // 2 else 4 - symbol_index
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "close": 8.0 + symbol_index + day_index * 0.2,
                    "composite_score": float(score),
                    "period_return": 0.01 * (symbol_index + 1),
                }
            )
    return pd.DataFrame(rows)


def _retail_costs(**overrides: object) -> CostsConfig:
    values: dict[str, object] = {
        "retail_mode": True,
        "commission": 0.0003,
        "slippage": 0,
        "min_commission": 1,
        "stamp_tax": 0.0005,
        "lot_size": 100,
        "initial_capital": 20_000,
        "max_holdings": 1,
        "partial_rebalance": True,
        "trade_freq": "daily",
        "min_holding_days": 1,
    }
    values.update(overrides)
    return CostsConfig(**values)


def test_calendar_all_frequencies_and_invalid_values() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-02-03"]))
    assert len(trading_dates(dates)) == 4
    assert len(weekly_dates(dates)) == 3
    assert len(rebalance_dates(dates, "monthly")) == 2
    assert len(rebalance_dates(dates, "weekly")) == 3
    assert len(score_schedule_dates(dates, "monthly", True, "daily")) == 4
    assert len(trade_schedule_dates(dates, "monthly", True, "weekly")) == 3
    assert len(trade_schedule_dates(dates, "monthly", True, "monthly")) == 2
    assert len(score_schedule_dates(dates, "monthly", False, "daily")) == 2
    with pytest.raises(ValueError, match="Unsupported rebalance"):
        rebalance_dates(dates, "yearly")
    with pytest.raises(ValueError, match="Unsupported retail"):
        _retail_trade_dates(dates, "hourly")


def test_cost_model_edge_cases_and_full_rebalance() -> None:
    costs = _retail_costs()
    assert round_to_lots(9.9, 0) == 9
    assert buy_trade_cost(0, costs) == 0
    assert sell_trade_cost(-1, costs) == 0
    assert (
        portfolio_value(100, {"good": 10, "missing": 20, "zero": 5}, {"good": 2, "zero": 0}) == 120
    )

    candidates = pd.DataFrame({"symbol": ["A", "B", "C"], "score": [3.0, 2.0, 1.0]})
    assert select_retail_targets(candidates, "score", 2, 0, {"A": 1}, costs) == []
    assert select_retail_targets(
        candidates, "score", 2, 20_000, {"A": 10, "B": -1, "C": 20}, costs
    ) == ["A", "C"]
    assert build_symbol_ranks(candidates, "score") == {"A": 1, "B": 2, "C": 3}
    assert sell_rank_limit(replace(costs, max_holdings=0, rank_change_threshold=-3)) == 1
    assert estimate_leg_rebalance_cost({"A"}, set(), 10_000, costs) == 0
    full_costs = replace(costs, partial_rebalance=False)
    assert estimate_leg_rebalance_cost({"A"}, {"A", "B"}, 20_000, full_costs) > 0

    cash, shares, fee = _buy_symbol(50, "A", 100, 50, costs)
    assert (cash, shares, fee) == (50, 0, 0)
    partial = retail_rebalance(
        10_000,
        {"STALE": 100, "A": 100},
        {"A": 10, "B": 20},
        ["B"],
        costs,
    )
    assert "STALE" not in partial.holdings
    assert partial.sells[0].symbol == "A"
    assert "B" in partial.holdings

    full = retail_rebalance(
        10_000,
        {"A": 100, "BAD": 100},
        {"A": 10, "BAD": 0, "B": 20},
        ["B"],
        full_costs,
    )
    assert full.holdings == {"B": full.holdings["B"]}
    assert full.buys and full.sells
    assert compute_period_return(0, {}, {}, {}) == 0
    assert compute_period_return(100, {"A": 10}, {"A": 10}, {"A": 0.1}) > 0
    assert retail_turnover(set(), set(), True) == 0
    assert retail_turnover(set(), {"A"}, True) == 1
    assert retail_turnover({"A"}, {"A", "B"}, True) == 0.5
    assert retail_turnover({"A"}, {"A"}, False) == 1


def test_holding_exit_helpers_cover_streak_and_missing_dates() -> None:
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    day_index = _trading_day_index(dates)
    costs = _retail_costs(
        early_exit_enabled=True,
        early_exit_single_day_return=0.08,
        early_exit_cumulative_return=0.2,
        early_exit_consecutive_days=2,
        early_exit_consecutive_daily=0.03,
    )
    meta = HoldingMeta(dates[0], 10.0, 1.0)
    _update_streak(meta, 0.04, costs)
    _update_streak(meta, 0.04, costs)
    assert _early_exit_triggered(meta, 10.5, 0.01, costs)
    _update_streak(meta, -0.01, costs)
    assert meta.consecutive_up_days == 0
    assert _early_exit_triggered(meta, 12.1, 0.01, costs)
    assert _early_exit_triggered(meta, 10.1, 0.09, costs)
    assert not _early_exit_triggered(replace(meta, buy_price=0), 10.0, 0.1, costs)
    assert not _early_exit_triggered(meta, 12.0, 0.1, replace(costs, early_exit_enabled=False))
    assert _holding_trading_days(dates[0], dates[2], day_index) == 2
    assert _holding_trading_days(pd.Timestamp("2024-01-01"), dates[2], day_index) == 0


def test_retail_daily_and_monthly_ledgers_execute_real_state_transitions() -> None:
    daily_dates = pd.date_range("2025-01-02", periods=10, freq="B")
    daily_panel = _retail_panel(daily_dates)
    daily_config = AppConfig(
        quantiles=5,
        holding_period="rebalance",
        costs=_retail_costs(trade_freq="daily", min_holding_days=1),
    )
    starts = pd.Series(20_000.0, index=daily_dates)
    daily_ledger = build_trade_ledger(
        daily_panel,
        daily_config,
        starts,
        long_only=True,
    )
    assert not daily_ledger.empty
    assert set(daily_ledger["exit_reason"]).issubset({"signal_exit", "early_exit", "open"})

    daily_returns = simulate_daily_retail_portfolio(
        daily_panel,
        daily_config,
        "composite_score",
        5,
        daily_dates,
    )
    assert not daily_returns.empty

    monthly_dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    monthly_panel = _retail_panel(pd.DatetimeIndex(monthly_dates))
    monthly_config = replace(
        daily_config,
        rebalance_freq="monthly",
        costs=replace(daily_config.costs, trade_freq="monthly"),
    )
    monthly_starts = pd.Series([20_000.0, 20_100.0, 20_200.0], index=monthly_dates)
    monthly_ledger = build_trade_ledger(
        monthly_panel,
        monthly_config,
        monthly_starts,
        long_only=True,
    )
    assert not monthly_ledger.empty
    assert monthly_ledger["buy_cost"].ge(0).all()

    with pytest.raises(ValueError, match="Score column"):
        build_trade_ledger(
            monthly_panel.drop(columns="composite_score"), monthly_config, monthly_starts
        )
    with pytest.raises(ValueError, match="Return column"):
        build_trade_ledger(
            monthly_panel.drop(columns="period_return"), monthly_config, monthly_starts
        )
    assert capital_curve_from_returns(pd.Series(dtype=float), 10_000).empty
    assert period_start_capitals(pd.Series(dtype=float), 10_000).empty
