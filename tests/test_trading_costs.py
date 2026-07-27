import pandas as pd
import pytest

from a_share_multifactor.config import CostsConfig
from a_share_multifactor.trading_costs import (
    HoldingMeta,
    buy_trade_cost,
    estimate_leg_rebalance_cost,
    retail_daily_step,
    retail_rebalance,
    retail_turnover,
    round_to_lots,
    select_retail_targets,
    sell_trade_cost,
    simulate_long_only_rebalance,
    _trading_day_index,
)


def test_round_to_lots() -> None:
    assert round_to_lots(250, 100) == 200
    assert round_to_lots(99, 100) == 0
    assert round_to_lots(100, 100) == 100


def test_min_commission_on_small_trade() -> None:
    costs = CostsConfig(commission=0.0003, min_commission=5.0, slippage=0.0)
    assert buy_trade_cost(1000, costs) == pytest.approx(5.0)
    assert sell_trade_cost(1000, costs) == pytest.approx(5.5)


def test_stamp_tax_only_on_sell() -> None:
    costs = CostsConfig(commission=0.0, slippage=0.0, min_commission=0.0, stamp_tax=0.0005)
    assert buy_trade_cost(10_000, costs) == 0.0
    assert sell_trade_cost(10_000, costs) == pytest.approx(5.0)


def test_estimate_leg_rebalance_cost_counts_trades() -> None:
    costs = CostsConfig(
        commission=0.0,
        slippage=0.0,
        min_commission=5.0,
        stamp_tax=0.0,
    )
    prev = {"000001", "000002", "000003"}
    curr = {"000002", "000003", "000004"}
    # sell 000001, buy 000004 => 2 trades * 5 yuan
    assert estimate_leg_rebalance_cost(prev, curr, 30_000, costs) == pytest.approx(10.0)


def test_simulate_long_only_rebalance_respects_lot_size() -> None:
    costs = CostsConfig(
        retail_mode=True,
        commission=0.0,
        slippage=0.0,
        min_commission=0.0,
        stamp_tax=0.0,
        lot_size=100,
        initial_capital=10_000,
    )
    cash = 10_000.0
    holdings: dict[str, int] = {}
    prices = {"000001": 50.0, "000002": 200.0}
    period_returns = {"000001": 0.1, "000002": 0.0}

    period_return, new_holdings, end_cash, _ = simulate_long_only_rebalance(
        cash=cash,
        holdings=holdings,
        prices=prices,
        period_returns=period_returns,
        target_symbols=["000001", "000002"],
        costs=costs,
    )

    assert new_holdings["000001"] == 100
    assert "000002" not in new_holdings
    assert end_cash == pytest.approx(10_000 - 5_000)
    assert period_return > 0


def test_partial_rebalance_reduces_trades() -> None:
    costs = CostsConfig(
        retail_mode=True,
        commission=0.0,
        slippage=0.0,
        min_commission=0.0,
        stamp_tax=0.0,
        lot_size=100,
        partial_rebalance=True,
    )
    holdings = {"000001": 100}
    cash = 5_000.0
    prices = {"000001": 10.0, "000002": 10.0}

    result = retail_rebalance(
        cash=cash,
        holdings=holdings.copy(),
        prices=prices,
        target_symbols=["000001"],
        costs=costs,
    )
    assert result.holdings == {"000001": 100}
    assert result.trade_cost == 0.0
    assert retail_turnover({"000001"}, {"000001"}, True) == 0.0


def test_select_retail_targets_caps_holdings() -> None:
    import pandas as pd

    costs = CostsConfig(lot_size=100, max_holdings=2)
    candidates = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003"],
            "composite_score": [3.0, 2.0, 1.0],
        }
    )
    prices = {"000001": 5.0, "000002": 8.0, "000003": 50.0}
    targets = select_retail_targets(candidates, "composite_score", 2, 10_000, prices, costs)
    assert targets == ["000001", "000002"]


def _daily_costs(**overrides: object) -> CostsConfig:
    base = dict(
        retail_mode=True,
        commission=0.0,
        slippage=0.0,
        min_commission=0.0,
        stamp_tax=0.0,
        lot_size=100,
        initial_capital=10_000,
        max_holdings=2,
        partial_rebalance=True,
        trade_freq="daily",
        min_holding_days=3,
        early_exit_single_day_return=0.08,
        early_exit_consecutive_days=3,
        early_exit_consecutive_daily=0.03,
        early_exit_cumulative_return=0.25,
    )
    base.update(overrides)
    return CostsConfig(**base)


def test_retail_daily_step_blocks_signal_sell_before_min_hold() -> None:
    costs = _daily_costs(min_holding_days=5)
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    day_index = _trading_day_index(pd.DatetimeIndex(dates))
    buy_date = dates[0]
    current_date = dates[2]

    holdings = {"000001": 100}
    meta = {
        "000001": HoldingMeta(buy_date=buy_date, buy_price=10.0, buy_cost=0.0),
    }
    prices = {"000001": 10.0}
    prev_prices = {"000001": 10.0}

    result = retail_daily_step(
        cash=9_000.0,
        holdings=holdings.copy(),
        meta=meta.copy(),
        prices=prices,
        prev_prices=prev_prices,
        target_symbols=[],
        costs=costs,
        current_date=current_date,
        day_index=day_index,
    )

    assert result.holdings == {"000001": 100}
    assert result.sells == []


def test_retail_daily_step_early_exit_on_big_single_day_gain() -> None:
    costs = _daily_costs(min_holding_days=20)
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    day_index = _trading_day_index(pd.DatetimeIndex(dates))
    buy_date = dates[0]
    current_date = dates[1]

    holdings = {"000001": 100}
    meta = {
        "000001": HoldingMeta(buy_date=buy_date, buy_price=10.0, buy_cost=0.0),
    }
    prices = {"000001": 10.9}
    prev_prices = {"000001": 10.0}

    result = retail_daily_step(
        cash=0.0,
        holdings=holdings.copy(),
        meta=meta.copy(),
        prices=prices,
        prev_prices=prev_prices,
        target_symbols=[],
        costs=costs,
        current_date=current_date,
        day_index=day_index,
    )

    assert "000001" not in result.holdings
    assert len(result.sells) == 1
    assert result.sells[0].exit_reason == "early_exit"


def test_retail_daily_step_allows_signal_sell_after_min_hold() -> None:
    costs = _daily_costs(min_holding_days=2)
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    day_index = _trading_day_index(pd.DatetimeIndex(dates))
    buy_date = dates[0]
    current_date = dates[2]

    holdings = {"000001": 100}
    meta = {
        "000001": HoldingMeta(buy_date=buy_date, buy_price=10.0, buy_cost=0.0),
    }
    prices = {"000001": 10.0}
    prev_prices = {"000001": 10.0}

    result = retail_daily_step(
        cash=0.0,
        holdings=holdings.copy(),
        meta=meta.copy(),
        prices=prices,
        prev_prices=prev_prices,
        target_symbols=[],
        costs=costs,
        current_date=current_date,
        day_index=day_index,
    )

    assert "000001" not in result.holdings
    assert len(result.sells) == 1
    assert result.sells[0].exit_reason == "signal_exit"


def test_retail_daily_step_rank_buffer_delays_signal_sell() -> None:
    costs = _daily_costs(min_holding_days=2, rank_change_threshold=5, max_holdings=10)
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    day_index = _trading_day_index(pd.DatetimeIndex(dates))
    buy_date = dates[0]
    current_date = dates[2]

    holdings = {"000001": 100}
    meta = {
        "000001": HoldingMeta(buy_date=buy_date, buy_price=10.0, buy_cost=0.0),
    }
    prices = {"000001": 10.0}
    prev_prices = {"000001": 10.0}
    # rank 12 is within top-10 + buffer(5)
    symbol_ranks = {"000001": 12}

    result = retail_daily_step(
        cash=0.0,
        holdings=holdings.copy(),
        meta=meta.copy(),
        prices=prices,
        prev_prices=prev_prices,
        target_symbols=[],
        costs=costs,
        current_date=current_date,
        day_index=day_index,
        symbol_ranks=symbol_ranks,
    )

    assert result.holdings == {"000001": 100}
    assert result.sells == []


def test_early_exit_disabled() -> None:
    costs = _daily_costs(
        min_holding_days=20,
        early_exit_enabled=False,
        early_exit_single_day_return=0.01,
    )
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    day_index = _trading_day_index(pd.DatetimeIndex(dates))
    holdings = {"000001": 100}
    meta = {"000001": HoldingMeta(buy_date=dates[0], buy_price=10.0, buy_cost=0.0)}
    prices = {"000001": 11.5}
    prev_prices = {"000001": 10.0}

    result = retail_daily_step(
        cash=0.0,
        holdings=holdings.copy(),
        meta=meta.copy(),
        prices=prices,
        prev_prices=prev_prices,
        target_symbols=[],
        costs=costs,
        current_date=dates[1],
        day_index=day_index,
    )

    assert result.holdings == {"000001": 100}
    assert result.sells == []
