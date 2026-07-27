import pandas as pd
import pytest

from a_share_multifactor.config import AppConfig, FilterConfig
from a_share_multifactor.trade_ledger import (
    build_trade_ledger,
    capital_curve_from_returns,
    period_start_capitals,
)


def test_capital_curve_from_returns() -> None:
    returns = pd.Series(
        [0.1, -0.05, 0.02], index=pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    )
    capital = capital_curve_from_returns(returns, 1_000_000)
    assert len(capital) == 3
    assert capital.iloc[0] == pytest.approx(1_100_000)
    assert capital.iloc[1] == pytest.approx(1_045_000)


def test_period_start_capitals() -> None:
    returns = pd.Series([0.1, -0.05], index=pd.to_datetime(["2020-01-31", "2020-02-28"]))
    starts = period_start_capitals(returns, 1_000_000)
    assert starts.iloc[0] == 1_000_000
    assert starts.iloc[1] == 1_100_000


def test_build_trade_ledger_basic() -> None:
    config = AppConfig(
        quantiles=5,
        holding_period="rebalance",
        filters=FilterConfig(use_historical_universe=False, min_list_days=0),
    )
    dates = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    rows = []
    for date in dates:
        for i, symbol in enumerate(["000001", "000002", "000003", "000004", "000005"]):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": 10.0 + i,
                    "composite_score": float(i),
                    "period_return": 0.01 * (i + 1),
                }
            )
    panel = pd.DataFrame(rows)
    starts = pd.Series([1_000_000, 1_010_000], index=dates[:2])
    ledger = build_trade_ledger(panel, config, starts, name_map={"000005": "测试五"})
    assert not ledger.empty
    assert "buy" in ledger["action_open"].values
    assert "sell_short" in ledger["action_open"].values
    assert (ledger["holding_days"] == 28).any()
    assert (ledger["name"] == "测试五").any()


def test_build_trade_ledger_long_only() -> None:
    config = AppConfig(
        quantiles=5,
        holding_period="rebalance",
        filters=FilterConfig(use_historical_universe=False, min_list_days=0),
    )
    dates = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    rows = []
    for date in dates:
        for i, symbol in enumerate(["000001", "000002", "000003", "000004", "000005"]):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": 10.0 + i,
                    "composite_score": float(i),
                    "period_return": 0.01 * (i + 1),
                }
            )
    panel = pd.DataFrame(rows)
    starts = pd.Series([10_000, 10_100], index=dates[:2])
    ledger = build_trade_ledger(panel, config, starts, long_only=True)
    assert not ledger.empty
    assert (ledger["side"] == "long").all()
    assert "sell_short" not in ledger["action_open"].values
    assert ledger["capital_allocated"].sum() > 0
