import pandas as pd

from a_share_multifactor.config import AppConfig
from a_share_multifactor.quantile_backtest import run_long_only_backtest, run_quantile_backtest


def _make_panel() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    for date in dates:
        for i in range(10):
            score = float(i)
            period_ret = score / 100.0
            rows.append(
                {
                    "date": date,
                    "symbol": f"{i:06d}",
                    "composite_score": score,
                    "period_return": period_ret,
                }
            )
    return pd.DataFrame(rows)


def test_quantile_backtest_ordering() -> None:
    panel = _make_panel()
    config = AppConfig(quantiles=5, rebalance_freq="monthly", holding_period="rebalance")
    result = run_quantile_backtest(panel, config)

    assert not result.quantile_returns.empty
    assert "Q5" in result.quantile_returns.columns
    assert result.quantile_returns["Q5"].mean() > result.quantile_returns["Q1"].mean()
    assert not result.long_short.empty
    assert not result.turnover.empty


def test_quantile_backtest_with_costs_and_benchmark() -> None:
    panel = _make_panel()
    config = AppConfig(quantiles=5, rebalance_freq="monthly", holding_period="rebalance")
    benchmark = pd.Series(
        [0.0, 0.01, 0.01],
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )
    result = run_quantile_backtest(panel, config, benchmark_returns=benchmark)
    assert not result.excess_returns.empty
    assert "long_short_excess" in result.stats["portfolio"].values


def test_long_only_backtest() -> None:
    panel = _make_panel()
    config = AppConfig(quantiles=5, rebalance_freq="monthly", holding_period="rebalance")
    result = run_long_only_backtest(panel, config)

    assert not result.quantile_returns.empty
    assert "Q5" in result.quantile_returns.columns
    assert "Q1" not in result.quantile_returns.columns
    assert not result.long_short.empty
    assert result.long_short.equals(result.quantile_returns["Q5"])
    assert "long_only" in result.stats["portfolio"].values


def test_long_only_backtest_retail_mode() -> None:
    panel = _make_panel()
    panel["close"] = 10.0
    from a_share_multifactor.config import CostsConfig

    config = AppConfig(
        quantiles=5,
        rebalance_freq="monthly",
        holding_period="rebalance",
        costs=CostsConfig(
            retail_mode=True,
            commission=0.0,
            slippage=0.0,
            min_commission=0.0,
            stamp_tax=0.0,
            lot_size=100,
            initial_capital=10_000,
        ),
    )
    result = run_long_only_backtest(panel, config)
    assert not result.long_short.empty
