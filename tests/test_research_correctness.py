"""Guards for biases that used to make backtests look better than the market."""

from pathlib import Path

import pandas as pd
import pytest

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import AppConfig, CostsConfig, DataPaths, FilterConfig
from a_share_multifactor.data_loader import (
    apply_tradability_filters,
    build_dataset,
    load_benchmark_returns,
    merge_earnings_with_max_age,
    save_parquet,
)
from a_share_multifactor.ic_analysis import summarize_ic
from a_share_multifactor.neutralize import neutralize_cross_section
from a_share_multifactor.preprocess import add_period_return, standardize_cross_section
from a_share_multifactor.synthesis import equal_weight_score
from a_share_multifactor.trading_costs import sell_trade_cost


def _bars(symbol: str, dates: list[str], closes: list[float], *, limit_up_on: set[str] | None = None):
    rows = []
    sealed = limit_up_on or set()
    for day, close in zip(dates, closes, strict=False):
        at_limit = day in sealed
        rows.append(
            {
                "symbol": symbol,
                "date": pd.Timestamp(day),
                "open": close,
                "high": close if at_limit else close + 1,
                "low": close if at_limit else close - 1,
                "close": close,
                "volume": 1000,
            }
        )
    return rows


def test_limit_up_is_not_bought_at_the_next_open() -> None:
    dates = ["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"]
    df = pd.DataFrame(
        _bars("000001", dates, [100.0, 110.0, 115.0, 121.0], limit_up_on={"2020-02-03"})
    )
    result = add_period_return(df, rebalance_dates(df["date"], "monthly"))
    january = result.loc[result["date"] == pd.Timestamp("2020-01-31"), "period_return"]
    assert january.isna().all()


def test_missing_st_identity_is_an_error() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["000001"],
            "date": pd.to_datetime(["2020-01-02"]),
            "close": [10.0],
        }
    )
    with pytest.raises(ValueError, match="exclude_st"):
        apply_tradability_filters(panel, AppConfig(filters=FilterConfig(min_list_days=0)))


def test_missing_industry_neutralization_is_an_error() -> None:
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-02"] * 2), "factor": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="industry"):
        neutralize_cross_section(df, ["factor"], by=["industry"])


def test_standardize_keeps_missing_factor_values() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 3),
            "factor": [1.0, pd.NA, 3.0],
        }
    )
    result = standardize_cross_section(df, ["factor"])
    assert pd.isna(result.loc[1, "factor"])
    scored = equal_weight_score(
        pd.DataFrame({"f1": [1.0], "f2": [pd.NA]}),
        ["f1", "f2"],
    )
    assert scored.loc[0, "composite_score"] == pytest.approx(1.0)


def test_forecast_score_expires() -> None:
    dates = pd.bdate_range("2023-01-02", periods=80)
    panel = pd.DataFrame({"symbol": ["000001"] * len(dates), "date": dates, "close": 10.0})
    forecasts = pd.DataFrame(
        {
            "symbol": ["000001"],
            "effective_date": [dates[0]],
            "forecast_score": [2],
        }
    )
    merged = merge_earnings_with_max_age(panel, forecasts, max_age_days=20)
    assert merged.loc[merged["date"] == dates[5], "forecast_score"].iloc[0] == 2
    assert pd.isna(merged.loc[merged["date"] == dates[40], "forecast_score"].iloc[0])


def test_historical_universe_requires_prices_for_departed_names(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    prices = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": dates,
            "name": ["平安银行", "平安银行"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.8, 9.9],
            "close": [10.1, 10.2],
            "volume": [1000, 1000],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": dates,
            "available_at": dates,
            "market_cap": [1.0, 1.0],
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["000001", "000009"],
            "date": dates,
            "in_universe": [1, 1],
        }
    )
    save_parquet(prices, tmp_path / "prices.parquet")
    save_parquet(fundamentals, tmp_path / "fundamentals.parquet")
    save_parquet(universe, tmp_path / "universe.parquet")
    config = AppConfig(
        start_date="2020-01-02",
        end_date="2020-01-03",
        filters=FilterConfig(use_historical_universe=True, min_list_days=0),
        data=DataPaths(
            price="prices.parquet",
            fundamentals="fundamentals.parquet",
            universe="universe.parquet",
        ),
    )
    with pytest.raises(ValueError, match="000009"):
        build_dataset(config, data_dir=tmp_path, include_alt=False)


def test_price_index_benchmark_is_rejected(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    save_parquet(
        pd.DataFrame({"date": dates, "benchmark_return": [0.01, -0.01]}),
        tmp_path / "benchmark.parquet",
    )
    config = AppConfig(
        start_date="2020-01-02",
        end_date="2020-01-03",
        data=DataPaths(benchmark="benchmark.parquet"),
    )
    with pytest.raises(ValueError, match="H00300"):
        load_benchmark_returns(config, data_dir=tmp_path)


def test_benchmark_month_end_follows_the_trading_calendar(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2025-01-27", "2025-02-05", "2025-02-28"])
    save_parquet(
        pd.DataFrame(
            {
                "date": dates,
                "benchmark_return": [0.01, 0.02, -0.01],
                "benchmark_kind": ["total_return"] * 3,
                "benchmark_symbol": ["H00300"] * 3,
            }
        ),
        tmp_path / "benchmark.parquet",
    )
    config = AppConfig(
        start_date="2025-01-01",
        end_date="2025-02-28",
        rebalance_freq="monthly",
        data=DataPaths(benchmark="benchmark.parquet"),
    )
    periods = load_benchmark_returns(config, data_dir=tmp_path)
    assert list(periods.index) == [pd.Timestamp("2025-01-27")]
    assert periods.iloc[0] == pytest.approx(1.02 * 0.99 - 1)


def test_stamp_duty_changes_on_2023_08_28() -> None:
    costs = CostsConfig(
        commission=0.0,
        slippage=0.0,
        min_commission=0.0,
        stamp_tax=0.0005,
        statutory_stamp_tax=True,
    )
    assert sell_trade_cost(10_000, costs, trade_date="2023-08-25") == pytest.approx(10.0)
    assert sell_trade_cost(10_000, costs, trade_date="2023-08-28") == pytest.approx(5.0)


def test_ic_summary_includes_t_stat() -> None:
    summary = summarize_ic(pd.Series([0.1, 0.2, 0.0, 0.1]))
    assert summary["n_obs"] == 4
    assert summary["ic_tstat"] > 0
