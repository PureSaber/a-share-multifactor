import pandas as pd

from a_share_multifactor.config import AppConfig, DataPaths, FilterConfig
from a_share_multifactor.data_loader import build_dataset
from a_share_multifactor.factors import compute_factors
from quant_data_kit.panel import (
    add_industry_relative_strength,
    merge_earnings_to_panel,
    merge_northbound_to_panel,
)
from quant_data_kit.storage import save_parquet


def test_alt_merge_columns(tmp_path) -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["000001"] * 30,
            "date": pd.date_range("2023-01-01", periods=30, freq="B"),
            "close": [10.0 + i * 0.1 for i in range(30)],
            "volume": [1000] * 30,
            "market_cap": [1e10] * 30,
            "pe_ratio": [10.0] * 30,
            "pb_ratio": [2.0] * 30,
            "industry": ["银行"] * 30,
        }
    )
    earnings = pd.DataFrame(
        {
            "symbol": ["000001"],
            "report_period": ["20230331"],
            "announce_date": pd.to_datetime(["2023-04-15"]),
            "effective_date": pd.to_datetime(["2023-04-17"]),
            "forecast_type": ["预增"],
            "forecast_score": [2],
            "change_pct_low": [50],
            "change_pct_high": [80],
        }
    )
    northbound = pd.DataFrame(
        {
            "symbol": ["000001"] * 10,
            "date": pd.date_range("2023-01-01", periods=10, freq="B"),
            "northbound_hold_ratio": [1.0 + i * 0.01 for i in range(10)],
        }
    )
    industry_returns = pd.DataFrame(
        {
            "industry": ["银行"] * 30,
            "date": pd.date_range("2023-01-01", periods=30, freq="B"),
            "industry_return": [0.001] * 30,
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=30, freq="B"),
            "benchmark_return": [0.0005] * 30,
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_parquet(panel, data_dir / "cn_a/daily/prices.parquet")
    save_parquet(panel[["symbol", "date", "market_cap", "pe_ratio", "pb_ratio"]], data_dir / "cn_a/fundamentals.parquet")
    save_parquet(earnings, data_dir / "cn_a/alt/earnings_forecast.parquet")
    save_parquet(northbound, data_dir / "cn_a/alt/northbound_holdings.parquet")
    save_parquet(industry_returns, data_dir / "cn_a/alt/industry_returns.parquet")
    save_parquet(benchmark, data_dir / "cn_a/benchmark/hs300_index.parquet")

    merged = merge_earnings_to_panel(panel, earnings)
    merged = merge_northbound_to_panel(merged, northbound)
    merged = add_industry_relative_strength(
        merged,
        industry_returns,
        benchmark.set_index("date")["benchmark_return"],
        window=5,
    )

    factors = compute_factors(
        merged,
        factor_names=[
            "forecast_score",
            "northbound_chg_5d",
            "industry_rs_20d",
            "momentum_20d",
        ],
    )
    assert "forecast_score" in factors.columns
    assert "northbound_chg_5d" in factors.columns
    assert "industry_rs_20d" in factors.columns


def test_build_dataset_without_alt(tmp_path) -> None:
    sample_prices = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002", "000002"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]),
            "open": [10.0, 10.1, 20.0, 20.2],
            "high": [10.5, 10.6, 20.5, 20.6],
            "low": [9.8, 9.9, 19.8, 19.9],
            "close": [10.2, 10.3, 20.1, 20.3],
            "volume": [1000, 1100, 2000, 2100],
        }
    )
    sample_fundamentals = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002", "000002"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]),
            "market_cap": [1e10, 1.01e10, 2e10, 2.01e10],
            "pe_ratio": [8.0, 8.1, 12.0, 12.1],
        }
    )
    data_dir = tmp_path
    save_parquet(sample_prices, data_dir / "cn_a/daily/prices.parquet")
    save_parquet(sample_fundamentals, data_dir / "cn_a/fundamentals.parquet")

    config = AppConfig(
        start_date="2020-01-02",
        end_date="2020-01-03",
        filters=FilterConfig(use_historical_universe=False, min_list_days=0),
        data=DataPaths(
            price="cn_a/daily/prices.parquet",
            fundamentals="cn_a/fundamentals.parquet",
        ),
    )
    panel = build_dataset(config, data_dir=data_dir, include_alt=False)
    assert len(panel) == 4
