from pathlib import Path

import pandas as pd

from a_share_multifactor.backtest import run_pipeline, write_outputs
from a_share_multifactor.config import AppConfig, DataPaths, FilterConfig
from a_share_multifactor.data_loader import save_parquet
from a_share_multifactor.quantile_backtest import BacktestResult, run_quantile_backtest


def test_write_outputs(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "symbol": ["A", "B"],
            "composite_score": [1.0, 2.0],
            "period_return": [0.01, 0.02],
        }
    )
    config = AppConfig(
        outputs_dir=str(tmp_path / "outputs"),
        holding_period="rebalance",
    )
    results = run_quantile_backtest(panel, config)
    ic_report = pd.DataFrame({"factor": ["momentum_20d"], "mean_ic": [0.1]})
    out_dir = write_outputs(results, ic_report, config)
    assert (out_dir / "ic_summary.csv").exists()
    assert (tmp_path / "outputs" / "latest" / "report.html").exists()


def test_run_pipeline(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    symbols = ["000001", "000002"]
    rows = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": 10 + i * 0.1,
                    "high": 10.5 + i * 0.1,
                    "low": 9.5 + i * 0.1,
                    "close": 10 + i * 0.1,
                    "volume": 1000,
                    "market_cap": 1e10,
                    "pe_ratio": 10.0,
                    "pb_ratio": 2.0,
                }
            )
    prices = pd.DataFrame(rows)
    fundamentals = prices[["symbol", "date", "market_cap", "pe_ratio", "pb_ratio"]].copy()
    fundamentals["available_at"] = fundamentals["date"]
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": [0.001] * len(dates),
        }
    )
    save_parquet(prices, tmp_path / "prices.parquet")
    save_parquet(fundamentals, tmp_path / "fundamentals.parquet")
    save_parquet(benchmark, tmp_path / "benchmark.parquet")

    config = AppConfig(
        start_date="2020-01-01",
        end_date="2020-03-01",
        outputs_dir=str(tmp_path / "outputs"),
        factors=["market_cap", "pe_ratio", "momentum_20d"],
        filters=FilterConfig(use_historical_universe=False, min_list_days=0),
        data=DataPaths(
            price="prices.parquet",
            fundamentals="fundamentals.parquet",
            benchmark="benchmark.parquet",
        ),
    )
    results, ic_report, output_dir = run_pipeline(config, data_dir=tmp_path)
    assert isinstance(results, BacktestResult)
    assert not ic_report.empty
    assert output_dir.exists()
