"""Seed minimal alt-data Parquet files for offline backtest smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_data_kit.panel import (
    add_industry_relative_strength,
    merge_earnings_to_panel,
    merge_northbound_to_panel,
)
from quant_data_kit.providers.industry import fetch_industry_returns
from quant_data_kit.storage import load_parquet, save_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed alt-data parquet from cached prices")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    args = parser.parse_args()

    price_path = args.data_dir / "cn_a/daily/prices.parquet"
    if not price_path.exists():
        raise SystemExit(f"Missing price cache: {price_path}")

    prices = load_parquet(price_path)
    symbols = sorted(prices["symbol"].unique())
    industries = sorted(prices["industry"].dropna().unique()) if "industry" in prices.columns else ["未知"]

    earnings = pd.DataFrame(
        {
            "symbol": symbols[:1],
            "report_period": ["20230331"],
            "announce_date": pd.to_datetime(["2023-04-15"]),
            "effective_date": pd.to_datetime(["2023-04-17"]),
            "forecast_type": ["预增"],
            "forecast_score": [2],
            "change_pct_low": [50],
            "change_pct_high": [80],
        }
    )

    dates = prices["date"].drop_duplicates().sort_values()
    northbound_rows = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            northbound_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "northbound_hold_ratio": 1.0 + i * 0.001,
                }
            )
    northbound = pd.DataFrame(northbound_rows)

    industry_returns = fetch_industry_returns(
        list(industries)[:3],
        str(dates.min().date()),
        str(dates.max().date()),
        fetch_fn=lambda industry, start, end: pd.DataFrame(
            {
                "date": dates,
                "close": 100 + pd.Series(range(len(dates))).values,
            }
        ),
        sleep_seconds=0,
    )

    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": [0.0005] * len(dates),
        }
    )

    save_parquet(earnings, args.data_dir / "cn_a/alt/earnings_forecast.parquet")
    save_parquet(northbound, args.data_dir / "cn_a/alt/northbound_holdings.parquet")
    save_parquet(industry_returns, args.data_dir / "cn_a/alt/industry_returns.parquet")
    save_parquet(benchmark, args.data_dir / "cn_a/benchmark/hs300_index.parquet")

    panel = merge_earnings_to_panel(prices, earnings)
    panel = merge_northbound_to_panel(panel, northbound)
    panel = add_industry_relative_strength(
        panel,
        industry_returns,
        benchmark.set_index("date")["benchmark_return"],
        window=20,
    )
    print(f"Seeded alt data for {len(symbols)} symbols, {len(panel)} panel rows")


if __name__ == "__main__":
    main()
