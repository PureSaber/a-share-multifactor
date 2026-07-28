"""CLI to fetch and cache market data from AKShare via quant-data-kit."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from quant_data_kit.providers.benchmark import fetch_hs300_benchmark
from quant_data_kit.providers.earnings_forecast import fetch_earnings_forecasts
from quant_data_kit.providers.fundamentals import fetch_fundamentals
from quant_data_kit.providers.industry import fetch_industry_returns
from quant_data_kit.providers.northbound import fetch_northbound_holdings
from quant_data_kit.providers.prices import fetch_daily_prices
from quant_data_kit.providers.universe import (
    fetch_hs300_constituents,
    fetch_hs300_constituents_history,
)

from a_share_multifactor.config import load_config
from a_share_multifactor.data_loader import (
    build_dataset,
    incremental_start_date,
    load_parquet,
    save_parquet,
    should_refresh_cache,
)

logger = logging.getLogger(__name__)


def _fetch_alt_data(
    config,
    args,
    symbols: list[str],
    data_dir: Path,
) -> None:
    earnings_path = data_dir / config.data.earnings_forecast
    if args.force or not earnings_path.exists():
        logger.info("Fetching earnings forecasts...")
        earnings = fetch_earnings_forecasts(
            config.start_date,
            config.end_date,
            sleep_seconds=config.fetch.sleep_seconds,
        )
        save_parquet(earnings, earnings_path)
        logger.info("Saved earnings forecasts: %s rows", len(earnings))

    northbound_path = data_dir / config.data.northbound
    if args.force or not northbound_path.exists():
        logger.info("Fetching northbound holdings for %s symbols...", len(symbols))
        northbound = fetch_northbound_holdings(
            symbols,
            sleep_seconds=config.fetch.sleep_seconds,
            max_workers=config.fetch.max_workers,
        )
        save_parquet(northbound, northbound_path)
        logger.info("Saved northbound: %s rows", len(northbound))

    industry_path = data_dir / config.data.industry_returns
    if args.force or not industry_path.exists():
        price_path = data_dir / config.data.price
        industries: list[str] = []
        if price_path.exists():
            prices = load_parquet(price_path)
            if "industry" in prices.columns:
                industries = sorted(prices["industry"].dropna().unique().tolist())
        if industries:
            logger.info("Fetching industry returns for %s industries...", len(industries))
            industry_returns = fetch_industry_returns(
                industries,
                config.start_date,
                config.end_date,
                sleep_seconds=config.fetch.sleep_seconds,
            )
            save_parquet(industry_returns, industry_path)
            logger.info("Saved industry returns: %s rows", len(industry_returns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch A-share data and cache as Parquet")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--force", action="store_true", help="Force refresh even if cache exists")
    parser.add_argument("--symbols-limit", type=int, default=0, help="Limit symbols for debugging")
    parser.add_argument("--fetch-alt", action="store_true", help="Also fetch alt data (earnings, northbound, industry)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = load_config(args.config)
    price_path = args.data_dir / config.data.price
    fundamentals_path = args.data_dir / config.data.fundamentals
    universe_path = args.data_dir / config.data.universe
    benchmark_path = args.data_dir / config.data.benchmark

    price_start = (
        config.start_date if args.force else incremental_start_date(price_path, config.start_date)
    )

    refresh_prices = args.force or should_refresh_cache(
        price_path, config.start_date, config.end_date
    )
    refresh_fundamentals = args.force or should_refresh_cache(
        fundamentals_path, config.start_date, config.end_date
    )
    refresh_universe = args.force or not universe_path.exists()
    refresh_benchmark = args.force or should_refresh_cache(
        benchmark_path, config.start_date, config.end_date
    )

    symbols: list[str] = []

    if refresh_prices:
        logger.info("Fetching HS300 constituents and daily prices...")
        symbols = fetch_hs300_constituents()
        if args.symbols_limit > 0:
            symbols = symbols[: args.symbols_limit]
        prices = fetch_daily_prices(
            symbols,
            price_start,
            config.end_date,
            sleep_seconds=config.fetch.sleep_seconds,
            max_workers=config.fetch.max_workers,
            max_retries=config.fetch.max_retries,
        )
        if not args.force and price_path.exists():
            existing = load_parquet(price_path)
            prices = (
                pd.concat([existing, prices], ignore_index=True)
                .drop_duplicates(subset=["symbol", "date"])
                .sort_values(["symbol", "date"])
            )
        save_parquet(prices, price_path)
        logger.info("Saved prices: %s (%s rows)", price_path, len(prices))
    else:
        logger.info("Price cache up to date: %s", price_path)

    if refresh_fundamentals:
        if refresh_prices:
            symbols = sorted(prices["symbol"].unique().tolist())  # type: ignore[name-defined]
        else:
            symbols = fetch_hs300_constituents()
            if args.symbols_limit > 0:
                symbols = symbols[: args.symbols_limit]
        fund_start = (
            config.start_date
            if args.force
            else incremental_start_date(fundamentals_path, config.start_date)
        )
        logger.info("Fetching fundamentals from %s...", fund_start)
        fundamentals = fetch_fundamentals(
            symbols,
            fund_start,
            config.end_date,
            sleep_seconds=config.fetch.sleep_seconds,
            max_workers=config.fetch.max_workers,
            max_retries=config.fetch.max_retries,
        )
        if not args.force and fundamentals_path.exists():
            existing = load_parquet(fundamentals_path)
            fundamentals = (
                pd.concat([existing, fundamentals], ignore_index=True)
                .drop_duplicates(subset=["symbol", "date"])
                .sort_values(["symbol", "date"])
            )
        save_parquet(fundamentals, fundamentals_path)
        logger.info("Saved fundamentals: %s (%s rows)", fundamentals_path, len(fundamentals))
    else:
        logger.info("Fundamentals cache up to date: %s", fundamentals_path)

    if refresh_universe and config.filters.use_historical_universe:
        logger.info("Building historical universe membership...")
        universe = fetch_hs300_constituents_history(config.start_date, config.end_date)
        save_parquet(universe, universe_path)
        logger.info("Saved universe: %s (%s rows)", universe_path, len(universe))

    if refresh_benchmark:
        logger.info("Fetching benchmark index returns...")
        bench_start = (
            config.start_date
            if args.force
            else incremental_start_date(benchmark_path, config.start_date)
        )
        benchmark = fetch_hs300_benchmark(bench_start, config.end_date)
        if not args.force and benchmark_path.exists():
            existing = load_parquet(benchmark_path)
            benchmark = (
                pd.concat([existing, benchmark], ignore_index=True)
                .drop_duplicates(subset=["date"])
                .sort_values("date")
            )
        save_parquet(benchmark, benchmark_path)
        logger.info("Saved benchmark: %s (%s rows)", benchmark_path, len(benchmark))

    if not symbols and price_path.exists():
        symbols = sorted(load_parquet(price_path)["symbol"].unique().tolist())
        if args.symbols_limit > 0:
            symbols = symbols[: args.symbols_limit]

    if args.fetch_alt and symbols:
        _fetch_alt_data(config, args, symbols, args.data_dir)

    panel = build_dataset(config, data_dir=args.data_dir, force_refresh=False, include_alt=args.fetch_alt)
    logger.info(
        "Dataset ready: %s rows, %s symbols, %s to %s",
        len(panel),
        panel["symbol"].nunique(),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )


if __name__ == "__main__":
    main()
