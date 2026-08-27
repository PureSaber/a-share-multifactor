"""Data loading and panel construction for multi-factor research."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from quant_data_kit.panel import (
    add_industry_relative_strength,
    merge_earnings_to_panel,
    merge_northbound_to_panel,
)
from quant_data_kit.providers.benchmark import fetch_hs300_benchmark
from quant_data_kit.providers.fundamentals import fetch_fundamentals
from quant_data_kit.providers.prices import fetch_daily_prices
from quant_data_kit.providers.universe import (
    fetch_hs300_constituents,
    fetch_hs300_constituents_history,
)
from quant_data_kit.storage import (
    cache_covers_range,
    incremental_start_date,
    load_parquet,
    parse_date,
    save_parquet,
    should_refresh_cache,
)
from quant_data_kit.snapshots import create_snapshot
from quant_data_kit.temporal import audit_point_in_time, point_in_time_join
from quant_data_kit.validate import validate_price_frame

from a_share_multifactor.config import AppConfig

logger = logging.getLogger(__name__)

# Re-export for tests and backward compatibility
__all__ = [
    "build_dataset",
    "cache_covers_range",
    "fetch_daily_prices",
    "fetch_fundamentals",
    "fetch_hs300_benchmark",
    "fetch_hs300_constituents",
    "fetch_hs300_constituents_history",
    "incremental_start_date",
    "load_benchmark_returns",
    "load_parquet",
    "merge_price_fundamentals",
    "save_parquet",
    "should_refresh_cache",
]


def merge_price_fundamentals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    pit: bool = True,
    fundamental_lag_days: int = 0,
    max_age_days: int | None = None,
    require_availability_timestamp: bool = True,
) -> pd.DataFrame:
    """Merge price and fundamentals; use merge_asof for point-in-time when pit=True."""
    if fundamentals.empty:
        return prices.copy()

    fund = fundamentals.copy()
    fund["date"] = pd.to_datetime(fund["date"]).dt.normalize()
    if "available_at" not in fund.columns:
        if require_availability_timestamp:
            raise ValueError(
                "PIT fundamentals require available_at; refresh the cache with quant-data-kit>=0.3"
            )
        fund["available_at"] = fund["date"]
    fund["available_at"] = pd.to_datetime(fund["available_at"]).dt.normalize()
    if fundamental_lag_days > 0:
        fund["available_at"] = fund["available_at"] + pd.Timedelta(days=fundamental_lag_days)

    if not pit:
        merged = prices.merge(
            fund,
            on=["symbol", "date"],
            how="left",
        )
        return merged.sort_values(["date", "symbol"]).reset_index(drop=True)

    fact_cols = [
        column
        for column in fund.columns
        if column not in {"symbol", "date", "available_at"}
    ]
    merged = point_in_time_join(
        prices,
        fund,
        observation_time="date",
        available_time="available_at",
        by=("symbol",),
        fact_columns=fact_cols,
        max_age=pd.Timedelta(days=max_age_days) if max_age_days else None,
    )
    audit_point_in_time(
        merged,
        max_age=pd.Timedelta(days=max_age_days) if max_age_days else None,
    )
    return merged


def apply_universe_filter(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return panel
    keys = panel.merge(
        universe[universe["in_universe"] == 1][["symbol", "date"]],
        on=["symbol", "date"],
        how="inner",
    )
    return keys.sort_values(["date", "symbol"]).reset_index(drop=True)


def apply_tradability_filters(panel: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    result = panel.copy()

    if config.filters.exclude_st and "name" in result.columns:
        result = result[~result["name"].astype(str).str.contains("ST", case=False, na=False)]

    if config.filters.min_list_days > 0:
        listing_counts = result.groupby("symbol").cumcount() + 1
        result = result[listing_counts >= config.filters.min_list_days]

    return result.reset_index(drop=True)


def _merge_alt_data(panel: pd.DataFrame, config: AppConfig, data_dir: Path) -> pd.DataFrame:
    result = panel.copy()

    earnings_path = data_dir / config.data.earnings_forecast
    if earnings_path.exists():
        earnings = load_parquet(earnings_path)
        result = merge_earnings_to_panel(result, earnings)

    northbound_path = data_dir / config.data.northbound
    if northbound_path.exists():
        northbound = load_parquet(northbound_path)
        result = merge_northbound_to_panel(result, northbound)

    industry_path = data_dir / config.data.industry_returns
    benchmark_path = data_dir / config.data.benchmark
    if industry_path.exists() and benchmark_path.exists():
        industry_returns = load_parquet(industry_path)
        benchmark = load_parquet(benchmark_path).set_index("date")["benchmark_return"]
        result = add_industry_relative_strength(result, industry_returns, benchmark, window=20)

    return result


def build_dataset(
    config: AppConfig,
    data_dir: Path | None = None,
    force_refresh: bool = False,
    include_alt: bool = True,
) -> pd.DataFrame:
    """Load cached Parquet or fetch from AKShare, then merge and slice."""
    root = data_dir or Path("./data")
    price_path = root / config.data.price
    fundamentals_path = root / config.data.fundamentals
    universe_path = root / config.data.universe

    universe = pd.DataFrame()
    if config.filters.use_historical_universe:
        if force_refresh or not universe_path.exists():
            universe = fetch_hs300_constituents_history(config.start_date, config.end_date)
            save_parquet(universe, universe_path)
        else:
            universe = load_parquet(universe_path)

    if force_refresh or not price_path.exists():
        symbols = (
            sorted(universe["symbol"].astype(str).unique().tolist())
            if not universe.empty
            else fetch_hs300_constituents()
        )
        prices = fetch_daily_prices(
            symbols,
            config.start_date,
            config.end_date,
            sleep_seconds=config.fetch.sleep_seconds,
            max_workers=config.fetch.max_workers,
            max_retries=config.fetch.max_retries,
        )
        save_parquet(prices, price_path)
    else:
        prices = load_parquet(price_path)
    price_quality = validate_price_frame(prices)

    if force_refresh or not fundamentals_path.exists():
        symbols = sorted(prices["symbol"].unique().tolist())
        fundamentals = fetch_fundamentals(
            symbols,
            config.start_date,
            config.end_date,
            sleep_seconds=config.fetch.sleep_seconds,
            max_workers=config.fetch.max_workers,
            max_retries=config.fetch.max_retries,
        )
        save_parquet(fundamentals, fundamentals_path)
    else:
        fundamentals = load_parquet(fundamentals_path)

    panel = merge_price_fundamentals(
        prices,
        fundamentals,
        pit=config.filters.pit_fundamentals,
        fundamental_lag_days=config.filters.fundamental_lag_days,
        max_age_days=config.filters.fundamental_max_age_days,
        require_availability_timestamp=config.filters.require_availability_timestamp,
    )
    start = parse_date(config.start_date)
    end = parse_date(config.end_date)
    panel = panel[(panel["date"] >= start) & (panel["date"] <= end)]

    if config.filters.use_historical_universe:
        panel = apply_universe_filter(panel, universe)

    panel = apply_tradability_filters(panel, config)

    if include_alt:
        panel = _merge_alt_data(panel, config, root)

    snapshot_root = root / config.data.snapshot_root
    price_snapshot = create_snapshot(
        prices,
        snapshot_root,
        dataset="cn_a_prices",
        source="akshare",
        as_of=config.end_date,
        adjustment="qfq",
        query={"start_date": config.start_date, "end_date": config.end_date},
    )
    fundamental_snapshot = create_snapshot(
        fundamentals,
        snapshot_root,
        dataset="cn_a_fundamentals",
        source="akshare",
        as_of=config.end_date,
        query={"start_date": config.start_date, "end_date": config.end_date},
    )
    universe_snapshot = None
    if not universe.empty:
        universe_snapshot = create_snapshot(
            universe,
            snapshot_root,
            dataset="hs300_historical_universe",
            source="akshare-cni",
            as_of=config.end_date,
            query={"start_date": config.start_date, "end_date": config.end_date},
        )
    result = panel.reset_index(drop=True)
    result.attrs["dataset_snapshots"] = {
        "prices": price_snapshot.snapshot_id,
        "fundamentals": fundamental_snapshot.snapshot_id,
    }
    if universe_snapshot is not None:
        result.attrs["dataset_snapshots"]["universe"] = universe_snapshot.snapshot_id
    result.attrs["data_quality"] = price_quality
    return result


def load_benchmark_returns(
    config: AppConfig,
    data_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.Series:
    from a_share_multifactor.calendar import rebalance_dates

    root = data_dir or Path("./data")
    benchmark_path = root / config.data.benchmark

    if force_refresh or not benchmark_path.exists():
        benchmark = fetch_hs300_benchmark(config.start_date, config.end_date)
        save_parquet(benchmark, benchmark_path)
    else:
        benchmark = load_parquet(benchmark_path)

    daily = benchmark.set_index("date")["benchmark_return"].sort_index()
    rebalance_idx = rebalance_dates(
        pd.Series(pd.date_range(config.start_date, config.end_date, freq="B")),
        config.rebalance_freq,
    )

    period_returns: dict[pd.Timestamp, float] = {}
    for idx in range(len(rebalance_idx) - 1):
        start_dt = rebalance_idx[idx]
        end_dt = rebalance_idx[idx + 1]
        window = daily[(daily.index > start_dt) & (daily.index <= end_dt)]
        if window.empty:
            continue
        period_returns[start_dt] = float((1 + window).prod() - 1)

    return pd.Series(period_returns).sort_index()
