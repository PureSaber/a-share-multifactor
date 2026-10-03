"""Data loading and panel construction for multi-factor research."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
from quant_data_kit.panel import (
    add_industry_relative_strength,
    merge_northbound_to_panel,
)
from quant_data_kit.providers.fundamentals import fetch_fundamentals
from quant_data_kit.providers.prices import fetch_daily_prices
from quant_data_kit.providers.universe import (
    fetch_hs300_constituents,
    fetch_hs300_constituents_history,
)
from quant_data_kit.snapshots import create_snapshot
from quant_data_kit.storage import (
    cache_covers_range,
    incremental_start_date,
    load_parquet,
    parse_date,
    save_parquet,
    should_refresh_cache,
)
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
    "history_start",
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
    # Published fundamentals own these columns. Embedded price-cache values must
    # never survive a PIT join or turn into silently ignored _x/_y columns.
    fact_cols = [column for column in fund if column not in {"symbol", "date", "available_at"}]
    market_fields = {"open", "high", "low", "close", "volume", "amount"}
    if market_fields.intersection(fact_cols):
        raise ValueError("Fundamental payload cannot overwrite market prices or volume")
    prices = prices.drop(columns=[column for column in fact_cols if column in prices])
    fund["date"] = pd.to_datetime(fund["date"]).dt.normalize()
    if "available_at" not in fund.columns:
        if require_availability_timestamp:
            raise ValueError(
                "PIT fundamentals require available_at; refresh the cache with quant-data-kit>=0.3"
            )
        fund["available_at"] = fund["date"]
    fund["available_at"] = pd.to_datetime(fund["available_at"])
    if pit and require_availability_timestamp and fund["available_at"].isna().any():
        raise ValueError(
            "PIT fundamentals require known publication timestamps; found unknown availability"
        )
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
        column for column in fund.columns if column not in {"symbol", "date", "available_at"}
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


def history_start(config: AppConfig) -> str:
    """Start price history early enough for ``min_list_days`` before the sample."""
    listed = config.filters.min_list_days
    if listed <= 0:
        return config.start_date
    start = pd.Timestamp(config.start_date) - pd.Timedelta(days=listed * 2)
    return start.date().isoformat()


def _listing_age_ok(panel: pd.DataFrame, min_list_days: int) -> pd.Series:
    ordered = panel.sort_values(["symbol", "date"])
    if "list_date" in ordered.columns and ordered["list_date"].notna().any():
        listed = pd.to_datetime(ordered["list_date"])
        on_market = ordered["date"] >= listed
        age = on_market.groupby(ordered["symbol"]).cumsum()
        return age.reindex(panel.index) >= min_list_days
    age = ordered.groupby("symbol").cumcount() + 1
    return age.reindex(panel.index) >= min_list_days


def apply_tradability_filters(panel: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"])

    if config.filters.exclude_st:
        if "is_st" in result.columns:
            result = result[~result["is_st"].fillna(False).astype(bool)]
        elif "name" in result.columns:
            result = result[~result["name"].astype(str).str.contains("ST", case=False, na=False)]
        else:
            raise ValueError(
                "exclude_st is enabled, but the panel has neither name nor is_st. "
                "ST names would stay in the universe."
            )

    if config.filters.min_list_days > 0:
        result = result.loc[_listing_age_ok(result, config.filters.min_list_days)]

    return result.reset_index(drop=True)


# Daily northbound stock-level disclosure stopped after this session.
_NORTHBOUND_DAILY_END = pd.Timestamp("2024-08-16")


def merge_earnings_with_max_age(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    max_age_days: int,
) -> pd.DataFrame:
    """Point-in-time forecast score that expires after ``max_age_days``."""
    result = panel.copy()
    if forecasts.empty or "forecast_score" not in forecasts.columns:
        result["forecast_score"] = pd.NA
        return result
    if "effective_date" not in forecasts.columns:
        raise ValueError(
            "Earnings forecasts require effective_date so an old pre-announcement can expire."
        )
    fund = forecasts[["symbol", "effective_date", "forecast_score"]].copy()
    fund["symbol"] = fund["symbol"].astype(str)
    fund["effective_date"] = pd.to_datetime(fund["effective_date"]).dt.normalize()
    fund = fund.dropna(subset=["effective_date"]).sort_values("effective_date")
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["symbol"] = result["symbol"].astype(str)
    parts: list[pd.DataFrame] = []
    for symbol, price_group in result.groupby("symbol", sort=False):
        fund_group = fund.loc[fund["symbol"] == symbol, ["effective_date", "forecast_score"]]
        if fund_group.empty:
            part = price_group.copy()
            part["forecast_score"] = pd.NA
            part["forecast_effective_date"] = pd.NaT
        else:
            part = pd.merge_asof(
                price_group.sort_values("date"),
                fund_group,
                left_on="date",
                right_on="effective_date",
                direction="backward",
            ).rename(columns={"effective_date": "forecast_effective_date"})
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)
    age = (merged["date"] - pd.to_datetime(merged["forecast_effective_date"])).dt.days
    if max_age_days > 0:
        merged.loc[age.isna() | (age > max_age_days), "forecast_score"] = pd.NA
    return (
        merged.drop(columns=["forecast_effective_date"])
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def _merge_alt_data(panel: pd.DataFrame, config: AppConfig, data_dir: Path) -> pd.DataFrame:
    result = panel.copy()

    earnings_path = data_dir / config.data.earnings_forecast
    if earnings_path.exists():
        earnings = load_parquet(earnings_path)
        result = merge_earnings_with_max_age(
            result,
            earnings,
            config.filters.forecast_max_age_days,
        )

    northbound_path = data_dir / config.data.northbound
    if northbound_path.exists():
        northbound = load_parquet(northbound_path)
        if not northbound.empty and "date" in northbound.columns:
            last_print = pd.to_datetime(northbound["date"]).max()
            sample_end = pd.to_datetime(result["date"]).max()
            if sample_end > _NORTHBOUND_DAILY_END and last_print < _NORTHBOUND_DAILY_END:
                logger.warning(
                    "Northbound holdings end on %s. Daily Stock Connect disclosure stopped "
                    "in August 2024, so later northbound_chg_5d values stay missing.",
                    pd.Timestamp(last_print).date(),
                )
        result = merge_northbound_to_panel(result, northbound)

    industry_path = data_dir / config.data.industry_returns
    benchmark_path = data_dir / config.data.benchmark
    if industry_path.exists() and benchmark_path.exists():
        if "industry" not in result.columns:
            raise ValueError(
                "Industry returns are cached, but prices have no industry column. "
                "industry_rs_20d would be entirely missing."
            )
        industry_returns = load_parquet(industry_path)
        benchmark = load_parquet(benchmark_path).set_index("date")["benchmark_return"]
        result = add_industry_relative_strength(result, industry_returns, benchmark, window=20)

    return result


def _require_historical_price_coverage(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    if universe.empty or "symbol" not in universe.columns:
        raise ValueError(
            "Historical index membership is empty. Refusing to price only today's constituents."
        )
    window = universe.copy()
    window["date"] = pd.to_datetime(window["date"])
    window = window[(window["date"] >= start) & (window["date"] <= end)]
    if "in_universe" in window.columns:
        window = window[window["in_universe"] == 1]
    needed = set(window["symbol"].astype(str))
    have = set(prices["symbol"].astype(str))
    missing = sorted(needed - have)
    if missing:
        sample = ", ".join(missing[:8])
        raise ValueError(
            f"Price cache is missing {len(missing)} historical index members ({sample}). "
            "A price file limited to today's constituents drops stocks that later left the index."
        )


def build_dataset(
    config: AppConfig,
    data_dir: Path | None = None,
    force_refresh: bool = False,
    include_alt: bool = True,
    allow_incomplete_universe: bool = False,
    *,
    read_only: bool = False,
) -> pd.DataFrame:
    """Load cached Parquet or fetch from AKShare, then merge and slice."""
    root = data_dir or Path("./data")
    price_path = root / config.data.price
    fundamentals_path = root / config.data.fundamentals
    universe_path = root / config.data.universe

    if read_only:
        if force_refresh:
            raise ValueError("Read-only data loading cannot refresh sources")
        required = [price_path, fundamentals_path]
        if config.filters.use_historical_universe:
            required.append(universe_path)
        for path in required:
            if not path.is_file():
                raise ValueError(f"Read-only data loading requires an existing file: {path}")

    universe = pd.DataFrame()
    if config.filters.use_historical_universe:
        if force_refresh or not universe_path.exists():
            universe = fetch_hs300_constituents_history(config.start_date, config.end_date)
            save_parquet(universe, universe_path)
        else:
            universe = load_parquet(universe_path)

    if force_refresh or not price_path.exists():
        if config.filters.use_historical_universe:
            if universe.empty:
                raise ValueError(
                    "Historical index membership is empty. "
                    "Refusing to price only today's constituents."
                )
            symbols = sorted(universe["symbol"].astype(str).unique().tolist())
        else:
            symbols = fetch_hs300_constituents()
        prices = fetch_daily_prices(
            symbols,
            history_start(config),
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
    panel = apply_tradability_filters(panel, config)
    panel = panel[(panel["date"] >= start) & (panel["date"] <= end)]

    if config.filters.use_historical_universe:
        if not allow_incomplete_universe:
            _require_historical_price_coverage(prices, universe, start, end)
        elif universe.empty:
            raise ValueError("Historical index membership is empty.")
        panel = apply_universe_filter(panel, universe)

    if include_alt:
        panel = _merge_alt_data(panel, config, root)

    result = panel.reset_index(drop=True)
    result.attrs["data_quality"] = price_quality
    if read_only:
        result.attrs["read_only"] = True
        return result

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
    result.attrs["dataset_snapshots"] = {
        "prices": price_snapshot.snapshot_id,
        "fundamentals": fundamental_snapshot.snapshot_id,
    }
    if universe_snapshot is not None:
        result.attrs["dataset_snapshots"]["universe"] = universe_snapshot.snapshot_id
    return result


def _normalize_index_history(hist: pd.DataFrame) -> pd.DataFrame:
    rename: dict[object, str] = {}
    for column in hist.columns:
        text = str(column)
        if text in {"日期", "date"}:
            rename[column] = "date"
        elif text in {"收盘", "close", "收盘价"}:
            rename[column] = "close"
    normalized = hist.rename(columns=rename)
    if not {"date", "close"}.issubset(normalized.columns):
        raise ValueError(
            "H00300 history is missing date or close. "
            "The price index sh000300 is not a substitute."
        )
    return normalized


def fetch_hs300_benchmark(
    start_date: str,
    end_date: str,
    fetch_fn=None,
) -> pd.DataFrame:
    """CSI 300 total-return index H00300.

    The price index sh000300 omits dividends, so excess return against it is
    high by about the dividend yield.
    """
    if fetch_fn is not None:
        hist = fetch_fn(start_date, end_date)
    else:
        import akshare as ak
        from quant_data_kit.providers._network import configure_network

        configure_network()
        hist = ak.stock_zh_index_hist_csindex(
            symbol="H00300",
            start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        )
    hist = _normalize_index_history(hist).sort_values("date")
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
    hist["benchmark_return"] = hist["close"].pct_change()
    start = parse_date(start_date)
    end = parse_date(end_date)
    hist = hist[(hist["date"] >= start) & (hist["date"] <= end)]
    hist["benchmark_kind"] = "total_return"
    hist["benchmark_symbol"] = "H00300"
    return hist[["date", "benchmark_return", "benchmark_kind", "benchmark_symbol"]].dropna(
        subset=["benchmark_return"]
    ).reset_index(drop=True)


def _require_total_return_benchmark(benchmark: pd.DataFrame) -> None:
    kind = benchmark["benchmark_kind"] if "benchmark_kind" in benchmark.columns else pd.Series(dtype=object)
    if kind.empty or not kind.eq("total_return").all():
        raise ValueError(
            "Benchmark cache is not the CSI 300 total-return index H00300. "
            "Delete it and refetch. The price index sh000300 omits dividends."
        )


def load_benchmark_returns(
    config: AppConfig,
    data_dir: Path | None = None,
    force_refresh: bool = False,
    *,
    read_only: bool = False,
) -> pd.Series:
    from a_share_multifactor.calendar import rebalance_dates

    root = data_dir or Path("./data")
    benchmark_path = root / config.data.benchmark

    if read_only and (force_refresh or not benchmark_path.is_file()):
        raise ValueError(f"Read-only benchmark loading requires an existing file: {benchmark_path}")

    if force_refresh or not benchmark_path.exists():
        benchmark = fetch_hs300_benchmark(config.start_date, config.end_date)
        save_parquet(benchmark, benchmark_path)
    else:
        benchmark = load_parquet(benchmark_path)
    _require_total_return_benchmark(benchmark)

    benchmark = benchmark.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["benchmark_return"] = pd.to_numeric(benchmark["benchmark_return"], errors="coerce")
    if benchmark["date"].isna().any() or benchmark["date"].duplicated().any():
        raise ValueError("Benchmark dates must be valid and unique")
    values = benchmark["benchmark_return"]
    if not values.map(lambda value: pd.notna(value) and math.isfinite(float(value))).all():
        raise ValueError("Benchmark returns must be finite")
    if (values < -1).any():
        raise ValueError("Benchmark returns cannot be below -100%")
    daily = benchmark.set_index("date")["benchmark_return"].sort_index()
    rebalance_idx = rebalance_dates(pd.Series(daily.index), config.rebalance_freq)

    period_returns: dict[pd.Timestamp, float] = {}
    for idx in range(len(rebalance_idx) - 1):
        start_dt = rebalance_idx[idx]
        end_dt = rebalance_idx[idx + 1]
        window = daily[(daily.index > start_dt) & (daily.index <= end_dt)]
        if window.empty:
            raise ValueError(
                f"Benchmark has no trading days between {pd.Timestamp(start_dt).date()} "
                f"and {pd.Timestamp(end_dt).date()}."
            )
        period_returns[start_dt] = float((1 + window).prod() - 1)

    return pd.Series(period_returns).sort_index()
