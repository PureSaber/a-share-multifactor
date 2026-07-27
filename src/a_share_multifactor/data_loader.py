"""Data loading, AKShare fetch helpers, and Parquet cache."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from a_share_multifactor.config import AppConfig

load_dotenv()
logger = logging.getLogger(__name__)

PRICE_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "name",
    "industry",
]
FUNDAMENTAL_COLUMNS = ["symbol", "date", "report_date", "market_cap", "pe_ratio", "pb_ratio"]
FUNDAMENTAL_RENAME = {
    "数据日期": "date",
    "总市值": "market_cap",
    "PE(TTM)": "pe_ratio",
    "市净率": "pb_ratio",
    "PB": "pb_ratio",
}
UNIVERSE_COLUMNS = ["symbol", "date", "in_universe"]
BENCHMARK_COLUMNS = ["date", "benchmark_return"]


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().zfill(6)


def _to_market_symbol(symbol: str) -> str:
    code = _normalize_symbol(symbol)
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _configure_network() -> None:
    """Prefer direct connections when system proxy is unavailable."""
    import os

    os.environ.setdefault("NO_PROXY", "*")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)


_configure_network()


def _parse_date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def cache_covers_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    date_col: str = "date",
) -> bool:
    if df.empty:
        return False
    dates = pd.to_datetime(df[date_col])
    return dates.min() <= _parse_date(start_date) and dates.max() >= _parse_date(end_date)


def fetch_hs300_constituents(fetch_fn: Callable[[], pd.DataFrame] | None = None) -> list[str]:
    """Return current HS300 constituent symbols (6-digit codes)."""
    if fetch_fn is not None:
        df = fetch_fn()
    else:
        import akshare as ak

        df = ak.index_stock_cons(symbol="000300")

    symbol_col = "品种代码" if "品种代码" in df.columns else df.columns[1]
    return [_normalize_symbol(code) for code in df[symbol_col].tolist()]


def fetch_hs300_constituents_history(
    start_date: str,
    end_date: str,
    fetch_fn: Callable[[], pd.DataFrame] | None = None,
    current_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build daily HS300 membership panel from index adjustment history.

    Returns DataFrame: symbol, date, in_universe (1/0)
    """
    if fetch_fn is not None:
        adjustments = fetch_fn()
    else:
        import akshare as ak

        adjustments = ak.index_detail_hist_adjust_cni(symbol="000300")

    events = adjustments.rename(
        columns={
            "日期": "date",
            "成分券代码": "symbol",
            "操作": "action",
        }
    )
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    events["symbol"] = events["symbol"].map(_normalize_symbol)
    events = events.sort_values("date")

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    all_dates = pd.date_range(start, end, freq="B")

    active = set(current_symbols or fetch_hs300_constituents())
    for _, row in events.sort_values("date", ascending=False).iterrows():
        event_date = row["date"]
        if event_date > end:
            continue
        if event_date <= start:
            continue
        symbol = row["symbol"]
        action = str(row["action"])
        if "纳入" in action or "进入" in action:
            active.discard(symbol)
        elif "剔除" in action or "退出" in action:
            active.add(symbol)

    rows: list[dict[str, object]] = []
    event_idx = 0
    event_list = events.reset_index(drop=True)

    for date in all_dates:
        while event_idx < len(event_list) and event_list.loc[event_idx, "date"] <= date:
            symbol = event_list.loc[event_idx, "symbol"]
            action = str(event_list.loc[event_idx, "action"])
            if event_list.loc[event_idx, "date"] > start:
                if "纳入" in action or "进入" in action:
                    active.add(symbol)
                elif "剔除" in action or "退出" in action:
                    active.discard(symbol)
            event_idx += 1
        for symbol in active:
            rows.append({"symbol": symbol, "date": date, "in_universe": 1})

    if not rows:
        for symbol in active:
            for date in all_dates:
                rows.append({"symbol": symbol, "date": date, "in_universe": 1})

    return pd.DataFrame(rows)


def fetch_hs300_benchmark(
    start_date: str,
    end_date: str,
    fetch_fn: Callable[[str, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fetch HS300 index daily returns."""
    start = _parse_date(start_date).strftime("%Y%m%d")
    end = _parse_date(end_date).strftime("%Y%m%d")

    if fetch_fn is not None:
        hist = fetch_fn(start, end)
    else:
        import akshare as ak

        try:
            hist = ak.stock_zh_index_daily_em(symbol="sh000300")
        except Exception:  # noqa: BLE001
            logger.warning("Eastmoney index API failed; falling back to stock_zh_index_daily")
            hist = ak.stock_zh_index_daily(symbol="sh000300")
        hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()

    hist = hist.sort_values("date")
    hist["benchmark_return"] = hist["close"].pct_change()
    hist = hist[(hist["date"] >= _parse_date(start_date)) & (hist["date"] <= _parse_date(end_date))]
    return hist[["date", "benchmark_return"]].dropna().reset_index(drop=True)


def _fetch_one_price(
    symbol: str,
    start: str,
    end: str,
    fetch_fn: Callable[[str, str, str], pd.DataFrame] | None,
    sleep_seconds: float,
) -> pd.DataFrame:
    if fetch_fn is not None:
        hist = fetch_fn(symbol, start, end)
    else:
        import akshare as ak

        code = _normalize_symbol(symbol)
        try:
            hist = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            hist = hist.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                }
            )
        except Exception:  # noqa: BLE001
            logger.debug("Eastmoney price API failed for %s; using stock_zh_a_hist_tx", code)
            hist = ak.stock_zh_a_hist_tx(
                symbol=_to_market_symbol(code),
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            if "amount" in hist.columns and "volume" not in hist.columns:
                hist = hist.rename(columns={"amount": "volume"})
        hist["symbol"] = code
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    keep = [col for col in PRICE_COLUMNS if col in hist.columns]
    frame = hist[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def fetch_daily_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    fetch_fn: Callable[[str, str, str], pd.DataFrame] | None = None,
    sleep_seconds: float = 0.2,
    max_workers: int = 1,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily OHLCV for symbols via AKShare."""
    start = _parse_date(start_date).strftime("%Y%m%d")
    end = _parse_date(end_date).strftime("%Y%m%d")
    frames: list[pd.DataFrame] = []

    def _task(symbol: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for _ in range(max_retries):
            try:
                return _fetch_one_price(symbol, start, end, fetch_fn, sleep_seconds)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(sleep_seconds * 2)
        raise RuntimeError(f"Failed to fetch prices for {symbol}") from last_error

    if max_workers <= 1:
        for symbol in symbols:
            frames.append(_task(symbol))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_task, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                frames.append(future.result())

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def _fetch_one_fundamental(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_fn: Callable[[str], pd.DataFrame] | None,
    sleep_seconds: float,
) -> pd.DataFrame:
    if fetch_fn is not None:
        values = fetch_fn(symbol)
    else:
        import akshare as ak

        values = ak.stock_value_em(symbol=_normalize_symbol(symbol))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    values = values.rename(columns=FUNDAMENTAL_RENAME)
    values["symbol"] = _normalize_symbol(symbol)

    keep = [col for col in FUNDAMENTAL_COLUMNS if col in values.columns]
    frame = values[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "report_date" not in frame.columns:
        frame["report_date"] = frame["date"]
    else:
        frame["report_date"] = pd.to_datetime(frame["report_date"]).dt.normalize()
    return frame[(frame["date"] >= start) & (frame["date"] <= end)]


def fetch_fundamentals(
    symbols: list[str],
    start_date: str,
    end_date: str,
    fetch_fn: Callable[[str], pd.DataFrame] | None = None,
    sleep_seconds: float = 0.2,
    max_workers: int = 1,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch market cap, PE, and PB history for symbols via AKShare."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    frames: list[pd.DataFrame] = []

    def _task(symbol: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for _ in range(max_retries):
            try:
                return _fetch_one_fundamental(symbol, start, end, fetch_fn, sleep_seconds)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(sleep_seconds * 2)
        raise RuntimeError(f"Failed to fetch fundamentals for {symbol}") from last_error

    if max_workers <= 1:
        for symbol in symbols:
            frames.append(_task(symbol))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_task, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                frames.append(future.result())

    if not frames:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

    fundamentals = pd.concat(frames, ignore_index=True)
    return fundamentals.sort_values(["symbol", "date"]).reset_index(drop=True)


def merge_price_fundamentals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    pit: bool = True,
    fundamental_lag_days: int = 0,
) -> pd.DataFrame:
    """Merge price and fundamentals; use merge_asof for point-in-time when pit=True."""
    if fundamentals.empty:
        return prices.copy()

    fund = fundamentals.copy()
    fund["date"] = pd.to_datetime(fund["date"]).dt.normalize()
    if "report_date" in fund.columns:
        fund["asof_date"] = pd.to_datetime(fund["report_date"]).dt.normalize()
    else:
        fund["asof_date"] = fund["date"]
    if fundamental_lag_days > 0:
        fund["asof_date"] = fund["asof_date"] + pd.Timedelta(days=fundamental_lag_days)

    if not pit:
        merged = prices.merge(
            fund.drop(columns=["asof_date"], errors="ignore"),
            on=["symbol", "date"],
            how="left",
        )
        return merged.sort_values(["date", "symbol"]).reset_index(drop=True)

    price_df = prices.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
    fund_cols = [c for c in fund.columns if c not in {"symbol", "date", "asof_date", "report_date"}]

    merged_parts: list[pd.DataFrame] = []
    for symbol, price_group in price_df.groupby("symbol", sort=False):
        fund_group = fund[fund["symbol"] == symbol][["asof_date"] + fund_cols].sort_values(
            "asof_date"
        )
        if fund_group.empty:
            merged_parts.append(price_group)
            continue
        part = pd.merge_asof(
            price_group.sort_values("date"),
            fund_group,
            left_on="date",
            right_on="asof_date",
            direction="backward",
        )
        merged_parts.append(part.drop(columns=["asof_date"], errors="ignore"))

    merged = pd.concat(merged_parts, ignore_index=True)
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def apply_universe_filter(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where symbol is in HS300 on that date."""
    if universe.empty:
        return panel
    keys = panel.merge(
        universe[universe["in_universe"] == 1][["symbol", "date"]],
        on=["symbol", "date"],
        how="inner",
    )
    return keys.sort_values(["date", "symbol"]).reset_index(drop=True)


def apply_tradability_filters(panel: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Filter ST names and stocks with insufficient listing history."""
    result = panel.copy()

    if config.filters.exclude_st and "name" in result.columns:
        result = result[~result["name"].astype(str).str.contains("ST", case=False, na=False)]

    if config.filters.min_list_days > 0:
        listing_counts = result.groupby("symbol").cumcount() + 1
        result = result[listing_counts >= config.filters.min_list_days]

    return result.reset_index(drop=True)


def incremental_start_date(path: Path, default_start: str) -> str:
    """Return day after cached max date, or default_start if no cache."""
    if not path.exists():
        return default_start
    df = load_parquet(path)
    if df.empty:
        return default_start
    next_day = pd.to_datetime(df["date"]).max() + pd.Timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")


def build_dataset(
    config: AppConfig,
    data_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load cached Parquet or fetch from AKShare, then merge and slice."""
    root = data_dir or Path("./data")
    price_path = root / config.data.price
    fundamentals_path = root / config.data.fundamentals
    universe_path = root / config.data.universe

    if force_refresh or not price_path.exists():
        symbols = fetch_hs300_constituents()
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
    )
    start = _parse_date(config.start_date)
    end = _parse_date(config.end_date)
    panel = panel[(panel["date"] >= start) & (panel["date"] <= end)]

    if config.filters.use_historical_universe:
        if force_refresh or not universe_path.exists():
            universe = fetch_hs300_constituents_history(config.start_date, config.end_date)
            save_parquet(universe, universe_path)
        else:
            universe = load_parquet(universe_path)
        panel = apply_universe_filter(panel, universe)

    panel = apply_tradability_filters(panel, config)
    return panel.reset_index(drop=True)


def load_benchmark_returns(
    config: AppConfig,
    data_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Load or fetch HS300 benchmark period returns aligned to rebalance dates."""
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


def should_refresh_cache(
    path: Path,
    start_date: str,
    end_date: str,
) -> bool:
    if not path.exists():
        return True
    df = load_parquet(path)
    return not cache_covers_range(df, start_date, end_date)
