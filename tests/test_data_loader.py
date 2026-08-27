from pathlib import Path

import pandas as pd
import pytest

from a_share_multifactor.config import AppConfig, DataPaths, FilterConfig
from a_share_multifactor.data_loader import (
    build_dataset,
    cache_covers_range,
    fetch_daily_prices,
    fetch_fundamentals,
    fetch_hs300_constituents,
    merge_price_fundamentals,
    save_parquet,
    should_refresh_cache,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
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


@pytest.fixture
def sample_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002", "000002"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]),
            "available_at": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]
            ),
            "market_cap": [1e10, 1.01e10, 2e10, 2.01e10],
            "pe_ratio": [8.0, 8.1, 12.0, 12.1],
        }
    )


def test_fetch_hs300_constituents_mock() -> None:
    def mock_fetch() -> pd.DataFrame:
        return pd.DataFrame({"品种代码": ["000001", "600519"]})

    symbols = fetch_hs300_constituents(fetch_fn=mock_fetch)
    assert symbols == ["000001", "600519"]


def test_fetch_daily_prices_mock() -> None:
    def mock_fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "date": pd.to_datetime(["2020-01-02"]),
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [100],
            }
        )

    prices = fetch_daily_prices(["000001"], "2020-01-01", "2020-12-31", fetch_fn=mock_fetch)
    assert len(prices) == 1
    assert prices.loc[0, "symbol"] == "000001"


def test_fetch_fundamentals_mock() -> None:
    def mock_fetch(symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": [symbol, symbol],
                "date": pd.to_datetime(["2020-01-02", "2020-06-01"]),
                "market_cap": [1e10, 1.1e10],
                "pe_ratio": [10.0, 11.0],
                "pb_ratio": [2.0, 2.1],
            }
        )

    fundamentals = fetch_fundamentals(["000001"], "2020-01-01", "2020-12-31", fetch_fn=mock_fetch)
    assert len(fundamentals) == 2
    assert fundamentals["pb_ratio"].notna().all()


def test_fetch_fundamentals_akshare_column_names() -> None:
    def mock_fetch(symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "数据日期": ["2020-01-02", "2020-06-01"],
                "总市值": [1e10, 1.1e10],
                "PE(TTM)": [10.0, 11.0],
                "市净率": [2.0, 2.1],
            }
        )

    fundamentals = fetch_fundamentals(["000001"], "2020-01-01", "2020-12-31", fetch_fn=mock_fetch)
    assert list(fundamentals.columns) == [
        "symbol",
        "date",
        "market_cap",
        "pe_ratio",
        "pb_ratio",
        "report_date",
        "available_at",
    ]
    assert fundamentals["pb_ratio"].tolist() == [2.0, 2.1]


def test_merge_price_fundamentals(sample_prices, sample_fundamentals) -> None:
    merged = merge_price_fundamentals(sample_prices, sample_fundamentals)
    assert len(merged) == 4
    assert "market_cap" in merged.columns
    assert merged.loc[0, "close"] == 10.2


def test_cache_covers_range(sample_prices) -> None:
    assert cache_covers_range(sample_prices, "2020-01-02", "2020-01-03")
    assert not cache_covers_range(sample_prices, "2019-01-01", "2020-01-03")


def test_save_and_should_refresh(tmp_path: Path, sample_prices) -> None:
    path = tmp_path / "prices.parquet"
    save_parquet(sample_prices, path)
    assert path.exists()
    assert not should_refresh_cache(path, "2020-01-02", "2020-01-03")
    assert should_refresh_cache(path, "2019-01-01", "2020-01-03")


def test_build_dataset_from_cache(tmp_path: Path, sample_prices, sample_fundamentals) -> None:
    config = AppConfig(
        start_date="2020-01-02",
        end_date="2020-01-03",
        filters=FilterConfig(use_historical_universe=False, min_list_days=0),
        data=DataPaths(price="prices.parquet", fundamentals="fundamentals.parquet"),
    )
    save_parquet(sample_prices, tmp_path / "prices.parquet")
    save_parquet(sample_fundamentals, tmp_path / "fundamentals.parquet")

    panel = build_dataset(config, data_dir=tmp_path)
    assert len(panel) == 4
    assert panel["market_cap"].notna().all()
