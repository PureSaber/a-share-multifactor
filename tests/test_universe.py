import pandas as pd

from a_share_multifactor.config import AppConfig, FilterConfig
from a_share_multifactor.data_loader import (
    apply_tradability_filters,
    apply_universe_filter,
    fetch_hs300_constituents_history,
)


def test_fetch_hs300_constituents_history_mock() -> None:
    def mock_fetch() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-02-01"]),
                "symbol": ["000002"],
                "action": ["纳入"],
            }
        )

    universe = fetch_hs300_constituents_history(
        "2020-01-01",
        "2020-02-29",
        fetch_fn=mock_fetch,
        current_symbols=["000001"],
    )
    assert not universe.empty
    assert "in_universe" in universe.columns


def test_apply_universe_filter() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000001"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
            "close": [1.0, 2.0, 1.1],
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "in_universe": [1, 1],
        }
    )
    filtered = apply_universe_filter(panel, universe)
    assert len(filtered) == 2
    assert "000002" not in filtered["symbol"].values


def test_apply_tradability_filters() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000001"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "name": ["平安银行", "ST测试", "平安银行"],
            "close": [1.0, 2.0, 1.1],
        }
    )
    config = AppConfig(filters=FilterConfig(exclude_st=True, min_list_days=2))
    filtered = apply_tradability_filters(panel, config)
    assert "ST测试" not in filtered["name"].values
    assert len(filtered) == 1
