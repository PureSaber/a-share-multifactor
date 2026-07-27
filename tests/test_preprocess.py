import pandas as pd

from a_share_multifactor.config import AppConfig, FilterConfig
from a_share_multifactor.preprocess import (
    add_forward_return,
    prepare_factor_panel,
    standardize_cross_section,
    winsorize_cross_section,
)


def test_winsorize_cross_section() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 5),
            "factor": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )
    result = winsorize_cross_section(df, ["factor"], (0.01, 0.99))
    assert result["factor"].max() < 100.0


def test_standardize_cross_section() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "factor": [1.0, 2.0, 3.0],
        }
    )
    result = standardize_cross_section(df, ["factor"])
    assert abs(result["factor"].mean()) < 1e-9
    assert abs(result["factor"].std(ddof=0) - 1.0) < 1e-9


def test_add_forward_return() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [100.0, 110.0, 121.0],
        }
    )
    result = add_forward_return(df, window=1)
    assert abs(result.loc[0, "forward_return_1d"] - 0.1) < 1e-9
    assert pd.isna(result.loc[2, "forward_return_1d"])


def test_prepare_factor_panel() -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["000001"] * 25,
            "date": pd.date_range("2020-01-01", periods=25, freq="B"),
            "close": [100 + i for i in range(25)],
            "market_cap": [1e10] * 25,
            "pe_ratio": [10.0] * 25,
            "pb_ratio": [2.0] * 25,
        }
    )
    config = AppConfig(
        forward_return_days=5,
        holding_period="rebalance",
        factors=["market_cap", "pe_ratio", "momentum_20d"],
        filters=FilterConfig(min_list_days=0),
    )
    panel = prepare_factor_panel(config, raw)
    assert "momentum_20d" in panel.columns
    assert "forward_return_5d" in panel.columns
    assert "period_return" in panel.columns
    assert panel.loc[0, "market_cap"] <= 0
