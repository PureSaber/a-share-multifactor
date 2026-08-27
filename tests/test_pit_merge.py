import pandas as pd

from a_share_multifactor.data_loader import merge_price_fundamentals


def test_merge_price_fundamentals_pit_sparse() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["000001"] * 4,
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
            "close": [10.0, 10.1, 10.2, 10.3],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-06"]),
            "report_date": pd.to_datetime(["2020-01-02", "2020-01-06"]),
            "available_at": pd.to_datetime(["2020-01-02", "2020-01-06"]),
            "market_cap": [1e10, 1.1e10],
            "pe_ratio": [8.0, 8.5],
        }
    )
    merged = merge_price_fundamentals(prices, fundamentals, pit=True)
    assert merged.loc[merged["date"] == "2020-01-03", "market_cap"].iloc[0] == 1e10
    assert merged.loc[merged["date"] == "2020-01-07", "market_cap"].iloc[0] == 1.1e10


def test_merge_price_fundamentals_exact_join() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["000001"],
            "date": pd.to_datetime(["2020-01-02"]),
            "close": [10.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["000001"],
            "date": pd.to_datetime(["2020-01-02"]),
            "report_date": pd.to_datetime(["2020-01-02"]),
            "available_at": pd.to_datetime(["2020-01-02"]),
            "market_cap": [1e10],
        }
    )
    merged = merge_price_fundamentals(prices, fundamentals, pit=False)
    assert merged.loc[0, "market_cap"] == 1e10
