import pandas as pd
import pytest

from a_share_multifactor.data_loader import merge_price_fundamentals


def test_price_cache_fundamentals_cannot_bypass_publication_time():
    prices = pd.DataFrame(
        {
            "symbol": ["000001"] * 2,
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "close": [10.0, 11.0],
            "pe_ratio": [999.0, 999.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["000001"],
            "date": pd.to_datetime(["2023-12-31"]),
            "available_at": pd.to_datetime(["2024-01-02"]),
            "pe_ratio": [10.0],
        }
    )
    result = merge_price_fundamentals(prices, fundamentals)
    assert pd.isna(result.pe_ratio.iloc[0])
    assert result.pe_ratio.iloc[1] == 10
    assert not any(c.endswith(("_x", "_y")) for c in result)
    with pytest.raises(ValueError, match="market prices"):
        merge_price_fundamentals(prices, fundamentals.assign(close=100.0))
