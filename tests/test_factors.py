import pandas as pd

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.factors import apply_factor_directions, compute_factors
from a_share_multifactor.preprocess import add_period_return


def test_compute_momentum():
    close = pd.Series([100, 102, 104, 103, 105])
    from a_share_multifactor.factors import compute_momentum

    mom = compute_momentum(close, window=2)
    assert abs(mom.iloc[-1] - (105 / 104 - 1)) < 1e-9


def test_compute_extended_factors() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 25,
            "date": pd.date_range("2020-01-01", periods=25, freq="B"),
            "close": [100 + i for i in range(25)],
            "volume": [1000 + i for i in range(25)],
            "market_cap": [1e9] * 25,
            "pe_ratio": [10.0] * 25,
            "pb_ratio": [2.0] * 25,
        }
    )
    result = compute_factors(
        df, factor_names=["momentum_20d", "volatility_20d", "reversal_5d", "turnover_20d"]
    )
    assert "volatility_20d" in result.columns
    assert "reversal_5d" in result.columns


def test_apply_factor_directions() -> None:
    df = pd.DataFrame({"market_cap": [1.0, 2.0], "momentum_20d": [0.1, 0.2]})
    result = apply_factor_directions(
        df,
        ["market_cap", "momentum_20d"],
        {"market_cap": -1, "momentum_20d": 1},
    )
    assert result.loc[0, "market_cap"] == -1.0
    assert result.loc[0, "momentum_20d"] == 0.1


def test_add_period_return() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "date": pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"]),
            "open": [100.0, 110.0, 115.0, 121.0],
            "high": [101.0, 112.0, 116.0, 122.0],
            "low": [99.0, 108.0, 114.0, 120.0],
            "close": [100.0, 111.0, 115.0, 121.5],
            "volume": [1000, 1000, 1000, 1000],
        }
    )
    rebalance_idx = rebalance_dates(df["date"], "monthly")
    result = add_period_return(df, rebalance_idx)
    january = result.loc[result["date"] == pd.Timestamp("2020-01-31"), "period_return"].iloc[0]
    assert abs(january - (121.0 / 110.0 - 1)) < 1e-9
