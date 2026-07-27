import pandas as pd

from a_share_multifactor.ic_analysis import (
    analyze_factors,
    calc_ic,
    calc_ic_series,
    calc_rank_ic,
    summarize_ic,
)


def test_calc_ic_perfect_positive() -> None:
    factor = pd.Series([1.0, 2.0, 3.0, 4.0])
    ret = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert abs(calc_ic(factor, ret) - 1.0) < 1e-9


def test_calc_rank_ic() -> None:
    factor = pd.Series([1.0, 2.0, 3.0, 4.0])
    ret = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert abs(calc_rank_ic(factor, ret) - 1.0) < 1e-9


def test_summarize_ic() -> None:
    ic_series = pd.Series([0.1, 0.2, -0.1, 0.05])
    summary = summarize_ic(ic_series)
    assert abs(summary["mean_ic"] - 0.0625) < 1e-9
    assert summary["ic_positive_ratio"] == 0.75


def test_calc_ic_series() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "factor": [1.0, 2.0, 1.0, 3.0],
            "forward_return_20d": [0.1, 0.2, 0.05, 0.25],
        }
    )
    ic_series = calc_ic_series(panel, "factor", "forward_return_20d")
    assert len(ic_series) == 2


def test_analyze_factors() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "f1": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "f2": [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
            "forward_return_20d": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        }
    )
    report = analyze_factors(panel, ["f1", "f2"], "forward_return_20d")
    assert len(report) == 2
    assert report.loc[report["factor"] == "f1", "mean_ic"].iloc[0] > 0
