import pandas as pd

from a_share_multifactor.ic_analysis import analyze_ic_decay


def test_analyze_ic_decay() -> None:
    rows = []
    for day in pd.date_range("2020-01-01", periods=30, freq="B"):
        rows.append(
            {
                "symbol": "A",
                "date": day,
                "close": 100 + len(rows),
                "factor": float(len(rows)),
            }
        )
    panel = pd.DataFrame(rows)
    decay = analyze_ic_decay(panel, ["factor"], [1, 5], price_col="close")
    assert set(decay["horizon_days"]) == {1, 5}
    assert "mean_ic" in decay.columns
