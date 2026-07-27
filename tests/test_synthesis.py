import pandas as pd

from a_share_multifactor.config import AppConfig, SynthesisConfig
from a_share_multifactor.synthesis import equal_weight_score, rolling_ic_weight_score, synthesize


def test_equal_weight_score() -> None:
    panel = pd.DataFrame({"f1": [1.0, -1.0], "f2": [1.0, 1.0]})
    result = equal_weight_score(panel, ["f1", "f2"])
    assert result.loc[0, "composite_score"] == 1.0
    assert result.loc[1, "composite_score"] == 0.0


def test_rolling_ic_weight_score() -> None:
    rows = []
    for day in pd.date_range("2020-01-01", periods=40, freq="B"):
        rows.append({"date": day, "f1": 1.0, "f2": -1.0, "forward_return_20d": 0.01})
        rows.append({"date": day, "f1": -1.0, "f2": 1.0, "forward_return_20d": -0.01})
    panel = pd.DataFrame(rows)
    result = rolling_ic_weight_score(panel, ["f1", "f2"], "forward_return_20d", lookback_months=1)
    assert result["composite_score"].notna().any()


def test_synthesize_rolling_ic_weight() -> None:
    config = AppConfig(synthesis=SynthesisConfig(method="rolling_ic_weight", lookback_months=1))
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 2),
            "f1": [1.0, -1.0],
            "forward_return_20d": [0.1, -0.1],
        }
    )
    result = synthesize(panel.assign(f2=[1.0, -1.0]), config)
    assert "composite_score" in result.columns
