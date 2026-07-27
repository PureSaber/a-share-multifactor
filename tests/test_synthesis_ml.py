import pandas as pd

from a_share_multifactor.config import AppConfig, SynthesisConfig
from a_share_multifactor.synthesis import rolling_ml_score, synthesize


def test_rolling_ml_score_ridge() -> None:
    rows = []
    for day in pd.date_range("2020-01-01", periods=60, freq="B"):
        rows.append({"date": day, "f1": 1.0, "f2": 0.5, "forward_return_20d": 0.02})
        rows.append({"date": day, "f1": -1.0, "f2": -0.5, "forward_return_20d": -0.02})
    panel = pd.DataFrame(rows)
    result = rolling_ml_score(
        panel, ["f1", "f2"], "forward_return_20d", method="ridge", lookback_months=1
    )
    assert result["composite_score"].notna().any()


def test_synthesize_ridge() -> None:
    config = AppConfig(synthesis=SynthesisConfig(method="ridge", lookback_months=1))
    rows = []
    for day in pd.date_range("2020-01-01", periods=60, freq="B"):
        rows.append({"date": day, "f1": 1.0, "forward_return_20d": 0.02})
    panel = pd.DataFrame(rows)
    result = synthesize(panel, config)
    assert "composite_score" in result.columns
