import pandas as pd
import pytest

from a_share_multifactor.return_forecasts import forecasts_at, read_forecasts


def test_frozen_forecasts_select_only_available_revision_and_exact_horizon(tmp_path):
    path = tmp_path / "forecasts.csv"
    path.write_text(
        "symbol,forecast_at,available_at,horizon_sessions,expected_return,model_version\n"
        "000001,2026-01-02T06:00Z,2026-01-02T06:00Z,5,0.01,v1\n"
        "000001,2026-01-02T06:00Z,2026-01-03T06:00Z,5,0.99,v2\n"
    )
    frame, payload, digest = read_forecasts(path)
    assert payload == path.read_bytes() and len(digest) == 64
    actual = forecasts_at(
        frame, decision_at="2026-01-02T07:00Z", assets=["000001"], horizon_sessions=5
    )
    assert actual.iloc[0] == 0.01
    frame.loc[frame.model_version == "v2", "expected_return"] = -100
    pd.testing.assert_series_equal(
        actual,
        forecasts_at(frame, decision_at="2026-01-02T07:00Z", assets=["000001"], horizon_sessions=5),
    )
    for cutoff, horizon in [
        ("2026-01-02T05:00Z", 5),
        ("2026-01-02T07:00Z", 1),
        ("2026-01-03T07:00Z", 5),
    ]:
        with pytest.raises(ValueError, match="missing contemporaneous"):
            forecasts_at(frame, decision_at=cutoff, assets=["000001"], horizon_sessions=horizon)
