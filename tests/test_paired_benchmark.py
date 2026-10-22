import pandas as pd
import pytest

from a_share_multifactor.research_workbench import paired_benchmark_evidence


def test_measured_benchmark_identity_and_pairs_are_required():
    dates = pd.bdate_range("2026-01-01", periods=5)
    returns = pd.Series([0, -0.01, 0.02, -0.02, 0.01], index=dates)
    benchmark = pd.DataFrame(
        {"date": dates, "benchmark_return": 2 * returns.values, "benchmark_id": "measured"}
    )
    assert paired_benchmark_evidence(returns, benchmark, "arbitrary-label") == {}
    report = paired_benchmark_evidence(returns, benchmark, "measured")
    assert report["volatility_ratio"] == pytest.approx(0.5)
    assert len(report["paired_returns"]) == 4
    with pytest.raises(ValueError, match="incomplete"):
        paired_benchmark_evidence(returns, benchmark.iloc[1:4], "measured")
