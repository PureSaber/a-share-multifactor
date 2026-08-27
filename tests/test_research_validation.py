import pandas as pd

from a_share_multifactor.config import AppConfig, ValidationConfig
from a_share_multifactor.research_validation import run_research_validation


def test_walk_forward_research_validation_outputs_fdr_and_leakage_audit() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx in range(8):
            factor = float(symbol_idx)
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{symbol_idx}",
                    "pe_ratio": factor,
                    "source_available_at": date,
                    "forward_return_5d": factor * 0.01 + date_idx * 0.00001,
                }
            )
    panel = pd.DataFrame(rows)
    config = AppConfig(
        forward_return_days=5,
        validation=ValidationConfig(
            enabled=True,
            train_size=15,
            test_size=8,
            step_size=8,
            embargo_size=2,
        ),
    )
    result = run_research_validation(panel, ["pe_ratio"], "forward_return_5d", config)
    assert len(result.fold_metrics[result.fold_metrics["factor"] == "__composite__"]) == 2
    assert result.multiple_testing.loc[0, "reject"]
    assert result.leakage_audit.loc[0, "future_rows"] == 0
