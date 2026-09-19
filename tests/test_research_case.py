import json

import pytest
import test_decision_workflow as decision_tests
from test_decision_workflow import _inputs

from a_share_multifactor import research_case

setup_decision = decision_tests.setup_decision


def test_fixed_comparison_retains_costs_and_failed_attempts(setup_decision, tmp_path, monkeypatch):
    config, _ = setup_decision
    inputs = tmp_path / "inputs"
    _inputs(inputs)
    monkeypatch.setattr(research_case, "_code_version", lambda *_: "a" * 40)
    monkeypatch.setattr(
        research_case, "_installed_internal_dependencies", lambda: {"test": "a" * 40}
    )
    out = tmp_path / "case"
    result = research_case.run_case(config, inputs, out, sessions=20)
    assert result["scope"].startswith("retrospective")
    assert result["baseline_fills"] > 0
    assert len(result["metrics"]) == 3
    assert (out / "strategy-cash_ledger.parquet").exists()
    json.dumps(result, allow_nan=False)
    with pytest.raises(FileExistsError):
        research_case.run_case(config, inputs, out, sessions=20)
