import json

import pandas as pd
import pytest
from quant_lab import load_and_validate_standard_run
from quant_lab.research import candidates
from test_decision_workflow import _inputs

from a_share_multifactor import run_contract
from a_share_multifactor.decision_workflow import write_watchlist_catalog
from a_share_multifactor.factors import compute_factors
from a_share_multifactor.research_workbench import EquityResearchExecutor


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    dates, _ = _inputs(tmp_path / "inputs")
    settings = {
        "watchlist": [
            {"symbol": s, "venue": v}
            for s, v in [
                ("000001", "SZSE"),
                ("000333", "SZSE"),
                ("600036", "SSE"),
                ("601318", "SSE"),
            ]
        ]
    }
    catalog = write_watchlist_catalog(tmp_path, settings, "2025-01-01", "2026-01-01")
    monkeypatch.setattr(run_contract, "_code_version", lambda *_: "a" * 40)
    monkeypatch.setattr(
        run_contract, "_installed_internal_dependencies", lambda: {"test": "a" * 40}
    )
    return {
        "schema_version": "quant.research-recipe/v1",
        "study_id": "integration-test",
        "hypothesis": "Fixed signals improve net returns",
        "mode": "exploratory",
        "backend": "equity",
        "inputs": {"bundle": str(tmp_path / "inputs"), "catalog": str(catalog)},
        "interval": {"start": str(dates[45].date()), "end": str(dates[79].date())},
        "factors": {"momentum_20d": 1, "volatility_20d": -1},
        "strategy": {
            "family": "rank",
            "frequency": "weekly",
            "top_n": 2,
            "max_weight": 0.25,
            "cash_buffer": 0.5,
            "trend_window": 20,
        },
        "costs": {
            "initial_capital": 100000,
            "commission": 0.0003,
            "min_commission": 5,
            "stamp_tax": 0.0005,
            "slippage": 0.001,
            "participation_rate": 0.01,
        },
        "diagnostics": {"cost_multipliers": [2], "signal_delays": [1]},
        "variants": [],
    }


def execute(executor, recipe, candidate, path):
    path.mkdir()
    return executor(recipe, candidate, path)


def test_candidate_ledgers_cost_stress_and_data_cache(recipe, tmp_path):
    executor = EquityResearchExecutor()
    planned = candidates(recipe)
    base = execute(executor, recipe, planned[0], tmp_path / "base")
    stress = execute(
        executor, recipe, next(c for c in planned if c["name"] == "cost_2x"), tmp_path / "stress"
    )
    assert load_and_validate_standard_run(tmp_path / "base").profile == "backtest-ledger"
    assert base["metrics"]["fills"] > 0
    assert stress["metrics"]["cost_total"] > base["metrics"]["cost_total"]
    assert stress["metrics"]["total_return"] < base["metrics"]["total_return"]
    assert len(executor._features) == 1
    assert base["comparison"]["start"] == recipe["interval"]["start"]
    json.dumps(base, allow_nan=False)


def test_delayed_signal_and_unknown_factor(recipe, tmp_path):
    executor = EquityResearchExecutor()
    candidate = next(c for c in candidates(recipe) if c["name"] == "delay_1")
    result = execute(executor, recipe, candidate, tmp_path / "delay")
    assert result["metrics"]["fills"] > 0
    with pytest.raises(ValueError, match="Unknown"):
        compute_factors(pd.DataFrame(), ["momentun_20d"])


def test_missing_history_and_input_corruption_block(recipe, tmp_path):
    executor = EquityResearchExecutor()
    recipe["required_history"] = {"tradable": "status"}
    with pytest.raises(ValueError, match="preflight"):
        execute(executor, recipe, candidates(recipe)[0], tmp_path / "blocked")
    report = json.loads((tmp_path / "blocked/preflight.json").read_text())
    assert any(r["code"] == "HISTORY_COVERAGE" for r in report["issues"])
    (tmp_path / "inputs/raw.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        execute(executor, recipe, candidates(recipe)[0], tmp_path / "corrupt")
