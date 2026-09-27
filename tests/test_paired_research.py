from copy import deepcopy

import pandas as pd
import pytest
from quant_lab.counterfactuals import intervention_plan
from quant_lab.research import candidates
from test_research_workbench import recipe as recipe_fixture

from a_share_multifactor.paired_research import run_paired

recipe = recipe_fixture


def plan_for(recipe):
    base = candidates(recipe)[0]
    base["risk"] = {"max_drawdown": 0.2}
    passive = {
        **deepcopy(base),
        "risk": {},
        "strategy": {**base["strategy"], "family": "buy_hold", "max_weight": 1, "cash_buffer": 0},
    }
    constrained = {**deepcopy(base), "strategy": {**base["strategy"], "family": "buy_hold"}}
    return intervention_plan(
        base,
        {
            "signal": {"momentum_20d": -1},
            "allocation": {"mode": "equal"},
            "risk_latch": {},
            "frequency": "daily",
            "fees": 0,
            "delay": 1,
        },
        benchmarks={
            "passive": passive,
            "same_risk_constrained": constrained,
            "cash": {"mode": "cash", "daily_return": 0},
        },
    )


def test_six_real_ledger_interventions_and_three_benchmarks_reconcile(recipe, tmp_path):
    report = run_paired(recipe, plan_for(recipe), tmp_path / "paired")
    assert report["available"], report
    assert len(report["attempts"]) == 9
    result = report["attribution"]
    assert len(result["one_at_a_time_effects"]) == 6
    assert sum(result["one_at_a_time_effects"].values()) + result[
        "unexplained_and_interaction_residual"
    ] == pytest.approx(result["return_gap"])
    values = pd.read_csv(tmp_path / "paired/net_returns.csv")
    assert values.cash.eq(0).all()
    assert (tmp_path / "paired/preregistration.json").exists()
    with pytest.raises(FileExistsError):
        run_paired(recipe, plan_for(recipe), tmp_path / "paired")


def test_failures_are_retained_and_disable_attribution(recipe, tmp_path):
    def fail(*args):
        raise ValueError("data gap")

    result = run_paired(recipe, plan_for(recipe), tmp_path / "failed", executor=fail)
    assert not result["available"] and len(result["attempts"]) == 9
    assert all(row["status"] == "failed" for row in result["attempts"])
