import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from quant_lab import load_and_validate_standard_run
from quant_lab.counterfactuals import intervention_plan, validate_plan
from quant_lab.research import candidates, digest, file_hash
from test_research_workbench import recipe as recipe_fixture

from a_share_multifactor.paired_research import run_paired
from a_share_multifactor.research_workbench import EquityResearchExecutor

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


def risk_plan(recipe):
    recipe["allocation"] = {"mode": "cost_aware", "max_turnover": 2.0}
    recipe["risk_model"] = {
        "model_kind": "statistical_proxy",
        "lookback": 30,
        "factor_bounds": {"market": [0, 0.6]},
        "benchmark_weights": {"000001": 0.25, "000333": 0.25, "600036": 0.25, "601318": 0.25},
        "active_factor_bounds": {"market": [-0.6, 0.0]},
        "max_tracking_error": 1.0,
    }
    original = plan_for(recipe)
    base = {**original["base"], "risk_model": deepcopy(recipe["risk_model"])}
    passive = {
        **original["benchmarks"]["passive"],
        "risk_model": None,
        "allocation": {"mode": "equal"},
    }
    constrained = {
        **original["benchmarks"]["same_risk_constrained"],
        "risk_model": deepcopy(recipe["risk_model"]),
    }
    return intervention_plan(
        base,
        {
            "signal": {"momentum_20d": -1},
            "allocation": {"mode": "equal", "max_turnover": 2.0},
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


def test_paired_equal_retains_risk_but_passive_explicitly_removes_model(recipe, tmp_path):
    plan = risk_plan(recipe)
    original = deepcopy(recipe)
    root = tmp_path / "risk-paired"
    report = run_paired(recipe, plan, root)
    assert report["available"], report
    assert recipe == original
    for attempt in report["attempts"]:
        run = root / attempt["name"]
        definition = run / "execution-definition.json"
        assert file_hash(definition) == attempt["definition_sha256"]
        resolved = json.loads(definition.read_text())
        if attempt["name"] == "passive":
            assert "risk_model" not in resolved["recipe"]
            assert resolved["recipe"]["risk"] == {}
            assert not (run / "risk_models.json").exists()
        else:
            assert resolved["recipe"]["risk_model"] == recipe["risk_model"]
            assert (run / "risk_models.json").exists()
        assert load_and_validate_standard_run(run).profile == "backtest-ledger"
    equal_result = json.loads((root / "allocation/result.json").read_text())
    assert equal_result["metrics"]["fills"] > 0
    definitions = json.loads((root / "preregistration.json").read_text())
    assert definitions["plan"] == plan


def test_single_dimension_intervention_cannot_also_remove_risk_model(recipe):
    plan = risk_plan(recipe)
    plan["variants"]["allocation"]["risk_model"] = None
    plan["sha256"] = digest({key: value for key, value in plan.items() if key != "sha256"})
    with pytest.raises(ValueError, match="only one declared dimension"):
        validate_plan(plan)


def test_empty_model_override_is_not_treated_as_disabling_risk(recipe, tmp_path):
    plan = plan_for(recipe)
    plan["benchmarks"]["passive"]["risk_model"] = {}
    plan["sha256"] = digest({key: value for key, value in plan.items() if key != "sha256"})
    calls = []

    def executor(spec, candidate, out):
        calls.append(candidate["name"])
        pd.DataFrame(
            {"return": [0.01, -0.01]}, index=pd.date_range("2025-01-01", periods=2)
        ).to_csv(out / "returns.csv")
        return {}

    report = run_paired(recipe, plan, tmp_path / "invalid", executor=executor)
    assert not report["available"]
    assert "passive" not in calls
    failure = next(item for item in report["attempts"] if item["name"] == "passive")
    assert failure["status"] == "failed" and "model_kind" in failure["error"]


def cash_trend_case(recipe):
    catalog = Path(recipe["inputs"]["catalog"])
    master = pd.read_csv(catalog, dtype=str)
    master["product_type"] = "etf"
    master.to_csv(catalog, index=False)
    recipe["strategy"].update(family="etf_trend", max_weight=0.4, cash_buffer=0.2)
    recipe["allocation"] = {"mode": "equal", "max_turnover": 2.0}
    recipe["risk"] = {"max_single_weight": 0.45, "max_gross_weight": 0.95}
    base = {**candidates(recipe)[0], "risk": deepcopy(recipe["risk"])}
    benchmarks = {
        "passive": {
            **deepcopy(base),
            "risk": {},
            "strategy": {
                **base["strategy"],
                "family": "buy_hold",
                "max_weight": 1,
                "cash_buffer": 0,
            },
        },
        "same_risk_constrained": {
            **deepcopy(base),
            "strategy": {**base["strategy"], "family": "buy_hold"},
        },
        "cash": {"mode": "cash", "daily_return": 0},
    }
    return base, benchmarks


def test_trend_changes_eligibility_but_not_ranks_or_rebalance_dates(recipe, tmp_path):
    base, benchmarks = cash_trend_case(recipe)
    plan = intervention_plan(base, {"trend_filter": "rank"}, benchmarks=benchmarks)
    executor = EquityResearchExecutor()
    plans = {}
    for name, candidate in {"base": base, **plan["variants"]}.items():
        out = tmp_path / name
        out.mkdir()
        plans[name] = executor(recipe, candidate, out, _plan_only=True)
    original, removed = plans["base"], plans["trend_filter"]
    columns = ["symbol", "date", "composite_score"]
    pd.testing.assert_frame_equal(original["scored"][columns], removed["scored"][columns])
    assert not original["scored"].eligible.all()
    assert removed["scored"].eligible.all()
    assert original["allocation"].keys() == removed["allocation"].keys()
    assert all(row["max_weight"] == 0.4 for row in removed["allocation"].values())


@pytest.mark.parametrize("reserve", [0, 0.4])
def test_cash_buffer_and_trend_execute_separately_through_native_ledgers(recipe, tmp_path, reserve):
    base, benchmarks = cash_trend_case(recipe)
    plan = intervention_plan(
        base, {"cash_buffer": reserve, "trend_filter": "rank"}, benchmarks=benchmarks
    )
    root = tmp_path / "paired"
    report = run_paired(recipe, plan, root)
    assert report["available"], report
    assert len(report["attempts"]) == 5
    for item in report["attempts"]:
        assert load_and_validate_standard_run(root / item["name"]).profile == "backtest-ledger"
    config = json.loads((root / "cash_buffer/standard/v2/config.json").read_text())
    assert config["costs"]["max_holdings"] == 2
    assert config["costs"]["max_position_weight"] == 0.4
    assert config["costs"]["cash_buffer"] == reserve
    for name in ("cash_buffer", "trend_filter"):
        definition = json.loads((root / name / "execution-definition.json").read_text())
        assert definition["recipe"]["risk"] == recipe["risk"]
        assert definition["candidate"]["factors"] == base["factors"]
    values = pd.read_csv(root / "net_returns.csv", index_col=0)
    decisions = {
        name: json.loads((root / name / "execution_diagnostics.json").read_text())[
            "allocation_decisions"
        ]
        for name in ("base", "cash_buffer")
    }
    largest = {
        name: max(sum(row["weights"].values()) for row in rows) for name, rows in decisions.items()
    }
    assert largest["base"] == pytest.approx(0.8)
    assert largest["cash_buffer"] == pytest.approx(0.8 if reserve == 0 else 0.6)
    if reserve == 0:
        pd.testing.assert_series_equal(values.base, values.cash_buffer, check_names=False)
    else:
        assert not values.base.equals(values.cash_buffer)
