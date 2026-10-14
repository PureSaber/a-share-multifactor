from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from quant_data_kit.research_coverage import validate_history

from a_share_multifactor.research_risk import (
    allocation_risk_inputs,
    build_risk_schedule,
    check_model_target,
)


def fixture():
    dates = pd.bdate_range("2025-01-02", periods=34)
    symbols = [f"S{i}" for i in range(12)]
    raw = pd.DataFrame(
        [
            {
                "date": day,
                "symbol": symbol,
                "close": 100 + index * (i + 1) / 30 + np.sin(index + i) * 0.2,
                "adjustment": "none",
            }
            for i, symbol in enumerate(symbols)
            for index, day in enumerate(dates)
        ]
    )
    histories = []
    for i, symbol in enumerate(symbols):
        for field, domain, value in (
            ("log_size", "fundamentals", np.log(i + 1)),
            ("cap", "fundamentals", (i + 1) * 100),
            ("sector", "classification", ["a", "b", "c"][i // 4]),
        ):
            histories.append(
                {
                    "symbol": symbol,
                    "domain": domain,
                    "field": field,
                    "value": str(value),
                    "effective_at": "2025-01-01T00:00:00Z",
                    "available_at": "2025-01-01T00:00:00Z",
                }
            )
    recipe = {
        "required_history": {
            "log_size": "fundamentals",
            "cap": "fundamentals",
            "sector": "classification",
        },
        "risk_model": {
            "model_kind": "fundamental_style",
            "lookback": 25,
            "min_periods": 12,
            "min_asset_observations": 12,
            "min_assets_per_period": 8,
            "exposure_pipeline": {
                "descriptor_fields": {"size": "log_size"},
                "market_cap_field": "cap",
                "industry_field": "sector",
                "config": {"styles": {"size": {"size": 1}}, "industry_universe": ["a", "b", "c"]},
            },
            "estimation": {
                "industry_constraint": True,
                "regression_weighting": "sqrt_market_cap",
                "factor_half_life": 15,
                "specific_half_life": 12,
                "newey_west_lags": 2,
                "specific_newey_west_lags": 2,
                "risk_horizon": 3,
                "regime_window": 4,
                "regime_min_periods": 8,
                "specific_target": "industry",
                "specific_shrinkage_method": "bayesian",
            },
            "factor_bounds": {"size": [-2, 2]},
            "benchmark_weights": {symbol: 1 / 12 for symbol in symbols},
            "max_tracking_error": 10,
        },
    }
    return raw, dates, symbols, validate_history(pd.DataFrame(histories)), recipe


def test_complete_history_to_model_to_allocator_and_target_check():
    raw, dates, symbols, history, recipe = fixture()
    schedule = build_risk_schedule(
        raw, recipe, [dates[-2], dates[-1]], raw_prices=raw, history=history
    )
    item = schedule[dates[-1].date()]
    model = item["model"]
    inputs = allocation_risk_inputs(item, symbols, {})
    snapshot = inputs["risk_snapshot"]
    pd.testing.assert_frame_equal(snapshot.exposures, model.exposures)
    pd.testing.assert_frame_equal(snapshot.factor_covariance, model.factor_covariance)
    pd.testing.assert_series_equal(snapshot.specific_variance, model.specific_variances)
    assert snapshot.matrix(symbols, model.as_of).variance(
        np.ones(len(symbols)) / len(symbols)
    ) == pytest.approx(
        float(
            (np.ones(len(symbols)) / len(symbols))
            @ model.asset_covariance
            @ (np.ones(len(symbols)) / len(symbols))
        )
    )
    assert inputs["factor_bounds"]["size"] == (-2, 2)
    report, alerts = check_model_target(item, {symbol: 1 / 12 for symbol in symbols})
    assert alerts == []
    assert report["risk_horizon"] == 3
    assert report["horizon_variance"] == pytest.approx(
        report["portfolio_risk"]["total_variance"] * 3 / 252
    )
    assert report["tracking_risk"]["volatility"] == pytest.approx(0, abs=1e-12)
    diagnostics = model.diagnostics.estimation
    assert diagnostics["specific_risk"]["method"] == "bayesian"
    assert "research_history_lineage" in diagnostics["exposure_provenance"]
    assert diagnostics["factor_covariance"]["newey_west_lags"] == 2


def test_future_history_revision_cannot_change_earlier_risk_forecasts():
    raw, dates, _, history, recipe = fixture()
    before = build_risk_schedule(raw, recipe, [dates[-2]], raw_prices=raw, history=history)
    future = history.copy()
    future["available_at"] = pd.Timestamp("2026-01-01T00:00:00Z")
    future.loc[future.field.eq("log_size"), "value"] = "1000"
    future.loc[future.field.eq("sector"), "value"] = "unknown"
    after = build_risk_schedule(
        raw, recipe, [dates[-2]], raw_prices=raw, history=pd.concat([history, future])
    )
    assert before[dates[-2].date()]["model"].to_dict() == after[dates[-2].date()]["model"].to_dict()


def test_three_percent_tracking_limit_reaches_solver_without_dense_covariance(monkeypatch):
    from quant_portfolio.allocation_v2 import allocate_research

    raw, dates, symbols, history, recipe = fixture()
    recipe["risk_model"]["max_tracking_error"] = 0.03
    schedule = build_risk_schedule(raw, recipe, [dates[-1]], raw_prices=raw, history=history)
    item = schedule[dates[-1].date()]

    def forbidden(_):
        raise AssertionError("optimizer bridge accessed full asset covariance")

    monkeypatch.setattr(type(item["model"]), "asset_covariance", property(forbidden))
    inputs = allocation_risk_inputs(item, symbols[:8], {})
    result = allocate_research(
        pd.Series(np.linspace(0, 1, 8), index=symbols[:8]),
        None,
        pd.Series({symbols[-1]: 0.1}),
        None,
        invested_limit=0.8,
        max_weight=0.2,
        config={
            "protocol_version": 2,
            "mode": "cost_aware",
            "risk_objective": "active",
            "forecast_horizon_sessions": 3,
            "expected_return_model": "provided",
            "max_turnover": 2,
        },
        expected_returns=pd.Series(0.002, index=symbols[:8]),
        **inputs,
    )
    report, alerts = check_model_target(item, result.weights)
    assert alerts == []
    assert result.diagnostics["annualized_tracking_error"] <= 0.03 + 1e-7
    assert report["tracking_risk"]["volatility"] == pytest.approx(
        result.diagnostics["annualized_tracking_error"]
    )
    assert set(result.weights.index) == set(symbols)
    assert result.weights.sum() == pytest.approx(0.8)


def test_original_history_age_and_source_coverage_are_not_reset_at_each_close():
    raw, dates, _, history, recipe = fixture()
    with pytest.raises(ValueError, match="original PIT history"):
        build_risk_schedule(raw, recipe, [dates[-1]], raw_prices=raw)
    stale = deepcopy(recipe)
    stale["risk_model"]["exposure_pipeline"]["config"]["max_descriptor_age_days"] = 2
    with pytest.raises(ValueError, match="Stale original"):
        build_risk_schedule(raw, stale, [dates[-1]], raw_prices=raw, history=history)
    incomplete = history.loc[~(history.symbol.eq("S0") & history.field.eq("cap"))]
    with pytest.raises(ValueError, match="Missing original PIT descriptor"):
        build_risk_schedule(raw, recipe, [dates[-1]], raw_prices=raw, history=incomplete)
