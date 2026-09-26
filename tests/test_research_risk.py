import numpy as np
import pandas as pd
import pytest
from quant_risk_monitor import BarraStyleRiskModel, FactorModelDiagnostics

from a_share_multifactor.config import AppConfig
from a_share_multifactor.research_risk import (
    build_risk_schedule,
    causal_asset_returns,
    check_model_target,
)
from a_share_multifactor.run_contract import _replay


def _market_risk_model():
    assets = ["000001", "000002", "000003", "000004"]
    exposures = pd.DataFrame({"market": 1.0}, index=assets)
    factor_covariance = pd.DataFrame([[0.16]], index=["market"], columns=["market"])
    specific_variances = pd.Series(0.04, index=assets)
    asset_covariance = pd.DataFrame(
        exposures.to_numpy() @ factor_covariance.to_numpy() @ exposures.to_numpy().T
        + np.diag(specific_variances),
        index=assets,
        columns=assets,
    )
    as_of = pd.Timestamp("2019-12-31T07:00:00Z")
    return BarraStyleRiskModel(
        model_kind="statistical_proxy",
        as_of=as_of,
        annualization=252,
        exposures=exposures,
        factor_returns=pd.DataFrame(
            {"market": [0.01, -0.01]},
            index=pd.to_datetime(["2019-12-27", "2019-12-30"], utc=True),
        ),
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        asset_covariance=asset_covariance,
        diagnostics=FactorModelDiagnostics(
            model_kind="statistical_proxy",
            as_of=as_of,
            factors=("market",),
            assets=tuple(assets),
            periods=2,
            first_period_start=pd.Timestamp("2019-12-26T07:00:00Z"),
            last_period_end=pd.Timestamp("2019-12-30T07:00:00Z"),
            exposure_effective_at=pd.Timestamp("2019-12-26T07:00:00Z"),
            exposure_available_at=pd.Timestamp("2019-12-26T07:00:00Z"),
            cross_section_sizes=(4, 4),
            regression_ranks=(1, 1),
            condition_numbers=(1.0, 1.0),
            residual_observations={asset: 2 for asset in assets},
            annualization=252,
            covariance_shrinkage=0.2,
            specific_variance_shrinkage=0.2,
        ),
    )


def _risk_item(*, absolute_lower=0.0):
    return {
        "model": _market_risk_model(),
        "settings": {
            "factor_bounds": {"market": [absolute_lower, 1.1]} if absolute_lower else {},
            "active_factor_bounds": {"market": [-0.5, 0.1]},
            "benchmark_weights": {
                "000001": 0.25,
                "000002": 0.25,
                "000003": 0.25,
                "000004": 0.25,
            },
            "max_tracking_error": 0.1,
        },
    }


def test_future_adjusted_vintage_does_not_change_past_risk_model():
    dates = pd.bdate_range("2025-01-02", periods=32)
    raw = pd.DataFrame(
        [
            {
                "date": day,
                "symbol": symbol,
                "close": 100 + index * loading + np.sin(index),
                "adjustment": "none",
            }
            for symbol, loading in zip("ABCD", [0.2, 0.5, 0.7, 1.0])
            for index, day in enumerate(dates)
        ]
    )
    recipe = {"risk_model": {"model_kind": "statistical_proxy", "lookback": 20}}
    original = build_risk_schedule(raw.copy(), recipe, [dates[-1]], raw_prices=raw)
    revised = raw.copy()
    revised.loc[revised.symbol.eq("A"), "close"] -= 10
    later = build_risk_schedule(revised, recipe, [dates[-1]], raw_prices=raw)
    pd.testing.assert_frame_equal(
        original[dates[-1].date()]["model"].asset_covariance,
        later[dates[-1].date()]["model"].asset_covariance,
    )


def test_entitlement_returns_use_ex_date_and_reject_unknown_announcement():
    dates = pd.bdate_range("2025-01-02", periods=3)
    raw = pd.DataFrame(
        {"date": dates, "symbol": "A", "close": [100.0, 49.0, 50.0], "adjustment": "none"}
    )
    actions = pd.DataFrame(
        [
            {
                "symbol": "A",
                "ex_date": dates[1],
                "announced_date": dates[0],
                "record_date": dates[0],
                "cash_per_share": 2,
                "share_ratio": 2,
            }
        ]
    )
    returns = causal_asset_returns(raw, actions)
    assert returns.loc[dates[1], "A"] == 0.0
    assert returns.loc[dates[2], "A"] == pytest.approx(50 / 49 - 1)
    actions["announced_date"] = dates[1]
    with pytest.raises(ValueError, match="announcement"):
        causal_asset_returns(raw, actions)
    with pytest.raises(ValueError, match="unadjusted"):
        causal_asset_returns(raw.assign(adjustment="qfq"))


def test_full_cash_only_downgrades_benchmark_relative_risk():
    _, alerts = check_model_target(
        _risk_item(),
        {},
        cash_policy_reason="full_cash_exit",
    )
    assert {alert["rule_id"] for alert in alerts} == {
        "portfolio.active_factor_bounds",
        "portfolio.max_tracking_error",
    }
    assert all(alert["severity"] == "warning" for alert in alerts)
    assert all(alert["details"]["policy_reason"] == "full_cash_exit" for alert in alerts)
    tracking_error = next(
        alert for alert in alerts if alert["rule_id"] == "portfolio.max_tracking_error"
    )
    assert tracking_error["details"]["actual"] > tracking_error["details"]["limit"]

    _, absolute_alerts = check_model_target(_risk_item(absolute_lower=0.5), {})
    absolute = next(
        alert for alert in absolute_alerts if alert["rule_id"] == "portfolio.factor_bounds"
    )
    assert absolute["severity"] == "critical"
    assert "policy_reason" not in absolute["details"]
    assert all(
        alert["severity"] == "warning"
        for alert in absolute_alerts
        if alert["rule_id"] != "portfolio.factor_bounds"
    )


def test_nonempty_tracking_error_stays_critical_and_invalid_model_fails_closed():
    _, alerts = check_model_target(_risk_item(), {"000001": 1.0})
    tracking_error = next(
        alert for alert in alerts if alert["rule_id"] == "portfolio.max_tracking_error"
    )
    assert tracking_error["severity"] == "critical"
    assert "policy_reason" not in tracking_error["details"]

    _, negative_alerts = check_model_target(_risk_item(), {"000001": -1.0})
    assert any(alert["severity"] == "critical" for alert in negative_alerts)
    assert all("policy_reason" not in alert["details"] for alert in negative_alerts)
    with pytest.raises(ValueError, match="finite"):
        check_model_target(_risk_item(), {"000001": np.nan})

    invalid = _risk_item()
    covariance = object.__getattribute__(invalid["model"], "_factor_covariance")
    covariance.iloc[0, 0] *= 2
    with pytest.raises(ValueError, match="inconsistent with X F X.T \\+ D"):
        check_model_target(invalid, {})


def test_full_cash_target_exits_real_ledger_and_realized_cash_is_warning_only():
    dates = pd.bdate_range("2020-01-02", periods=5)
    assets = ["000001", "000002", "000003", "000004"]
    panel = pd.DataFrame(
        [
            {
                "date": day,
                "symbol": symbol,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1_000_000,
                "composite_score": 1.0,
                "adjustment": "none",
                "decision_allowed": True,
            }
            for day in dates
            for symbol in assets
        ]
    )
    item = _risk_item()
    replay = _replay(
        panel,
        AppConfig(rebalance_freq="daily"),
        "full-cash-risk-policy",
        target_schedule={
            dates[0].date(): {symbol: 2_400 for symbol in assets},
            dates[2].date(): {},
        },
        risk_schedule={day.date(): item for day in dates},
    )

    exit_orders = replay.frames["orders"].loc[lambda frame: frame["reduce_only"]]
    assert len(exit_orders) == len(assets)
    assert exit_orders["side"].eq("sell").all()
    assert exit_orders["status"].eq("filled").all()
    assert (
        replay.frames["fills"].loc[lambda frame: frame.side.eq("sell")]["quantity_units"].sum()
        == 9_600
    )
    assert all(
        quantity.units == 0
        for quantity in replay.ledger.snapshot(replay.events[-1].available_at).positions.values()
    )

    exit_check = next(
        check
        for check in replay.risk_checks
        if check.get("stage") == "target"
        and any(
            alert.get("details", {}).get("policy_reason") == "full_cash_exit"
            for alert in check["alerts"]
        )
    )
    assert not exit_check["has_critical"]
    assert all(alert["severity"] == "warning" for alert in exit_check["alerts"])
    realized_cash = [
        check
        for check in replay.risk_checks
        if check.get("stage") == "realized"
        and any(
            alert.get("details", {}).get("policy_reason") == "full_cash_state"
            for alert in check["alerts"]
        )
    ]
    assert realized_cash
    assert all(not check["has_critical"] for check in realized_cash)
    assert all(
        alert["severity"] == "warning" for check in realized_cash for alert in check["alerts"]
    )
