"""Risk estimates must not turn vendor delivery time into exchange close time."""

import numpy as np
import pandas as pd
import pytest

from a_share_multifactor.research_risk import build_risk_schedule


def _inputs():
    dates = pd.bdate_range("2025-01-02", periods=35)
    raw = pd.DataFrame(
        [
            {
                "date": day,
                "symbol": symbol,
                "close": 100 + index * loading + np.sin(index),
                "adjustment": "none",
                "available_at": day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21),
            }
            for symbol, loading in zip("ABCD", [0.2, 0.5, 0.7, 1.0])
            for index, day in enumerate(dates)
        ]
    )
    recipe = {"risk_model": {"model_kind": "statistical_proxy", "lookback": 20}}
    return raw, recipe, dates


def _model(raw, recipe, dates, research=None):
    return build_risk_schedule(
        raw.copy() if research is None else research,
        recipe,
        [dates[-1]],
        raw_prices=raw,
    )[dates[-1].date()]["model"]


def test_evening_arrival_cannot_change_same_close_estimate():
    raw, recipe, dates = _inputs()
    model = _model(raw, recipe, dates)
    assert model.diagnostics.periods == 20
    assert model.diagnostics.last_period_end == dates[-2].tz_localize("UTC") + pd.Timedelta(hours=7)
    poisoned = raw.copy()
    poisoned.loc[poisoned.date.eq(dates[-1]), "close"] *= 100
    changed = _model(poisoned, recipe, dates)
    pd.testing.assert_frame_equal(model.asset_covariance, changed.asset_covariance)


def test_both_return_endpoint_arrivals_are_required():
    raw, recipe, dates = _inputs()
    # A late correction to yesterday's closing price also makes today's return late.
    affected = raw.date.eq(dates[-2]) & raw.symbol.eq("A")
    raw.loc[affected, "available_at"] = dates[-1].tz_localize("Asia/Shanghai") + pd.Timedelta(
        days=1
    )
    model = _model(raw, recipe, dates)
    assert model.diagnostics.last_period_end == dates[-3].tz_localize("UTC") + pd.Timedelta(hours=7)
    assert model.diagnostics.periods == 20


@pytest.mark.parametrize("bad", [None, "2025-01-02T15:00:00", "2025-01-02T06:59:59Z"])
def test_explicit_invalid_price_availability_is_rejected(bad):
    raw, recipe, dates = _inputs()
    raw["available_at"] = raw.available_at.astype(object)
    raw.loc[0, "available_at"] = bad
    with pytest.raises(ValueError, match="available_at"):
        _model(raw, recipe, dates)


def test_explicit_late_feature_is_not_selected_for_current_model():
    raw, recipe, dates = _inputs()
    raw = raw.drop(columns="available_at")
    research = raw.copy()
    research["size"] = research.symbol.map(dict(zip("ABCD", [-2, -1, 1, 2])))
    research["size_available_at"] = research.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(
        hours=15
    )
    latest = research.date.eq(dates[-1])
    research.loc[latest, "size_available_at"] += pd.Timedelta(days=1)
    recipe["risk_model"]["exposure_fields"] = {"size": "size"}
    recipe["required_history"] = {"size": "fundamentals"}
    model = _model(raw, recipe, dates, research)
    research.loc[latest, "size"] = [13, -9, 8, -1]
    changed = _model(raw, recipe, dates, research)
    pd.testing.assert_frame_equal(model.asset_covariance, changed.asset_covariance)
    assert model.diagnostics.exposure_effective_at < dates[-1].tz_localize("UTC")


def test_equal_close_arrival_preserves_legacy_estimate():
    raw, recipe, dates = _inputs()
    raw["available_at"] -= pd.Timedelta(hours=6)
    actual = _model(raw, recipe, dates)
    legacy = _model(raw.drop(columns="available_at"), recipe, dates)
    pd.testing.assert_frame_equal(actual.asset_covariance, legacy.asset_covariance)


def test_late_entitlement_does_not_enter_an_earlier_risk_fit():
    raw, recipe, dates = _inputs()
    raw = raw.drop(columns="available_at")
    actions = pd.DataFrame(
        [
            {
                "symbol": "A",
                "ex_date": dates[-1],
                "announced_date": dates[-3],
                "record_date": dates[-2],
                "cash_per_share": 2.0,
                "share_ratio": 1.0,
                "available_at": dates[-1].tz_localize("UTC") + pd.Timedelta(days=1),
            }
        ]
    )

    def run():
        return build_risk_schedule(raw, recipe, [dates[-1]], raw_prices=raw, actions=actions)[
            dates[-1].date()
        ]["model"]

    original = run()
    actions["cash_per_share"] = 200.0
    changed = run()
    pd.testing.assert_frame_equal(original.asset_covariance, changed.asset_covariance)
    assert original.diagnostics.last_period_end == dates[-2].tz_localize("UTC") + pd.Timedelta(
        hours=7
    )
