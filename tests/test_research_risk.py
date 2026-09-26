import numpy as np
import pandas as pd
import pytest

from a_share_multifactor.research_risk import build_risk_schedule, causal_asset_returns


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
