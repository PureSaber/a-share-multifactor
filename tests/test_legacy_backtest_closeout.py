import pandas as pd
import pytest

from a_share_multifactor.config import AppConfig, CostsConfig
from a_share_multifactor.quantile_backtest import (
    _prepare_rebalance_day,
    run_long_only_backtest,
    run_quantile_backtest,
)


def panel():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-30"] * 4),
            "symbol": list("ABCD"),
            "composite_score": [1, 2, 3, 4],
            "period_return": [-0.03, -0.03, 0.03, 0.03],
        }
    )


def test_long_short_pays_both_legs_costs():
    values = []
    for commission in (0, 0.01, 0.05):
        config = AppConfig(
            quantiles=2,
            rebalance_freq="monthly",
            holding_period="rebalance",
            costs=CostsConfig(commission=commission, slippage=0, retail_mode=False),
        )
        values.append(run_quantile_backtest(panel(), config).long_short.iloc[0])
    assert values == pytest.approx([0.06, 0.02, -0.14])


def test_future_missingness_cannot_change_selected_holdings():
    data = panel()
    config = AppConfig(quantiles=2, rebalance_freq="monthly", holding_period="rebalance")
    before = _prepare_rebalance_day(
        data, data.date.iloc[0], "composite_score", "period_return", config
    )
    data.loc[3, "period_return"] = float("nan")
    after = _prepare_rebalance_day(
        data, data.date.iloc[0], "composite_score", "period_return", config
    )
    assert before[["symbol", "quantile"]].equals(after[["symbol", "quantile"]])
    with pytest.raises(ValueError, match="no reweighting"):
        run_quantile_backtest(data, config)


def test_missing_selected_leg_and_internal_gaps_do_not_disappear():
    config = AppConfig(
        quantiles=2,
        rebalance_freq="monthly",
        holding_period="rebalance",
        costs=CostsConfig(retail_mode=False),
    )
    data = panel()
    data.loc[data.composite_score > 2, "period_return"] = float("nan")
    with pytest.raises(ValueError, match="no reweighting"):
        run_long_only_backtest(data, config)
    missing = panel().assign(date=pd.Timestamp("2026-02-27"), period_return=float("nan"))
    future = panel().assign(date=pd.Timestamp("2026-03-31"))
    with pytest.raises(ValueError, match="no reweighting"):
        run_quantile_backtest(pd.concat([panel(), missing, future]), config)
    result = run_quantile_backtest(pd.concat([panel(), missing]), config)
    assert list(result.long_short.index) == [pd.Timestamp("2026-01-30")]
