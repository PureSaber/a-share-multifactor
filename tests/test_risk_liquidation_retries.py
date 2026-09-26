import pandas as pd

from a_share_multifactor.config import AppConfig
from a_share_multifactor.run_contract import _replay


def _partial_liquidation_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=6)
    prices = (10.0, 10.0, 5.0, 5.0, 5.0, 5.0)
    volumes = (100_000, 100_000, 100_000, 1_000, 100_000, 100_000)
    return pd.DataFrame(
        [
            {
                "date": date,
                "symbol": "000001",
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "market_cap": 1_000_000.0,
                "composite_score": 1.0,
                "adjustment": "none",
                "decision_allowed": True,
            }
            for date, price, volume in zip(dates, prices, volumes, strict=True)
        ]
    )


def test_liquidation_retries_from_ledger_after_partial_ioc_fill():
    panel = _partial_liquidation_panel()
    dates = tuple(pd.to_datetime(panel["date"]).dt.date)
    replay = _replay(
        panel,
        AppConfig(rebalance_freq="daily"),
        "liquidation-retries",
        target_schedule={dates[0]: {"000001": 9_000}},
        risk_limits={"max_drawdown": 0.05, "drawdown_action": "liquidate", "max_turnover": 2},
    )

    orders = replay.frames["orders"]
    exits = orders.loc[orders["reduce_only"]].reset_index(drop=True)
    assert exits[["side", "quantity_units", "filled_quantity_units", "status"]].to_dict(
        "records"
    ) == [
        {
            "side": "sell",
            "quantity_units": 9_000,
            "filled_quantity_units": 100,
            "status": "expired",
        },
        {
            "side": "sell",
            "quantity_units": 8_900,
            "filled_quantity_units": 8_900,
            "status": "filled",
        },
    ]
    assert pd.to_datetime(exits["event_time"], utc=True).dt.date.tolist() == [dates[2], dates[3]]

    first_exit_id = exits.iloc[0]["order_id"]
    first_exit_events = replay.frames["order_events"].loc[
        lambda frame: frame["order_id"].eq(first_exit_id)
    ]
    assert first_exit_events["to_status"].tolist() == [
        "accepted",
        "partially_filled",
        "expired",
    ]
    assert first_exit_events.iloc[-1]["reason"] == "IOC remainder expired"

    fills = replay.frames["fills"]
    sell_fills = fills.loc[fills["side"].eq("sell")].reset_index(drop=True)
    assert sell_fills["quantity_units"].tolist() == [100, 8_900]
    assert sell_fills["quantity_units"].sum() == 9_000
    first_sell_day = pd.to_datetime(sell_fills.iloc[0]["event_time"], utc=True).date()
    assert first_sell_day == dates[3]

    risk_events = [check for check in replay.risk_checks if check.get("halt_reason") == "drawdown"]
    assert risk_events
    assert all(check["has_critical"] for check in risk_events)
    assert all(
        any(alert["rule_id"] == "portfolio.drawdown" for alert in check["alerts"])
        for check in risk_events
    )
    halt_time = pd.Timestamp(risk_events[0]["checked_at"])
    post_halt_orders = orders.loc[pd.to_datetime(orders["event_time"], utc=True) >= halt_time]
    assert post_halt_orders["side"].eq("sell").all()
    assert post_halt_orders["reduce_only"].all()

    post_halt_positions = replay.frames["positions"].loc[
        lambda frame: pd.to_datetime(frame["event_time"], utc=True) >= halt_time
    ]
    assert post_halt_positions["quantity_units"].tolist() == [9_000, 8_900, 0, 0]
    assert post_halt_positions["quantity_units"].is_monotonic_decreasing
    final_positions = replay.ledger.snapshot(replay.events[-1].available_at).positions
    assert final_positions["000001"].units == 0
