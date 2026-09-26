from dataclasses import replace

import pandas as pd
import pytest

from a_share_multifactor.config import AppConfig
from a_share_multifactor.corporate_actions import action_events
from a_share_multifactor.decision_workflow import write_watchlist_catalog
from a_share_multifactor.run_contract import _replay


@pytest.mark.parametrize("ratio,ex_price,quantity", [("2", 4.5, 1000), ("1", 9, 500)])
def test_cash_and_split_replay_preserves_nav_and_balances(tmp_path, ratio, ex_price, quantity):
    dates = pd.bdate_range("2026-07-08", periods=4)
    rows = []
    for day, price in zip(dates, [10, 10, ex_price, ex_price]):
        rows.append(
            {
                "symbol": "600036",
                "date": day,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000000,
                "composite_score": 1.0 if day == dates[0] else float("nan"),
            }
        )
    raw = pd.DataFrame(rows)
    adjusted = raw.copy()
    adjusted["close"] = ex_price
    actions = pd.DataFrame(
        [
            {
                "event_id": "distribution",
                "symbol": "600036",
                "announced_date": dates[0] - pd.Timedelta(days=3),
                "record_date": dates[1],
                "ex_date": dates[2],
                "pay_date": dates[2],
                "shares_available_date": dates[2],
                "cash_per_share": "1",
                "share_ratio": ratio,
                "source": "fixture",
            }
        ]
    )
    events = action_events(actions, raw, adjusted, dates[0], dates[-1])
    cfg = AppConfig()
    cfg = replace(
        cfg,
        quantiles=1,
        rebalance_freq="daily",
        costs=replace(
            cfg.costs,
            retail_mode=False,
            initial_capital=10000,
            commission=0,
            min_commission=0,
            stamp_tax=0,
            slippage=0,
            max_position_weight=0.5,
            cash_buffer=0,
            max_holdings=1,
        ),
    )
    catalog = write_watchlist_catalog(
        tmp_path, {"watchlist": [{"symbol": "600036", "venue": "SSE"}]}, "2026-07-01", "2026-08-01"
    )
    replay = _replay(raw, cfg, "action-test", catalog_path=catalog, corporate_actions=events)
    snapshot = replay.ledger.snapshot(replay.events[-1].available_at)
    assert snapshot.nav.to_decimal() == 10000
    assert snapshot.positions["600036"].to_decimal() == quantity
    assert snapshot.cash_balances["CNY"].to_decimal() == 5500
    journal = replay.frames["cash_ledger"]
    assert (journal.groupby(["transaction_id", "currency"]).amount_units.sum() == 0).all()
    with pytest.raises(ValueError, match="matching cashflow"):
        action_events(None, raw, adjusted, dates[0], dates[-1])
    actions.loc[0, "pay_date"] = dates[-1]
    deferred = action_events(actions, raw, adjusted, dates[0], dates[-1])
    assert [event.action_type for event in deferred] == [
        "cash_dividend_entitlement",
        "cash_dividend_payment",
    ]
    replay = _replay(raw, cfg, "deferred-action", catalog_path=catalog, corporate_actions=deferred)
    payment_time = deferred[1].available_at
    entitled = max(
        (
            snapshot
            for snapshot in replay.ledger.recorded_snapshots
            if pd.Timestamp(snapshot.event_time).tz_convert("Asia/Shanghai").date()
            == deferred[0].effective_date
        ),
        key=lambda snapshot: snapshot.event_time,
    )
    paid = next(
        snapshot
        for snapshot in replay.ledger.recorded_snapshots
        if snapshot.event_time == payment_time
    )
    assert replay.ledger.dividend_receivable_balance("CNY", instrument_id="600036") == 0
    assert entitled.nav.to_decimal() == paid.nav.to_decimal() == 10000
    assert paid.cash_balances["CNY"].to_decimal() == 5500

    actions.loc[0, "pay_date"] = dates[-1] + pd.Timedelta(days=10)
    unpaid = action_events(actions, raw, adjusted, dates[0], dates[-1])
    assert [event.action_type for event in unpaid] == ["cash_dividend_entitlement"]
    replay = _replay(raw, cfg, "unpaid-action", catalog_path=catalog, corporate_actions=unpaid)
    assert replay.ledger.dividend_receivable_balance("CNY", instrument_id="600036") == 500
    final = replay.ledger.snapshot(replay.events[-1].available_at)
    assert final.nav.to_decimal() == 10000
    assert final.cash_balances["CNY"].to_decimal() == 5000
