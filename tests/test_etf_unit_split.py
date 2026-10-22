import hashlib
from dataclasses import replace
from decimal import Decimal

import pandas as pd
from quant_data_kit.etf_events import adjust_etf_prices, normalize_etf_splits

from a_share_multifactor.config import AppConfig
from a_share_multifactor.corporate_actions import action_events, split_share_rules
from a_share_multifactor.decision_workflow import write_watchlist_catalog
from a_share_multifactor.run_contract import _replay


def test_end_of_day_split_in_native_ledger_and_repeatability(tmp_path):
    source = b"synthetic terms mirror issuer ratio and per-holder ceil"
    (tmp_path / "issuer.txt").write_bytes(source)
    item = {
        "event_id": "split",
        "symbol": "510500",
        "announced_date": "2022-08-23",
        "record_date": "2022-08-26",
        "ex_date": "2022-08-29",
        "credit_at": "2022-08-26T16:00:00+08:00",
        "known_at": "2022-08-24T00:00:00+08:00",
        "sellable_date": "2022-08-29",
        "ratio": "1.14539",
        "fractional_rule": "ceil",
        "source": "synthetic-issuer",
        "file": "issuer.txt",
        "sha256": hashlib.sha256(source).hexdigest(),
    }
    days = pd.bdate_range("2022-08-24", periods=5)
    raw = pd.DataFrame(
        [
            {
                "symbol": "510500",
                "date": day,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000000,
                "amount": price * 1000000,
                "adjustment": "none",
                "source": "synthetic",
                "composite_score": 1.0,
            }
            for day, price in zip(days, [7.2, 7.2, 7.2, 6.3, 6.3])
        ]
    )
    actions = normalize_etf_splits(
        [item], evidence_root=tmp_path, captured_at="2026-10-22T00:00:00Z"
    )
    adjusted = adjust_etf_prices(raw, actions)
    events = action_events(actions, raw, adjusted, days[0], days[-1])
    config = AppConfig()
    config = replace(
        config,
        quantiles=1,
        costs=replace(
            config.costs,
            initial_capital=10000,
            retail_mode=False,
            commission=0,
            min_commission=0,
            stamp_tax=0,
            slippage=0,
            max_position_weight=1,
            cash_buffer=0,
            max_holdings=1,
        ),
    )
    catalog = write_watchlist_catalog(
        tmp_path,
        {"watchlist": [{"symbol": "510500", "venue": "SSE", "asset_class": "etf"}]},
        "2022-08-01",
        "2022-09-01",
    )
    kwargs = {
        "catalog_path": catalog,
        "corporate_actions": events,
        "share_rules": split_share_rules(actions, raw),
        "target_schedule": {days[0].date(): {"510500": 100}},
    }
    result = _replay(raw, config, "unit-split", **kwargs)
    final = result.ledger.snapshot()
    assert final.positions["510500"].to_decimal() == 115
    assert final.cash_balances["CNY"].to_decimal() == 9280
    assert final.nav.to_decimal() == Decimal("10004.5")
    assert all(
        sum(p.amount.to_decimal() for p in t.postings) == 0 for t in result.ledger.transactions
    )
    assert _replay(raw, config, "unit-split", **kwargs).ledger.snapshot() == final
