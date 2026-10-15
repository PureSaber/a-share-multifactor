import pandas as pd
import pytest
from test_certified_execution import _certified_panel

from a_share_multifactor.config import AppConfig
from a_share_multifactor.run_contract import _replay


def test_daily_recording_preserves_exact_execution_and_fee_attribution():
    panel = _certified_panel()
    config = AppConfig(factors=["market_cap"], quantiles=5, rebalance_freq="monthly")
    full = _replay(panel, config, "recording-parity")
    daily = _replay(panel, config, "recording-parity", snapshot_frequency="daily")
    assert full.result.ledger_sha256 == daily.result.ledger_sha256
    assert full.result.fill_sha256 == daily.result.fill_sha256
    for name in ("orders", "fills", "costs", "cash_ledger"):
        pd.testing.assert_frame_equal(full.frames[name], daily.frames[name])
    pd.testing.assert_frame_equal(
        full.frames["returns"], daily.frames["returns"], atol=1e-14, rtol=1e-12
    )
    assert len(daily.ledger.recorded_snapshots) == panel.date.nunique()
    assert len(daily.frames["positions"]) < len(full.frames["positions"])
    state = daily.ledger.capture_state()
    before = daily.ledger.recorded_snapshots
    daily.ledger.restore_state(state)
    assert daily.ledger.recorded_snapshots == before


def test_unknown_snapshot_frequency_is_rejected():
    with pytest.raises(ValueError, match="snapshot_frequency"):
        _replay(_certified_panel(), AppConfig(), "bad-frequency", snapshot_frequency="weekly")


def test_event_scoped_risk_view_cache_preserves_every_account_fact():
    from quant_data_kit import FixedPoint

    panel = _certified_panel()
    config = AppConfig(factors=["market_cap"], quantiles=2, rebalance_freq="monthly")
    original = _replay(panel, config, "cached-parity", snapshot_frequency="daily")
    cached = _replay(
        panel, config, "cached-parity", snapshot_frequency="daily", cache_risk_views=True
    )
    assert original.result == cached.result
    for key in original.frames:
        pd.testing.assert_frame_equal(original.frames[key], cached.frames[key])
    ledger = cached.ledger
    at = cached.events[-1].available_at
    snapshot = ledger.snapshot(at)
    assert ledger.snapshot(at) is snapshot
    assert ledger.portfolio_risk_snapshot(at) is ledger.portfolio_risk_snapshot(at)
    state = ledger.capture_state()
    ledger.book_external_cash(
        transfer_id="test-deposit", amount=FixedPoint(10000, 2), currency="CNY", event_time=at
    )
    assert ledger.snapshot(at).nav.to_decimal() == snapshot.nav.to_decimal() + 100
    ledger.restore_state(state)
    assert ledger.snapshot(at) == snapshot
