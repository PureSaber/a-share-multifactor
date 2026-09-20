import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from quant_lab import load_and_validate_standard_run

from a_share_multifactor import decision_workflow as flow
from a_share_multifactor import run_contract
from a_share_multifactor.decision_contract import validate_decision


def _inputs(path, *, last=79, origin="live_public_api"):
    path.mkdir()
    dates = pd.bdate_range("2025-01-02", periods=100)
    rng = np.random.default_rng(22)
    rows = []
    for k, symbol in enumerate(["000001", "000333", "600036", "601318"]):
        series = (10 + k * 5) * np.exp(np.cumsum(rng.normal(0.001, 0.005, 100)))
        for day, price in zip(dates[: last + 1], series[: last + 1]):
            price = round(price, 2)
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "open": price,
                    "high": price + 0.1,
                    "low": price - 0.1,
                    "close": price,
                    "volume": 1000000,
                    "volume_unit": "share",
                    "source": "test_fixture",
                    "adjustment": "none",
                }
            )
    raw = pd.DataFrame(rows)
    frames = {
        "raw": raw,
        "adjusted": raw.assign(adjustment="qfq"),
        "calendar": pd.DataFrame({"date": dates}),
        "benchmark": pd.DataFrame({"date": dates[: last + 1], "benchmark_return": 0.001}),
    }
    manifest = {"origin": origin, "captured_at": str(dates[last]), "files": {}}
    for name, frame in frames.items():
        file = path / f"{name}.parquet"
        frame.to_parquet(file, index=False)
        manifest["files"][name] = {"file": file.name, "sha256": flow.sha256(file)}
    flow._save_json(path / "manifest.json", manifest)
    return dates, frames


@pytest.fixture
def setup_decision(tmp_path, monkeypatch):
    config = yaml.safe_load(Path("configs/decision_watchlist.yaml").read_text(encoding="utf-8"))
    config["frequency"] = "daily"
    config["app"]["validation"].update(train_size=25, test_size=15, step_size=15, embargo_size=5)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(run_contract, "_code_version", lambda *_: "a" * 40)
    monkeypatch.setattr(flow, "_code_version", lambda *_: "a" * 40)
    monkeypatch.setattr(flow, "_dependency_revisions", lambda: {"quant-data-kit": "a" * 40})
    return path, tmp_path / "output"


def _run(config, output, inputs, dates, last=79):
    return flow.run_decision(
        config,
        output,
        as_of=str(dates[last].date()),
        inputs=inputs,
        now=dates[last].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=17),
    )


def test_trading_status_must_be_complete_same_session_and_fresh():
    now = pd.Timestamp("2026-09-18T17:00:00+08:00")
    rows = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "session": "2026-09-18",
                "status": "no_reported_restriction",
                "captured_at": "2026-09-18T16:30:00+08:00",
                "source": "fixture",
            }
        ]
    )
    result = flow.validate_trading_status(
        rows,
        symbols=["000001"],
        as_of=pd.Timestamp("2026-09-18"),
        now=now,
        required=True,
        max_age_hours=8,
    )
    assert result[0]["symbol"] == "000001"

    stale = rows.assign(captured_at="2026-09-17T16:30:00+08:00")
    with pytest.raises(ValueError, match="Stale"):
        flow.validate_trading_status(
            stale,
            symbols=["000001"],
            as_of=pd.Timestamp("2026-09-18"),
            now=now,
            required=True,
            max_age_hours=8,
        )

    restricted = rows.assign(status="suspended")
    with pytest.raises(ValueError, match="restriction"):
        flow.validate_trading_status(
            restricted,
            symbols=["000001"],
            as_of=pd.Timestamp("2026-09-18"),
            now=now,
            required=True,
            max_age_hours=8,
        )


def test_forward_account_freezes_signals_and_advances_the_same_ledger(setup_decision, tmp_path):
    config, output = setup_decision
    first = tmp_path / "first"
    dates, _ = _inputs(first)
    run = _run(config, output, first, dates)
    card = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    assert card["status"] == "paper_ready", card["reasons"]
    validate_decision(card)
    assert card["validation"]["forward_observation_days"] == 0
    assert card["current_positions"] == []
    assert card["proposed_trades"]
    assert card["risk"]["rebalance_frequency"] == "daily"
    assert "early_exit_enabled" not in card["risk"]["allocation"]
    for trade in card["proposed_trades"]:
        assert trade["estimated_execution_price"] >= trade["reference_close"] * 1.001
        assert trade["estimated_slippage"] == pytest.approx(
            (trade["estimated_execution_price"] - trade["reference_close"]) * trade["quantity"]
        )
    manifest = load_and_validate_standard_run(run)
    assert manifest.profile == "backtest-ledger"
    metrics = json.loads((run / "standard/v2/metrics.json").read_text(encoding="utf-8"))
    assert metrics["ic_summary"] and metrics["ic_decay"]
    assert metrics["backtest_stats"][0]["total_return"] == 0
    saved = pd.read_parquet(run / "scored_panel.parquet")

    second = tmp_path / "second"
    _inputs(second, last=80)
    next_run = _run(config, output, second, dates, last=80)
    next_card = json.loads((next_run / "decision.json").read_text(encoding="utf-8"))
    assert next_card["status"] != "blocked", next_card["reasons"]
    assert next_card["validation"]["forward_observation_days"] == 1
    assert next_card["current_positions"]
    new = pd.read_parquet(next_run / "scored_panel.parquet")
    pd.testing.assert_frame_equal(saved, new.loc[new.date <= dates[79]].reset_index(drop=True))
    fills = pd.read_parquet(next_run / "standard/v2/fills.parquet")
    planned_ids = {trade["order_id"] for trade in card["proposed_trades"]}
    assert set(fills.order_id) == planned_ids
    cash = pd.read_parquet(next_run / "standard/v2/cash_ledger.parquet")
    assert (cash.groupby(["transaction_id", "currency"]).amount_units.sum() == 0).all()


@pytest.mark.parametrize("problem", ["stale", "duplicate", "unit", "negative", "missing_session"])
def test_bad_market_data_cannot_create_advice(tmp_path, problem):
    dates, frames = _inputs(tmp_path / "input")
    raw = frames["raw"].copy()
    if problem == "stale":
        raw = raw[raw.date < dates[79]]
    elif problem == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif problem == "unit":
        raw["volume_unit"] = "lot"
    elif problem == "negative":
        raw.loc[0, "close"] = -1
    else:
        raw = raw.drop(index=5)
    frames["raw"] = raw
    with pytest.raises(ValueError):
        flow.validate_inputs(frames, list(raw.symbol.unique()), dates[79])


def test_input_tampering_blocks_and_replaces_latest_pointer(setup_decision, tmp_path):
    config, output = setup_decision
    inputs = tmp_path / "inputs"
    dates, _ = _inputs(inputs)
    (inputs / "raw.parquet").write_bytes(b"tampered")
    output.mkdir()
    flow._save_json(output / "latest.json", {"status": "paper_ready", "decision": "old"})
    run = _run(config, output, inputs, dates)
    card = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    assert card["status"] == "blocked" and not card["targets"]
    assert "integrity" in card["reasons"][0]
    pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    assert pointer["status"] == "blocked"


def test_provider_timeout_produces_a_blocked_card(setup_decision, monkeypatch):
    config, output = setup_decision

    def unavailable(*_args):
        raise subprocess.TimeoutExpired("provider", 1)

    monkeypatch.setattr(flow, "fetch_inputs", unavailable)
    run = flow.run_decision(config, output, now=pd.Timestamp("2025-04-30T09:00:00Z"))
    card = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    assert card["status"] == "blocked" and card["proposed_trades"] == []


def test_corporate_action_window_remains_blocked(setup_decision, tmp_path):
    config, output = setup_decision
    settings = yaml.safe_load(config.read_text())
    settings["simulation_sessions"] = 2
    config.write_text(yaml.safe_dump(settings))
    inputs = tmp_path / "inputs"
    dates, frames = _inputs(inputs)
    adjusted = frames["adjusted"]
    adjusted.loc[adjusted.date == dates[79], "close"] *= 0.9
    file = inputs / "adjusted.parquet"
    adjusted.to_parquet(file, index=False)
    manifest = json.loads((inputs / "manifest.json").read_text())
    manifest["files"]["adjusted"]["sha256"] = flow.sha256(file)
    flow._save_json(inputs / "manifest.json", manifest)
    run = _run(config, output, inputs, dates)
    card = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    assert card["status"] == "blocked"
    assert "Corporate-action" in card["reasons"][0]
    assert not (output / "paper_state.json").exists()


@pytest.mark.parametrize("change", ["code", "dependency", "legacy"])
def test_account_cannot_rewrite_history_with_changed_execution(
    setup_decision, tmp_path, monkeypatch, change
):
    config, output = setup_decision
    inputs = tmp_path / "inputs"
    dates, _ = _inputs(inputs)
    _run(config, output, inputs, dates)
    state_path = output / "paper_state.json"
    if change == "code":
        monkeypatch.setattr(flow, "_code_version", lambda *_: "b" * 40)
    elif change == "dependency":
        monkeypatch.setattr(flow, "_dependency_revisions", lambda: {"quant-data-kit": "b" * 40})
    else:
        state = json.loads(state_path.read_text())
        del state["execution_identity"]
        flow._save_json(state_path, state)
    original_state = state_path.read_bytes()
    run = flow.run_decision(
        config,
        output,
        inputs=inputs,
        now=dates[79].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=18),
    )
    card = json.loads((run / "decision.json").read_text())
    assert card["status"] == "blocked" and "code/dependencies" in card["reasons"][0]
    assert not card["proposed_trades"]
    assert state_path.read_bytes() == original_state
    assert json.loads((output / "latest.json").read_text())["status"] == "blocked"


def test_dependency_provenance_supports_installed_distributions(monkeypatch):
    # Installed VCS dependencies are wheels, not sibling Git checkouts.
    monkeypatch.setattr(flow, "_installed_internal_dependencies", lambda: {"quant-lab": "a" * 40})
    assert flow._dependency_revisions() == {"quant-lab": "a" * 40}


@pytest.mark.parametrize("count", [0, 100000])
def test_invalid_replay_window_replaces_old_actionable_card(setup_decision, tmp_path, count):
    config, output = setup_decision
    settings = yaml.safe_load(config.read_text())
    settings["simulation_sessions"] = count
    config.write_text(yaml.safe_dump(settings))
    inputs = tmp_path / "inputs"
    dates, _ = _inputs(inputs)
    output.mkdir()
    flow._save_json(output / "latest.json", {"status": "paper_ready"})
    run = _run(config, output, inputs, dates)
    card = json.loads((run / "decision.json").read_text())
    assert card["status"] == "blocked" and "simulation_sessions" in card["reasons"][0]
    assert json.loads((output / "latest.json").read_text())["status"] == "blocked"


@pytest.mark.parametrize("problem", ["missing", "malformed", "empty", "unsupported"])
def test_config_refresh_failure_cannot_leave_an_old_buy_card(setup_decision, problem):
    config, output = setup_decision
    output.mkdir()
    flow._save_json(output / "latest.json", {"status": "paper_ready"})
    if problem == "missing":
        config.unlink()
    elif problem == "malformed":
        config.write_text("app: [invalid")
    elif problem == "empty":
        config.write_text("")
    else:
        settings = yaml.safe_load(config.read_text())
        settings["frequency"] = "unsupported"
        config.write_text(yaml.safe_dump(settings))
    run = flow.run_decision(config, output, now=pd.Timestamp("2025-04-30T09:00:00Z"))
    card = json.loads((run / "decision.json").read_text())
    assert card["status"] == "blocked" and card["proposed_trades"] == []
    assert json.loads((output / "latest.json").read_text())["status"] == "blocked"
