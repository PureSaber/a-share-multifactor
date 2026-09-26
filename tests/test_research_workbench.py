import json
from pathlib import Path

import pandas as pd
import pytest
from quant_data_kit.research_coverage import import_history
from quant_execution import ReplayError
from quant_lab import load_and_validate_standard_run
from quant_lab.research import candidates, file_hash
from test_decision_workflow import _inputs

from a_share_multifactor import run_contract
from a_share_multifactor.decision_workflow import write_watchlist_catalog
from a_share_multifactor.factors import compute_factors
from a_share_multifactor.research_workbench import (
    EquityResearchExecutor,
    preflight_recipe,
    validate_research_execution,
)


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    dates, _ = _inputs(tmp_path / "inputs")
    settings = {
        "watchlist": [
            {"symbol": s, "venue": v}
            for s, v in [
                ("000001", "SZSE"),
                ("000333", "SZSE"),
                ("600036", "SSE"),
                ("601318", "SSE"),
            ]
        ]
    }
    catalog = write_watchlist_catalog(tmp_path, settings, "2025-01-01", "2026-01-01")
    monkeypatch.setattr(run_contract, "_code_version", lambda *_: "a" * 40)
    monkeypatch.setattr(
        run_contract, "_installed_internal_dependencies", lambda: {"test": "a" * 40}
    )
    return {
        "schema_version": "quant.research-recipe/v1",
        "study_id": "integration-test",
        "hypothesis": "Fixed signals improve net returns",
        "mode": "exploratory",
        "backend": "equity",
        "inputs": {"bundle": str(tmp_path / "inputs"), "catalog": str(catalog)},
        "interval": {"start": str(dates[45].date()), "end": str(dates[79].date())},
        "factors": {"momentum_20d": 1, "volatility_20d": -1},
        "strategy": {
            "family": "rank",
            "frequency": "weekly",
            "top_n": 2,
            "max_weight": 0.25,
            "cash_buffer": 0.5,
            "trend_window": 20,
        },
        "costs": {
            "initial_capital": 100000,
            "commission": 0.0003,
            "min_commission": 5,
            "stamp_tax": 0.0005,
            "slippage": 0.001,
            "participation_rate": 0.01,
        },
        "diagnostics": {"cost_multipliers": [2], "signal_delays": [1]},
        "variants": [],
    }


def execute(executor, recipe, candidate, path):
    path.mkdir()
    return executor(recipe, candidate, path)


def test_candidate_ledgers_cost_stress_and_data_cache(recipe, tmp_path):
    executor = EquityResearchExecutor()
    planned = candidates(recipe)
    base = execute(executor, recipe, planned[0], tmp_path / "base")
    stress = execute(
        executor, recipe, next(c for c in planned if c["name"] == "cost_2x"), tmp_path / "stress"
    )
    assert load_and_validate_standard_run(tmp_path / "base").profile == "backtest-ledger"
    assert base["metrics"]["fills"] > 0
    assert stress["metrics"]["cost_total"] > base["metrics"]["cost_total"]
    assert stress["metrics"]["total_return"] < base["metrics"]["total_return"]
    assert len(executor._features) == 1
    assert base["comparison"]["start"] == recipe["interval"]["start"]
    json.dumps(base, allow_nan=False)


def test_delayed_signal_and_unknown_factor(recipe, tmp_path):
    executor = EquityResearchExecutor()
    candidate = next(c for c in candidates(recipe) if c["name"] == "delay_1")
    result = execute(executor, recipe, candidate, tmp_path / "delay")
    assert result["metrics"]["fills"] > 0
    with pytest.raises(ValueError, match="Unknown"):
        compute_factors(pd.DataFrame(), ["momentun_20d"])


def test_missing_history_and_input_corruption_block(recipe, tmp_path):
    executor = EquityResearchExecutor()
    recipe["required_history"] = {"tradable": "status"}
    with pytest.raises(ValueError, match="preflight"):
        execute(executor, recipe, candidates(recipe)[0], tmp_path / "blocked")
    report = json.loads((tmp_path / "blocked/preflight.json").read_text())
    assert any(r["code"] == "HISTORY_COVERAGE" for r in report["issues"])
    (tmp_path / "inputs/raw.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        execute(executor, recipe, candidates(recipe)[0], tmp_path / "corrupt")


def freeze_history(recipe, tmp_path, *, domain, field, values):
    source = tmp_path / "history.csv"
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "domain": domain,
                "field": field,
                "effective_at": "2025-01-01T00:00:00Z",
                "available_at": "2025-01-01T00:00:00Z",
                "value": value,
            }
            for symbol, value in values.items()
        ]
    ).to_csv(source, index=False)
    root = tmp_path / "history"
    import_history(source, root, provider="test", source_uri="test://history", license_note="test")
    recipe["inputs"]["history"] = str(root)


def freeze_execution_history(
    recipe,
    tmp_path,
    dates,
    *,
    universe_exit=None,
    suspended_until=None,
    delist_day=None,
    unlisted_symbols=(),
):
    symbols = ["000001", "000333", "600036", "601318"]
    fields = {
        "listed": "status_listed",
        "delisted": "status_delisted",
        "tradable": "status_tradable",
        "limit_up": "status_limit_up",
        "limit_down": "status_limit_down",
    }
    rows = []
    for symbol in symbols:
        for semantic, field in fields.items():
            rows.append(
                {
                    "symbol": symbol,
                    "domain": "status",
                    "field": field,
                    "effective_at": "2025-01-01T00:00:00Z",
                    "available_at": "2025-01-01T00:00:00Z",
                    "value": (
                        "true"
                        if semantic in {"listed", "tradable"} and symbol not in unlisted_symbols
                        else "false"
                    ),
                }
            )
        rows.append(
            {
                "symbol": symbol,
                "domain": "universe",
                "field": "research_member",
                "effective_at": "2025-01-01T00:00:00Z",
                "available_at": "2025-01-01T00:00:00Z",
                "value": "true",
            }
        )
        if suspended_until is not None:
            for day, value in ((dates[45], "false"), (suspended_until, "true")):
                timestamp = pd.Timestamp(day).tz_localize("UTC").isoformat()
                rows.append(
                    {
                        "symbol": symbol,
                        "domain": "status",
                        "field": "status_tradable",
                        "effective_at": timestamp,
                        "available_at": timestamp,
                        "value": value,
                    }
                )
        if delist_day is not None and symbol == "000001":
            timestamp = pd.Timestamp(delist_day).tz_localize("UTC").isoformat()
            for field, value in (
                ("status_listed", "false"),
                ("status_delisted", "true"),
                ("status_tradable", "false"),
            ):
                rows.append(
                    {
                        "symbol": symbol,
                        "domain": "status",
                        "field": field,
                        "effective_at": timestamp,
                        "available_at": timestamp,
                        "value": value,
                    }
                )
    if universe_exit is not None:
        timestamp = pd.Timestamp(universe_exit).tz_localize("UTC").isoformat()
        rows.append(
            {
                "symbol": "000001",
                "domain": "universe",
                "field": "research_member",
                "effective_at": timestamp,
                "available_at": timestamp,
                "value": "false",
            }
        )
    source = tmp_path / "execution-history.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    root = tmp_path / "execution-history"
    import_history(
        source, root, provider="test", source_uri="test://execution", license_note="test"
    )
    recipe["inputs"]["history"] = str(root)
    recipe["required_history"] = {
        "research_member": "universe",
        **{field: "status" for field in fields.values()},
    }
    recipe["execution"] = {
        "mode": "dynamic",
        "universe_field": "research_member",
        "status_fields": fields,
        "max_retry_sessions": 5,
    }


def test_embedded_fundamentals_cannot_use_unrelated_history(recipe, tmp_path):
    raw_path = Path(recipe["inputs"]["bundle"]) / "raw.parquet"
    pd.read_parquet(raw_path).assign(pe_ratio=10.0).to_parquet(raw_path, index=False)
    manifest_path = raw_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["raw"]["sha256"] = file_hash(raw_path)
    manifest_path.write_text(json.dumps(manifest))
    freeze_history(
        recipe, tmp_path, domain="classification", field="industry", values={"000001": "bank"}
    )
    recipe["factors"] = {"pe_inv": 1}
    with pytest.raises(ValueError, match="publication-time mapping.*pe_ratio"):
        execute(EquityResearchExecutor(), recipe, candidates(recipe)[0], tmp_path / "blocked-pit")


def test_registered_fundamental_history_is_used(recipe, tmp_path):
    recipe["factors"] = {"pe_inv": 1}
    recipe["required_history"] = {"pe_ratio": "fundamentals"}
    freeze_history(
        recipe,
        tmp_path,
        domain="fundamentals",
        field="pe_ratio",
        values={"000001": 10, "000333": 20, "600036": 30, "601318": 40},
    )
    result = execute(EquityResearchExecutor(), recipe, candidates(recipe)[0], tmp_path / "pit")
    assert result["metrics"]["fills"] > 0
    assert result["factor_evidence"]["coverage"][0]["coverage"] == 1


def test_preflight_and_current_nav_allocation_retry_suspended_orders(recipe, tmp_path):
    calendar = pd.read_parquet(Path(recipe["inputs"]["bundle"]) / "calendar.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    freeze_execution_history(
        recipe,
        tmp_path,
        dates,
        universe_exit=dates[55],
        suspended_until=dates[47],
    )
    recipe["allocation"] = {"mode": "equal"}
    report = preflight_recipe(recipe)
    assert report["passed"], report
    assert report["allocation"]["mode"] == "equal"
    assert report["execution"]["mode"] == "dynamic"

    candidate = next(item for item in candidates(recipe) if item["name"] == "buy_hold")
    result = execute(EquityResearchExecutor(), recipe, candidate, tmp_path / "dynamic")
    diagnostics = json.loads((tmp_path / "dynamic/execution_diagnostics.json").read_text())
    assert diagnostics["allocation_decisions"][0]["nav"] == recipe["costs"]["initial_capital"]
    assert any("MARKET_NOT_TRADABLE" in row["reason"] for row in diagnostics["unfilled_orders"])
    assert result["metrics"]["fills"] > 0
    returns = pd.read_csv(tmp_path / "dynamic/returns.csv", index_col=0)
    expected_sessions = dates[(dates >= dates[45]) & (dates <= dates[79])]
    assert list(pd.to_datetime(returns.index)) == list(expected_sessions)


def test_suspended_ioc_orders_stop_after_bounded_session_retries(recipe, tmp_path):
    calendar = pd.read_parquet(Path(recipe["inputs"]["bundle"]) / "calendar.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    freeze_execution_history(recipe, tmp_path, dates, suspended_until=dates[79])
    recipe["allocation"] = {"mode": "equal"}
    recipe["execution"]["max_retry_sessions"] = 1
    candidate = next(item for item in candidates(recipe) if item["name"] == "buy_hold")
    result = execute(EquityResearchExecutor(), recipe, candidate, tmp_path / "retry-limit")
    diagnostics = json.loads((tmp_path / "retry-limit/execution_diagnostics.json").read_text())
    assert result["metrics"]["fills"] == 0
    assert diagnostics["retry_diagnostics"]
    assert all(item["attempts"] == 2 for item in diagnostics["retry_diagnostics"])
    assert all(
        item["reason"] == "max_retry_sessions_exhausted"
        for item in diagnostics["retry_diagnostics"]
    )


def test_execution_schema_and_missing_status_fail_closed(recipe, tmp_path):
    fields = {
        "listed": "listed",
        "delisted": "delisted",
        "tradable": "tradable",
        "limit_up": "limit_up",
        "limit_down": "limit_down",
    }
    normalized = validate_research_execution(
        {"mode": "fixed", "status_fields": fields, "max_retry_sessions": 0}
    )
    assert normalized["universe_field"] is None
    with pytest.raises(ValueError, match="exactly"):
        validate_research_execution({"mode": "fixed", "status_fields": {"listed": "listed"}})
    freeze_history(
        recipe,
        tmp_path,
        domain="status",
        field="listed",
        values={"000001": "true"},
    )
    recipe["required_history"] = {field: "status" for field in fields.values()}
    recipe["execution"] = {"mode": "fixed", "status_fields": fields}
    report = preflight_recipe(recipe)
    assert not report["passed"]
    assert "missing" in report["issues"][0]["detail"]


def test_preflight_rejects_late_master_but_allows_never_listed_symbol(recipe, tmp_path):
    calendar = pd.read_parquet(Path(recipe["inputs"]["bundle"]) / "calendar.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    freeze_execution_history(
        recipe,
        tmp_path,
        dates,
        unlisted_symbols={"000001"},
    )
    for name in ("raw", "adjusted"):
        path = Path(recipe["inputs"]["bundle"]) / f"{name}.parquet"
        frame = pd.read_parquet(path)
        frame = frame[~frame.symbol.astype(str).eq("000001")]
        frame.to_parquet(path, index=False)
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][name]["sha256"] = file_hash(path)
        manifest_path.write_text(json.dumps(manifest))
    report = preflight_recipe(recipe)
    assert report["passed"], report
    dormant = next(row for row in report["instrument_master"] if row["symbol"] == "000001")
    assert dormant["required_sessions"] == 0

    catalog = Path(recipe["inputs"]["catalog"])
    master = pd.read_csv(catalog, dtype=str)
    master.loc[master.symbol.eq("000333"), "available_at"] = "2027-01-01T00:00:00Z"
    master.to_csv(catalog, index=False)
    report = preflight_recipe(recipe)
    assert not report["passed"]
    assert any(issue["code"] == "INSTRUMENT_MASTER_PIT_COVERAGE" for issue in report["issues"])


def test_expression_factors_share_preflight_execution_and_report_path(recipe, tmp_path):
    recipe["factor_expressions"] = {"momentum_minus_volatility": "momentum_20d - volatility_20d"}
    recipe["factors"] = {"momentum_minus_volatility": 1}
    report = preflight_recipe(recipe)
    assert report["passed"], report
    assert "momentum_minus_volatility" in report["requirements"]
    result = execute(EquityResearchExecutor(), recipe, candidates(recipe)[0], tmp_path / "expr")
    assert result["metrics"]["fills"] > 0
    assert result["factor_evidence"]["coverage"][0]["factor"] == "momentum_minus_volatility"


def test_etf_template_accepts_equity_asset_class_with_etf_product_type(recipe, tmp_path):
    catalog = Path(recipe["inputs"]["catalog"])
    master = pd.read_csv(catalog, dtype=str)
    master["asset_class"] = "equity"
    master["product_type"] = "etf"
    master.to_csv(catalog, index=False)
    recipe["strategy"]["family"] = "etf_trend"
    report = preflight_recipe(recipe)
    assert report["passed"], report
    result = execute(
        EquityResearchExecutor(),
        recipe,
        candidates(recipe)[0],
        tmp_path / "etf-product",
    )
    assert result["metrics"]["fills"] > 0


@pytest.mark.parametrize("mode", ["inverse_vol", "cost_aware"])
def test_optimizer_modes_rebalance_against_current_ledger_nav(recipe, tmp_path, mode):
    recipe["allocation"] = {
        "mode": mode,
        "lookback": 20,
        "min_observations": 20,
    }
    output = tmp_path / mode
    result = execute(EquityResearchExecutor(), recipe, candidates(recipe)[0], output)
    diagnostics = json.loads((output / "execution_diagnostics.json").read_text())
    decisions = diagnostics["allocation_decisions"]
    assert result["metrics"]["fills"] > 0
    assert len(decisions) > 1
    assert any(decision["nav"] != recipe["costs"]["initial_capital"] for decision in decisions[1:])
    assert all(sum(decision["weights"].values()) <= 0.5 + 1e-9 for decision in decisions)


def test_held_delisting_without_disposition_fails_instead_of_fake_sale(recipe, tmp_path):
    calendar = pd.read_parquet(Path(recipe["inputs"]["bundle"]) / "calendar.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    delist_day = dates[60]
    freeze_execution_history(recipe, tmp_path, dates, delist_day=delist_day)
    recipe["allocation"] = {"mode": "equal"}
    for name in ("raw", "adjusted"):
        path = Path(recipe["inputs"]["bundle"]) / f"{name}.parquet"
        frame = pd.read_parquet(path)
        frame = frame[
            ~(frame.symbol.astype(str).eq("000001") & pd.to_datetime(frame.date).ge(delist_day))
        ]
        frame.to_parquet(path, index=False)
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][name]["sha256"] = file_hash(path)
        manifest_path.write_text(json.dumps(manifest))
    report = preflight_recipe(recipe)
    assert report["passed"], report
    candidate = next(item for item in candidates(recipe) if item["name"] == "buy_hold")
    with pytest.raises(ReplayError, match="disposition evidence"):
        execute(EquityResearchExecutor(), recipe, candidate, tmp_path / "delisted")


def test_restrictions_outside_observation_pool_do_not_block_replay(recipe, tmp_path):
    recipe["required_history"] = {"tradable": "status"}
    freeze_history(
        recipe,
        tmp_path,
        domain="status",
        field="tradable",
        values={
            "000001": "true",
            "000333": "true",
            "600036": "true",
            "601318": "true",
            "OTHER": "false",
        },
    )
    result = execute(EquityResearchExecutor(), recipe, candidates(recipe)[0], tmp_path / "scoped")
    assert result["metrics"]["fills"] > 0
