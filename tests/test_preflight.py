from dataclasses import asdict

import pandas as pd
import pytest
import yaml
from quant_data_kit.exceptions import ValidationError

from a_share_multifactor.config import AppConfig, DataPaths, FilterConfig
from a_share_multifactor.data_loader import build_dataset
from a_share_multifactor.preflight import inspect_inputs, main


def cached_case(root):
    dates = pd.bdate_range("2024-01-02", periods=30)
    prices = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": date,
                "open": 10.0 + index / 100,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0 + index / 100,
                "volume": 10000.0,
                "is_st": False,
            }
            for symbol in ["000001", "000002"]
            for index, date in enumerate(dates)
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {"symbol": symbol, "date": dates[0], "available_at": dates[0], "pe_ratio": pe}
            for symbol, pe in [("000001", 10.0), ("000002", 20.0)]
        ]
    )
    universe = prices[["symbol", "date"]].assign(in_universe=1)
    benchmark = pd.DataFrame(
        {"date": dates, "benchmark_return": 0.001, "benchmark_kind": "total_return"}
    )
    for name, frame in [
        ("prices", prices),
        ("fundamentals", fundamentals),
        ("universe", universe),
        ("benchmark", benchmark),
    ]:
        frame.to_parquet(root / f"{name}.parquet", index=False)
    cfg = AppConfig(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        factors=["pe_ratio"],
        rebalance_freq="weekly",
        filters=FilterConfig(min_list_days=0),
        outputs_dir=str(root / "outputs"),
        data=DataPaths(
            price="prices.parquet",
            fundamentals="fundamentals.parquet",
            universe="universe.parquet",
            benchmark="benchmark.parquet",
        ),
    )
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")
    return config, cfg


def snapshot(root):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_preflight_uses_native_loaders_without_fetch_snapshots_or_strategy(tmp_path, monkeypatch):
    from a_share_multifactor import backtest, data_loader

    config, cfg = cached_case(tmp_path)
    before = snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must only read inputs")

    for name in (
        "fetch_daily_prices",
        "fetch_fundamentals",
        "fetch_hs300_constituents_history",
        "fetch_hs300_constituents",
        "fetch_hs300_benchmark",
        "save_parquet",
        "create_snapshot",
    ):
        monkeypatch.setattr(data_loader, name, forbidden)
    for name in ("run_quantile_backtest", "synthesize", "write_outputs"):
        monkeypatch.setattr(backtest, name, forbidden)
    evidence = inspect_inputs(config, tmp_path, 1)
    assert evidence["software_preflight"] == "pass"
    assert evidence["symbols"] == 1 and evidence["rows"] == 30
    assert evidence["factor_non_null_rows"] == {"pe_ratio": 30}
    assert evidence["read_only"] is True and evidence["investable"] is False
    assert len(evidence["input_files"]) == 4
    assert snapshot(tmp_path) == before
    assert not (tmp_path / "snapshots").exists() and not (tmp_path / "outputs").exists()
    with pytest.raises(ValueError, match="cannot refresh"):
        build_dataset(cfg, tmp_path, force_refresh=True, read_only=True)


@pytest.mark.parametrize("missing", ["prices", "fundamentals", "universe", "benchmark"])
def test_missing_sources_fail_without_download(tmp_path, monkeypatch, missing):
    from a_share_multifactor import data_loader

    config, _ = cached_case(tmp_path)
    (tmp_path / f"{missing}.parquet").unlink()
    before = snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("missing cache must not trigger download")

    for name in (
        "fetch_daily_prices",
        "fetch_fundamentals",
        "fetch_hs300_constituents_history",
        "fetch_hs300_benchmark",
    ):
        monkeypatch.setattr(data_loader, name, forbidden)
    with pytest.raises(ValueError, match="existing file"):
        inspect_inputs(config, tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "bad",
    [
        "future_fundamentals",
        "missing_factor",
        "universe_member",
        "benchmark_kind",
        "empty_window",
        "invalid_price",
    ],
)
def test_preflight_retains_business_input_checks(tmp_path, bad):
    config, _ = cached_case(tmp_path)
    if bad == "missing_factor":
        raw = yaml.safe_load(config.read_text())
        raw["factors"] = ["forecast_score"]
        config.write_text(yaml.safe_dump(raw))
    elif bad == "empty_window":
        raw = yaml.safe_load(config.read_text())
        raw["start_date"], raw["end_date"] = "2025-01-01", "2025-02-01"
        config.write_text(yaml.safe_dump(raw))
    else:
        name = {
            "future_fundamentals": "fundamentals",
            "universe_member": "universe",
            "benchmark_kind": "benchmark",
            "invalid_price": "prices",
        }[bad]
        path = tmp_path / f"{name}.parquet"
        frame = pd.read_parquet(path)
        if bad == "future_fundamentals":
            frame["available_at"] = pd.Timestamp("2030-01-01")
        elif bad == "universe_member":
            frame.loc[0, "symbol"] = "999999"
        elif bad == "benchmark_kind":
            frame["benchmark_kind"] = "price_only"
        else:
            frame.loc[0, "close"] = -1
        frame.to_parquet(path, index=False)
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, KeyError, ValidationError)):
        inspect_inputs(config, tmp_path)
    assert snapshot(tmp_path) == before


def test_cli_prints_json_and_missing_inputs_are_nonzero(tmp_path, capsys):
    import json

    config, _ = cached_case(tmp_path)
    main(["--config", str(config), "--data-dir", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["read_only"] is True
    (tmp_path / "prices.parquet").unlink()
    with pytest.raises(SystemExit) as error:
        main(["--config", str(config), "--data-dir", str(tmp_path)])
    assert error.value.code == 2 and capsys.readouterr().out == ""


@pytest.mark.parametrize("bad", ["nan", "duplicate", "wrong_window"])
def test_preflight_requires_usable_benchmark_in_research_window(tmp_path, bad):
    config, _ = cached_case(tmp_path)
    path = tmp_path / "benchmark.parquet"
    frame = pd.read_parquet(path)
    if bad == "nan":
        frame.loc[1, "benchmark_return"] = float("nan")
    elif bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    else:
        frame["date"] = frame["date"] - pd.DateOffset(years=2)
    frame.to_parquet(path, index=False)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="Benchmark"):
        inspect_inputs(config, tmp_path)
    assert snapshot(tmp_path) == before


def test_preflight_detects_input_change(tmp_path, monkeypatch):
    from a_share_multifactor import preflight

    config, _ = cached_case(tmp_path)
    original = preflight.prepare_factor_inputs

    def changing(*args):
        result = original(*args)
        config.write_text(config.read_text() + "\n# changed\n")
        return result

    monkeypatch.setattr(preflight, "prepare_factor_inputs", changing)
    with pytest.raises(ValueError, match="changed during preflight"):
        inspect_inputs(config, tmp_path)
    assert not (tmp_path / "outputs").exists()
