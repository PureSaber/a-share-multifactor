import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from a_share_multifactor import backtest, fetch_data, grid_search
from a_share_multifactor.config import (
    AppConfig,
    DataPaths,
    FetchConfig,
    FilterConfig,
    GridSearchConfig,
    ValidationConfig,
)


def _fetch_config(data_dir: Path) -> AppConfig:
    return AppConfig(
        start_date="2020-01-01",
        end_date="2020-01-03",
        outputs_dir=str(data_dir / "outputs"),
        data=DataPaths(
            price="prices.parquet",
            fundamentals="fundamentals.parquet",
            universe="universe.parquet",
            benchmark="benchmark.parquet",
            earnings_forecast="earnings.parquet",
            northbound="northbound.parquet",
            industry_returns="industry.parquet",
        ),
        filters=FilterConfig(use_historical_universe=True),
        fetch=FetchConfig(max_workers=1, sleep_seconds=0.0, max_retries=1),
    )


def test_fetch_cli_force_incremental_and_cached_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fetch_config(tmp_path)
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    prices = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "date": dates,
            "open": [10.0, 20.0],
            "high": [10.2, 20.2],
            "low": [9.8, 19.8],
            "close": [10.1, 20.1],
            "volume": [1000, 2000],
            "industry": ["银行", "科技"],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "date": dates,
            "available_at": dates,
            "market_cap": [100.0, 200.0],
        }
    )
    benchmark = pd.DataFrame({"date": dates, "benchmark_return": [0.01, -0.01]})
    panel = prices[["symbol", "date", "close"]].copy()

    monkeypatch.setattr(fetch_data, "load_config", lambda _path: config)
    monkeypatch.setattr(fetch_data, "fetch_hs300_constituents", lambda: ["000001", "000002"])
    monkeypatch.setattr(fetch_data, "fetch_daily_prices", lambda *_args, **_kw: prices.copy())
    monkeypatch.setattr(fetch_data, "fetch_fundamentals", lambda *_args, **_kw: fundamentals.copy())
    monkeypatch.setattr(
        fetch_data,
        "fetch_hs300_constituents_history",
        lambda *_args: pd.DataFrame({"symbol": ["000001"], "date": [dates[0]]}),
    )
    monkeypatch.setattr(fetch_data, "fetch_hs300_benchmark", lambda *_args: benchmark.copy())
    monkeypatch.setattr(
        fetch_data,
        "fetch_earnings_forecasts",
        lambda *_args, **_kw: pd.DataFrame({"symbol": ["000001"], "date": [dates[0]]}),
    )
    monkeypatch.setattr(
        fetch_data,
        "fetch_northbound_holdings",
        lambda *_args, **_kw: pd.DataFrame({"symbol": ["000001"], "date": [dates[0]]}),
    )
    monkeypatch.setattr(
        fetch_data,
        "fetch_industry_returns",
        lambda industries, *_args, **_kw: pd.DataFrame(
            {"industry": industries, "date": [dates[0]] * len(industries)}
        ),
    )
    monkeypatch.setattr(fetch_data, "build_dataset", lambda *_args, **_kw: panel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asm-fetch",
            "--config",
            "unused.yaml",
            "--data-dir",
            str(tmp_path),
            "--force",
            "--fetch-alt",
            "--symbols-limit",
            "2",
            "--verbose",
        ],
    )
    fetch_data.main()
    assert (tmp_path / "prices.parquet").exists()
    assert (tmp_path / "industry.parquet").exists()

    monkeypatch.setattr(fetch_data, "should_refresh_cache", lambda *_args: True)
    monkeypatch.setattr(fetch_data, "incremental_start_date", lambda *_args: "2020-01-02")
    monkeypatch.setattr(
        sys,
        "argv",
        ["asm-fetch", "--config", "unused.yaml", "--data-dir", str(tmp_path)],
    )
    fetch_data.main()
    assert len(fetch_data.load_parquet(tmp_path / "prices.parquet")) == 2

    monkeypatch.setattr(fetch_data, "should_refresh_cache", lambda *_args: False)
    fetch_data.main()


def test_grid_search_business_loop_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(grid_search=GridSearchConfig(quantiles=[3, 5], winsorize_upper=[0.95, 0.99]))
    stats = pd.DataFrame(
        {
            "portfolio": ["long_short"],
            "ann_return": [0.12],
            "sharpe": [1.5],
        }
    )
    ic_report = pd.DataFrame({"mean_ic": [0.03, 0.05]})
    calls = []

    def fake_pipeline(trial: AppConfig, **_kwargs: object):
        calls.append((trial.quantiles, trial.preprocess.winsorize))
        return SimpleNamespace(stats=stats), ic_report, Path("unused")

    monkeypatch.setattr(grid_search, "run_pipeline", fake_pipeline)
    summary = grid_search.run_grid_search(config, data_dir=tmp_path)
    assert len(summary) == 4
    assert summary["long_short_sharpe"].eq(1.5).all()
    assert len(calls) == 4

    output = tmp_path / "grid" / "summary.csv"
    monkeypatch.setattr(grid_search, "load_config", lambda _path: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asm-grid-search",
            "--config",
            "unused.yaml",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--verbose",
        ],
    )
    grid_search.main()
    assert output.exists()


def test_backtest_cli_dry_run_and_write_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(
        outputs_dir=str(tmp_path / "report"),
        factors=["market_cap"],
        validation=ValidationConfig(enabled=True, train_size=2, test_size=1, step_size=1),
    )
    panel = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "close": [10.0, 11.0],
            "market_cap": [1.0, 2.0],
            "forward_return_20d": [0.01, 0.02],
        }
    )
    panel.attrs["dataset_snapshots"] = {"prices": "sha256:test"}
    panel.attrs["data_quality"] = {"rows": 2}
    ic_report = pd.DataFrame({"factor": ["market_cap"], "mean_ic": [0.1]})
    results = SimpleNamespace(stats=pd.DataFrame({"portfolio": ["Q5"], "sharpe": [1.0]}))

    monkeypatch.setattr(backtest, "load_config", lambda _path: config)
    monkeypatch.setattr(backtest, "build_dataset", lambda *_args, **_kw: panel.copy())
    monkeypatch.setattr(backtest, "prepare_factor_panel", lambda _cfg, frame: frame)
    monkeypatch.setattr(backtest, "analyze_factors", lambda *_args, **_kw: ic_report)
    monkeypatch.setattr(backtest, "synthesize", lambda frame, *_args, **_kw: frame)
    monkeypatch.setattr(backtest, "load_benchmark_returns", lambda *_args, **_kw: pd.Series())
    monkeypatch.setattr(backtest, "run_quantile_backtest", lambda *_args, **_kw: results)
    monkeypatch.setattr(backtest, "run_research_validation", lambda *_args: "validated")
    monkeypatch.setattr(
        backtest,
        "analyze_ic_decay",
        lambda *_args, **_kw: pd.DataFrame({"factor": ["market_cap"], "ic": [0.1]}),
    )
    monkeypatch.setattr(
        backtest,
        "export_ic_series",
        lambda _panel, _factors, _return_col, path: Path(path).mkdir(parents=True, exist_ok=True),
    )
    run_dir = tmp_path / "report" / "run"

    def fake_write_outputs(*_args: object, **_kwargs: object) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(backtest, "write_outputs", fake_write_outputs)
    monkeypatch.setattr(backtest, "write_equity_standard_run", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        backtest,
        "write_html_report",
        lambda _results, _ic, path, _title: Path(path).write_text("report", encoding="utf-8"),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["asm-backtest", "--config", "unused.yaml", "--data-dir", str(tmp_path), "--dry-run"],
    )
    backtest.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asm-backtest",
            "--config",
            "unused.yaml",
            "--data-dir",
            str(tmp_path),
            "--symbols-limit",
            "1",
            "--verbose",
        ],
    )
    backtest.main()
    assert (tmp_path / "report" / "ic_summary.csv").exists()
    assert (tmp_path / "report" / "report.html").exists()
