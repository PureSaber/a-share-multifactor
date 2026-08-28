import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from a_share_multifactor import retail_param_grid, synthesis_compare
from a_share_multifactor.config import AppConfig, CostsConfig, SynthesisConfig


def _scored_panel() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    rows = []
    for day_index, day in enumerate(dates):
        for symbol_index, symbol in enumerate(["000001", "000002", "000003", "000004", "000005"]):
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "close": 10.0 + symbol_index + day_index * 0.1,
                    "market_cap": float(symbol_index),
                    "forward_return_20d": 0.01 * (symbol_index - 2),
                    "composite_score": float(symbol_index),
                }
            )
    return pd.DataFrame(rows)


def test_retail_param_grid_cache_run_plot_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _scored_panel()
    config = AppConfig(
        outputs_dir=str(tmp_path / "outputs"),
        factors=["market_cap"],
        costs=CostsConfig(
            retail_mode=True,
            initial_capital=10_000,
            max_holdings=2,
            min_commission=0,
            commission=0,
            slippage=0,
            stamp_tax=0,
        ),
        synthesis=SynthesisConfig(method="ols", lookback_months=1),
    )
    synth_calls = []

    def fake_synthesize(frame: pd.DataFrame, trial: AppConfig, **_kwargs: object) -> pd.DataFrame:
        synth_calls.append(trial.costs.trade_freq)
        return frame.copy()

    monkeypatch.setattr(retail_param_grid, "synthesize", fake_synthesize)
    cached = retail_param_grid._load_or_build_scored_panel(
        panel, config, "daily", tmp_path / "cache", pd.Timestamp("2025-01-01")
    )
    loaded = retail_param_grid._load_or_build_scored_panel(
        panel, config, "daily", tmp_path / "cache", None
    )
    pd.testing.assert_frame_equal(cached, loaded)
    assert synth_calls == ["daily"]
    assert len(retail_param_grid.build_param_grid()) == 81
    assert (
        retail_param_grid._stats_from_returns(pd.Series(dtype=float), 10_000, "daily")[
            "trade_periods"
        ]
        == 0
    )

    monkeypatch.setattr(retail_param_grid, "build_dataset", lambda *_args, **_kw: panel)
    monkeypatch.setattr(retail_param_grid, "prepare_factor_panel", lambda _cfg, frame: frame)
    monkeypatch.setattr(
        retail_param_grid,
        "_load_or_build_scored_panel",
        lambda frame, *_args, **_kw: frame,
    )

    def fake_simulation(
        _panel: pd.DataFrame,
        trial: AppConfig,
        _score: str,
        _quantile: int,
        trade_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        magnitude = 0.0001 * (trial.costs.min_holding_days + len(trade_dates))
        return pd.Series([magnitude] * len(trade_dates), index=trade_dates, dtype=float)

    monkeypatch.setattr(retail_param_grid, "simulate_daily_retail_portfolio", fake_simulation)
    summary = retail_param_grid.run_retail_param_grid(
        config,
        data_dir=tmp_path,
        eval_start_date="2025-01-01",
        initial_capital=10_000,
        cache_dir=tmp_path / "run-cache",
    )
    assert len(summary) == 81
    assert summary["total_return_pct"].is_monotonic_decreasing

    chart_dir = tmp_path / "charts"
    retail_param_grid.plot_param_grid_results(summary, chart_dir)
    assert (chart_dir / "param_grid_top20.png").exists()
    assert (chart_dir / "param_grid_heatmaps.png").exists()
    assert (chart_dir / "param_grid_marginal.png").exists()

    monkeypatch.setattr(retail_param_grid, "load_config", lambda _path: config)
    monkeypatch.setattr(retail_param_grid, "run_retail_param_grid", lambda *_args, **_kw: summary)
    plotted = []
    monkeypatch.setattr(
        retail_param_grid,
        "plot_param_grid_results",
        lambda frame, path: plotted.append((len(frame), Path(path))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asm-retail-grid",
            "--config",
            "unused.yaml",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "cli-grid"),
            "--eval-start-date",
            "2025-01-01",
            "--initial-capital",
            "12000",
            "--verbose",
        ],
    )
    retail_param_grid.main()
    assert plotted == [(81, tmp_path / "cli-grid")]


def test_synthesis_comparison_run_reports_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_symbol_names = synthesis_compare._load_symbol_names
    panel = _scored_panel()
    config = AppConfig(
        end_date="2025-01-31",
        outputs_dir=str(tmp_path / "outputs"),
        factors=["market_cap"],
        costs=CostsConfig(retail_mode=True, trade_freq="daily", max_holdings=2),
    )
    returns = pd.Series(
        [0.01, -0.005, 0.02],
        index=pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08"]),
    )
    ledger = pd.DataFrame(
        {
            "open_date": pd.to_datetime(["2025-01-06", "2025-01-07"]),
            "close_date": pd.to_datetime(["2025-01-07", "2025-01-08"]),
            "symbol": ["000005", "000001"],
            "name": ["五", "一"],
            "side": ["long", "short"],
            "quantile": [5, 1],
        }
    )
    ic_report = pd.DataFrame({"factor": ["market_cap"], "mean_ic": [0.1], "ir": [1.0]})

    monkeypatch.setattr(synthesis_compare, "build_dataset", lambda *_args, **_kw: panel)
    monkeypatch.setattr(synthesis_compare, "prepare_factor_panel", lambda _cfg, frame: frame)
    monkeypatch.setattr(synthesis_compare, "analyze_factors", lambda *_args, **_kw: ic_report)
    monkeypatch.setattr(
        synthesis_compare, "load_benchmark_returns", lambda *_args, **_kw: pd.Series(dtype=float)
    )
    monkeypatch.setattr(
        synthesis_compare, "_load_symbol_names", lambda symbols: dict.fromkeys(symbols, "名称")
    )
    monkeypatch.setattr(synthesis_compare, "synthesize", lambda frame, *_args, **_kw: frame)
    monkeypatch.setattr(
        synthesis_compare,
        "run_quantile_backtest",
        lambda *_args, **_kw: SimpleNamespace(long_short=returns),
    )
    monkeypatch.setattr(
        synthesis_compare,
        "run_long_only_backtest",
        lambda *_args, **_kw: SimpleNamespace(long_short=returns),
    )
    monkeypatch.setattr(synthesis_compare, "build_trade_ledger", lambda *_args, **_kw: ledger)

    curves, returned_ic, ledgers, summary = synthesis_compare.run_synthesis_comparison(
        config,
        data_dir=tmp_path,
        initial_capital=10_000,
        eval_start_date="2025-01-06",
        long_only=False,
    )
    assert list(curves.columns) == synthesis_compare.SYNTHESIS_METHODS
    assert returned_ic.equals(ic_report)
    assert set(ledgers) == set(synthesis_compare.SYNTHESIS_METHODS)
    assert len(summary) == 5

    long_curves, _, _, _ = synthesis_compare.run_synthesis_comparison(
        config, data_dir=tmp_path, initial_capital=10_000, long_only=True
    )
    assert not long_curves.empty
    assert (
        synthesis_compare._stats_from_returns(pd.Series(dtype=float), 10_000)["rebalance_periods"]
        == 0
    )

    output_dir = tmp_path / "comparison"
    synthesis_compare.write_outputs(
        output_dir,
        curves,
        summary,
        ledgers,
        10_000,
        eval_period="2025-01-06 ~ 2025-01-31",
        long_only=False,
        retail_mode=True,
        max_holdings=2,
        partial_rebalance=True,
        trade_freq="daily",
        min_holding_days=3,
    )
    assert (output_dir / "capital_comparison.png").exists()
    assert (output_dir / "trade_events_equal_weight.png").exists()
    html = (output_dir / "synthesis_comparison.html").read_text(encoding="utf-8")
    assert "最少持有" in html
    assert "data:image/png;base64" in html
    synthesis_compare.plot_trade_events(
        curves,
        pd.DataFrame(),
        "missing",
        10_000,
        tmp_path / "missing.png",
    )

    fake_ak = SimpleNamespace(
        index_stock_cons=lambda **_kw: pd.DataFrame({"品种代码": [1], "品种名称": ["平安"]})
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_ak)
    assert load_symbol_names(["000001"]) == {"000001": "平安"}
    monkeypatch.setattr(fake_ak, "index_stock_cons", lambda **_kw: (_ for _ in ()).throw(OSError()))
    assert load_symbol_names(["000001"]) == {"000001": "000001"}

    monkeypatch.setattr(synthesis_compare, "load_config", lambda _path: config)
    monkeypatch.setattr(
        synthesis_compare,
        "run_synthesis_comparison",
        lambda *_args, **_kw: (curves, ic_report, ledgers, summary),
    )
    written = []
    monkeypatch.setattr(
        synthesis_compare,
        "write_outputs",
        lambda output, *_args, **_kw: written.append(Path(output)),
    )
    cli_output = tmp_path / "cli-comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asm-compare",
            "--config",
            "unused.yaml",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(cli_output),
            "--initial-capital",
            "10000",
            "--eval-start-date",
            "2025-01-06",
            "--long-only",
            "--verbose",
        ],
    )
    synthesis_compare.main()
    assert written == [cli_output]


def test_synthesis_comparison_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _scored_panel()
    config = replace(AppConfig(factors=["market_cap"]), costs=CostsConfig())
    monkeypatch.setattr(synthesis_compare, "build_dataset", lambda *_args, **_kw: panel)
    monkeypatch.setattr(synthesis_compare, "prepare_factor_panel", lambda _cfg, frame: frame)
    monkeypatch.setattr(
        synthesis_compare,
        "analyze_factors",
        lambda *_args, **_kw: pd.DataFrame({"factor": ["market_cap"], "mean_ic": [0.0]}),
    )
    monkeypatch.setattr(
        synthesis_compare, "load_benchmark_returns", lambda *_args, **_kw: pd.Series(dtype=float)
    )
    monkeypatch.setattr(synthesis_compare, "_load_symbol_names", lambda symbols: {})
    monkeypatch.setattr(synthesis_compare, "synthesize", lambda frame, *_args, **_kw: frame)
    monkeypatch.setattr(
        synthesis_compare,
        "run_quantile_backtest",
        lambda *_args, **_kw: SimpleNamespace(long_short=pd.Series(dtype=float)),
    )
    curves, _, ledgers, summary = synthesis_compare.run_synthesis_comparison(config)
    assert curves.empty
    assert ledgers == {}
    assert summary.empty
