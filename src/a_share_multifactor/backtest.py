"""Quantile backtest entry point."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from a_share_multifactor.config import AppConfig, load_config
from a_share_multifactor.data_loader import build_dataset, load_benchmark_returns
from a_share_multifactor.ic_analysis import analyze_factors, analyze_ic_decay, export_ic_series
from a_share_multifactor.preprocess import prepare_factor_panel
from a_share_multifactor.quantile_backtest import BacktestResult, run_quantile_backtest
from a_share_multifactor.report import write_html_report
from a_share_multifactor.synthesis import synthesize

logger = logging.getLogger(__name__)


def _write_run_metadata(
    run_dir: Path,
    config: AppConfig,
    config_source: Path | None,
    run_id: str,
) -> None:
    snapshot_path = run_dir / "config.snapshot.yaml"
    if config_source and config_source.is_file():
        shutil.copy2(config_source, snapshot_path)
    else:
        snapshot_path.write_text(
            yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    meta = {
        "project": "a-share-multifactor",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "factors": list(config.factors),
        "universe": config.universe,
        "outputs_dir": str(config.outputs_dir),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_outputs(
    results: BacktestResult,
    ic_report: pd.DataFrame,
    config: AppConfig,
    output_root: Path | None = None,
    config_source: Path | None = None,
) -> Path:
    """Write IC summary and backtest results to outputs directory."""
    root = output_root or Path(config.outputs_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    latest_dir = root / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    ic_report.to_csv(run_dir / "ic_summary.csv", index=False)
    results.quantile_returns.to_csv(run_dir / "quantile_returns.csv")
    results.cumulative_returns.to_csv(run_dir / "cumulative_returns.csv")
    results.stats.to_csv(run_dir / "backtest_stats.csv", index=False)
    if not results.long_short.empty:
        results.long_short.to_csv(run_dir / "long_short.csv", header=["long_short"])
    if not results.turnover.empty:
        results.turnover.to_csv(run_dir / "turnover.csv")
    if not results.excess_returns.empty:
        results.excess_returns.to_csv(run_dir / "excess_returns.csv")

    ic_report.to_csv(latest_dir / "ic_summary.csv", index=False)
    results.quantile_returns.to_csv(latest_dir / "quantile_returns.csv")
    results.cumulative_returns.to_csv(latest_dir / "cumulative_returns.csv")
    results.stats.to_csv(latest_dir / "backtest_stats.csv", index=False)
    if not results.long_short.empty:
        results.long_short.to_csv(latest_dir / "long_short.csv", header=["long_short"])
    if not results.turnover.empty:
        results.turnover.to_csv(latest_dir / "turnover.csv")
    if not results.excess_returns.empty:
        results.excess_returns.to_csv(latest_dir / "excess_returns.csv")

    config_summary = (
        f"Universe: {config.universe} | Factors: {config.factors} | "
        f"Synthesis: {config.synthesis.method} | Holding: {config.holding_period}"
    )
    write_html_report(results, ic_report, latest_dir / "report.html", config_summary)
    write_html_report(results, ic_report, run_dir / "report.html", config_summary)

    _write_run_metadata(run_dir, config, config_source, run_id)
    _write_run_metadata(latest_dir, config, config_source, run_id)

    return run_dir


def run_pipeline(
    config: AppConfig,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[BacktestResult, pd.DataFrame, Path]:
    """Run full factor research pipeline."""
    panel = build_dataset(config, data_dir=data_dir)
    logger.info("Loaded panel: %s rows, %s symbols", len(panel), panel["symbol"].nunique())

    panel = prepare_factor_panel(config, panel)
    factor_cols = [col for col in config.factors if col in panel.columns]
    ic_report = analyze_factors(panel, factor_cols, config.forward_return_col)
    scored = synthesize(panel, config, ic_summary=ic_report)

    benchmark = load_benchmark_returns(config, data_dir=data_dir)
    results = run_quantile_backtest(scored, config, benchmark_returns=benchmark)

    if dry_run:
        logger.info("Dry run complete — skipping output write")
        return results, ic_report, Path(config.outputs_dir)

    root = Path(config.outputs_dir)
    latest_dir = root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    export_ic_series(panel, factor_cols, config.forward_return_col, latest_dir / "ic_series")
    ic_decay = analyze_ic_decay(panel, factor_cols, config.ic_decay_horizons, price_col="close")
    ic_decay.to_csv(latest_dir / "ic_decay.csv", index=False)

    run_dir = write_outputs(results, ic_report, config)
    export_ic_series(panel, factor_cols, config.forward_return_col, run_dir / "ic_series")
    ic_decay.to_csv(run_dir / "ic_decay.csv", index=False)
    return results, ic_report, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share multi-factor quantile backtest")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--symbols-limit", type=int, default=0, help="Limit symbols for debugging")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = load_config(args.config)
    logger.info("Loaded config: universe=%s, factors=%s", config.universe, config.factors)

    panel = build_dataset(config, data_dir=args.data_dir)
    if args.symbols_limit > 0:
        keep = sorted(panel["symbol"].unique())[: args.symbols_limit]
        panel = panel[panel["symbol"].isin(keep)].reset_index(drop=True)
        logger.info("Limited to %s symbols", len(keep))

    panel = prepare_factor_panel(config, panel)
    factor_cols = [col for col in config.factors if col in panel.columns]
    ic_report = analyze_factors(panel, factor_cols, config.forward_return_col)
    scored = synthesize(panel, config, ic_summary=ic_report)
    benchmark = load_benchmark_returns(config, data_dir=args.data_dir)
    results = run_quantile_backtest(scored, config, benchmark_returns=benchmark)

    if args.dry_run:
        logger.info("Dry run complete — skipping output write")
        print("\nIC Summary:")
        print(ic_report.to_string(index=False))
        return

    root = Path(config.outputs_dir)
    latest_dir = root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    export_ic_series(panel, factor_cols, config.forward_return_col, latest_dir / "ic_series")
    ic_decay = analyze_ic_decay(panel, factor_cols, config.ic_decay_horizons, price_col="close")
    ic_decay.to_csv(latest_dir / "ic_decay.csv", index=False)

    run_dir = write_outputs(
        results, ic_report, config, output_root=root, config_source=args.config
    )
    export_ic_series(panel, factor_cols, config.forward_return_col, run_dir / "ic_series")
    ic_decay.to_csv(run_dir / "ic_decay.csv", index=False)

    ic_report.to_csv(root / "ic_summary.csv", index=False)
    write_html_report(
        results,
        ic_report,
        root / "report.html",
        f"Four-category factors | {config.factors}",
    )

    print("\nIC Summary:")
    print(ic_report.to_string(index=False))
    print("\nBacktest Stats:")
    print(results.stats.to_string(index=False))
    print(f"\nOutputs written to: {root}")


if __name__ == "__main__":
    main()
