"""Parameter grid search for factor research pipeline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from a_share_multifactor.backtest import run_pipeline
from a_share_multifactor.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def run_grid_search(
    config: AppConfig,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Scan quantiles and winsorize settings; return summary table."""
    rows: list[dict[str, float | int | str]] = []

    for quantiles in config.grid_search.quantiles:
        for upper in config.grid_search.winsorize_upper:
            lower = config.preprocess.winsorize[0]
            trial = replace(
                config,
                quantiles=quantiles,
                preprocess=replace(config.preprocess, winsorize=(lower, upper)),
            )
            logger.info("Grid search: quantiles=%s winsorize=(%s, %s)", quantiles, lower, upper)
            results, ic_report, _ = run_pipeline(trial, data_dir=data_dir, dry_run=True)
            long_short = results.stats.loc[results.stats["portfolio"] == "long_short"]
            mean_ic = ic_report["mean_ic"].mean() if not ic_report.empty else float("nan")
            row: dict[str, float | int | str] = {
                "quantiles": quantiles,
                "winsorize_upper": upper,
                "mean_ic": float(mean_ic),
            }
            if not long_short.empty:
                row["long_short_ann_return"] = float(long_short.iloc[0]["ann_return"])
                row["long_short_sharpe"] = float(long_short.iloc[0]["sharpe"])
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search quantiles and winsorize settings")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grid_search.csv"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = load_config(args.config)
    summary = run_grid_search(config, data_dir=args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
