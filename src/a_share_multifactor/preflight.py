"""Read-only native preflight for the quantile-research data path."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
from quant_data_kit.exceptions import ValidationError

from a_share_multifactor.calendar import rebalance_dates
from a_share_multifactor.config import load_config
from a_share_multifactor.data_loader import build_dataset, load_benchmark_returns
from a_share_multifactor.preprocess import prepare_factor_inputs
from a_share_multifactor.quantile_backtest import align_benchmark_returns
from a_share_multifactor.run_contract import (
    _build_events,
    build_instrument_master,
    execution_catalog,
    validate_execution_profile,
)


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_inputs(config_path: Path, data_dir: Path, symbols_limit: int = 0) -> dict:
    config_hash = _digest(config_path)
    config = load_config(config_path)
    paths = {
        name: data_dir / value
        for name, value in asdict(config.data).items()
        if name != "snapshot_root"
    }
    paths["instrument_catalog"] = execution_catalog(config)
    identities = {name: _digest(path) for name, path in paths.items() if path.is_file()}
    panel = build_dataset(config, data_dir=data_dir, read_only=True)
    if symbols_limit > 0:
        symbols = sorted(panel["symbol"].unique())[:symbols_limit]
        panel = panel[panel["symbol"].isin(symbols)].reset_index(drop=True)
    if panel.empty:
        raise ValueError(
            "No rows remain in the requested window after universe/tradability filters"
        )
    if not config.factors:
        raise ValueError("At least one configured factor is required")
    prepared = prepare_factor_inputs(config, panel)
    benchmark = load_benchmark_returns(config, data_dir=data_dir, read_only=True)
    if benchmark.empty:
        raise ValueError("No benchmark holding periods are available")
    periods = rebalance_dates(panel["date"], config.rebalance_freq)[:-1]
    if periods.empty:
        raise ValueError("No complete holding period is available in the requested sample")
    benchmark = align_benchmark_returns(benchmark, periods)
    validate_execution_profile(panel, config)
    instruments, mappings = build_instrument_master(
        panel, catalog_path=paths["instrument_catalog"]
    )
    bars = _build_events(panel, instruments)
    if _digest(config_path) != config_hash or identities != {
        name: _digest(path) for name, path in paths.items() if path.is_file()
    }:
        raise ValueError("Input files changed during preflight; retry with stable inputs")
    dates = pd.to_datetime(panel["date"])
    return {
        "schema_version": "asm.quantile-preflight/v1",
        "software_preflight": "pass",
        "read_only": True,
        "investable": False,
        "scope": "cached_data_factors_benchmark_and_execution_inputs",
        "execution_rules": {
            "symbols": len(instruments),
            "mappings": len(mappings),
            "bars": len(bars),
            "catalog_scope": "declared-rules-not-exchange-history-certification",
        },
        "symbols": int(panel["symbol"].nunique()),
        "rows": len(panel),
        "requested_window": {"start": config.start_date, "end": config.end_date},
        "observed_window": {"start": dates.min().isoformat(), "end": dates.max().isoformat()},
        "factor_non_null_rows": {
            factor: int(prepared[factor].notna().sum()) for factor in config.factors
        },
        "benchmark_periods": len(benchmark),
        "config_sha256": config_hash,
        "input_files": {
            name: {"path": str(paths[name].resolve()), "sha256": digest}
            for name, digest in identities.items()
        },
        "data_quality": panel.attrs.get("data_quality", {}),
        "limitations": [
            "No network refresh, snapshot publication, strategy scoring, backtest or output writes",
            "Cached input checks do not certify complete exchange history or independent PIT vintages",
            "Preflight does not lock input files; execution reads and validates them again",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only A-share research data preflight")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--symbols-limit", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = inspect_inputs(args.config, args.data_dir, args.symbols_limit)
    except (OSError, ValueError, KeyError, ValidationError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
