"""Small subprocess boundary so an unavailable free provider cannot hang a run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["raw", "adjusted", "benchmark", "calendar"])
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind in {"raw", "adjusted"}:
        from quant_data_kit.providers.prices import fetch_daily_prices

        frame = fetch_daily_prices(
            args.symbols,
            args.start,
            args.end,
            adjust="" if args.kind == "raw" else "qfq",
            max_workers=2,
            max_retries=1,
            sleep_seconds=0.2,
        )
    elif args.kind == "benchmark":
        from quant_data_kit.providers.benchmark import fetch_hs300_benchmark

        frame = fetch_hs300_benchmark(args.start, args.end)
    else:
        from quant_data_kit.calendar import load_sse_trade_dates

        frame = pd.DataFrame({"date": load_sse_trade_dates()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
