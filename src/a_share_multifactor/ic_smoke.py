"""IC smoke pipeline using quant-factors compute output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from a_share_multifactor.factors import compute_factors
from a_share_multifactor.ic_analysis import analyze_factors, export_ic_series


def load_ic_smoke_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_smoke_panel(symbols: list[str], periods: int) -> pd.DataFrame:
    """Synthetic OHLCV panel large enough for quant-factors windows."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            close = 10.0 + i * 0.05 + (hash(symbol) % 3) * 0.1
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000 + i * 10,
                }
            )
    return pd.DataFrame(rows)


def add_forward_return(
    panel: pd.DataFrame,
    days: int,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "close",
) -> pd.DataFrame:
    result = panel.copy()
    col = f"forward_return_{days}d"
    result[col] = result.groupby(symbol_col)[price_col].transform(
        lambda s, d=days: s.shift(-d) / s - 1
    )
    return result


def run_ic_smoke(config_path: Path | str, output_dir: Path | str | None = None) -> Path:
    """
    Compute factors via quant-factors-backed compute_factors, run IC analysis,
    and write reproducible outputs.
    """
    cfg = load_ic_smoke_config(config_path)
    factors: list[str] = cfg["factors"]
    forward_days: int = int(cfg.get("forward_return_days", 20))
    return_col = f"forward_return_{forward_days}d"

    panel_cfg = cfg.get("panel", {})
    symbols = panel_cfg.get("symbols", ["000001", "000002", "000003"])
    periods = int(panel_cfg.get("periods", 60))

    out = Path(output_dir or cfg.get("outputs_dir", "outputs/ic_smoke"))
    out.mkdir(parents=True, exist_ok=True)

    raw = build_smoke_panel(symbols, periods)
    with_factors = compute_factors(raw, factor_names=factors)
    panel = add_forward_return(with_factors, forward_days)

    missing = [f for f in factors if f not in panel.columns]
    if missing:
        raise ValueError(f"quant-factors compute missing columns: {missing}")

    ic_report = analyze_factors(panel, factors, return_col)
    ic_report.to_csv(out / "ic_summary.csv", index=False)
    export_ic_series(panel, factors, return_col, out / "ic_series")

    manifest = {
        "factors": factors,
        "forward_return_days": forward_days,
        "return_col": return_col,
        "rows": len(panel),
        "symbols": symbols,
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run IC smoke pipeline")
    parser.add_argument(
        "--config",
        default="configs/ic_smoke.yaml",
        help="Path to ic_smoke.yaml",
    )
    parser.add_argument("--output", default=None, help="Override outputs directory")
    args = parser.parse_args()
    out = run_ic_smoke(args.config, args.output)
    print(f"IC smoke outputs: {out}")


if __name__ == "__main__":
    main()
