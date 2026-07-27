"""Compute retail OLS buy list for a given trade date (fresh portfolio)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from a_share_multifactor.config import load_config
from a_share_multifactor.data_loader import build_dataset
from a_share_multifactor.preprocess import prepare_factor_panel
from a_share_multifactor.quantile_backtest import _assign_quantiles
from a_share_multifactor.synthesis import synthesize
from a_share_multifactor.synthesis_compare import _load_symbol_names
from a_share_multifactor.trading_costs import (
    _buy_symbol,
    build_symbol_ranks,
    select_retail_targets,
)


def compute_buy_list(
    trade_date: str,
    config_path: Path,
    data_dir: Path,
    initial_capital: float = 10_000.0,
    early_exit: str = "off",
    min_holding_days: int = 10,
    trade_freq: str = "weekly",
    rank_change_threshold: int = 0,
) -> pd.DataFrame:
    from a_share_multifactor.retail_param_grid import EARLY_EXIT_PRESETS

    config = load_config(config_path)
    costs_kw: dict[str, object] = {
        "retail_mode": True,
        "initial_capital": initial_capital,
        "min_holding_days": min_holding_days,
        "trade_freq": trade_freq,
        "rank_change_threshold": rank_change_threshold,
        "max_holdings": 10,
        "partial_rebalance": True,
    }
    costs_kw.update(EARLY_EXIT_PRESETS[early_exit])
    config = replace(config, costs=replace(config.costs, **costs_kw))

    panel = prepare_factor_panel(config, build_dataset(config, data_dir=data_dir))
    trial = replace(config, synthesis=replace(config.synthesis, method="ols"))
    scored = synthesize(panel, trial)

    ts = pd.Timestamp(trade_date)
    day = scored[scored["date"] == ts].copy()
    if day.empty:
        available = sorted(scored["date"].dropna().unique())
        raise ValueError(f"No data for {trade_date}. Nearest dates: {available[-3:]}")

    day["quantile"] = _assign_quantiles(day["composite_score"], config.quantiles)
    day = day.dropna(subset=["quantile", "close"])
    longs = day[day["quantile"] == float(config.quantiles)].copy()
    if longs.empty:
        raise ValueError(f"No Q5 candidates on {trade_date}")

    prices = day.drop_duplicates("symbol").set_index("symbol")["close"].astype(float).to_dict()
    prices = {str(k): float(v) for k, v in prices.items()}

    targets = select_retail_targets(
        longs,
        "composite_score",
        config.costs.max_holdings,
        initial_capital,
        prices,
        config.costs,
    )
    ranks = build_symbol_ranks(longs, "composite_score")

    names = _load_symbol_names(list(prices.keys()))
    factor_cols = [c for c in config.factors if c in longs.columns]

    rows: list[dict[str, object]] = []
    cash = float(initial_capital)
    budget_each = cash / max(len(targets), 1)

    for rank, symbol in enumerate(targets, start=1):
        sym = str(symbol).zfill(6)
        price = prices.get(symbol) or prices.get(sym)
        if price is None:
            continue
        cash_after, shares, trade_cost = _buy_symbol(cash, sym, price, budget_each, config.costs)
        notional = shares * price if shares > 0 else 0.0
        row_data = longs[longs["symbol"].astype(str).str.zfill(6) == sym]
        if row_data.empty:
            row_data = longs[longs["symbol"].astype(str) == symbol]
        factor_vals = {}
        if not row_data.empty:
            r = row_data.iloc[0]
            factor_vals = {f: round(float(r[f]), 4) for f in factor_cols if pd.notna(r.get(f))}

        rows.append(
            {
                "rank": rank,
                "symbol": sym,
                "name": names.get(sym, sym),
                "composite_score": round(float(row_data.iloc[0]["composite_score"]), 6)
                if not row_data.empty
                else None,
                "q5_rank": ranks.get(symbol, ranks.get(sym)),
                "close": round(price, 4),
                "shares": shares,
                "notional": round(notional, 2),
                "buy_cost": round(trade_cost, 2),
                "total_cash": round(notional + trade_cost, 2),
                **factor_vals,
            }
        )
        if shares > 0:
            cash = cash_after
            budget_each = cash / max(len(targets) - rank, 1)

    result = pd.DataFrame(rows)
    leftover = round(cash, 2)
    print(f"Trade date: {trade_date}")
    print(f"Initial capital: {initial_capital:,.0f} | Remaining cash after buys: {leftover:,.2f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute OLS retail buy list")
    parser.add_argument("--trade-date", type=str, default="2026-07-10")
    parser.add_argument("--config", type=Path, default=Path("configs/run_retail_daily_10k.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/buy_list_2026-07-10.csv"))
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    args = parser.parse_args()

    result = compute_buy_list(
        args.trade_date,
        args.config,
        args.data_dir,
        initial_capital=args.initial_capital,
        early_exit="off",
        min_holding_days=10,
        trade_freq="weekly",
        rank_change_threshold=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
