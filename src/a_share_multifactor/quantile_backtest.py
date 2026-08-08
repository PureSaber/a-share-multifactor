"""Quantile portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from a_share_multifactor.calendar import rebalance_dates, trade_schedule_dates
from a_share_multifactor.config import AppConfig
from a_share_multifactor.trading_costs import (
    estimate_leg_rebalance_cost,
    portfolio_value,
    retail_turnover,
    select_retail_targets,
    simulate_daily_retail_portfolio,
    simulate_long_only_rebalance,
)


@dataclass
class BacktestResult:
    quantile_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    long_short: pd.Series
    stats: pd.DataFrame
    turnover: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    excess_returns: pd.DataFrame = field(default_factory=pd.DataFrame)


def _assign_quantiles(scores: pd.Series, quantiles: int) -> pd.Series:
    valid = scores.dropna()
    if valid.empty or valid.nunique() < quantiles:
        return pd.Series(index=scores.index, dtype="float")
    ranks = pd.qcut(valid, quantiles, labels=False, duplicates="drop")
    result = pd.Series(index=scores.index, dtype="float")
    result.loc[valid.index] = ranks.astype(float) + 1
    return result


def _return_col(config: AppConfig) -> str:
    if config.holding_period == "rebalance":
        return config.period_return_col
    return config.forward_return_col


def _compute_turnover(
    prev_symbols: dict[int, set[str]],
    curr_symbols: dict[int, set[str]],
    quantiles: int,
) -> float:
    turnovers: list[float] = []
    for q in range(1, quantiles + 1):
        prev = prev_symbols.get(q, set())
        curr = curr_symbols.get(q, set())
        if not curr:
            continue
        if not prev:
            turnovers.append(1.0)
        else:
            overlap = len(prev.intersection(curr))
            turnovers.append(1.0 - overlap / max(len(curr), len(prev)))
    return float(np.mean(turnovers)) if turnovers else 0.0


def _apply_costs(
    gross_return: float,
    turnover: float,
    config: AppConfig,
    portfolio_value: float | None = None,
    prev_symbols: set[str] | None = None,
    curr_symbols: set[str] | None = None,
    leg_capital: float | None = None,
) -> float:
    """Apply trading costs; retail mode uses per-trade minimums and stamp tax."""
    if not config.costs.retail_mode:
        cost_rate = config.costs.commission + config.costs.slippage
        return gross_return - turnover * cost_rate * 2

    capital = portfolio_value if portfolio_value is not None else config.costs.initial_capital
    if capital <= 0:
        return gross_return

    if prev_symbols is not None and curr_symbols is not None and leg_capital is not None:
        yuan_cost = estimate_leg_rebalance_cost(
            prev_symbols, curr_symbols, leg_capital, config.costs
        )
        return gross_return - yuan_cost / capital

    cost_rate = config.costs.commission + config.costs.slippage + config.costs.stamp_tax * turnover
    return gross_return - turnover * cost_rate * 2


def _compute_leg_turnover(prev_symbols: set[str], curr_symbols: set[str]) -> float:
    if not curr_symbols:
        return 0.0
    if not prev_symbols:
        return 1.0
    overlap = len(prev_symbols.intersection(curr_symbols))
    return 1.0 - overlap / max(len(curr_symbols), len(prev_symbols))


def _empty_backtest_result() -> BacktestResult:
    empty = pd.DataFrame()
    return BacktestResult(
        quantile_returns=empty,
        cumulative_returns=empty,
        long_short=pd.Series(dtype=float),
        stats=empty,
    )


def _periods_per_year(config: AppConfig, *, daily: bool = False) -> int:
    if daily:
        return 252
    return 12 if config.rebalance_freq == "monthly" else 252


def _portfolio_stats_row(
    series: pd.Series,
    periods_per_year: int,
    portfolio: str,
) -> dict[str, float | str]:
    mean_ret = float(series.mean())
    vol = float(series.std(ddof=0))
    ann_return = (1 + mean_ret) ** periods_per_year - 1
    ann_vol = vol * np.sqrt(periods_per_year) if vol > 0 else float("nan")
    sharpe = ann_return / ann_vol if ann_vol and not np.isnan(ann_vol) else float("nan")
    return {
        "portfolio": portfolio,
        "mean_return": mean_ret,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


def _build_excess_returns(
    quantile_returns: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    excess = pd.DataFrame(index=quantile_returns.index)
    if benchmark.empty:
        return excess
    aligned_benchmark = benchmark.reindex(quantile_returns.index).fillna(0.0)
    for col in quantile_returns.columns:
        excess[col] = quantile_returns[col] - aligned_benchmark
    return excess


def _prepare_rebalance_day(
    panel: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    score_col: str,
    return_col: str,
    config: AppConfig,
) -> pd.DataFrame:
    day = panel[panel["date"] == rebalance_date].copy()
    if day.empty:
        return day
    day["quantile"] = _assign_quantiles(day[score_col], config.quantiles)
    return day.dropna(subset=["quantile", return_col])


def _long_only_retail_daily(
    panel: pd.DataFrame,
    config: AppConfig,
    score_col: str,
    long_q: int,
    q_col: str,
    benchmark_returns: pd.Series | None,
    trade_start_date: str | pd.Timestamp | None,
) -> BacktestResult:
    trade_idx = trade_schedule_dates(
        panel["date"],
        config.rebalance_freq,
        config.costs.retail_mode,
        config.costs.trade_freq,
    )
    if trade_start_date is not None:
        trade_idx = trade_idx[trade_idx >= pd.Timestamp(trade_start_date)]
    portfolio = simulate_daily_retail_portfolio(
        panel,
        config,
        score_col,
        long_q,
        trade_idx,
    )
    if portfolio.empty:
        return _empty_backtest_result()

    quantile_returns = pd.DataFrame({q_col: portfolio})
    cumulative_returns = (1 + quantile_returns).cumprod()
    periods_per_year = _periods_per_year(config, daily=True)
    stats_rows: list[dict[str, float | str]] = []
    series = portfolio.dropna()
    if not series.empty:
        stats_rows.append(_portfolio_stats_row(series, periods_per_year, "long_only"))

    benchmark = benchmark_returns if benchmark_returns is not None else pd.Series(dtype=float)
    return BacktestResult(
        quantile_returns=quantile_returns,
        cumulative_returns=cumulative_returns,
        long_short=portfolio,
        stats=pd.DataFrame(stats_rows),
        turnover=pd.DataFrame(),
        benchmark_returns=benchmark,
        excess_returns=pd.DataFrame(),
    )


def _long_only_rebalance_loop(
    panel: pd.DataFrame,
    config: AppConfig,
    score_col: str,
    return_col: str,
    long_q: int,
    q_col: str,
    benchmark_returns: pd.Series | None,
) -> BacktestResult:
    rebalance_idx = rebalance_dates(panel["date"], config.rebalance_freq)
    period_returns: list[dict[str, float | pd.Timestamp]] = []
    turnover_rows: list[dict[str, float | pd.Timestamp]] = []
    prev_long: set[str] = set()
    price_col = "close"

    cash = float(config.costs.initial_capital)
    holdings: dict[str, int] = {}

    for rebalance_date in rebalance_idx:
        day = _prepare_rebalance_day(panel, rebalance_date, score_col, return_col, config)
        if day.empty:
            continue

        longs = day[day["quantile"] == float(long_q)]
        if longs.empty:
            continue

        if config.costs.retail_mode:
            if price_col not in day.columns:
                raise ValueError(f"Price column not found for retail mode: {price_col}")
            prices = (
                day.drop_duplicates("symbol").set_index("symbol")[price_col].astype(float).to_dict()
            )
            total_value = portfolio_value(cash, holdings, prices)
            target_symbols = select_retail_targets(
                longs,
                score_col,
                config.costs.max_holdings,
                total_value,
                prices,
                config.costs,
            )
            curr_long = set(target_symbols)
            turnover = retail_turnover(prev_long, curr_long, config.costs.partial_rebalance)
        else:
            curr_long = set(longs["symbol"].astype(str))
            turnover = _compute_leg_turnover(prev_long, curr_long)

        turnover_rows.append({"date": rebalance_date, "turnover": turnover})

        if config.costs.retail_mode:
            period_rets = (
                longs.drop_duplicates("symbol")
                .set_index("symbol")[return_col]
                .astype(float)
                .to_dict()
            )
            net, holdings, cash, _trade_cost = simulate_long_only_rebalance(
                cash=cash,
                holdings=holdings,
                prices=prices,
                period_returns=period_rets,
                target_symbols=target_symbols,
                costs=config.costs,
            )
        else:
            gross = float(longs[return_col].mean())
            net = _apply_costs(gross, turnover, config)
        period_returns.append({"date": rebalance_date, q_col: net})
        prev_long = curr_long

    if not period_returns:
        return _empty_backtest_result()

    quantile_returns = pd.DataFrame(period_returns).set_index("date").sort_index()
    cumulative_returns = (1 + quantile_returns).cumprod()
    turnover_df = pd.DataFrame(turnover_rows).set_index("date") if turnover_rows else pd.DataFrame()
    portfolio = quantile_returns[q_col]

    benchmark = benchmark_returns if benchmark_returns is not None else pd.Series(dtype=float)
    excess = _build_excess_returns(quantile_returns, benchmark)
    if not benchmark.empty:
        excess = excess[[q_col]]

    periods_per_year = _periods_per_year(config)
    stats_rows: list[dict[str, float | str]] = []
    series = portfolio.dropna()
    if not series.empty:
        stats_rows.append(_portfolio_stats_row(series, periods_per_year, "long_only"))
        if not benchmark.empty:
            excess_series = portfolio - benchmark.reindex(portfolio.index).fillna(0.0)
            stats_rows.append(
                _portfolio_stats_row(excess_series, periods_per_year, "long_only_excess")
            )

    return BacktestResult(
        quantile_returns=quantile_returns,
        cumulative_returns=cumulative_returns,
        long_short=portfolio,
        stats=pd.DataFrame(stats_rows),
        turnover=turnover_df,
        benchmark_returns=benchmark,
        excess_returns=excess,
    )


def run_long_only_backtest(
    panel: pd.DataFrame,
    config: AppConfig,
    score_col: str = "composite_score",
    return_col: str | None = None,
    benchmark_returns: pd.Series | None = None,
    long_quantile: int | None = None,
    trade_start_date: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Run long-only backtest on the top quantile (default Q5)."""
    return_col = return_col or _return_col(config)
    long_q = long_quantile if long_quantile is not None else config.quantiles
    q_col = f"Q{long_q}"

    if score_col not in panel.columns:
        raise ValueError(f"Score column not found: {score_col}")

    if config.costs.retail_mode and config.costs.trade_freq in {"daily", "weekly"}:
        return _long_only_retail_daily(
            panel,
            config,
            score_col,
            long_q,
            q_col,
            benchmark_returns,
            trade_start_date,
        )

    if return_col not in panel.columns:
        raise ValueError(f"Return column not found: {return_col}")

    return _long_only_rebalance_loop(
        panel,
        config,
        score_col,
        return_col,
        long_q,
        q_col,
        benchmark_returns,
    )


def _quantile_rebalance_loop(
    panel: pd.DataFrame,
    config: AppConfig,
    score_col: str,
    return_col: str,
) -> tuple[list[dict[str, float | pd.Timestamp]], list[dict[str, float | pd.Timestamp]]]:
    rebalance_idx = rebalance_dates(panel["date"], config.rebalance_freq)
    period_returns: list[dict[str, float | pd.Timestamp]] = []
    turnover_rows: list[dict[str, float | pd.Timestamp]] = []
    prev_holdings: dict[int, set[str]] = {}
    portfolio_value = float(config.costs.initial_capital)

    for rebalance_date in rebalance_idx:
        day = _prepare_rebalance_day(panel, rebalance_date, score_col, return_col, config)
        if day.empty:
            continue

        curr_holdings: dict[int, set[str]] = {}
        for quantile, group in day.groupby("quantile"):
            curr_holdings[int(quantile)] = set(group["symbol"].astype(str))

        turnover = _compute_turnover(prev_holdings, curr_holdings, config.quantiles)
        turnover_rows.append({"date": rebalance_date, "turnover": turnover})

        grouped = day.groupby("quantile")[return_col].mean()
        row: dict[str, float | pd.Timestamp] = {"date": rebalance_date}
        leg_capital = portfolio_value * 0.5
        for quantile, value in grouped.items():
            gross = float(value)
            if config.costs.retail_mode:
                q = int(quantile)
                prev_set = prev_holdings.get(q, set())
                curr_set = curr_holdings.get(q, set())
                yuan_cost = estimate_leg_rebalance_cost(
                    prev_set, curr_set, leg_capital, config.costs
                )
                net = gross - yuan_cost / portfolio_value if portfolio_value > 0 else gross
            else:
                net = _apply_costs(gross, turnover, config)
            row[f"Q{int(quantile)}"] = net
        period_returns.append(row)
        prev_holdings = curr_holdings

    return period_returns, turnover_rows


def _quantile_backtest_stats(
    quantile_returns: pd.DataFrame,
    long_short: pd.Series,
    benchmark: pd.Series,
    config: AppConfig,
) -> pd.DataFrame:
    stats_rows: list[dict[str, float | str]] = []
    periods_per_year = _periods_per_year(config)

    for col in quantile_returns.columns:
        series = quantile_returns[col].dropna()
        if series.empty:
            continue
        stats_rows.append(_portfolio_stats_row(series, periods_per_year, col))

    if not long_short.empty:
        stats_rows.append(_portfolio_stats_row(long_short, periods_per_year, "long_short"))

    if not benchmark.empty and not long_short.empty:
        aligned_benchmark = benchmark.reindex(long_short.index).fillna(0.0)
        excess_ls = long_short - aligned_benchmark
        stats_rows.append(
            _portfolio_stats_row(excess_ls, periods_per_year, "long_short_excess")
        )

    return pd.DataFrame(stats_rows)


def run_quantile_backtest(
    panel: pd.DataFrame,
    config: AppConfig,
    score_col: str = "composite_score",
    return_col: str | None = None,
    benchmark_returns: pd.Series | None = None,
) -> BacktestResult:
    """Run cross-sectional quantile backtest on composite score."""
    return_col = return_col or _return_col(config)
    if score_col not in panel.columns:
        raise ValueError(f"Score column not found: {score_col}")
    if return_col not in panel.columns:
        raise ValueError(f"Return column not found: {return_col}")

    period_returns, turnover_rows = _quantile_rebalance_loop(panel, config, score_col, return_col)

    if not period_returns:
        return _empty_backtest_result()

    quantile_returns = pd.DataFrame(period_returns).set_index("date").sort_index()
    cumulative_returns = (1 + quantile_returns).cumprod()
    turnover_df = pd.DataFrame(turnover_rows).set_index("date") if turnover_rows else pd.DataFrame()

    if "Q1" in quantile_returns.columns and f"Q{config.quantiles}" in quantile_returns.columns:
        long_short = quantile_returns[f"Q{config.quantiles}"] - quantile_returns["Q1"]
    else:
        long_short = pd.Series(dtype=float)

    benchmark = benchmark_returns if benchmark_returns is not None else pd.Series(dtype=float)
    excess = _build_excess_returns(quantile_returns, benchmark)
    stats = _quantile_backtest_stats(quantile_returns, long_short, benchmark, config)

    return BacktestResult(
        quantile_returns=quantile_returns,
        cumulative_returns=cumulative_returns,
        long_short=long_short,
        stats=stats,
        turnover=turnover_df,
        benchmark_returns=benchmark,
        excess_returns=excess,
    )
