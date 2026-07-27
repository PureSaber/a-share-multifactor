"""A-share retail trading cost helpers and lot-size portfolio simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from a_share_multifactor.config import CostsConfig

if TYPE_CHECKING:
    from a_share_multifactor.config import AppConfig


def round_to_lots(shares: float, lot_size: int) -> int:
    """Round share count down to the nearest tradable lot."""
    if lot_size <= 0:
        return int(math.floor(shares))
    return int(shares // lot_size) * lot_size


def buy_trade_cost(notional: float, costs: CostsConfig) -> float:
    """One-way buy cost: commission (with minimum), plus slippage."""
    if notional <= 0:
        return 0.0
    commission = max(notional * costs.commission, costs.min_commission)
    slippage = notional * costs.slippage
    return commission + slippage


def sell_trade_cost(notional: float, costs: CostsConfig) -> float:
    """One-way sell cost: commission (with minimum), stamp tax, plus slippage."""
    if notional <= 0:
        return 0.0
    commission = max(notional * costs.commission, costs.min_commission)
    stamp = notional * costs.stamp_tax
    slippage = notional * costs.slippage
    return commission + stamp + slippage


def portfolio_value(
    cash: float,
    holdings: dict[str, int],
    prices: dict[str, float],
) -> float:
    """Mark-to-market portfolio value."""
    invested = sum(
        shares * prices[symbol]
        for symbol, shares in holdings.items()
        if symbol in prices and prices[symbol] > 0
    )
    return cash + invested


def select_retail_targets(
    candidates: pd.DataFrame,
    score_col: str,
    max_holdings: int,
    portfolio_value: float,
    prices: dict[str, float],
    costs: CostsConfig,
) -> list[str]:
    """
    Pick top-scored Q5 names up to max_holdings that fit lot-size constraints.

    Scans candidates in score order and keeps names where one lot fits the
    equal-weight slot budget (portfolio_value / max_holdings).
    """
    ranked = candidates.sort_values(score_col, ascending=False)
    limit = max_holdings if max_holdings > 0 else len(ranked)
    if limit <= 0 or portfolio_value <= 0:
        return []

    per_slot = portfolio_value / limit
    targets: list[str] = []
    for _, row in ranked.iterrows():
        if len(targets) >= limit:
            break
        symbol = str(row["symbol"])
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        if price * costs.lot_size <= per_slot:
            targets.append(symbol)
    return targets


def build_symbol_ranks(candidates: pd.DataFrame, score_col: str) -> dict[str, int]:
    """Map symbol -> 1-based score rank within candidates."""
    ranked = candidates.sort_values(score_col, ascending=False)
    return {str(symbol): rank for rank, symbol in enumerate(ranked["symbol"].astype(str), start=1)}


def sell_rank_limit(costs: CostsConfig) -> int:
    """Keep holdings while rank stays within top N plus buffer."""
    buffer = max(costs.rank_change_threshold, 0)
    return max(costs.max_holdings, 1) + buffer


def estimate_leg_rebalance_cost(
    prev_symbols: set[str],
    curr_symbols: set[str],
    leg_capital: float,
    costs: CostsConfig,
) -> float:
    """
    Estimate yuan rebalance cost for one portfolio leg using per-trade minimums.

    With partial_rebalance, only symbols that enter or leave incur trades.
    """
    if not curr_symbols or leg_capital <= 0:
        return 0.0

    per_name = leg_capital / len(curr_symbols)
    if costs.partial_rebalance:
        to_sell = prev_symbols - curr_symbols
        to_buy = curr_symbols - prev_symbols
    else:
        to_sell = prev_symbols
        to_buy = curr_symbols

    total = 0.0
    for _ in to_sell:
        total += sell_trade_cost(per_name, costs)
    for _ in to_buy:
        total += buy_trade_cost(per_name, costs)
    return total


@dataclass
class HoldingMeta:
    buy_date: pd.Timestamp
    buy_price: float
    buy_cost: float
    consecutive_up_days: int = 0


@dataclass
class TradeFill:
    symbol: str
    shares: int
    price: float
    cost: float
    exit_reason: str = ""


@dataclass
class RetailRebalanceResult:
    holdings: dict[str, int]
    cash: float
    trade_cost: float
    buys: list[TradeFill] = field(default_factory=list)
    sells: list[TradeFill] = field(default_factory=list)


def _buy_symbol(
    cash: float,
    symbol: str,
    price: float,
    budget: float,
    costs: CostsConfig,
) -> tuple[float, int, float]:
    """Try to buy one symbol within budget; return (cash, shares, cost)."""
    shares = round_to_lots(budget / price, costs.lot_size)
    if shares <= 0:
        return cash, 0, 0.0

    notional = shares * price
    trade_cost = buy_trade_cost(notional, costs)
    if notional + trade_cost > cash:
        shares = round_to_lots((cash - costs.min_commission) / price, costs.lot_size)
        if shares <= 0:
            return cash, 0, 0.0
        notional = shares * price
        trade_cost = buy_trade_cost(notional, costs)
        if notional + trade_cost > cash:
            return cash, 0, 0.0

    cash -= notional + trade_cost
    return cash, shares, trade_cost


def retail_rebalance(
    cash: float,
    holdings: dict[str, int],
    prices: dict[str, float],
    target_symbols: list[str],
    costs: CostsConfig,
) -> RetailRebalanceResult:
    """Rebalance toward target_symbols with optional partial (low-turnover) mode."""
    total_trade_cost = 0.0
    buys: list[TradeFill] = []
    sells: list[TradeFill] = []
    target_set = set(target_symbols)

    if costs.partial_rebalance:
        for symbol, shares in list(holdings.items()):
            if symbol in target_set:
                continue
            price = prices.get(symbol)
            if price is None or price <= 0 or shares <= 0:
                del holdings[symbol]
                continue
            notional = shares * price
            trade_cost = sell_trade_cost(notional, costs)
            cash += notional - trade_cost
            total_trade_cost += trade_cost
            sells.append(TradeFill(symbol, shares, price, trade_cost))
            del holdings[symbol]

        new_symbols = [sym for sym in target_symbols if sym not in holdings]
        if new_symbols and cash > 0:
            budget_each = cash / len(new_symbols)
            for symbol in new_symbols:
                price = prices.get(symbol)
                if price is None or price <= 0:
                    continue
                cash, shares, trade_cost = _buy_symbol(cash, symbol, price, budget_each, costs)
                if shares <= 0:
                    continue
                total_trade_cost += trade_cost
                holdings[symbol] = shares
                buys.append(TradeFill(symbol, shares, price, trade_cost))
    else:
        for symbol, shares in list(holdings.items()):
            price = prices.get(symbol)
            if price is None or price <= 0 or shares <= 0:
                continue
            notional = shares * price
            trade_cost = sell_trade_cost(notional, costs)
            cash += notional - trade_cost
            total_trade_cost += trade_cost
            sells.append(TradeFill(symbol, shares, price, trade_cost))
        holdings = {}

        if target_symbols:
            budget_each = cash / len(target_symbols)
            for symbol in target_symbols:
                price = prices.get(symbol)
                if price is None or price <= 0:
                    continue
                cash, shares, trade_cost = _buy_symbol(cash, symbol, price, budget_each, costs)
                if shares <= 0:
                    continue
                total_trade_cost += trade_cost
                holdings[symbol] = shares
                buys.append(TradeFill(symbol, shares, price, trade_cost))

    return RetailRebalanceResult(
        holdings=holdings,
        cash=cash,
        trade_cost=total_trade_cost,
        buys=buys,
        sells=sells,
    )


def compute_period_return(
    cash: float,
    holdings: dict[str, int],
    prices: dict[str, float],
    period_returns: dict[str, float],
) -> float:
    """Return for the holding period after rebalance."""
    invested = sum(
        holdings[sym] * prices[sym] for sym in holdings if sym in prices and prices[sym] > 0
    )
    start_value = cash + invested
    if start_value <= 0:
        return 0.0

    end_invested = 0.0
    for symbol, shares in holdings.items():
        price = prices.get(symbol, 0.0)
        period_ret = period_returns.get(symbol, 0.0)
        end_invested += shares * price * (1.0 + period_ret)

    end_value = cash + end_invested
    return end_value / start_value - 1.0


def simulate_long_only_rebalance(
    cash: float,
    holdings: dict[str, int],
    prices: dict[str, float],
    period_returns: dict[str, float],
    target_symbols: list[str],
    costs: CostsConfig,
) -> tuple[float, dict[str, int], float, float]:
    """
    Rebalance a long-only retail portfolio for one period.

    Returns (period_return, new_holdings, end_cash, total_trade_cost_yuan).
    """
    result = retail_rebalance(cash, holdings, prices, target_symbols, costs)
    period_return = compute_period_return(result.cash, result.holdings, prices, period_returns)
    return period_return, result.holdings, result.cash, result.trade_cost


def retail_turnover(
    prev_symbols: set[str],
    curr_symbols: set[str],
    partial_rebalance: bool,
) -> float:
    """Fraction of names traded this rebalance."""
    if not curr_symbols:
        return 0.0
    if not prev_symbols:
        return 1.0
    if partial_rebalance:
        traded = len(prev_symbols - curr_symbols) + len(curr_symbols - prev_symbols)
        return traded / max(len(curr_symbols), len(prev_symbols))
    return 1.0


def _trading_day_index(trading_dates: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(date): idx for idx, date in enumerate(trading_dates)}


def _holding_trading_days(
    buy_date: pd.Timestamp,
    current_date: pd.Timestamp,
    day_index: dict[pd.Timestamp, int],
) -> int:
    buy = pd.Timestamp(buy_date)
    current = pd.Timestamp(current_date)
    if buy not in day_index or current not in day_index:
        return 0
    return day_index[current] - day_index[buy]


def _update_streak(meta: HoldingMeta, daily_return: float, costs: CostsConfig) -> None:
    if daily_return >= costs.early_exit_consecutive_daily:
        meta.consecutive_up_days += 1
    else:
        meta.consecutive_up_days = 0


def _early_exit_triggered(
    meta: HoldingMeta,
    current_price: float,
    daily_return: float,
    costs: CostsConfig,
) -> bool:
    if not costs.early_exit_enabled:
        return False
    if meta.buy_price <= 0:
        return False
    cumulative = current_price / meta.buy_price - 1.0
    if daily_return >= costs.early_exit_single_day_return:
        return True
    if cumulative >= costs.early_exit_cumulative_return:
        return True
    if meta.consecutive_up_days >= costs.early_exit_consecutive_days:
        return True
    return False


def _can_sell_locked_position(
    meta: HoldingMeta,
    current_date: pd.Timestamp,
    current_price: float,
    daily_return: float,
    costs: CostsConfig,
    day_index: dict[pd.Timestamp, int],
) -> tuple[bool, str]:
    if _early_exit_triggered(meta, current_price, daily_return, costs):
        return True, "early_exit"
    held = _holding_trading_days(meta.buy_date, current_date, day_index)
    if held >= costs.min_holding_days:
        return True, "signal_exit"
    return False, "locked"


def retail_daily_step(
    cash: float,
    holdings: dict[str, int],
    meta: dict[str, HoldingMeta],
    prices: dict[str, float],
    prev_prices: dict[str, float],
    target_symbols: list[str],
    costs: CostsConfig,
    current_date: pd.Timestamp,
    day_index: dict[pd.Timestamp, int],
    symbol_ranks: dict[str, int] | None = None,
) -> RetailRebalanceResult:
    """One-day retail rebalance with minimum holding and early-exit exceptions."""
    total_trade_cost = 0.0
    buys: list[TradeFill] = []
    sells: list[TradeFill] = []
    target_set = set(target_symbols)
    rank_limit = sell_rank_limit(costs)
    ranks = symbol_ranks or {}

    for symbol, shares in list(holdings.items()):
        position_meta = meta.get(symbol)
        if position_meta is None:
            continue

        price = prices.get(symbol)
        if price is None or price <= 0 or shares <= 0:
            holdings.pop(symbol, None)
            meta.pop(symbol, None)
            continue

        prev = prev_prices.get(symbol, price)
        daily_return = price / prev - 1.0 if prev and prev > 0 else 0.0
        _update_streak(position_meta, daily_return, costs)

        if _early_exit_triggered(position_meta, price, daily_return, costs):
            allowed, reason = True, "early_exit"
        elif symbol in target_set or ranks.get(symbol, 999) <= rank_limit:
            continue
        else:
            allowed, reason = _can_sell_locked_position(
                position_meta,
                current_date,
                price,
                daily_return,
                costs,
                day_index,
            )
            if not allowed:
                continue

        notional = shares * price
        trade_cost = sell_trade_cost(notional, costs)
        cash += notional - trade_cost
        total_trade_cost += trade_cost
        sells.append(TradeFill(symbol, shares, price, trade_cost, reason))
        holdings.pop(symbol, None)
        meta.pop(symbol, None)

    new_symbols = [sym for sym in target_symbols if sym not in holdings]
    if new_symbols and cash > 0 and len(holdings) < max(costs.max_holdings, 1):
        slots = max(costs.max_holdings - len(holdings), 0)
        for symbol in new_symbols[:slots]:
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            budget_each = cash / max(len(new_symbols[:slots]), 1)
            cash, shares, trade_cost = _buy_symbol(cash, symbol, price, budget_each, costs)
            if shares <= 0:
                continue
            total_trade_cost += trade_cost
            holdings[symbol] = shares
            meta[symbol] = HoldingMeta(
                buy_date=current_date,
                buy_price=price,
                buy_cost=trade_cost,
            )
            buys.append(TradeFill(symbol, shares, price, trade_cost, "buy"))

    return RetailRebalanceResult(
        holdings=holdings,
        cash=cash,
        trade_cost=total_trade_cost,
        buys=buys,
        sells=sells,
    )


def simulate_daily_retail_portfolio(
    panel: pd.DataFrame,
    config: "AppConfig",
    score_col: str,
    long_quantile: int,
    trade_dates: pd.DatetimeIndex,
    price_col: str = "close",
) -> pd.Series:
    """Run daily retail long-only simulation; returns daily portfolio returns."""
    from a_share_multifactor.quantile_backtest import _assign_quantiles

    costs = config.costs
    day_index = _trading_day_index(trade_dates)
    by_date = {pd.Timestamp(date): group for date, group in panel.groupby("date", sort=True)}

    cash = float(costs.initial_capital)
    holdings: dict[str, int] = {}
    meta: dict[str, HoldingMeta] = {}
    prev_prices: dict[str, float] = {}
    prev_value = float(costs.initial_capital)
    daily_returns: dict[pd.Timestamp, float] = {}

    for trade_date in trade_dates:
        trade_date = pd.Timestamp(trade_date)
        day = by_date.get(trade_date)
        if day is None or day.empty:
            continue
        if score_col not in day.columns or price_col not in day.columns:
            continue

        scored = day.copy()
        scored["quantile"] = _assign_quantiles(scored[score_col], config.quantiles)
        scored = scored.dropna(subset=["quantile"])
        longs = scored[scored["quantile"] == float(long_quantile)]
        if longs.empty and not holdings:
            continue

        prices = (
            day.drop_duplicates("symbol").set_index("symbol")[price_col].astype(float).to_dict()
        )
        prices = {str(k): float(v) for k, v in prices.items()}

        if prev_prices:
            prev_value = cash + sum(
                holdings[sym] * prev_prices.get(sym, prices.get(sym, 0.0)) for sym in holdings
            )

        total_value = portfolio_value(cash, holdings, prices)
        target_symbols: list[str] = []
        symbol_ranks: dict[str, int] = {}
        if not longs.empty:
            symbol_ranks = build_symbol_ranks(longs, score_col)
            target_symbols = select_retail_targets(
                longs,
                score_col,
                costs.max_holdings,
                total_value,
                prices,
                costs,
            )

        result = retail_daily_step(
            cash=cash,
            holdings=holdings,
            meta=meta,
            prices=prices,
            prev_prices=prev_prices,
            target_symbols=target_symbols,
            costs=costs,
            current_date=trade_date,
            day_index=day_index,
            symbol_ranks=symbol_ranks,
        )
        cash = result.cash
        holdings = result.holdings

        end_value = portfolio_value(cash, holdings, prices)
        daily_returns[trade_date] = end_value / prev_value - 1.0 if prev_value > 0 else 0.0
        prev_value = end_value
        prev_prices = prices

    return pd.Series(daily_returns).sort_index()
