"""Trade ledger for quantile long-short backtests."""

from __future__ import annotations

import pandas as pd

from a_share_multifactor.calendar import rebalance_dates, trade_schedule_dates
from a_share_multifactor.config import AppConfig
from a_share_multifactor.quantile_backtest import _assign_quantiles, _return_col
from a_share_multifactor.trading_costs import (
    HoldingMeta,
    _trading_day_index,
    build_symbol_ranks,
    buy_trade_cost,
    portfolio_value,
    retail_daily_step,
    retail_rebalance,
    round_to_lots,
    select_retail_targets,
    sell_trade_cost,
)


def capital_curve_from_returns(
    period_returns: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Convert per-period long-short returns into a capital curve."""
    clean = period_returns.dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    growth = (1 + clean).cumprod()
    return initial_capital * growth


def period_start_capitals(
    period_returns: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Capital available at the open of each rebalance period."""
    clean = period_returns.dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    capitals = [initial_capital]
    for ret in clean:
        capitals.append(capitals[-1] * (1 + ret))
    return pd.Series(capitals[:-1], index=clean.index, dtype=float)


def _close_retail_position(
    rows: list[dict[str, object]],
    symbol: str,
    pos: dict[str, object],
    close_date: pd.Timestamp,
    close_price: float,
    sell_cost: float,
    names: dict[str, str],
    long_q: int,
    exit_reason: str = "",
) -> None:
    open_date = pos["open_date"]
    shares = int(pos["shares"])
    open_price = float(pos["open_price"])
    buy_cost = float(pos["buy_cost"])
    notional = shares * open_price
    gross_pnl = shares * (close_price - open_price)
    net_pnl = gross_pnl - buy_cost - sell_cost
    period_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
    holding_days = (close_date - open_date).days

    rows.append(
        {
            "open_date": open_date,
            "close_date": close_date,
            "symbol": symbol,
            "name": names.get(symbol, symbol),
            "side": "long",
            "quantile": long_q,
            "action_open": "buy",
            "action_close": "sell",
            "open_price": round(open_price, 4),
            "close_price": round(close_price, 4),
            "capital_allocated": round(notional + buy_cost, 2),
            "shares": shares,
            "holding_days": holding_days,
            "period_return": round(period_return, 6),
            "buy_cost": round(buy_cost, 2),
            "sell_cost": round(sell_cost, 2),
            "pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "exit_reason": exit_reason,
        }
    )


def _build_retail_long_ledger(
    panel: pd.DataFrame,
    config: AppConfig,
    period_start_capital: pd.Series,
    score_col: str,
    long_q: int,
    names: dict[str, str],
    price_col: str,
) -> pd.DataFrame:
    rebalance_idx = list(rebalance_dates(panel["date"], config.rebalance_freq))
    rows: list[dict[str, object]] = []
    cash = float(config.costs.initial_capital)
    holdings: dict[str, int] = {}
    open_positions: dict[str, dict[str, object]] = {}

    for i, open_date in enumerate(rebalance_idx):
        if open_date not in period_start_capital.index:
            continue

        day = panel[panel["date"] == open_date].copy()
        if day.empty:
            continue

        day["quantile"] = _assign_quantiles(day[score_col], config.quantiles)
        day = day.drop_duplicates(subset=["symbol"])
        day = day.dropna(subset=["quantile", price_col])
        if day.empty:
            continue

        longs = day[day["quantile"] == float(long_q)]
        if longs.empty:
            continue

        prices = day.set_index("symbol")[price_col].astype(float).to_dict()
        prices = {str(k): float(v) for k, v in prices.items()}
        total_value = portfolio_value(cash, holdings, prices)
        target_symbols = select_retail_targets(
            longs,
            score_col,
            config.costs.max_holdings,
            total_value,
            prices,
            config.costs,
        )

        result = retail_rebalance(cash, holdings, prices, target_symbols, config.costs)
        cash = result.cash
        holdings = result.holdings

        for sell in result.sells:
            pos = open_positions.pop(sell.symbol, None)
            if pos is None:
                continue
            _close_retail_position(
                rows,
                sell.symbol,
                pos,
                open_date,
                sell.price,
                sell.cost,
                names,
                long_q,
                exit_reason=sell.exit_reason,
            )

        for buy in result.buys:
            open_positions[buy.symbol] = {
                "open_date": open_date,
                "shares": buy.shares,
                "open_price": buy.price,
                "buy_cost": buy.cost,
            }

    last_date = rebalance_idx[-1] if rebalance_idx else None
    for symbol, pos in open_positions.items():
        close_date = None
        close_price = None
        sell_cost = 0.0
        if last_date is not None:
            close_row = panel[
                (panel["date"] == last_date) & (panel["symbol"].astype(str) == symbol)
            ]
            if not close_row.empty:
                close_date = last_date
                close_price = float(close_row.iloc[0][price_col])
                sell_cost = sell_trade_cost(pos["shares"] * close_price, config.costs)

        open_date = pos["open_date"]
        shares = int(pos["shares"])
        open_price = float(pos["open_price"])
        buy_cost = float(pos["buy_cost"])
        notional = shares * open_price
        if close_price is not None:
            gross_pnl = shares * (close_price - open_price)
            net_pnl = gross_pnl - buy_cost - sell_cost
            period_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
            holding_days = (close_date - open_date).days if close_date is not None else None
        else:
            gross_pnl = 0.0
            net_pnl = -buy_cost
            period_return = 0.0
            holding_days = None

        rows.append(
            {
                "open_date": open_date,
                "close_date": close_date,
                "symbol": symbol,
                "name": names.get(symbol, symbol),
                "side": "long",
                "quantile": long_q,
                "action_open": "buy",
                "action_close": "sell" if close_date is not None else None,
                "open_price": round(open_price, 4),
                "close_price": round(close_price, 4) if close_price is not None else None,
                "capital_allocated": round(notional + buy_cost, 2),
                "shares": shares,
                "holding_days": holding_days,
                "period_return": round(period_return, 6),
                "buy_cost": round(buy_cost, 2),
                "sell_cost": round(sell_cost, 2),
                "pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
            }
        )

    return pd.DataFrame(rows)


def _build_retail_daily_ledger(
    panel: pd.DataFrame,
    config: AppConfig,
    period_start_capital: pd.Series,
    score_col: str,
    long_q: int,
    names: dict[str, str],
    price_col: str,
) -> pd.DataFrame:
    trade_idx = trade_schedule_dates(
        panel["date"],
        config.rebalance_freq,
        config.costs.retail_mode,
        config.costs.trade_freq,
    )
    day_index = _trading_day_index(trade_idx)
    by_date = {pd.Timestamp(d): g for d, g in panel.groupby("date", sort=True)}
    rows: list[dict[str, object]] = []
    cash = float(config.costs.initial_capital)
    holdings: dict[str, int] = {}
    meta: dict[str, HoldingMeta] = {}
    open_positions: dict[str, dict[str, object]] = {}
    prev_prices: dict[str, float] = {}

    for trade_date in trade_idx:
        trade_date = pd.Timestamp(trade_date)
        if trade_date not in period_start_capital.index:
            continue

        day = by_date.get(trade_date)
        if day is None or day.empty:
            continue

        scored = day.copy()
        scored["quantile"] = _assign_quantiles(scored[score_col], config.quantiles)
        scored = scored.drop_duplicates(subset=["symbol"])
        scored = scored.dropna(subset=["quantile", price_col])
        longs = scored[scored["quantile"] == float(long_q)]

        prices = (
            day.drop_duplicates("symbol").set_index("symbol")[price_col].astype(float).to_dict()
        )
        prices = {str(k): float(v) for k, v in prices.items()}
        total_value = portfolio_value(cash, holdings, prices)
        target_symbols: list[str] = []
        symbol_ranks: dict[str, int] = {}
        if not longs.empty:
            symbol_ranks = build_symbol_ranks(longs, score_col)
            target_symbols = select_retail_targets(
                longs,
                score_col,
                config.costs.max_holdings,
                total_value,
                prices,
                config.costs,
            )

        result = retail_daily_step(
            cash=cash,
            holdings=holdings,
            meta=meta,
            prices=prices,
            prev_prices=prev_prices,
            target_symbols=target_symbols,
            costs=config.costs,
            current_date=trade_date,
            day_index=day_index,
            symbol_ranks=symbol_ranks,
        )
        cash = result.cash
        holdings = result.holdings

        for sell in result.sells:
            pos = open_positions.pop(sell.symbol, None)
            if pos is None:
                continue
            _close_retail_position(
                rows,
                sell.symbol,
                pos,
                trade_date,
                sell.price,
                sell.cost,
                names,
                long_q,
                exit_reason=sell.exit_reason,
            )

        for buy in result.buys:
            open_positions[buy.symbol] = {
                "open_date": trade_date,
                "shares": buy.shares,
                "open_price": buy.price,
                "buy_cost": buy.cost,
            }

        prev_prices = prices

    last_date = trade_idx[-1] if len(trade_idx) else None
    for symbol, pos in open_positions.items():
        close_date = None
        close_price = None
        sell_cost = 0.0
        if last_date is not None:
            close_row = panel[
                (panel["date"] == last_date) & (panel["symbol"].astype(str) == symbol)
            ]
            if not close_row.empty:
                close_date = last_date
                close_price = float(close_row.iloc[0][price_col])
                sell_cost = sell_trade_cost(int(pos["shares"]) * close_price, config.costs)

        open_date = pos["open_date"]
        shares = int(pos["shares"])
        open_price = float(pos["open_price"])
        buy_cost = float(pos["buy_cost"])
        notional = shares * open_price
        if close_price is not None:
            gross_pnl = shares * (close_price - open_price)
            net_pnl = gross_pnl - buy_cost - sell_cost
            period_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
            holding_days = (close_date - open_date).days if close_date is not None else None
        else:
            gross_pnl = 0.0
            net_pnl = -buy_cost
            period_return = 0.0
            holding_days = None

        rows.append(
            {
                "open_date": open_date,
                "close_date": close_date,
                "symbol": symbol,
                "name": names.get(symbol, symbol),
                "side": "long",
                "quantile": long_q,
                "action_open": "buy",
                "action_close": "sell" if close_date is not None else None,
                "open_price": round(open_price, 4),
                "close_price": round(close_price, 4) if close_price is not None else None,
                "capital_allocated": round(notional + buy_cost, 2),
                "shares": shares,
                "holding_days": holding_days,
                "period_return": round(period_return, 6),
                "buy_cost": round(buy_cost, 2),
                "sell_cost": round(sell_cost, 2),
                "pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "exit_reason": "open",
            }
        )

    return pd.DataFrame(rows)


def build_trade_ledger(
    panel: pd.DataFrame,
    config: AppConfig,
    period_start_capital: pd.Series,
    score_col: str = "composite_score",
    long_quantile: int | None = None,
    short_quantile: int = 1,
    name_map: dict[str, str] | None = None,
    price_col: str = "close",
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Build open-close trade records.

    long_only=False: dollar-neutral long-short (Q5 long, Q1 short, 50/50 capital).
    long_only=True: full capital equally allocated to top quantile (Q5) only.
    """
    if score_col not in panel.columns:
        raise ValueError(f"Score column not found: {score_col}")

    return_col = _return_col(config)
    if return_col not in panel.columns:
        raise ValueError(f"Return column not found: {return_col}")

    long_q = long_quantile if long_quantile is not None else config.quantiles
    names = name_map or {}

    if config.costs.retail_mode and long_only:
        if config.costs.trade_freq in {"daily", "weekly"}:
            return _build_retail_daily_ledger(
                panel,
                config,
                period_start_capital,
                score_col,
                long_q,
                names,
                price_col,
            )
        return _build_retail_long_ledger(
            panel, config, period_start_capital, score_col, long_q, names, price_col
        )

    rebalance_idx = list(rebalance_dates(panel["date"], config.rebalance_freq))
    rows: list[dict[str, object]] = []

    for i, open_date in enumerate(rebalance_idx):
        if open_date not in period_start_capital.index:
            continue

        close_date = rebalance_idx[i + 1] if i + 1 < len(rebalance_idx) else None
        day = panel[panel["date"] == open_date].copy()
        if day.empty:
            continue

        day["quantile"] = _assign_quantiles(day[score_col], config.quantiles)
        day = day.drop_duplicates(subset=["symbol"])
        day = day.dropna(subset=["quantile", return_col, price_col])
        if day.empty:
            continue

        longs = day[day["quantile"] == float(long_q)]
        if longs.empty:
            continue
        if not long_only:
            shorts = day[day["quantile"] == float(short_quantile)]
            if shorts.empty:
                continue

        capital = float(period_start_capital.loc[open_date])
        long_each = capital / len(longs) if long_only else capital * 0.5 / len(longs)
        short_each = 0.0 if long_only else capital * 0.5 / len(shorts)
        holding_days = (close_date - open_date).days if close_date is not None else None

        for _, row in longs.iterrows():
            symbol = str(row["symbol"])
            open_price = float(row[price_col])
            period_return = float(row[return_col])
            close_price = None
            sell_cost = 0.0
            if config.costs.retail_mode:
                shares = round_to_lots(long_each / open_price, config.costs.lot_size)
                if shares <= 0:
                    continue
                notional = shares * open_price
                buy_cost = buy_trade_cost(notional, config.costs)
                capital_used = notional + buy_cost
            else:
                shares = long_each / open_price if open_price > 0 else 0.0
                notional = long_each
                buy_cost = 0.0
                capital_used = long_each

            if close_date is not None and close_price is None:
                close_row = panel[
                    (panel["date"] == close_date) & (panel["symbol"].astype(str) == symbol)
                ]
                if not close_row.empty:
                    close_price = float(close_row.iloc[0][price_col])
                    if config.costs.retail_mode:
                        sell_notional = shares * close_price
                        sell_cost = sell_trade_cost(sell_notional, config.costs)

            gross_pnl = (
                notional * period_return if config.costs.retail_mode else long_each * period_return
            )
            net_pnl = gross_pnl - buy_cost - sell_cost

            rows.append(
                {
                    "open_date": open_date,
                    "close_date": close_date,
                    "symbol": symbol,
                    "name": names.get(symbol, symbol),
                    "side": "long",
                    "quantile": int(long_q),
                    "action_open": "buy",
                    "action_close": "sell" if close_date is not None else None,
                    "open_price": round(open_price, 4),
                    "close_price": round(close_price, 4) if close_price is not None else None,
                    "capital_allocated": round(capital_used, 2),
                    "shares": int(shares) if config.costs.retail_mode else round(shares, 2),
                    "holding_days": holding_days,
                    "period_return": round(period_return, 6),
                    "buy_cost": round(buy_cost, 2),
                    "sell_cost": round(sell_cost, 2),
                    "pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                }
            )

        if long_only:
            continue

        for _, row in shorts.iterrows():
            symbol = str(row["symbol"])
            open_price = float(row[price_col])
            period_return = float(row[return_col])
            close_price = None
            close_cost = 0.0
            if config.costs.retail_mode:
                shares = round_to_lots(short_each / open_price, config.costs.lot_size)
                if shares <= 0:
                    continue
                notional = shares * open_price
                open_cost = sell_trade_cost(notional, config.costs)
                capital_used = notional
            else:
                shares = short_each / open_price if open_price > 0 else 0.0
                notional = short_each
                open_cost = 0.0
                capital_used = short_each

            if close_date is not None and close_price is None:
                close_row = panel[
                    (panel["date"] == close_date) & (panel["symbol"].astype(str) == symbol)
                ]
                if not close_row.empty:
                    close_price = float(close_row.iloc[0][price_col])
                    if config.costs.retail_mode:
                        cover_notional = shares * close_price
                        close_cost = buy_trade_cost(cover_notional, config.costs)

            gross_pnl = (
                -notional * period_return
                if config.costs.retail_mode
                else -short_each * period_return
            )
            net_pnl = gross_pnl - open_cost - close_cost

            rows.append(
                {
                    "open_date": open_date,
                    "close_date": close_date,
                    "symbol": symbol,
                    "name": names.get(symbol, symbol),
                    "side": "short",
                    "quantile": int(short_quantile),
                    "action_open": "sell_short",
                    "action_close": "buy_cover" if close_date is not None else None,
                    "open_price": round(open_price, 4),
                    "close_price": round(close_price, 4) if close_price is not None else None,
                    "capital_allocated": round(capital_used, 2),
                    "shares": int(shares) if config.costs.retail_mode else round(shares, 2),
                    "holding_days": holding_days,
                    "period_return": round(period_return, 6),
                    "buy_cost": round(close_cost, 2),
                    "sell_cost": round(open_cost, 2),
                    "pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                }
            )

    return pd.DataFrame(rows)
