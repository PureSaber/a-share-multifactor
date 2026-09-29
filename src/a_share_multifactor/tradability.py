"""Next-session execution and A-share limit/halt flags.

Signals are known at the close. Fills use the next session's open. A limit-up
or a halted name cannot be bought; a limit-down or a halt delays the exit to
the next session where a sell is allowed, and that worse price stays in the
period return.
"""

from __future__ import annotations

import bisect

import pandas as pd

_LIMIT_TOLERANCE = 0.995
_CHINEXT_WIDE_LIMIT = pd.Timestamp("2020-08-24")


def price_limit_ratio(symbol: object, date: object, name: object = None) -> float:
    """Board limit as a fraction of the previous close.

    ST uses 5% when a name is present. STAR is 20%. ChiNext is 20% from
    2020-08-24 and 10% before that. Beijing listings use 30%. The main board
    is 10%. Forward-adjusted returns can mis-label an ex-rights session.
    """
    text = "" if name is None or pd.isna(name) else str(name)
    if "ST" in text.upper():
        return 0.05
    code = str(symbol).zfill(6)[-6:]
    if code.startswith("688"):
        return 0.20
    if code.startswith(("8", "4", "92")):
        return 0.30
    if code.startswith("300") and pd.Timestamp(date).normalize() >= _CHINEXT_WIDE_LIMIT:
        return 0.20
    return 0.10


def annotate_tradability(df: pd.DataFrame) -> pd.DataFrame:
    """Add suspended, limit_up, limit_down, can_buy and can_sell columns."""
    required = {"symbol", "date", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Tradability checks need "
            + ", ".join(missing)
            + ". Limit and halt filters are not skipped when those fields are absent."
        )
    ordered = df.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values(["symbol", "date"])
    prev_close = ordered.groupby("symbol", sort=False)["close"].shift(1)
    ret = ordered["close"] / prev_close - 1
    ret = ret.where(prev_close > 0)
    names = ordered["name"] if "name" in ordered.columns else pd.Series(pd.NA, index=ordered.index)
    limits = pd.Series(
        [
            price_limit_ratio(symbol, day, name)
            for symbol, day, name in zip(ordered["symbol"], ordered["date"], names, strict=False)
        ],
        index=ordered.index,
        dtype=float,
    )
    close = ordered["close"].abs()
    at_high = (ordered["high"] - ordered["close"]).abs() <= close * 1e-4 + 1e-8
    at_low = (ordered["close"] - ordered["low"]).abs() <= close * 1e-4 + 1e-8
    suspended = pd.to_numeric(ordered["volume"], errors="coerce").fillna(0) <= 0
    known = prev_close.notna() & ret.notna()
    limit_up = (~suspended) & known & at_high & (ret >= limits * _LIMIT_TOLERANCE) & (ret < limits + 0.05)
    limit_down = (
        (~suspended) & known & at_low & (ret <= -limits * _LIMIT_TOLERANCE) & (ret > -(limits + 0.05))
    )
    ordered["suspended"] = suspended.to_numpy()
    ordered["limit_up"] = limit_up.fillna(False).to_numpy()
    ordered["limit_down"] = limit_down.fillna(False).to_numpy()
    ordered["can_buy"] = ~(ordered["suspended"] | ordered["limit_up"])
    ordered["can_sell"] = ~(ordered["suspended"] | ordered["limit_down"])
    return ordered


def _session_after(dates: list[pd.Timestamp], day: pd.Timestamp) -> pd.Timestamp | None:
    position = bisect.bisect_right(dates, pd.Timestamp(day))
    if position >= len(dates):
        return None
    return dates[position]


def _execution_open(bar: pd.Series) -> float:
    return float(bar["open"])


def _lot_price(bar: pd.Series) -> float:
    """Cash needed for one share at the traded price, not the forward-adjusted close."""
    if "amount" in bar.index and "volume" in bar.index:
        volume = bar["volume"]
        amount = bar["amount"]
        if pd.notna(amount) and pd.notna(volume) and float(volume) > 0:
            return float(amount) / float(volume)
    return _execution_open(bar)


def assign_executable_period_returns(
    df: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    symbol_col: str = "symbol",
    date_col: str = "date",
    col_name: str = "period_return",
) -> pd.DataFrame:
    """Return from the next session open to the next tradable exit open.

    The value is stored on the signal date. Names that cannot be bought stay
    missing, so they drop out of the portfolio. An exit that is limit-down or
    halted walks forward until a sell is allowed, and stops before the
    following period's entry.
    """
    if "open" not in df.columns:
        raise ValueError(
            "Period returns fill at the next session open. The panel has no open prices."
        )
    annotated = annotate_tradability(df)
    annotated[col_name] = pd.NA
    annotated["entry_price"] = pd.NA
    dates = [pd.Timestamp(day) for day in sorted(annotated[date_col].dropna().unique())]
    rebalance_list = [pd.Timestamp(day) for day in pd.to_datetime(rebalance_dates)]
    indexed = annotated.drop_duplicates([symbol_col, date_col]).set_index([symbol_col, date_col])

    def bar(symbol: object, day: pd.Timestamp) -> pd.Series | None:
        key = (symbol, day)
        if key not in indexed.index:
            return None
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row

    for idx, start_date in enumerate(rebalance_list[:-1]):
        next_signal = rebalance_list[idx + 1]
        entry_date = _session_after(dates, start_date)
        scheduled_exit = _session_after(dates, next_signal)
        if entry_date is None or scheduled_exit is None:
            continue
        window_end = None
        if idx + 2 < len(rebalance_list):
            window_end = _session_after(dates, rebalance_list[idx + 2])
        start_rows = indexed.index.get_level_values(date_col) == start_date
        symbols = indexed.index.get_level_values(symbol_col)[start_rows]
        for symbol in symbols:
            entry = bar(symbol, entry_date)
            if entry is None or not bool(entry["can_buy"]) or not pd.notna(entry["open"]):
                continue
            cursor = scheduled_exit
            exit_open = None
            last_close = None
            while cursor is not None and (window_end is None or cursor < window_end):
                exit_bar = bar(symbol, cursor)
                if exit_bar is not None and pd.notna(exit_bar["close"]):
                    last_close = float(exit_bar["close"])
                if (
                    exit_bar is not None
                    and bool(exit_bar["can_sell"])
                    and pd.notna(exit_bar["open"])
                    and float(exit_bar["open"]) > 0
                ):
                    exit_open = _execution_open(exit_bar)
                    break
                cursor = _session_after(dates, cursor)
            entry_open = _execution_open(entry)
            if entry_open <= 0:
                continue
            exit_price = exit_open if exit_open is not None else last_close
            if exit_price is None or exit_price <= 0:
                continue
            mask = (annotated[symbol_col] == symbol) & (annotated[date_col] == start_date)
            annotated.loc[mask, col_name] = exit_price / entry_open - 1
            annotated.loc[mask, "entry_price"] = _lot_price(entry)

    return annotated.sort_values([date_col, symbol_col]).reset_index(drop=True)
