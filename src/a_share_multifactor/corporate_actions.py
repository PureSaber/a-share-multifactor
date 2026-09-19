"""Conservative daily-paper bridge for evidenced, same-day distributions."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
from quant_data_kit import CorporateActionEvent, FixedPoint


def action_events(
    actions: pd.DataFrame | None,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple:
    selected = pd.DataFrame() if actions is None else actions.copy()
    if not selected.empty:
        for key in (
            "announced_date",
            "record_date",
            "ex_date",
            "pay_date",
            "shares_available_date",
        ):
            selected[key] = pd.to_datetime(selected[key])
        if selected.ex_date.isna().any():
            raise ValueError("Corporate-action feed has unknown ex-dates")
        selected = selected[(selected.ex_date > start) & (selected.ex_date <= end)]
        if selected.duplicated(["symbol", "ex_date"]).any():
            raise ValueError("Duplicate corporate actions")
    events, by_key = [], {}
    for row in selected.itertuples():
        cash, ratio = Decimal(row.cash_per_share), Decimal(row.share_ratio)
        if not cash.is_finite() or not ratio.is_finite() or cash < 0 or ratio <= 0:
            raise ValueError("Invalid corporate-action amounts")
        dates = sorted(pd.to_datetime(raw.loc[raw.symbol == row.symbol, "date"]).unique())
        previous = [pd.Timestamp(x) for x in dates if pd.Timestamp(x) < row.ex_date]
        if (
            pd.isna(row.announced_date)
            or row.announced_date >= row.ex_date
            or not previous
            or row.record_date != previous[-1]
        ):
            raise ValueError(
                "Corporate-action record/announcement dates do not establish entitlement"
            )
        if (cash and row.pay_date != row.ex_date) or (
            ratio != 1 and row.shares_available_date != row.ex_date
        ):
            raise ValueError(
                "Deferred dividend/share delivery requires a receivables ledger; blocked"
            )
        at = (row.ex_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)).tz_convert("UTC")

        def fixed(value):
            scaled = value * 10**8
            if scaled != scaled.to_integral_value():
                raise ValueError("Corporate-action precision exceeds eight decimals")
            return FixedPoint(int(scaled), 8)

        events.append(
            CorporateActionEvent(
                event_id=row.event_id,
                instrument_id=row.symbol,
                event_time=at.to_pydatetime(),
                available_at=at.to_pydatetime(),
                received_at=at.to_pydatetime(),
                source=row.source,
                trading_day=row.ex_date.date(),
                session_id="CN-A-SHARE:" + str(row.ex_date.date()),
                sequence=0,
                action_type="cash_and_share_distribution",
                effective_date=row.ex_date.date(),
                cash_amount=fixed(cash) if cash else None,
                currency="CNY" if cash else None,
                ratio=fixed(ratio) if ratio != 1 else None,
            )
        )
        by_key[(row.symbol, row.ex_date)] = (float(cash), float(ratio))
    check = raw[(raw.date >= start) & (raw.date <= end)].merge(
        adjusted[["symbol", "date", "close"]],
        on=["symbol", "date"],
        suffixes=("", "_qfq"),
        validate="one_to_one",
    )
    if len(check) != len(raw[(raw.date >= start) & (raw.date <= end)]):
        raise ValueError("Corporate-action adjusted-price coverage is incomplete")
    for symbol, group in check.groupby("symbol"):
        group = group.sort_values("date")
        factors = group.close_qfq / group.close
        if not np.isfinite(factors).all() or (factors <= 0).any():
            raise ValueError("Corporate-action adjustment factors invalid")
        # Only CHANGES in the factor imply events inside the account window.
        # A historical qfq vintage can have a constant scale different from one.
        multiplicative_matches = True
        for i in range(1, len(group)):
            day = pd.Timestamp(group.iloc[i].date)
            cash, ratio = by_key.get((symbol, day), (0.0, 1.0))
            previous_close = float(group.iloc[i - 1].close)
            if cash >= previous_close:
                raise ValueError("Corporate-action cash exceeds previous price")
            expected = previous_close * ratio / (previous_close - cash)
            observed = float(factors.iloc[i] / factors.iloc[i - 1])
            tolerance = 0.025 / min(float(group.iloc[i].close), previous_close)
            if abs(observed - expected) > tolerance:
                multiplicative_matches = False
        # Tencent uses a subtract-dividend affine qfq series, unlike providers
        # that multiply by reinvestment factors. Validate one model over the
        # WHOLE window, never select whichever model fits each individual day.
        affine_matches = True
        multiplier = 1.0
        offset = float(group.iloc[-1].close_qfq - group.iloc[-1].close)
        for i in range(len(group) - 1, -1, -1):
            row = group.iloc[i]
            if abs(float(row.close) * multiplier + offset - float(row.close_qfq)) > 0.025:
                affine_matches = False
            cash, ratio = by_key.get((symbol, pd.Timestamp(row.date)), (0.0, 1.0))
            offset -= multiplier * cash / ratio
            multiplier /= ratio
        if not (multiplicative_matches or affine_matches):
            raise ValueError(
                "Corporate-action/adjustment differences require a matching cashflow feed"
            )
    return tuple(sorted(events, key=lambda e: (e.available_at, e.instrument_id)))
