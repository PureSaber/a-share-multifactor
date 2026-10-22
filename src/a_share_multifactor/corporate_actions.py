"""Conservative daily-paper bridge for evidenced, same-day distributions."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
from quant_data_kit import CorporateActionEvent, FixedPoint


def resolve_share_event(event, rule, *, account_id, recorded_snapshots):
    """Resolve an ordinary-holder distribution using this account's record date.

    Strict evidence and sensitivity assumptions are carried in the native event;
    neither changes instrument precision or invents fractional cash payments.
    """
    from quant_data_kit.share_entitlements import ShareEntitlement

    record_date = pd.Timestamp(rule["record_date"]).date()
    eligible = [
        s
        for s in recorded_snapshots
        if pd.Timestamp(s.event_time).tz_convert("Asia/Shanghai").date() == record_date
    ]
    if not eligible:
        raise ValueError("share entitlement requires the actual record-date account snapshot")
    snapshot = max(eligible, key=lambda s: s.event_time)
    quantity = snapshot.positions.get(event.instrument_id, FixedPoint(0, 0))
    ratio = Decimal(str(rule["ordinary_additional_ratio"]))
    precision = max(0, -ratio.as_tuple().exponent)
    status = rule.get("evidence_status", "unresolved")
    confirmed = rule.get("confirmed_quantity")
    if confirmed is not None and Decimal(str(confirmed)) != int(confirmed):
        raise ValueError("confirmed share quantity must be an exact integer")
    reference = rule.get("reference_price")
    reference_fp = None
    if reference is not None:
        value = Decimal(str(reference))
        scale = max(0, -value.as_tuple().exponent)
        reference_fp = FixedPoint(int(value.scaleb(scale)), scale)
    terms = ShareEntitlement(
        entitlement_id=rule["entitlement_id"],
        account_id=account_id,
        record_date=record_date,
        eligible_quantity=quantity,
        ordinary_additional_ratio=FixedPoint(int(ratio.scaleb(precision)), precision),
        known_at=pd.Timestamp(rule["known_at"]).tz_convert("UTC").to_pydatetime(),
        credit_at=pd.Timestamp(rule["credit_at"]).tz_convert("UTC").to_pydatetime(),
        sellable_date=pd.Timestamp(rule["sellable_date"]).date(),
        evidence_source=rule["evidence_source"],
        evidence_sha256=rule["evidence_sha256"],
        evidence_status=status,
        fractional_rule=rule["fractional_rule"],
        confirmed_quantity=None if confirmed is None else FixedPoint(int(confirmed), 0),
        reference_price=reference_fp,
    )
    if confirmed is not None and rule.get("confirmed_account_id") != account_id:
        raise ValueError("confirmed registry quantity must identify the exact account")
    return replace(
        event, event_id=terms.entitlement_id + ":" + account_id, ratio=None, share_entitlement=terms
    )


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
        credit_days = selected.ex_date.copy()
        has_credit = pd.Series(False, index=selected.index)
        if "credit_at" in selected:
            has_credit = selected.credit_at.notna()
            credit_days.loc[has_credit] = (
                pd.to_datetime(selected.loc[has_credit, "credit_at"], utc=True)
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
                .dt.normalize()
            )
        # Unit credit can occur on the record-date close, before the ex-date.
        # Keep factor evidence for the ex-date even when credit precedes the window.
        selected = selected[
            ((selected.ex_date > start) & (selected.ex_date <= end))
            | (has_credit & credit_days.between(start, end))
        ]
        if selected.duplicated(["symbol", "ex_date"]).any():
            raise ValueError("Duplicate corporate actions")
    events, by_key = [], {}
    for row in selected.itertuples():
        cash, ratio = Decimal(row.cash_per_share), Decimal(row.share_ratio)
        if not cash.is_finite() or not ratio.is_finite() or cash < 0 or ratio <= 0:
            raise ValueError("Invalid corporate-action amounts")
        dates = sorted(pd.to_datetime(raw.loc[raw.symbol == row.symbol, "date"]).unique())
        previous = [pd.Timestamp(x) for x in dates if pd.Timestamp(x) < row.ex_date]
        if (cash or ratio != 1) and (
            pd.isna(row.announced_date)
            or row.announced_date >= row.ex_date
            or not previous
            or row.record_date != previous[-1]
        ):
            raise ValueError(
                "Corporate-action record/announcement dates do not establish entitlement"
            )
        if cash and (pd.isna(row.pay_date) or row.pay_date < row.ex_date):
            raise ValueError("Cash dividend requires a known payment date on or after ex-date")
        if ratio != 1 and row.shares_available_date != row.ex_date:
            raise ValueError("Deferred share delivery requires a share receivables ledger; blocked")
        at = (row.ex_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)).tz_convert("UTC")
        credit = getattr(row, "credit_at", None)
        if pd.notna(credit):
            at = pd.Timestamp(credit).tz_convert("UTC")

        def fixed(value):
            scaled = value * 10**8
            if scaled != scaled.to_integral_value():
                raise ValueError("Corporate-action precision exceeds eight decimals")
            return FixedPoint(int(scaled), 8)

        deferred_cash = bool(cash and row.pay_date != row.ex_date)
        credit_day = pd.Timestamp(at).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
        holder_events = (
            [] if (cash == 0 and ratio == 1) or not start <= credit_day <= end else events
        )
        holder_events.append(
            CorporateActionEvent(
                event_id=(row.event_id + ":entitlement" if deferred_cash else row.event_id),
                instrument_id=row.symbol,
                event_time=at.to_pydatetime(),
                available_at=at.to_pydatetime(),
                received_at=at.to_pydatetime(),
                source=row.source,
                trading_day=at.tz_convert("Asia/Shanghai").date(),
                session_id="CN-A-SHARE:" + str(at.tz_convert("Asia/Shanghai").date()),
                sequence=0,
                action_type=(
                    "cash_dividend_entitlement" if deferred_cash else "cash_and_share_distribution"
                ),
                effective_date=credit_day.date(),
                cash_amount=fixed(cash) if cash else None,
                currency="CNY" if cash else None,
                ratio=fixed(ratio) if ratio != 1 or cash == 0 else None,
            )
        )
        if deferred_cash and row.pay_date <= end:
            paid_at = (
                row.pay_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)
            ).tz_convert("UTC")
            events.append(
                CorporateActionEvent(
                    event_id=row.event_id + ":payment",
                    instrument_id=row.symbol,
                    event_time=paid_at.to_pydatetime(),
                    available_at=paid_at.to_pydatetime(),
                    received_at=paid_at.to_pydatetime(),
                    source=row.source,
                    trading_day=row.pay_date.date(),
                    session_id="CN-A-SHARE:" + str(row.pay_date.date()),
                    sequence=1,
                    action_type="cash_dividend_payment",
                    effective_date=row.ex_date.date(),
                    cash_amount=fixed(cash),
                    currency="CNY",
                )
            )
        reference_cash = getattr(row, "adjustment_cash_per_share", None)
        reference_ratio = getattr(row, "adjustment_share_ratio", None)
        overridden = pd.notna(reference_cash) or pd.notna(reference_ratio)
        if overridden:
            source = getattr(row, "adjustment_reference_source", None)
            sha = getattr(row, "adjustment_reference_sha256", None)
            if (
                not isinstance(source, str)
                or not source.strip()
                or not isinstance(sha, str)
                or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)
            ):
                raise ValueError("adjustment reference requires source evidence and SHA-256")
        reference_cash = Decimal(str(reference_cash)) if pd.notna(reference_cash) else cash
        reference_ratio = Decimal(str(reference_ratio)) if pd.notna(reference_ratio) else ratio
        if (
            not reference_cash.is_finite()
            or not reference_ratio.is_finite()
            or reference_cash < 0
            or reference_ratio <= 0
        ):
            raise ValueError("invalid exchange adjustment reference")
        # Differential distributions can use a diluted exchange reference while
        # entitled shareholders receive the full cash/share distribution above.
        by_key[(row.symbol, row.ex_date)] = (float(reference_cash), float(reference_ratio))
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


def split_share_rules(actions, raw):
    """Bind per-account rounding and reference marks to exact issuer terms."""
    rules = {}
    if actions is None or "credit_at" not in actions:
        return rules
    for row in actions[actions.credit_at.notna()].itertuples():
        ratio = Decimal(str(row.share_ratio))
        previous = raw[(raw.symbol == row.symbol) & (raw.date == pd.Timestamp(row.record_date))]
        if len(previous) != 1 or ratio <= 1:
            raise ValueError("split requires the real record-date raw close and expansion ratio")
        reference = (Decimal(str(previous.iloc[0].close)) / ratio).quantize(Decimal("0.00000001"))
        rules[row.event_id] = {
            "entitlement_id": row.event_id,
            "record_date": str(pd.Timestamp(row.record_date).date()),
            "ordinary_additional_ratio": str(ratio - 1),
            "known_at": row.known_at,
            "credit_at": row.credit_at,
            "sellable_date": str(pd.Timestamp(row.shares_available_date).date()),
            "evidence_source": row.evidence_source,
            "evidence_sha256": row.evidence_sha256,
            "evidence_status": "rule_determined",
            "fractional_rule": row.fractional_rule,
            "reference_price": str(reference),
        }
    return rules
