"""Configured A-share simulation models using QExec's one exact ledger."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from quant_data_kit import AssetClass, BarEvent, FixedPoint
from quant_execution import BarMatchingModel, OrderType, RuleBookRiskGate, Side

from a_share_multifactor.config import CostsConfig
from a_share_multifactor.trading_costs import stamp_duty_rate


def decimal(value: FixedPoint) -> Decimal:
    return Decimal(value.units).scaleb(-value.scale)


def commission_increment(notional, prior, rate, floor):
    """Exact order-level commission shared by target quotes and partial fills."""
    paid = max(prior * rate, floor) if prior else Decimal(0)
    return max((prior + notional) * rate, floor) - paid


def quote_target_trade(spec, quantity, mark, costs, at, *, money_scale=8):
    """Known-price estimate; next-bar price and availability remain unknown."""
    if not quantity:
        return Decimal(0), Decimal(0)
    direction = 1 if quantity > 0 else -1
    tick = decimal(spec.price_tick)
    slipped = mark * (1 + Decimal(direction) * Decimal(str(costs.slippage)))
    rounding = ROUND_CEILING if direction > 0 else ROUND_FLOOR
    execution_price = (slipped / tick).to_integral_value(rounding=rounding) * tick
    notional = Decimal(abs(quantity)) * execution_price * decimal(spec.contract_multiplier)
    commission = commission_increment(
        notional, Decimal(0), Decimal(str(costs.commission)), Decimal(str(costs.min_commission))
    )
    tax = (
        Decimal(str(stamp_duty_rate(costs, at)))
        if quantity < 0 and spec.asset_class is AssetClass.EQUITY
        else Decimal(0)
    )
    from decimal import ROUND_HALF_UP

    fee = (commission + tax * notional).quantize(
        Decimal(1).scaleb(-money_scale), rounding=ROUND_HALF_UP
    )
    change = -Decimal(direction) * notional - fee
    friction = -change - Decimal(quantity) * mark
    return change, friction


class ConfiguredAShareRiskGate(RuleBookRiskGate):
    """Charge the configured commission floor once per order, across partial fills.

    The effective rate is also used by QExec's fill-time cash check. All fees
    remain native QExec Fee events and balance through its ExactAccountLedger.
    """

    def reset(self):
        super().reset()
        self._commission_notionals = {}

    def capture_state(self):
        state = super().capture_state()
        state["commission_notionals"] = self._commission_notionals.copy()
        return state

    def restore_state(self, state):
        super().restore_state(state)
        self._commission_notionals = state["commission_notionals"].copy()

    def _fee_rate_for(self, fill, order, state, spec):
        if spec.asset_class not in {AssetClass.EQUITY, AssetClass.ETF}:
            return super()._fee_rate_for(fill, order, state, spec)
        notional = decimal(fill.quantity) * decimal(fill.price) * decimal(spec.contract_multiplier)
        rate = Decimal(spec.metadata["commission_rate"])
        floor = Decimal(spec.metadata.get("min_commission", "0"))
        prior = self._commission_notionals.get(order.order_id, Decimal(0))
        commission = commission_increment(notional, prior, rate, floor)
        stamp = Decimal(0)
        if fill.side is Side.SELL and spec.asset_class is AssetClass.EQUITY:
            if spec.metadata.get("statutory_stamp_tax") == "true":
                duty = stamp_duty_rate(
                    CostsConfig(
                        statutory_stamp_tax=True,
                        stamp_tax=float(spec.metadata["stamp_duty_rate"]),
                    ),
                    fill.event_time,
                )
                stamp = Decimal(str(duty))
            else:
                stamp = Decimal(spec.metadata["stamp_duty_rate"])
        return commission / notional + stamp

    def fee_for(self, fill, order):
        fee = super().fee_for(fill, order)
        spec = self.instruments[fill.instrument_id]
        notional = decimal(fill.quantity) * decimal(fill.price) * decimal(spec.contract_multiplier)
        self._commission_notionals[order.order_id] = (
            self._commission_notionals.get(order.order_id, Decimal(0)) + notional
        )
        return fee


class ConfiguredBarMatchingModel(BarMatchingModel):
    """Next-bar open fills with adverse proportional slippage and volume limits."""

    def __init__(self, instruments, *, slippage: float, participation_rate: float):
        super().__init__(instruments, participation_rate=str(participation_rate))
        self.slippage = Decimal(str(slippage))
        if not Decimal(0) <= self.slippage < Decimal(1):
            raise ValueError("slippage must be in [0, 1)")

    def eligible(self, order, event):
        # A signal made from today's completed close can never fill today's open.
        return (
            isinstance(event, BarEvent)
            and super().eligible(order, event)
            and event.bar_start > order.intent.created_at
        )

    def _execution_price(self, order, bar):
        price = super()._execution_price(order, bar)
        if price is None or order.intent.order_type not in {OrderType.MARKET, OrderType.STOP}:
            return price
        direction = Decimal(1) if order.intent.side is Side.BUY else Decimal(-1)
        tick = self._price_tick(order.intent.instrument_id)
        rounding = ROUND_CEILING if direction > 0 else ROUND_FLOOR
        slipped = decimal(price) * (1 + direction * self.slippage)
        slipped = (slipped / tick).to_integral_value(rounding=rounding) * tick
        return FixedPoint(int(slipped.scaleb(price.scale)), price.scale)
