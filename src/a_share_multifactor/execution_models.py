"""Configured A-share simulation models using QExec's one exact ledger."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from quant_data_kit import AssetClass, FixedPoint
from quant_execution import BarMatchingModel, OrderType, RuleBookRiskGate, Side


def decimal(value: FixedPoint) -> Decimal:
    return Decimal(value.units).scaleb(-value.scale)


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
        paid = max(prior * rate, floor) if prior else Decimal(0)
        commission = max((prior + notional) * rate, floor) - paid
        stamp = Decimal(spec.metadata["stamp_duty_rate"]) if fill.side is Side.SELL else Decimal(0)
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
        return super().eligible(order, event) and event.bar_start > order.intent.created_at

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
