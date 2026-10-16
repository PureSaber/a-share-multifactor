"""Bridge the optimizer to QExec account facts and exact simulated cost rules."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from time import perf_counter

import numpy as np
import pandas as pd
from quant_portfolio.allocation_v2 import allocate_research
from quant_portfolio.executable import repair_integer_target
from quant_portfolio.feasibility import minimum_risk_seed, problem_payload
from quant_portfolio.solver import AllocationInputError, OptimizationError, constraint_residuals

from a_share_multifactor.execution_models import decimal, quote_target_trade


def executable_allocation(strategy, plan, snapshot, nav):
    started = perf_counter()
    timeout = plan["allocation"].get("timeout_seconds", 30)
    gate = strategy.risk_gate
    if gate is None:
        raise ValueError("executable allocation requires the execution account risk gate")
    current = {str(a): int(decimal(q)) for a, q in snapshot.positions.items() if q.units}
    risk_snapshot = plan.get("risk_snapshot")
    benchmark = None if risk_snapshot is None else risk_snapshot.benchmark
    assets = sorted(
        set(plan["scores"].index)
        | set(current)
        | (set() if benchmark is None else set(benchmark.index))
    )
    # Benchmark-only zero positions need no executable quote or tradable mark.
    traded_assets = set(plan["scores"].index) | set(current)
    missing_marks = traded_assets - set(strategy._closing_prices)
    if missing_marks:
        raise AllocationInputError(
            "known account marks are missing", reason="missing_mark", assets=missing_marks
        )
    marks = {a: decimal(strategy._closing_prices[a]) for a in traded_assets}
    held_weights = pd.Series(
        {a: float(Decimal(q) * marks[a] / nav) for a, q in current.items()}, dtype=float
    )
    states = gate.capture_state()
    lower, upper, buy_limits, sell_limits = {}, {}, {}, {}
    lots = {}
    fixed_targets = {}
    odd_lot_sell_assets = set()
    for asset in traded_assets:
        spec = gate.instruments[asset]
        lots[asset] = int(spec.metadata["lot_size"])
        if spec.asset_class.value == "equity" and spec.venue in {"SSE", "SZSE"}:
            odd_lot_sell_assets.add(asset)
        held = current.get(asset, 0)
        acquired = (
            gate.ledger.acquired_today(asset, snapshot.event_time.date())
            if spec.asset_class.value == "equity"
            else Decimal(0)
        )
        reserved = sum(
            original for a, original, _ in states["position_reservations"].values() if a == asset
        )
        sellable = max(0, held - int(acquired + reserved))
        buyable = int(nav / marks[asset]) if asset in plan["scores"].index else 0
        state = states["states"].get(asset)
        if state is None or state.status not in {"open", "trading", "normal"}:
            sellable, buyable = 0, 0
        if asset not in plan["scores"].index:
            # Leave only the known unsellable/odd-lot remainder outside the
            # investment universe. Neither the solver nor repair may retain a
            # freely tradable excluded holding merely to save turnover costs.
            remainder = held % lots[asset] if asset in odd_lot_sell_assets else 0
            sellable = max(
                (sellable // lots[asset]) * lots[asset],
                remainder + ((sellable - remainder) // lots[asset]) * lots[asset]
                if sellable >= remainder
                else 0,
            )
            fixed_targets[asset] = held - sellable
        sell_limits[asset], buy_limits[asset] = sellable, buyable
        lower[asset] = float(Decimal(held - sellable) * marks[asset] / nav)
        upper[asset] = (
            min(plan["max_weight"], float(Decimal(held + buyable) * marks[asset] / nav))
            if buyable
            else float(Decimal(held) * marks[asset] / nav)
        )
        if asset in fixed_targets:
            upper[asset] = lower[asset]
    costs = strategy.costs
    reproduction = {
        "current_quantities": current,
        "marks": {a: str(v) for a, v in marks.items()},
        "lots": lots,
        "cash": str(decimal(snapshot.cash_balances["CNY"])),
        "nav": str(nav),
        "max_buy": buy_limits,
        "max_sell": sell_limits,
        "fixed_targets": fixed_targets,
        "odd_lot_sell_assets": sorted(odd_lot_sell_assets),
        "costs": asdict(costs),
        "decision_at": snapshot.event_time.isoformat(),
        "allocation": dict(plan["allocation"]),
        "non_cash_assets": str(gate.ledger.dividend_receivable_balance("CNY")),
    }
    objective_costs = plan.get("objective_costs", True)
    if type(objective_costs) is not bool:
        raise ValueError("objective_costs must be an explicit boolean")
    from a_share_multifactor.trading_costs import stamp_duty_rate

    buy_rates = pd.Series({a: costs.commission + costs.slippage for a in traded_assets})
    sell_rates = pd.Series(
        {
            a: costs.commission
            + costs.slippage
            + (
                stamp_duty_rate(costs, snapshot.event_time)
                if gate.instruments[a].asset_class.value == "equity"
                else 0
            )
            for a in traded_assets
        }
    )
    kwargs = {
        key: plan[key]
        for key in (
            "factor_exposures",
            "factor_bounds",
            "risk_snapshot",
            "max_tracking_error_override",
            "expected_returns",
            "ic_labels",
            "factor_bounds_absolute",
            "signal_cross_section",
        )
        if plan.get(key) is not None
    }
    try:
        remaining = timeout - (perf_counter() - started)
        if remaining <= 0:
            raise OptimizationError(
                "timeout", "account input construction exhausted rebalance deadline"
            )
        result = allocate_research(
            plan["scores"],
            plan["trailing_returns"],
            held_weights,
            None,
            invested_limit=plan["invested_limit"],
            max_weight=plan["max_weight"],
            config=plan["allocation"],
            remaining_timeout_seconds=remaining,
            decision_at=snapshot.event_time,
            lower_bounds=pd.Series(lower),
            upper_bounds=pd.Series(upper),
            buy_costs=buy_rates if objective_costs else buy_rates * 0,
            sell_costs=sell_rates if objective_costs else sell_rates * 0,
            **kwargs,
        )
    except OptimizationError as exc:
        exc.diagnostics = {**exc.diagnostics, "account_input": reproduction}
        raise
    validation = result.validation_context
    settings, risk, wb = validation["settings"], validation["risk"], validation["benchmark"]
    assets = validation["assets"]
    bounds, x = validation["factor_bounds"], validation["factor_exposures"]
    # The risk bridge has already converted active exposure limits to absolute.
    current_vector = held_weights.reindex(assets, fill_value=0).to_numpy()

    def check(w):
        vector = w.reindex(assets, fill_value=0).to_numpy()
        # Quantity lower bounds are checked exactly by the repair. Weight bounds
        # are checked after fee-induced NAV changes.
        residuals = constraint_residuals(
            vector,
            current_vector,
            settings,
            assets=assets,
            budget=plan["invested_limit"],
            budget_equality=False,
            factor_exposures=x,
            factor_bounds=bounds,
            risk=risk,
            benchmark=wb,
            detailed_asset_bounds=True,
        )
        if costs.max_holdings > 0:
            residuals["max_holdings"] = int((vector > 1e-9).sum()) - costs.max_holdings
        return residuals

    def quote(asset, delta):
        return quote_target_trade(
            gate.instruments[asset], delta, marks[asset], costs, snapshot.event_time
        )

    def seed(remaining):
        point, info = minimum_risk_seed(validation, min(5.0, remaining / 2))
        return pd.Series(point, index=assets).reindex(sorted(traded_assets)), info

    # Reserve existing buy commitments; they must be cancelled/reconciled before
    # a new full-account target can be planned, rather than double-spending cash.
    if states["cash_reservations"] or states["position_reservations"]:
        raise AllocationInputError(
            "allocation requires outstanding orders to be reconciled first",
            reason="outstanding_orders",
        )
    cash = decimal(snapshot.cash_balances["CNY"])
    remaining = timeout - (perf_counter() - started)
    if remaining <= 0:
        raise OptimizationError(
            "timeout", "rebalance deadline exceeded before integer repair", result.diagnostics
        )
    try:
        repaired = repair_integer_target(
            result.weights.reindex(sorted(traded_assets), fill_value=0),
            current=current,
            prices=marks,
            lots=lots,
            cash=cash,
            nav=nav,
            non_cash_assets=gate.ledger.dividend_receivable_balance("CNY"),
            fixed_quantities=fixed_targets,
            quote_trade=quote,
            constraint_residuals=check,
            max_invested=plan["invested_limit"],
            max_buy=buy_limits,
            max_sell=sell_limits,
            timeout_seconds=remaining,
            feasibility_seed=seed if risk is not None else None,
            odd_lot_sell_assets=odd_lot_sell_assets,
        )
    except OptimizationError as exc:
        exc.diagnostics = {
            "stage": "integer_repair",
            "continuous_weights": result.weights.to_dict(),
            "continuous_solver": result.diagnostics,
            "problem": problem_payload(validation),
            "account_input": reproduction,
            **exc.diagnostics,
        }
        raise
    diagnostics = {**result.diagnostics, "execution": repaired.diagnostics}
    diagnostics["objective_includes_cost_estimates"] = objective_costs
    diagnostics["rebalance_seconds"] = perf_counter() - started
    if risk is not None:
        w = repaired.weights.reindex(assets, fill_value=0).to_numpy()
        diagnostics["integer_annualized_volatility"] = float(np.sqrt(max(risk.variance(w), 0)))
        if wb is not None:
            diagnostics["integer_annualized_tracking_error"] = float(
                np.sqrt(max(risk.variance(w - wb), 0))
            )
    return result.weights, repaired.quantities, diagnostics
