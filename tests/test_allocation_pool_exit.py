from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest
from quant_data_kit import AssetClass, FixedPoint

from a_share_multifactor.allocation_execution import executable_allocation
from a_share_multifactor.config import CostsConfig


@pytest.mark.parametrize("status", ["open", "suspended"])
@pytest.mark.parametrize("objective_costs", [True, False])
def test_excluded_holding_exits_unless_known_untradable(status, objective_costs):
    spec = SimpleNamespace(
        metadata={"lot_size": "100"},
        asset_class=AssetClass.ETF,
        price_tick=FixedPoint(1, 3),
        contract_multiplier=FixedPoint(1, 0),
    )
    gate = SimpleNamespace(
        instruments={"A": spec, "B": spec},
        ledger=SimpleNamespace(dividend_receivable_balance=lambda currency: Decimal(0)),
        capture_state=lambda: {
            "position_reservations": {},
            "cash_reservations": {},
            "states": {"A": SimpleNamespace(status=status), "B": SimpleNamespace(status="open")},
        },
    )
    strategy = SimpleNamespace(
        risk_gate=gate,
        _closing_prices={"A": FixedPoint(10, 0), "B": FixedPoint(10, 0)},
        costs=CostsConfig(commission=0, min_commission=5, stamp_tax=0, slippage=0),
    )
    plan = {
        "objective_costs": objective_costs,
        "scores": pd.Series({"B": 1.0}),
        "trailing_returns": None,
        "invested_limit": 0.8,
        "max_weight": 1.0,
        "allocation": {
            "protocol_version": 2,
            "mode": "equal",
            "risk_objective": "absolute",
            "forecast_horizon_sessions": 1,
            "expected_return_model": "none",
            "max_turnover": 2.0,
        },
    }
    snapshot = SimpleNamespace(
        positions={"A": FixedPoint(100, 0)},
        event_time=datetime(2026, 1, 5, 7, tzinfo=timezone.utc),
        cash_balances={"CNY": FixedPoint(1000, 0)},
    )
    weights, targets, diagnostics = executable_allocation(strategy, plan, snapshot, Decimal(2000))
    expected = 0 if status == "open" else 100
    assert targets.get("A", 0) == expected
    assert weights.A == pytest.approx(expected / 200, abs=1e-7)
    assert Decimal(diagnostics["execution"]["cash"]) >= 0
    assert diagnostics["objective_includes_cost_estimates"] is objective_costs
    assert Decimal(diagnostics["execution"]["costs"]) == (10 if status == "open" else 0)
