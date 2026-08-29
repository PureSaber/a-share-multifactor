"""Wave 3: multifactor compute_factors must match quant_factors for shared ids."""

from __future__ import annotations

import numpy as np
import pandas as pd
from quant_factors.core import compute_factors as qf_compute

from a_share_multifactor.factors import SHARED_QF_FACTORS, compute_factors, quant_factors_version


def _panel(n_days: int = 60, n_symbols: int = 3) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for i, sym in enumerate([f"S{i:03d}" for i in range(n_symbols)]):
        close = 10 + i + np.cumsum(rng.normal(0, 0.02, n_days))
        for d in range(n_days):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=d),
                    "symbol": sym,
                    "open": close[d],
                    "high": close[d] * 1.01,
                    "low": close[d] * 0.99,
                    "close": close[d],
                    "volume": 1_000_000 + 1000 * d,
                }
            )
    return pd.DataFrame(rows)


def test_quant_factors_version_pinned() -> None:
    assert quant_factors_version() == "0.2.1"


def test_shared_factors_numeric_parity() -> None:
    panel = _panel()
    names = list(SHARED_QF_FACTORS)
    via_asm = compute_factors(panel, factor_names=names)
    via_qf = qf_compute(panel, factors=names)
    for name in names:
        a = via_asm[name].to_numpy(dtype=float)
        b = via_qf[name].to_numpy(dtype=float)
        # Allow NaN alignment on warm-up windows
        mask = ~(np.isnan(a) | np.isnan(b))
        assert mask.any(), f"{name} produced no overlapping values"
        np.testing.assert_allclose(a[mask], b[mask], rtol=1e-10, atol=1e-12)
