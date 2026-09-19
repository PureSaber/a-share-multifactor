"""One return convention for research reports and decision evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd


def return_statistics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Geometric annual growth and arithmetic Sharpe (zero risk-free rate).

    Inputs must be non-overlapping, equally spaced return observations. Missing
    observations are not zero returns. The initial NAV of one is included in DD.
    """
    clean = pd.to_numeric(returns, errors="raise").dropna().astype(float)
    if periods_per_year <= 0 or not np.isfinite(clean).all() or (clean < -1).any():
        raise ValueError("Invalid returns or sampling frequency")
    if clean.empty:
        return {
            **dict.fromkeys(
                [
                    "mean_return",
                    "total_return",
                    "ann_return",
                    "annual_mean_return",
                    "ann_vol",
                    "sharpe",
                    "max_drawdown",
                ],
                float("nan"),
            ),
            "total_return": 0.0,
            "max_drawdown": 0.0,
        }
    nav = np.r_[1.0, (1 + clean).cumprod().to_numpy()]
    total_return = float(nav[-1] - 1)
    mean_return = float(clean.mean())
    ann_vol = float(clean.std(ddof=0) * np.sqrt(periods_per_year))
    return {
        "mean_return": mean_return,
        "total_return": total_return,
        "ann_return": float(nav[-1] ** (periods_per_year / len(clean)) - 1),
        "annual_mean_return": mean_return * periods_per_year,
        "ann_vol": ann_vol,
        "sharpe": mean_return * periods_per_year / ann_vol if ann_vol > 0 else float("nan"),
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
    }
