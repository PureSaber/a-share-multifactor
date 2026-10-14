"""Frozen, explicitly timestamped return forecasts for allocation protocol v2."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from quant_portfolio.risk_snapshot import timestamp


def read_forecasts(path):
    payload = Path(path).read_bytes()
    frame = pd.read_csv(BytesIO(payload), dtype={"symbol": str, "model_version": str})
    required = {
        "symbol",
        "forecast_at",
        "available_at",
        "horizon_sessions",
        "expected_return",
        "model_version",
    }
    if set(frame.columns) != required or frame.empty:
        raise ValueError(
            "return forecast CSV requires symbol, forecast_at, available_at, horizon_sessions, expected_return, model_version"
        )
    for key in ("forecast_at", "available_at"):
        frame[key] = frame[key].map(timestamp)
    if (frame.available_at < frame.forecast_at).any():
        raise ValueError("forecasts cannot be available before their estimation time")
    if any(
        frame[key].isna().any() or frame[key].str.strip().eq("").any()
        for key in ("symbol", "model_version")
    ):
        raise ValueError("forecasts require asset and model identifiers")
    if (
        not np.isfinite(frame[["expected_return", "horizon_sessions"]]).all().all()
        or (frame.horizon_sessions < 1).any()
        or (frame.horizon_sessions % 1 != 0).any()
    ):
        raise ValueError("forecasts must be finite with a positive integer horizon")
    if frame.duplicated(["symbol", "forecast_at", "available_at", "horizon_sessions"]).any():
        raise ValueError("ambiguous duplicate forecast revisions")
    return frame, payload, hashlib.sha256(payload).hexdigest()


def forecasts_at(frame, *, decision_at, assets, horizon_sessions):
    cutoff = timestamp(decision_at)
    selected = (
        frame.loc[
            (frame.available_at <= cutoff)
            & (frame.forecast_at <= cutoff)
            & (
                frame.forecast_at.dt.tz_convert("Asia/Shanghai").dt.date
                == cutoff.tz_convert("Asia/Shanghai").date()
            )
            & (frame.horizon_sessions == horizon_sessions)
        ]
        .sort_values(["forecast_at", "available_at"])
        .drop_duplicates("symbol", keep="last")
    )
    result = selected.set_index("symbol").expected_return.reindex(assets)
    if result.isna().any():
        raise ValueError("missing contemporaneous, available return forecasts for candidates")
    return result
