"""Causal daily risk snapshots shared by allocation and decision checks.

Risk returns use raw traded prices and previously announced entitlements. A market-only
model is a statistical proxy, never a licensed Barra model or real descriptors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quant_lab.research_options import validate_risk_model


def neutralize_signals(features, names, by):
    from quant_factors.neutralize import neutralize_cross_section

    if not by:
        return features
    for field in by:
        if field not in features or features[field].isna().any():
            raise ValueError(f"Neutralization is missing PIT {field}")
        if field == "market_cap":
            values = pd.to_numeric(features[field], errors="raise")
            if not np.isfinite(values).all() or not values.gt(0).all():
                raise ValueError("Neutralization requires positive finite market_cap")
            features = features.assign(market_cap=values)
    return neutralize_cross_section(features, cols=names, by=by)


def causal_asset_returns(raw, actions=None):
    """Ex-date entitlement returns; future qfq vintages never enter estimation."""
    if "adjustment" not in raw or not raw.adjustment.eq("none").all():
        raise ValueError("Risk estimation requires unadjusted traded prices")
    prices = raw.pivot(index="date", columns="symbol", values="close").sort_index()
    returns = prices.pct_change(fill_method=None)
    if actions is not None and not actions.empty:
        if actions.duplicated(["symbol", "ex_date"]).any():
            raise ValueError("Duplicate risk-return corporate actions")
        for row in actions.itertuples():
            ex_date = pd.Timestamp(row.ex_date)
            if ex_date <= prices.index.min() or ex_date > prices.index.max():
                continue
            announced = pd.Timestamp(row.announced_date)
            cash, ratio = float(row.cash_per_share), float(row.share_ratio)
            if (
                pd.isna(announced)
                or announced >= ex_date
                or not np.isfinite([cash, ratio]).all()
                or cash < 0
                or ratio <= 0
            ):
                raise ValueError("Risk returns require an evidenced announcement before ex-date")
            if ex_date not in prices.index or row.symbol not in prices:
                raise ValueError("Risk-return action lacks traded-price coverage")
            prior = prices.index[prices.index < ex_date][-1]
            if pd.Timestamp(row.record_date) != prior:
                raise ValueError("Risk-return entitlement record date is inconsistent")
            returns.loc[ex_date, row.symbol] = (
                prices.loc[ex_date, row.symbol] * ratio + cash
            ) / prices.loc[prior, row.symbol] - 1
    return returns


def build_risk_schedule(research, recipe, sessions, *, raw_prices, actions=None):
    """Fit on lagged exposures and returns available at each completed close."""
    from quant_risk_monitor.factor_model import (
        AssetReturnObservation,
        ExposureSnapshot,
        FactorModelConfig,
        fit_barra_style_risk_model,
    )

    config = recipe.get("risk_model")
    industry_field = recipe.get("risk", {}).get("industry_field")
    if config is None and industry_field is None:
        return {}
    settings = validate_risk_model(config, recipe.get("required_history", {})) if config else None
    returns = causal_asset_returns(raw_prices, actions)
    symbols = returns.columns
    snapshots, observations = [], []
    schedule = {}
    session_set = {pd.Timestamp(day) for day in sessions}
    for offset, (day, values) in enumerate(returns.iterrows()):
        cutoff = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=7)
        rows = research.loc[research.date.eq(day)].set_index("symbol").reindex(symbols)
        if settings:
            exposures = pd.DataFrame(index=symbols)
            if settings["market_factor"]:
                exposures["market"] = 1.0
            for factor, field in settings["exposure_fields"].items():
                exposures[factor] = pd.to_numeric(rows[field], errors="raise")
            snapshots.append(
                ExposureSnapshot(
                    effective_at=cutoff,
                    available_at=cutoff,
                    values=exposures,
                    source="PIT-history-at-close",
                )
            )
            if offset:
                prior = pd.Timestamp(returns.index[offset - 1]).tz_localize("UTC") + pd.Timedelta(
                    hours=7
                )
                observations.append(
                    AssetReturnObservation(
                        period_start=prior,
                        period_end=cutoff,
                        available_at=cutoff,
                        values=values,
                        source="raw-traded-close+previously-announced-entitlement",
                    )
                )
        if day not in session_set:
            continue
        item = {"classifications": {}, "settings": settings, "model": None}
        if industry_field:
            if industry_field not in rows or rows[industry_field].isna().any():
                raise ValueError(f"Missing PIT classification at {day}")
            groups = rows[industry_field].astype(str)
            if groups.str.strip().eq("").any():
                raise ValueError("Empty PIT classification")
            item["classifications"] = groups.to_dict()
        if settings:
            config_keys = (
                "model_kind",
                "min_periods",
                "min_assets_per_period",
                "min_asset_observations",
                "covariance_shrinkage",
                "specific_variance_shrinkage",
                "annualization",
            )
            selected = observations[-settings["lookback"] :]
            model = fit_barra_style_risk_model(
                exposure_snapshots=snapshots[-len(selected) - 1 :],
                return_observations=selected,
                as_of=cutoff,
                config=FactorModelConfig(**{key: settings[key] for key in config_keys}),
            )
            if set(settings["benchmark_weights"]) - set(symbols):
                raise ValueError("Benchmark references assets missing from risk model")
            item["model"] = model
        schedule[day.date()] = item
    return schedule


def allocation_risk_inputs(item, symbols, risk):
    """Convert active bounds to absolute bounds; industry caps use one-hot X."""
    exposures = pd.DataFrame(index=symbols)
    bounds = {}
    model = item.get("model")
    if model is not None:
        settings = item["settings"]
        bounds = {key: tuple(value) for key, value in settings["factor_bounds"].items()}
        benchmark = pd.Series(settings["benchmark_weights"], dtype=float).reindex(
            model.exposures.index, fill_value=0
        )
        baseline = model.exposures.T @ benchmark
        for factor, (lower, upper) in settings["active_factor_bounds"].items():
            active = (lower + baseline[factor], upper + baseline[factor])
            absolute = bounds.get(factor, (-np.inf, np.inf))
            bounds[factor] = (max(absolute[0], active[0]), min(absolute[1], active[1]))
            if bounds[factor][0] > bounds[factor][1]:
                raise ValueError("Absolute and active factor bounds are infeasible")
        for factor in bounds:
            exposures[factor] = model.exposures.loc[symbols, factor]
    if "max_industry_weight" in risk:
        groups = item["classifications"]
        for group in sorted(set(groups.values())):
            factor = "industry:" + group
            if factor in bounds:
                raise ValueError("Industry constraint name collides with risk factor")
            exposures[factor] = [float(groups[symbol] == group) for symbol in symbols]
            bounds[factor] = (0.0, risk["max_industry_weight"])
    return {
        "factor_exposures": exposures if bounds else None,
        "factor_bounds": bounds or None,
        "covariance_override": model.asset_covariance.loc[symbols, symbols]
        if model is not None
        else None,
    }


def check_model_target(item, weights, *, cash_policy_reason=None):
    model = item.get("model")
    if model is None:
        return {}, []
    settings = item["settings"]
    report = model.analyze(
        weights={key: float(value) for key, value in weights.items()},
        benchmark_weights=settings["benchmark_weights"] or None,
    )
    full_cash = not any(float(value) != 0 for value in weights.values())
    if full_cash:
        cash_policy_reason = cash_policy_reason or "full_cash_state"
    else:
        cash_policy_reason = None
    alerts = []
    for key, values in (
        ("factor_bounds", report.portfolio_exposures),
        ("active_factor_bounds", report.active_exposures),
    ):
        for factor, (lower, upper) in settings[key].items():
            actual = float(values[factor])
            if actual < lower - 1e-9 or actual > upper + 1e-9:
                cash_relative_warning = key == "active_factor_bounds" and full_cash
                details = {
                    "factor": factor,
                    "actual": actual,
                    "lower": lower,
                    "upper": upper,
                }
                if cash_relative_warning:
                    details["policy_reason"] = cash_policy_reason
                alerts.append(
                    {
                        "rule_id": "portfolio." + key,
                        "severity": "warning" if cash_relative_warning else "critical",
                        "message": "Rounded target violates factor exposure bound",
                        "details": details,
                    }
                )
    if (
        "max_tracking_error" in settings
        and report.tracking_risk.volatility > settings["max_tracking_error"] + 1e-9
    ):
        details = {
            "actual": report.tracking_risk.volatility,
            "limit": settings["max_tracking_error"],
        }
        if full_cash:
            details["policy_reason"] = cash_policy_reason
        alerts.append(
            {
                "rule_id": "portfolio.max_tracking_error",
                "severity": "warning" if full_cash else "critical",
                "message": "Target exceeds annualized tracking-error limit",
                "details": details,
            }
        )
    return report.to_dict(), alerts
