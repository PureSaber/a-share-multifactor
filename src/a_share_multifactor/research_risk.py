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


def _price_availability(raw):
    """Preserve explicit vendor arrival times instead of relabeling them as close.

    Missing columns retain the historical caller-assumed close contract. This
    compatibility path is not evidence of historical PIT source availability.
    An explicitly supplied but missing, naive, or pre-close timestamp is invalid.
    """
    closes = pd.to_datetime(raw.date).dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    closes = closes.dt.tz_convert("UTC")
    if "available_at" not in raw:
        return closes
    values = []
    for value in raw.available_at:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("Price available_at must be complete and timezone-aware")
        values.append(timestamp.tz_convert("UTC"))
    available = pd.Series(values, index=raw.index, dtype="datetime64[ns, UTC]")
    if available.lt(closes).any():
        raise ValueError("Daily close available_at cannot precede the market close")
    return available


def _exposure_availability(rows, cutoff, fields):
    """Honor explicitly supplied per-feature and underlying price availability."""
    latest = cutoff
    for field in ["available_at", *(name + "_available_at" for name in fields)]:
        if field not in rows:
            continue
        for value in rows[field]:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError("Exposure availability must be complete and timezone-aware")
            latest = max(latest, timestamp.tz_convert("UTC"))
    return latest


def _action_availability(actions):
    if actions is None or actions.empty or "available_at" not in actions:
        return {}
    arrivals = {}
    for row in actions.itertuples():
        timestamp = pd.Timestamp(row.available_at)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("Action available_at must be complete and timezone-aware")
        day = pd.Timestamp(row.ex_date)
        timestamp = timestamp.tz_convert("UTC")
        arrivals[day] = max(arrivals.get(day, timestamp), timestamp)
    return arrivals


def build_risk_schedule(research, recipe, sessions, *, raw_prices, actions=None, history=None):
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
    price_arrivals = (
        raw_prices.assign(_risk_available_at=_price_availability(raw_prices))
        .groupby("date")
        ._risk_available_at.max()
    )
    action_arrivals = _action_availability(actions)
    symbols = returns.columns
    snapshots, observations = [], []
    schedule = {}
    session_set = {pd.Timestamp(day) for day in sessions}
    for offset, (day, values) in enumerate(returns.iterrows()):
        cutoff = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=7)
        rows = research.loc[research.date.eq(day)].set_index("symbol").reindex(symbols)
        if settings:
            if settings.get("exposure_pipeline") is not None:
                snapshots.append(
                    _pipeline_snapshot(rows, cutoff, settings["exposure_pipeline"], history)
                )
            else:
                exposures = pd.DataFrame(index=symbols)
                if settings["market_factor"]:
                    exposures["market"] = 1.0
                for factor, field in settings["exposure_fields"].items():
                    exposures[factor] = pd.to_numeric(rows[field], errors="raise")
                snapshots.append(
                    ExposureSnapshot(
                        effective_at=cutoff,
                        available_at=max(
                            _exposure_availability(
                                rows, cutoff, settings["exposure_fields"].values()
                            ),
                            price_arrivals.loc[day] if settings["exposure_fields"] else cutoff,
                        ),
                        values=exposures,
                        source="caller-exposure-history; explicit-arrivals-when-supplied",
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
                        available_at=max(
                            price_arrivals.loc[day],
                            price_arrivals.loc[returns.index[offset - 1]],
                            action_arrivals.get(day, cutoff),
                        ),
                        values=values,
                        source="raw-traded-close+previously-announced-entitlement; "
                        + (
                            "explicit-price-arrival"
                            if "available_at" in raw_prices
                            else "caller-assumed-close-availability"
                        ),
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
            selected = [row for row in observations if row.available_at <= cutoff][
                -settings["lookback"] :
            ]
            model = fit_barra_style_risk_model(
                exposure_snapshots=[row for row in snapshots if row.available_at <= cutoff],
                return_observations=selected,
                as_of=cutoff,
                config=FactorModelConfig(
                    **{key: settings[key] for key in config_keys}, **settings["estimation"]
                ),
            )
            if set(settings["benchmark_weights"]) - set(symbols):
                raise ValueError("Benchmark references assets missing from risk model")
            item["model"] = model
        schedule[day.date()] = item
    return schedule


def _pipeline_snapshot(rows, cutoff, pipeline, history):
    """Materialize each descriptor and classification from its original vintage."""
    from dataclasses import replace

    from quant_data_kit.research_coverage import asof_history
    from quant_risk_monitor import ExposureBuildConfig, build_exposure_snapshot

    if history is None:
        raise ValueError("exposure_pipeline requires original PIT history with source vintages")
    mappings = {"market_cap": pipeline["market_cap_field"], **pipeline["descriptor_fields"]}
    descriptors = pd.DataFrame(index=rows.index)
    lineage, effective, available = {}, [], []
    for descriptor, field in mappings.items():
        selected = (
            asof_history(history, as_of=cutoff, domain="fundamentals", field=field)
            .set_index("symbol")
            .reindex(rows.index)
        )
        if selected.effective_at.isna().any() or selected.available_at.isna().any():
            raise ValueError(f"Missing original PIT descriptor history for {field}")
        maximum_age = pipeline["config"].get("max_descriptor_age_days")
        if (
            maximum_age is not None
            and ((cutoff - selected.effective_at).dt.total_seconds() > maximum_age * 86400).any()
        ):
            raise ValueError(f"Stale original PIT descriptor history for {field}")
        descriptors[descriptor] = pd.to_numeric(selected.value, errors="raise")
        effective.append(selected.effective_at)
        available.append(selected.available_at)
        lineage[field] = selected.reset_index().astype(str).to_dict("records")
    descriptors["effective_at"] = pd.concat(effective, axis=1).max(axis=1)
    descriptors["available_at"] = pd.concat(available, axis=1).max(axis=1)
    descriptors["source"] = "research-original-PIT-descriptor-vintages"
    descriptors["asset"] = rows.index
    field = pipeline["industry_field"]
    selected = (
        asof_history(history, as_of=cutoff, domain="classification", field=field)
        .set_index("symbol")
        .reindex(rows.index)
    )
    if selected.effective_at.isna().any():
        raise ValueError("Missing original PIT classification history")
    classifications = pd.DataFrame(
        {
            "asset": rows.index,
            "effective_at": selected.effective_at.to_numpy(),
            "available_at": selected.available_at.to_numpy(),
            "source": "research-original-PIT-classification-vintage",
            "industry": selected.value.to_numpy(),
        }
    )
    lineage[field] = selected.reset_index().astype(str).to_dict("records")
    snapshot = build_exposure_snapshot(
        descriptor_history=descriptors,
        classification_history=classifications,
        assets=list(rows.index),
        as_of=cutoff,
        config=ExposureBuildConfig(**pipeline["config"]),
    )
    return replace(
        snapshot, provenance={**snapshot.provenance, "research_history_lineage": lineage}
    )


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
