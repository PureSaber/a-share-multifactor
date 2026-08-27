"""Strict out-of-sample factor validation for the A-share research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd
from quant_factors.validation import (
    audit_feature_availability,
    benjamini_hochberg,
    summarize_fold_stability,
    walk_forward_splits,
)

from a_share_multifactor.config import AppConfig


@dataclass
class ResearchValidationResult:
    fold_metrics: pd.DataFrame
    multiple_testing: pd.DataFrame
    summary: dict
    leakage_audit: pd.DataFrame


def _daily_rank_ic(frame: pd.DataFrame, factor: str, target: str) -> pd.Series:
    return (
        frame.groupby("date", sort=True)
        .apply(
            lambda group: group[factor].corr(group[target], method="spearman"),
            include_groups=False,
        )
        .dropna()
    )


def _two_sided_normal_pvalue(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return 1.0
    std = float(clean.std(ddof=1))
    if std == 0:
        return 0.0 if float(clean.mean()) != 0 else 1.0
    z = abs(float(clean.mean()) / (std / sqrt(len(clean))))
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def run_research_validation(
    panel: pd.DataFrame,
    factors: list[str],
    target_col: str,
    config: AppConfig,
) -> ResearchValidationResult:
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"]).sort_values().unique())
    settings = config.validation
    splits = walk_forward_splits(
        dates,
        train_size=settings.train_size,
        test_size=settings.test_size,
        step_size=settings.step_size,
        embargo_size=settings.embargo_size,
        expanding=True,
    )
    if not splits:
        raise ValueError(
            "No walk-forward folds; increase history or reduce validation train/test windows"
        )

    rows: list[dict[str, float | int | str]] = []
    for split in splits:
        train_dates = dates[split.train_indices]
        test_dates = dates[split.test_indices]
        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"].isin(test_dates)].copy()
        signed_scores: list[pd.Series] = []
        for factor in factors:
            train_ic = _daily_rank_ic(train, factor, target_col)
            direction = 1.0 if float(train_ic.mean()) >= 0 else -1.0
            test_ic = _daily_rank_ic(test, factor, target_col)
            rows.append(
                {
                    "fold": split.fold,
                    "factor": factor,
                    "train_start": split.train_start.date().isoformat(),
                    "train_end": split.train_end.date().isoformat(),
                    "test_start": split.test_start.date().isoformat(),
                    "test_end": split.test_end.date().isoformat(),
                    "train_mean_ic": float(train_ic.mean()) if not train_ic.empty else np.nan,
                    "test_mean_ic": float(test_ic.mean()) if not test_ic.empty else np.nan,
                    "direction": direction,
                }
            )
            signed_scores.append(
                test.groupby("date")[factor].rank(pct=True, method="average") * direction
            )
        if signed_scores:
            test["oos_composite_score"] = pd.concat(signed_scores, axis=1).mean(axis=1)
            composite_ic = _daily_rank_ic(test, "oos_composite_score", target_col)
            rows.append(
                {
                    "fold": split.fold,
                    "factor": "__composite__",
                    "train_start": split.train_start.date().isoformat(),
                    "train_end": split.train_end.date().isoformat(),
                    "test_start": split.test_start.date().isoformat(),
                    "test_end": split.test_end.date().isoformat(),
                    "train_mean_ic": np.nan,
                    "test_mean_ic": float(composite_ic.mean())
                    if not composite_ic.empty
                    else np.nan,
                    "direction": 1.0,
                }
            )

    fold_metrics = pd.DataFrame(rows)
    factor_rows = fold_metrics[fold_metrics["factor"] != "__composite__"]
    hypotheses = []
    for factor, group in factor_rows.groupby("factor"):
        hypotheses.append(
            {"factor": factor, "p_value": _two_sided_normal_pvalue(group["test_mean_ic"])}
        )
    hypothesis_frame = pd.DataFrame(hypotheses)
    adjusted = benjamini_hochberg(
        hypothesis_frame["p_value"], alpha=settings.multiple_testing_alpha
    )
    multiple_testing = pd.concat(
        [hypothesis_frame.reset_index(drop=True), adjusted[["adjusted_p_value", "reject"]]],
        axis=1,
    )

    composite = fold_metrics[fold_metrics["factor"] == "__composite__"]
    summary = summarize_fold_stability(composite, "test_mean_ic")
    summary["method"] = "expanding_walk_forward"
    summary["embargo_size"] = settings.embargo_size
    summary["discoveries_after_fdr"] = int(multiple_testing["reject"].sum())

    leakage_audit = pd.DataFrame()
    fundamental_features = [
        factor for factor in factors if factor in {"market_cap", "pe_ratio", "pb_ratio"}
    ]
    if fundamental_features and "source_available_at" in panel.columns:
        leakage_audit = audit_feature_availability(
            panel,
            {factor: "source_available_at" for factor in fundamental_features},
        )
        if int(leakage_audit["future_rows"].sum()) > 0:
            raise ValueError("Feature availability audit found future data")
    return ResearchValidationResult(
        fold_metrics=fold_metrics,
        multiple_testing=multiple_testing,
        summary=summary,
        leakage_audit=leakage_audit,
    )
