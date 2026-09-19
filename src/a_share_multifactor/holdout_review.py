"""Seal a preregistered forward-paper evaluation only after its interval ends."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from quant_lab import load_and_validate_standard_run
from quant_lab.trials import TrialRegistry

from a_share_multifactor.decision_contract import clean_json
from a_share_multifactor.decision_workflow import load_inputs, sha256
from a_share_multifactor.performance import return_statistics


def evaluate(db: Path, study_id: str, run: Path, *, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    registry = TrialRegistry(db)
    spec = registry.definition(study_id)["definition"]
    start, end = pd.Timestamp(spec["holdout_start"]), pd.Timestamp(spec["holdout_end"])
    if now.date() <= end.date():
        raise ValueError("Holdout has not ended; no partial evaluation or premature alpha claim")
    card = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    if pd.Timestamp(card["as_of"]) < end:
        raise ValueError("Run does not reach the end of the registered holdout")
    if card["status"] == "blocked" or card["evidence"].get("study_id") != study_id:
        raise ValueError("Run is not a completed observation from this study")
    identity = {key: card["evidence"][key] for key in ("code_version", "internal_dependencies")}
    if identity != spec["code_identity"]:
        raise ValueError("Holdout strategy/dependencies differ from preregistration")
    manifest = load_and_validate_standard_run(run)
    if manifest.code_version != identity["code_version"]:
        raise ValueError("Decision and immutable ledger code identity disagree")
    source = Path(card["evidence"]["inputs"])
    if sha256(source / "manifest.json") != card["evidence"]["input_manifest_sha256"]:
        raise ValueError("Holdout source manifest changed")
    _, frames = load_inputs(source)
    expected = pd.DatetimeIndex(pd.to_datetime(frames["calendar"].date))
    expected = expected[(expected >= start) & (expected <= end)]
    returns = pd.read_parquet(run / "standard/v2/returns.parquet")
    returns["date"] = (
        pd.to_datetime(returns.event_time, utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    selected = returns[(returns.date >= start) & (returns.date <= end)]
    if not len(expected) or selected.date.duplicated().any() or set(selected.date) != set(expected):
        raise ValueError("Holdout does not contain every exchange session")
    if pd.Timestamp(card["validation"]["simulation_start"]) >= start:
        raise ValueError("Forward account was not initialized before holdout")
    benchmark = frames["benchmark"].set_index("date").benchmark_return.reindex(expected)
    if benchmark.isna().any():
        raise ValueError("Holdout benchmark has gaps")
    evidence = clean_json(
        {
            "start": spec["holdout_start"],
            "end": spec["holdout_end"],
            "code_identity": identity,
            "input_sha256": sha256(source / "manifest.json"),
            "standard_manifest_sha256": sha256(run / "standard/v2/run_manifest.json"),
            "observations": len(expected),
            "strategy_net": return_statistics(selected.net_return, 252),
            "hs300_price_index_before_costs": return_statistics(benchmark, 252),
            "scope": "single preregistered forward sample; not automatic investment approval",
        }
    )
    registry.seal_holdout(study_id, evidence, now=now)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.db, args.study, args.run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
