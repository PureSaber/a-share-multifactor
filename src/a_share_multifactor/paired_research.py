"""Execute a frozen economic intervention matrix through the real research ledger."""

import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd
from quant_lab.counterfactuals import attribution, validate_plan
from quant_lab.research import VARIANT_FIELDS, canonical, digest, file_hash, validate_recipe

from a_share_multifactor.research_workbench import EquityResearchExecutor


def run_paired(recipe, plan, output: Path, *, executor=None):
    """No auto-selection, live orders or changed production risk configuration.

    Non-cash definitions are full candidate dictionaries plus an optional risk
    subtree. Cash is explicitly zero-interest native cash, not a Treasury proxy.
    The supplied snapshot and recipe are frozen before the first execution.
    """
    validate_recipe(recipe)
    if recipe.get("validation") or recipe["backend"] != "equity":
        raise ValueError("paired runner requires a single common equity interval")
    validate_plan(plan)
    if plan["benchmarks"].get("cash") != {"mode": "cash", "daily_return": 0}:
        raise ValueError("cash benchmark must explicitly declare zero-interest native cash")
    output.mkdir(parents=True, exist_ok=False)
    frozen = {
        "recipe": recipe,
        "plan": plan,
        "input_manifest_sha256": file_hash(Path(recipe["inputs"]["bundle"]) / "manifest.json"),
    }
    (output / "preregistration.json").write_text(canonical(frozen), encoding="utf-8")
    executor = executor or EquityResearchExecutor()
    requests = {"base": plan["base"], **plan["variants"], **plan["benchmarks"]}
    series, attempts = {}, []
    for name, request in requests.items():
        if name == "cash":
            continue
        out = output / name
        out.mkdir()
        try:
            spec = deepcopy(recipe)
            spec["risk"] = deepcopy(request.get("risk", recipe.get("risk", {})))
            candidate = deepcopy(request)
            candidate.pop("risk", None)
            if set(candidate) - (VARIANT_FIELDS | {"candidate_id"}):
                raise ValueError("unknown paired candidate fields")
            candidate["name"] = name
            candidate["candidate_id"] = digest(candidate)[:20]
            # Validate each changed allocation/frequency/risk using the same
            # closed recipe contract before handing it to the strategy adapter.
            checked = deepcopy(spec)
            checked["variants"] = [{k: v for k, v in candidate.items() if k in VARIANT_FIELDS}]
            checked.update(
                {k: candidate[k] for k in ("factors", "strategy", "allocation") if k in candidate}
            )
            validate_recipe(checked)
            result = executor(spec, candidate, out)
            result["artifacts"] = {
                str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            }
            (out / "result.json").write_text(canonical(result), encoding="utf-8")
            returns = pd.read_csv(out / "returns.csv", index_col=0, parse_dates=True)
            if returns.shape[1] != 1:
                raise ValueError("one net-return column required")
            series[name] = returns.iloc[:, 0]
            attempts.append(
                {
                    "name": name,
                    "status": "completed",
                    "result_sha256": file_hash(out / "result.json"),
                }
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Paired research failed: %s", name)
            attempts.append({"name": name, "status": "failed", "error": str(exc)})
        # Persist failures immediately rather than losing them if another replay fails.
        (output / "attempts.json").write_text(canonical(attempts), encoding="utf-8")
    report = {
        "schema": "quant.paired-evidence/v1",
        "plan_sha256": plan["sha256"],
        "preregistration_sha256": file_hash(output / "preregistration.json"),
        "attempts": attempts,
        "available": len(series) == len(requests) - 1,
    }
    if report["available"]:
        values = pd.DataFrame(series).sort_index()  # outer alignment; gaps fail closed
        values["cash"] = 0.0
        try:
            report["attribution"] = attribution(plan, values)
            values.to_csv(output / "net_returns.csv")
            report["returns_sha256"] = file_hash(output / "net_returns.csv")
        except ValueError as exc:
            report.update(available=False, reason=str(exc))
    (output / "paired-evidence.json").write_text(canonical(report), encoding="utf-8")
    return report
