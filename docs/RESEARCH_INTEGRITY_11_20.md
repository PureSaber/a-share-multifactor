# Continuous OOS and economic diagnosis

## 15: uninterrupted simulated account

`EquityResearchExecutor.continuous(recipe, selections, output)` accepts a frozen list of
`{train: {start,end}, test: {start,end}, candidate: ...}`. Training must precede test;
test sessions must cover the whole requested interval exactly once, in order. Pipeline
`validation.account_policy: continuous` constructs the list without using test outcomes.

All causal decision plans are compiled, then one quant-execution replay processes the entire
OOS event sequence. Cash, holdings, unsettled amounts, fees, pending orders and risk latches
are not reset between folds. An unchanged parameter sequence matches one-pass returns,
orders and ledger. New parameters act at future scheduled rebalances, not by retrospectively
rewriting positions. Fee policy and allocation mode cannot change mid-account. This is batch
continuous replay, not live execution or crash-resumable checkpointing.

## 16: controlled reasons for underperformance

```python
from copy import deepcopy
from pathlib import Path
from quant_lab.research import candidates, load_recipe
from quant_lab.counterfactuals import intervention_plan
from a_share_multifactor.paired_research import run_paired

recipe = load_recipe(Path("recipe.yaml"))  # one fixed equity interval, no validation wrapper
base = candidates(recipe)[0]
base["risk"] = recipe.get("risk", {})
base["risk_model"] = deepcopy(recipe.get("risk_model"))
passive = {**deepcopy(base), "risk": {}, "risk_model": None,
    "allocation": {"mode": "equal", "max_turnover": 2}, "strategy": {
    **base["strategy"], "family": "buy_hold", "cash_buffer": 0, "max_weight": 1}}
constrained = {**deepcopy(base), "strategy": {**base["strategy"], "family": "buy_hold"}}
plan = intervention_plan(base, {"fees": 0, "delay": 1}, benchmarks={
    "passive": passive, "same_risk_constrained": constrained,
    "cash": {"mode": "cash", "daily_return": 0},
})
evidence = run_paired(recipe, plan, Path("paired-new-run"))
```

Other dimensions: `signal` replaces the factor map; `allocation` replaces allocation settings;
`risk_latch` replaces the risk subtree; `frequency` replaces strategy frequency. Each changes
exactly one declared dimension. Give explicit distinct executable benchmark definitions:
passive is an unhedged buy-and-hold portfolio under the supplied market/execution model;
same-risk preserves the original position/cash/risk constraints; cash has zero interest in
native currency. All use the same interval and data. Costs remain realistic except in the
explicit fee counterfactual. A passive portfolio is not guaranteed to be a licensed index.
An omitted `risk_model` inherits the recipe; an explicit null removes it. Empty
model mappings are invalid. Equal allocation retains absolute factor and PIT
industry bounds through joint projection; the original tracking-error gate still
runs. A variant cannot change its risk model while changing another dimension.
The cost multiplier scales commission, minimum commission, tax and fill-price
slippage together; it does not isolate commission alone.

`preregistration.json` is written before execution. Each resolved recipe/candidate
pair is saved in `execution-definition.json` and hashed in its successful attempt.
Every attempt and failed replay is kept;
any failure/gap disables attribution. `paired-evidence.json` reconciles the benchmark return
gap to six optional one-at-a-time effects plus an unexplained/interaction residual. It cannot
establish causality or justify weakening real risk limits. `net_returns.csv` and per-run byte
manifests let reviewers reproduce the calculation.

## 20: objective evidence

The executor supplies observed net return, Sharpe, drawdown magnitude, initial capital and
fees/initial-capital. It intentionally leaves capacity, excess to an external benchmark,
turnover convention, data-condition approval and stop conditions unfilled unless separately
measured. The registry's acceptance decision remains insufficient when these are missing.
Read the quant-lab objective contract before supplying external evidence.

Retain the vendored AKShare `1.18.88.post1` wheel and mini-racer `0.14.1`; these features do not
replace the audited data runtime or enable live trading.
