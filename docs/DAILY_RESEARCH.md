# Reproducible daily research

`configs/decision_daily.yaml` fixes a four-stock research watchlist, cost model,
factor specification and future holdout. `quant-pipeline` runs the producer,
indexes successful and blocked decisions, and regenerates the report-hub page.
Use the integrated install profile in quant-workspace, not mixed historical tags.

```sh
python -m a_share_multifactor.decision_workflow --config configs/decision_daily.yaml --output ../daily-runs
python -m a_share_multifactor.research_case --config configs/decision_daily.yaml --inputs INPUT_DIRECTORY --output NEW_CASE_DIRECTORY --sessions 40
```

Each normal decision attempt is registered before provider access and remains in
the experiment database on failure. Studies freeze configuration and code identity.
The default future holdout is 2026-09-21 through 2026-12-31: registration must occur
before it starts. Changing strategy/dependencies requires a new study and paper
account. Diagnostic labels are limited to those maturing strictly before holdout.
The review command refuses early, incomplete, changed-code and repeated evaluations:

```sh
python -m a_share_multifactor.holdout_review --db ../daily-runs/experiments.db --study ashare-momentum-volatility-2026q4 --run FINAL_RUN_DIRECTORY
```

Corporate actions retain the source record and announcement/record/ex/payment/share
delivery dates. Only verified same-day delivery can enter this ledger; deferred
receivables require a separate implementation. Cash distributions are gross: personal
holding-period dividend tax is excluded. Raw bars determine executions; adjusted
prices determine factors. The entire account window must match a single multiplicative
or affine adjustment model. Missing or inconsistent actions block the run.

The exploratory case compares net strategy returns with an exposure-matched,
equal-weight buy-and-hold portfolio using the same next-bar execution and cost engine.
The HS300 comparison is a price index before costs and is labelled accordingly.
Outputs include source/code identities, both fill and ledger histories, walk-forward
factor statistics and FDR results. This is a retrospective current-watchlist example,
not an untouched holdout and not evidence of investment-ready alpha.

Observed on 2026-09-19: CNInfo supplied 109 records for the four stocks. For 000333,
the 2026-06-29 cash distribution is 3.80/share while the captured Tencent qfq series
changes by 3.72. Windows crossing that event remain blocked. A shorter window starting
after that event can demonstrate supported ledger behavior; it must not replace the
failed full-window result or be presented as performance-selected validation.
The current ST source was unavailable. Advisory runs explicitly show unverified
trading status; strict mode blocks. These public feeds do not supply a complete
historical universe, delisting history or authoritative point-in-time status archive.
