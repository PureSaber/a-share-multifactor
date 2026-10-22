# Evidenced ETF units and research results

The equity adapter consumes normalized issuer split terms from quant-data-kit.
It resolves the actual account's record-date quantity, applies deterministic
per-holder rounding at end-of-record-day credit, and preserves the ex/sellable
date. Pending orders and deferred share targets for the pre-split unit are cleared.
The record-date credit is included even if the replay ends before the next ex-date.
Unsupported consolidations and uncertain share distributions remain blocked.

After an optimizer failure, the adapter cancels outstanding orders and evaluates
the current portfolio using the serialized original problem and account-level
risk checks. Noncompliant or unknown holdings latch a halt; later liquidation
settings cannot convert that stop into an automatic sale. Native diagnostics keep
solver-reported infeasibility separate from search failure and from a mathematical
proof. Publishing summaries include all unsuccessful decisions and consecutive
gaps; a published all-cash allocation is still a publication.

Research results expose evidence status and paired returns only when the measured
benchmark frame has the exact declared `benchmark_id`. A benchmark kind or a
recipe label alone cannot establish that identity. Initial-close account setup is
excluded from the return sample. Unknown failure evidence prevents future-research
acceptance. Capacity, turnover, data conditions and stop conditions still require
actual measurements; they are not filled in from performance or a successful replay.

These are research-code changes, not an installation into a frozen runtime or a
reset/repair of any existing account. Tests and short historical split replays do
not count toward native engineering, future research or promotion gates.
