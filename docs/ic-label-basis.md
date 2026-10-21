# IC labels and execution timing

New `AppConfig` research runs default to `ic_return_basis: next_open`. A signal
observed at session t close is scored against open(t+1+h) / open(t+1) - 1 for
horizon h. The same setting is used by factor preparation, IC decay and decision
workflow diagnostics. The IC smoke command also defaults to next-open labels.

These are **gross price diagnostics**, not fills or net account returns. Actual
execution still applies its own halt, price-limit, lot and cost rules. The common
session grid comes from the supplied panel: a missing symbol's scheduled entry or
exit bar stays missing; it is not replaced by that symbol's next observed bar.
Supply the complete trading-session panel when comparing horizons.

Every generated label records its basis, endpoint and availability. Next-open
labels mature at the exit session at the earliest; late entry/exit data delays
maturity. Unknown availability stays unknown. Training still requires availability
strictly before the training cutoff. There is no close-price fallback when open
prices are absent.

IC summaries expose the minimum and median cross-section with observed labels,
plus dates without labels. These counts describe the actual sample; no arbitrary
universe-size cutoff is treated as statistical certification.

`ic_return_basis: same_close` reproduces the earlier close-to-close diagnostic
convention. The low-level `add_forward_return` and `analyze_ic_decay` functions
retain that default for existing callers, and accept `basis="next_open"` explicitly.
Do not compare old and new IC values as if the label were unchanged. Register a
new study/code identity for a rerun and preserve existing research artifacts.
