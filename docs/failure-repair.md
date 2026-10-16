# Research execution failure repair

The adapter retains complete continuous/account inputs on failures, uses typed
input errors and passes one remaining deadline through solve and integer repair.
Legacy allocation dictionaries are not silently upgraded to protocol v2.

`_replay(..., share_rules=...)` resolves rights from the existing ledger's actual
record-date snapshots. It does not maintain a second account. UTC normalization,
credit timing, confirmed account identity and sellable dates are validated.
Native snapshots use the resolved event's independent reference price. Original
unverified events stop exposed accounts; unheld assets do not invent entitlements.

Strict and sensitivity runs must have different identities and separate output
directories. Later announcements may corroborate the audit but cannot supply
earlier strategy information. Unidentified fractional rights remain unresolved
in strict runs; rounding assumptions are explicit in sensitivity branches.

Restore a checkpoint before the action to apply corrected evidence. Keep the
previous event stream, ledger and outputs. Retrying the same action does not
credit shares twice. Freeze accounts and default models until their separate
promotion gates pass.

## Ordinary-install fixture catalog

The default versioned instrument catalog is included inside the wheel. Its bytes
must match the retained research fixture at `data/fixtures/a_share_instrument_catalog_v1.csv`.
Explicit caller-supplied catalogs keep their existing behavior. This only fixes
resource lookup for installed packages; it does not change instrument facts or
the historical study inputs.
