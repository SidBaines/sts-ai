# Unreleased canonical RNG-resampled search condition

Candidate environment/package 0.1.4 changes seeded teacher queries on copies,
with observation label `sts_env_public_v2` retained. Ordinary game execution and
unsampled privileged search are unchanged by design; verify matched rollouts
before reusing a baseline. No release tag is created or moved.

Before a seeded draw-order intervention, sort draw/discard/exhaust by static
card identity, public upgrade count, base/current cost, free-once and retain.
The query reports `hidden_pile_order_canonicalized`. Returned discard/exhaust
root choices are remapped to the corresponding live card instance before their
bits, descriptions and legality are exposed. Instance identity is private routing
metadata; the policy receives only the current public display-menu index.
Diagnostic best-sequence bits still refer to sampled teacher states, not a plan
to execute blindly in the live battle. Existing draw-index query abstentions remain.

The research adapter reseeds battle RNGs on copies and derives search seeds
from the public observation hash, replacing the private world-seed calculation.
It requires the canonicalization provenance flag. Its protocol is distinct from
the historical 0.1.3 sampled-native search. State/history-dependent native fields,
including latent card-instance payloads and equal-public-key tie behaviour, have
not been certified as a function of public observations alone. Neither ordinary
privileged search nor this resampled variant is a certified public-only planner.
Do not silently convert simulator-condition evidence into vanilla competence.

The test-only extension creates native copies with reversed unordered piles and
constructed discard/exhaust selection states. It is separate from the production
binary. Normal-state tests compare identical full public views and exact query
statistics; selection tests compare equal card multisets/semantic menus while
preserving each visible routing order, then execute returned actions on copies.
The tests do not assert byte-identical selection observations when menus reorder.
Full-suite, bounded smoke, fresh-process and paired ordinary-rollout gates remain
required. Older builds, source snapshots and failed test receipts are retained.

Offline builds reuse independently cloned local dependencies with verified pins.
Core build is `external/sts_lightspeed/build_v2`; research build is
`external/sts_lightspeed_rl_v2/build014_canonical`. Always pass the chosen build
explicitly. Queue entries in the original worktree use a preparation bootstrap
because its old Python core cannot validate a new binary; the isolated helper
writes its own strict run record and performs actual native verification.
