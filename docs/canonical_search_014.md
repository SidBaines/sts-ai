# Canonical sampled native search

Release 0.1.4 changes seeded native search queries on copies; the observation
label `sts_env_public_v2` is retained. Ordinary game execution and unsampled
privileged search are unchanged by design (checked by comparing ordinary
rollouts between 0.1.3 and 0.1.4; see the [release record](release_v0.1.4.md)).

Before a seeded draw-order intervention, sort draw/discard/exhaust by static
card identity, public upgrade count, base/current cost, free-once and retain.
The query reports `hidden_pile_order_canonicalized`. Returned discard/exhaust
root choices are remapped to the corresponding live card instance before their
bits, descriptions and legality are exposed. Instance identity is private routing
metadata; a policy should only receive the current public display-menu index.
Diagnostic best-sequence bits still refer to sampled teacher states, not a plan
to execute blindly in the live battle. Existing draw-index query abstentions remain.

Without this, the 0.1.3 sampler shuffled from the live hidden pile order, so a
fixed finite set of sampled worlds could depend on the true draw order. Adapters
that use sampled search should also derive their search seeds from public
information (for example, a hash of the public observation) rather than the
private world seed. State/history-dependent native fields, including latent
card-instance payloads and equal-public-key tie behaviour, have not been
certified as a function of public observations alone. Neither ordinary
privileged search nor this resampled variant is a certified public-only planner.
Do not present results that use it as unassisted competence.

The test-only extension `tests/native/hidden_pile_fixture.cpp`, built by
`scripts/build_test_fixtures.sh`, creates native copies with reversed unordered
piles and constructed discard/exhaust selection states. It is separate from the
production binary. Normal-state tests compare identical full public views and
exact query statistics; selection tests compare equal card multisets/semantic
menus while preserving each visible routing order, then execute returned actions
on copies. The tests do not assert byte-identical selection observations when
menus reorder.

`scripts/build_lightspeed.sh` can reuse already populated pinned dependencies
with `STS_OFFLINE_DEPS=1`; it verifies their identities and refuses mismatches.
