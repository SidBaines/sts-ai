# Environment base: contract and maintenance

`main` holds the reusable Slay the Spire environment, extracted from the
original research harness. Its scope is game state, legal actions, stepping,
baseline policies, build provenance, replay, and an agent CLI. Model clients,
datasets, training recipes, framing prompts, and historical experimental
artifacts remain on workstream branches; see [WORKSTREAMS.md](../WORKSTREAMS.md).

## Interface commitments

- Observations contain public game information and the current legal menu
  (`index` and `description`). Raw native action bits are excluded: draw-selection
  encodings can reveal hidden draw positions. Privileged native search operates
  behind the environment boundary.
- `info` is diagnostic metadata and may contain privileged executed-action bits
  and simulator identity. Policies must not receive it as input. The optional
  Gymnasium `info["observation"]` field is the safe public observation and can be
  used by richer encoders.
- Action indices identify positions in the current legal menu, not globally
  stable action types. Agents must use the menu from their latest observation.
- World and policy seeds are separate. Search settings and native build identity
  are part of an episode's reproducibility contract.
- The dependency-free core rejects invalid actions. The optional Gymnasium
  adapter uses explicit zero-reward no-ops for padded masked slots inside its
  fixed action space; all attempts count toward its decision horizon. Changing
  `invalid_action_reward` is an explicit adapter setting. Episode termination and truncation are distinct.
  Native failures raise exceptions and are recorded explicitly rather than scored as ordinary losses.
- JSONL traces and replay are versioned artifacts. Future incompatible observation,
  action, reward, or trace changes require a new interface/release version.
- The core stays dependency-free. Gymnasium is optional and pinned; user model
  clients and RL packages belong in their own adapters.

The default sparse reward is one on victory and zero otherwise. Changing it
through a supplied reward function is an adapter choice, not an environment
default. Event option descriptions are derived from the pinned simulator's actual
handlers. Common Act 1 events are described, and dynamic visible HP amounts,
selected cards, and trade offers are public fields. Uncovered options explicitly
say their effects are unavailable. This release does not claim full event or
vanilla-game fidelity: for example, upstream Mushrooms and Mysterious Sphere
alternative choices award gold, Mausoleum does not award a relic, and Woman in
Blue does not charge for potions. These are simulator behaviours, recorded
explicitly instead of silently describing different mechanics.

The optional Gymnasium observation has two fields: `features` (15 compact
numeric features) and `action_mask` (a fixed menu mask). It omits strategic
card/enemy/map/event detail and raw native action bits. The full public
observation in `info["observation"]` is available to richer experiment encoders;
the remaining `info` diagnostics must not be used as policy input. The compact
baseline is not a complete Markov representation.

For parallel RL environments, configure outer process workers as non-daemonic.
The core creates an isolated native simulator child, and Python daemonic
processes cannot create children. Gymnasium `AsyncVectorEnv` defaults to
`daemon=True`; use `daemon=False`, `context="spawn"`, and
`shared_memory=False` with the configuration below. Native call timeouts remain
active inside each vector worker. Other process-based RL wrappers must also
support non-daemonic workers.

```python
from gymnasium.vector import AsyncVectorEnv
from sts_env.gymnasium import GymnasiumEnv

def make_env():
    return GymnasiumEnv(mode="combat", max_act=1, battle_simulations=50)

if __name__ == "__main__":
    envs = AsyncVectorEnv([make_env, make_env], daemon=False,
                         context="spawn", shared_memory=False)
    try:
        observations, infos = envs.reset(seed=[3, 4])
        actions = [int(mask.argmax()) for mask in observations["action_mask"]]
        observations, rewards, terminated, truncated, infos = envs.step(actions)
    finally:
        envs.close()
```

A `daemonic processes are not allowed to have children` error indicates that
the outer RL wrapper needs this configuration.

The native OOC policy seed is reduced to 32 bits and that effective seed is
recorded. Values differing only above bit 31 therefore alias. On libc++, native
search seeds zero and one also share an RNG stream; do not count them as
independent search votes. These are separate from the external agent RNG. The native action-query
baseline uses the world seed for combat search and the configured OOC seed for
its scripted choices; `--policy-seed` configures external agent resets.

The clean base intentionally fixes a missing break in upstream's weighted native
reward-action sampler and missing breaks in Designer event options that formerly
fell through into additional charges and HP loss. Native policy trajectories can therefore differ from the
old research harness even under the same seeds; the original branch and frozen
research artifacts retain their prior behaviour. New releases record the native
patch identity and reject incompatible build/replay provenance.

The initial character is Ironclad and Acts 1–3 are the standard scope. Optional
Act 4 is experimental: key ownership and burning-elite flags are not yet fully
represented in public observations. Native timeout/error isolation and reported
undefined-behaviour flags do not certify all reachable game states as free of
upstream undefined-memory or phantom-power bugs. Unflagged corrupted states may
still evade detection.

The clean interface additionally exposes the complete visible OOC map routes
and individual card-reward skip choices. These observation/menu improvements,
chest-policy recovery, and the native-policy fixes above deliberately change
behaviour relative to the old harness. The Python interface file hash and exact
runtime version protect replay from unversioned dirty serializer changes;
adapters remain responsible for recording their own code/model identity. Full, out-of-combat-only, and combat-only
control share the same stepping interface. In combat-only mode, the built-in
out-of-combat policy is scripted/seeded rather than a planning search.

## Migration from the research harness

This clean base uses distribution `slay-the-spire-env`, package `sts_env`, public
observation contract `sts_env_public_v2` (`sts_env_public_v1` before 0.1.3), and
trace schema `sts_env_trace_v1`.
It does not reinterpret the old `sts_ai` DecisionRecord schema. Frozen historical
traces and their original replay/training tools stay on the LLM workstream (`ws/llm`).
Port experiments by writing agent/encoder/reward adapters around the new API;
record this new interface identity when collecting new data. The internal
legacy combat renderers are implementation details, not a compatibility promise
for old training inputs.

## Maintenance workflow

Start new experiments from an immutable release tag. Land fixes on `main`
through pull requests, test them there, and publish a new release tag instead of
moving old tags. Experiments deliberately merge or cherry-pick later fixes and
record the version used. Do not use an existing tag to represent new code.

The native simulator is a pinned upstream checkout plus
`patches/sts_lightspeed_python_api.patch`. Changes to native code must update that
patch, apply cleanly to the pinned upstream source, and be rebuilt. Build outputs
and upstream source are ignored; never depend on an untracked edit for a fix.

## Release verification

Run these checks from a fresh source checkout before tagging. The build must be
present; `STS_REQUIRE_SIMULATOR=1` prevents missing binaries from being silently
accepted as skipped integration tests.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[gym]'
scripts/build_lightspeed.sh
export STS_SEARCH_TEST_FIXTURE="$(scripts/build_test_fixtures.sh)"
STS_REQUIRE_SIMULATOR=1 STS_REQUIRE_SEARCH_TEST_FIXTURE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -t .
.venv/bin/sts-env doctor
.venv/bin/sts-env run --agent random --seed 3 --mode ooc \
  --max-act 1 --battle-simulations 50 --output data/release-smoke.jsonl
.venv/bin/sts-env replay data/release-smoke.jsonl
```

Also run bounded full/combat mode episodes, custom-agent import/configuration,
fresh-process deterministic replay, timeout/failure handling, and wheel/entry
point installation. Ensure packaging does not include models, experiment code,
outputs, or a platform-specific native binary. Check the version is consistent
in package metadata and trace provenance. Inspect `git diff --check` and record
native compiler/Python identity alongside the verification results.

A successful local release test does not establish all-platform deterministic
behaviour. CI builds independently on Linux and macOS to catch build and API
regressions. Keep scientific evaluations separate from release smoke tests.

## First release status

Version 0.1.0 passed all local release gates on 11 October 2026.
[The release verification record](release_v0.1.0.md) records the 191-test suite,
CLI/replay, privacy, timeout, vectorization and packaging checks. The immutable `env-v0.1.0` tag identifies
the tested source; no historical training results are claimed for this release.

## Release 0.1.2: Discovery

Release 0.1.2 adds validated Ironclad Discovery support; see the
[repair and verification record](discovery_fix.md). The old native fault flag
unconditionally marked Discovery as unsupported. It does not diagnose memory
UB. Unsupported class-card generation and invalid Discovery offers still fail
explicitly, and all environment fault/timeout guards remain active. Discovery
trajectories intentionally continue beyond their former rejection points. The
public schemas are unchanged; the release/native/interface identities change.
Historical experiments and running studies keep their frozen backends.

## Release 0.1.3: shop maintenance

Release 0.1.3 repairs upstream shop pricing and purchase validation: Courier and
Membership discounts, restock and replacement prices, and potion purchases that
could not take effect. The public observation label advances to
`sts_env_public_v2` because shop menus no longer offer ineffective potion
purchases, so menu indices can differ from 0.1.2. See the
[migration record](shop_maintenance_013.md) and
[release record](release_v0.1.3.md).

## Release 0.1.4: canonical sampled search

Release 0.1.4 changes only seeded native search queries
(`search_best_action(..., draw_order_seed=...)`). Before sampling a draw order,
the query sorts the unordered draw, discard and exhaust piles by public card
attributes, so a fixed number of sampled worlds no longer starts from the true
hidden pile order. Returned discard/exhaust selections are mapped back to the
live card instance. Ordinary execution and unsampled search are unchanged, and
`sts_env_public_v2` is retained. Neither search mode is a certified public-only
planner; see the [search record](canonical_search_014.md) and
[release record](release_v0.1.4.md). `scripts/build_test_fixtures.sh` builds
the test-only native fixture its regression tests need.
