# Environment release 0.1.0 verification

Release reference: `env-v0.1.0` on branch `environment-base`.
Status: verified locally on 11 October 2026. The immutable release tag identifies this tested source.

## Release contents

- Dependency-free `SlayTheSpireEnv` reset/step API; full/OOC/combat control modes.
- Optional Gymnasium 1.4.0 adapter, masked menu actions, finite action-attempt
  horizon, and a documented compact numeric encoder.
- Agent CLI (`run`, `evaluate`, `play`, `inspect`, `replay`, `doctor`), custom
  module factories, per-episode resets, and immutable output creation.
- Pinned sts_lightspeed/pybind11 plus the versioned native patch, build manifest,
  verified binary identity, subprocess timeouts, and distinct failure reporting.
- Public OOC event effects/offers and combat state; neutral text observations.
- Versioned new traces with replay verification. Historical experiments and
  frozen traces remain on the original research branch.

## Verification record

All required local gates passed on Apple Silicon with Python 3.12.13.
The complete suite ran 191 tests with no failures or skips, including required
native integration and the Gymnasium checker. CI is configured for independent
Linux/macOS builds; those hosted jobs have not been run because this release
was created locally.

| Gate | Result |
| --- | --- |
| Unit suite | Passed as part of the complete 191-test suite |
| Required native integration suite | Passed, with STS_REQUIRE_SIMULATOR=1 and zero skipped tests |
| Gymnasium environment checker | Passed; two-world AsyncVectorEnv composition and worker cleanup also passed |
| Bounded random Act 1 OOC rollout | Passed: seed 3, 50 battle simulations, 21 decisions, ordinary defeat with no simulator error |
| Full and flipped-hybrid combat rollouts | Passed: random agents in all three modes; native full/reverse both reached the Act 1 boss and ended normally |
| Custom agent import/configuration/reset | Passed: an external-directory factory, JSON config, eight decisions, and exact replay; stateful reset and model reuse regressions passed |
| Fresh-process deterministic replay | Passed: five profiles repeated identically excluding elapsed time; nine traces replayed |
| Native timeout/failure containment | Passed: worker startup/step timeout, crash, cleanup and reset recovery; forced native startup deadline returned explicit SimulatorTimeout/error |
| Wheel and CLI entry point installation | Passed: core-only wheel installed in a fresh virtualenv; console doctor and custom-agent rollout run from an external working directory |
| Source archive and extracted archive wheel build | Passed: all native build helpers, patch, exact release native manifest, examples, and tests included; wheel built from extracted archive |
| `git diff --check` | Passed |

## Local result details

At world seed 3, native full control executed 239 decisions. Flipped hybrid
executed 172 external combat decisions and 67 automatic OOC decisions. Both
reached floor 16 and ended in ordinary defeat. These are release mechanics
checks, not evidence of policy competence. Evaluation accounted for all three
requested worlds (3, 4, 5) with no simulator errors.

The privacy audit checked 663 public observations: every legal action contains
only `index` and `description`. All 654 completed action records retain native
bits only in privileged diagnostics. The optional Gym observation has only
`features` and `action_mask`.

Pinned/build identities used for the checks:

- Upstream: `7476a81954020087da31d41d16fddf475746ec2d`
- JSON submodule: `0b345b20c888f7dc8888485768e4bf9a6be29de0`; pybind11: `v2.13.6`
- Native patch SHA-256: `e60eb5aaefcdb6f83f6b71b83ae00682598ca88255ae4dc732b3547a76bb2d66`
- Local binary SHA-256: `8dba3d574ea3d4cd4c91b7031ffb38821b3e18c03358263407df3c05329c1724`
- Python interface SHA-256: `49c112e2e41ffe3b8267e92e50db0dc6d83c68881633d3ac11c1c8570d6630b6`
- Python: `3.12.13`; CMake: `cmake version 3.31.6`
- Compiler: `Apple clang version 21.0.0 (clang-2100.1.1.101)`
- Tested wheel SHA-256: `8b8c7f727fc43e0e3209b9007b168946c36b98a4a025ddcaa2a95c8c2bf49098`

The wheel was force-installed without dependencies in a fresh virtualenv;
Gymnasium and NumPy were absent there. Its custom-agent rollout and replay both
passed. The source archive includes build helpers, patch, matching native
manifest, examples, tests and docs, and its runtime payload matches the wheel.
Fresh upstream patch apply and reverse-apply checks passed.

The original `astra-6sep` checkout stayed on its original commit and all 60
modified/untracked source files matched their pre-extraction SHA-256 hashes.

## Provenance and qualifications

The pinned upstream commit, pybind11 tag, patch digest, Python ABI, and compiled
module digest are recorded by the native build manifest. `doctor` verifies that
identity without launching gameplay. Replay rejects an incompatible interface or
native build rather than silently accepting it. Python interface file contents
and the exact runtime Python version are also recorded and compared, so source
edits cannot silently reuse the same version label. Agent/model code identity is
an adapter responsibility.

The initial release supports Ironclad, with Acts 1–3 as the standard scope.
Optional Act 4 remains experimental because key and burning-elite observations
are incomplete. Known upstream undefined-memory/phantom-power bugs are not
certified as eliminated in every state. Timeout, native-error, and reported-UB
containment do not detect every unflagged corrupt state. Native OOC choices are scripted/seeded;
only combat uses native search. The default reward is sparse victory (1) and
zero otherwise. Native faults are errors rather than valid losses. The optional
Gymnasium encoder omits card/enemy/map/event details; adapters can encode the
full public observation in `info["observation"]`. Gymnasium observations have
only `features` and `action_mask`. Raw native action bits are excluded from public
menus because draw-selection encodings can leak hidden positions. Executed-action
bits remain privileged diagnostics; policies must not receive the remaining
`info` metadata. Some upstream event mechanics differ from
vanilla Slay the Spire and uncovered descriptions are explicitly marked.

The weighted-reward sampler, chest-policy, individual card-reward skip, and
Designer fallthrough fixes deliberately change native-policy/menu trajectories
relative to the research branch. Full visible OOC map routes and event effects
are intentional new-observation improvements. World, external policy, and native
OOC seeds are separate; effective native OOC seed width and build-specific
search aliases are documented in the environment contract. Cross-build or
cross-platform byte-identical trajectories are not promised.
