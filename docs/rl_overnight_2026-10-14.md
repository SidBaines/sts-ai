# Overnight player improvement study 14 October 2026

## Current summary

The user authorized continued local training and experiments until an explicit stop in the morning. Work is in `codex/reproducibility-fault-records`; historical campaigns, their budgets, final cohorts and frozen native binaries are preserved. The Mac is on AC power. Each job remains bounded and records source/build/data/checkpoint identities, commands and results. No new final cohort is opened for tuning.

The accepted experimental backend is a separately identified environment 0.1.2 condition: the validated clean-base Ironclad Discovery maintenance patch plus the existing public-search bindings, rebuilt in this worktree. The old backend remains available. This is an intentional trajectory boundary, so all policy comparisons will be rerun together on the new backend before comparing gains.

The incumbent is round-2 OOC update 12 (historical larger-panel 154/4,036 valid wins); update 20 is a candidate (57/1,015 on a reused smaller panel, not confirmed). Combat parameters remain frozen initially. Fresh matched 4,096-world development comparisons of both models are queued; neither is promoted on the new backend yet.

## Protocol and work order

1. Build and test the isolated Discovery backend, replay known faults and verify fresh-process determinism. Record any remaining native failures separately from defeats.
2. Compare updates 12 and 20 on a new fixed 4,096-world TRAIN-split development cohort starting at 5,800,000,000. Use paired full-game wins as the headline and paired progression as secondary evidence. The fresh starting-model reference is part of this condition.
3. Evaluate public search on the selected starting model, using a matched subset before scaling the comparison. Learned-only and search-assisted agents remain distinct.
4. Diagnose strategic habits and combat representation; run bounded OOC strategy/learning pilots with explicit controls. Adopt or extend only when live-game evidence supports it. Do not repeat failed combat recipes without a tested change.
5. Continue selecting experiments from evidence until the user stops the work. Maintain this reverse-chronological log and its top summary, preserving negative results and recovery information.

New training worlds begin at 20,000,000,000, with per-seed/per-iteration offsets recorded by the trainer. New label collection begins at 30,000,000,000. A separate confirmation cohort starting at 5,900,000,000 is reserved for a frozen selected candidate and will not be used for pilot selection. The prior 5e9 and 6e9 cohorts remain historical development data. The old curriculum final cohorts remain closed.

Initial OOC recipe: frozen combat, OOC macro credit, gamma 1, lambda 1, 768 games/update, learning rate 3e-4, three epochs, 4,096-row minibatches, 1,024-row microbatches, zero entropy bonus, KL guard .05, full Acts 1–3, progress reward +.05/floor and terminal +/-1. Each new condition gets a fresh output and declared optimizer initialization. Keep learned-only policy evaluation exact argmax. Evaluate meaningful choices and whole-game outcomes, not only proxy losses.

## 14 October 01:02 London Backend accepted and checkpoint confirmation queued

The full 0.1.2 core plus public-search build passes 498 tests with two declared optional/historical-artifact skips. The bounded smoke has no error, known Discovery prefixes pass their old failure point, and the ordered fresh/reverse patch checks pass. Compiler/build identities are saved in `backend012_identity.json`. No clean release tag is created; this is the new experimental condition. Local default build aliases now point at the owned new build, with legacy aliases retained.

Fresh 4,096-world development comparisons of updates 12 and 20 use identical worlds from 5.8e9, full Act 3, exact argmax and the same corrected backend. Both are queued with three collectors/nine workers each; this preserves a matched comparison while fitting the laptop resource envelope. The 5.9e9 confirmation cohort remains closed.

## 14 October 01:00 London Discovery acceptance expectation repairs

The first full suite ran 498 tests and found two historical expectations that treated world 2005 as an invalid native trajectory. Its archived 0.1.1 fault packet confirms that the rejected action was Discovery. On 0.1.2 the world completes correctly in both cache paths. Tests now require that healthy parity and preserve fail-closed slow/fast prefix coverage using an explicit illegal index. The child error is the existing `IndexError: list index out of range`, identical in both paths. The acceptance rerun and compiler identity are preserved under `data/overnight_20261014/`. No native guard was removed to satisfy these tests.

## 14 October 00:57 London Backend build and ordered patch checks

The new module built successfully with SHA256 `9aeac602912d3f455b17b34e9a549a10b544893936087e9ce2d0bce077a87d41`. The actual captured Discovery sequence passed its former rejection point with the new core and search bindings. Bounded core smoke passed. A direct reverse check of the environment patch while the search overlay remained applied failed at the shared binding context; the proper ordered check applied environment then overlay to a fresh upstream tree, reversed overlay then environment, and passed. The experiment binary was not altered by these checks. Full native suite acceptance is in `data/overnight_20261014/backend012_tests.log`.

## 14 October 00:52 London Preparation and authority

The previous round is complete and the new overnight authorization supersedes the earlier review-only state. The user explicitly asked to continue until stopped in the morning; no automatic overall stop time or extension of an old allocation has been inferred. The queue runner is live and idle before preparation. AC power and approximately 22 GiB free disk were checked.

The 0.1.2 Discovery change and native mechanics regressions were imported from the separate clean-base maintenance branch. Its public schema remains unchanged, while package/native/interface identities change. A new local native checkout was cloned from existing pinned upstream history, then received the tracked environment and public-search patches. Read-only pinned dependency sources are reused; CMake 3.31.6 was copied into an owned build-only runtime, without modifying the frozen training runtime.

Queued `night012_build` builds into this worktree's owned `external/sts_lightspeed_rl/build`. Existing live/historical build directories are not overwritten. The build job's bootstrap identity points at the already verified maintenance candidate; the produced module will have its own manifest and acceptance record before any simulation/training job uses it.
