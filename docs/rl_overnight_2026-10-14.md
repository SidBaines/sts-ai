# Overnight player improvement study 14 October 2026

## Current summary

The user authorized continued local training and experiments until an explicit stop in the morning. The active backend is now the independently repaired 0.1.3 shop condition; learning must use its rebaselined scores. Work is in `codex/reproducibility-fault-records`; historical campaigns, their budgets, final cohorts and frozen native binaries are preserved. The Mac is on AC power. Each job remains bounded and records source/build/data/checkpoint identities, commands and results. No new final cohort is opened for tuning.

The first accepted experimental backend was a separately identified environment 0.1.2 condition: the validated clean-base Ironclad Discovery maintenance patch plus the existing public-search bindings, rebuilt in this worktree. The old backend remains available. This is an intentional trajectory boundary, so all policy comparisons will be rerun together on the new backend before comparing gains.

The incumbent is round-2 OOC update 12 (historical larger-panel 154/4,036 valid wins); update 20 is a candidate (57/1,015 on a reused smaller panel, not confirmed). Combat parameters remain frozen initially. The fresh 0.1.2 comparison selected update 20 provisionally (194/4,096 versus 139/4,096), but a confirmed shop defect requires a new 0.1.3 baseline before training.

## Protocol and work order

1. Build and test the isolated Discovery backend, replay known faults and verify fresh-process determinism. Record any remaining native failures separately from defeats.
2. Compare updates 12 and 20 on a new fixed 4,096-world TRAIN-split development cohort starting at 5,800,000,000. Use paired full-game wins as the headline and paired progression as secondary evidence. The fresh starting-model reference is part of this condition.
3. Evaluate public search on the selected starting model, using a matched subset before scaling the comparison. Learned-only and search-assisted agents remain distinct.
4. Diagnose strategic habits and combat representation; run bounded OOC strategy/learning pilots with explicit controls. Adopt or extend only when live-game evidence supports it. Do not repeat failed combat recipes without a tested change.
5. Continue selecting experiments from evidence until the user stops the work. Maintain this reverse-chronological log and its top summary, preserving negative results and recovery information.

New training worlds begin at 20,000,000,000, with per-seed/per-iteration offsets recorded by the trainer. New label collection begins at 30,000,000,000. A separate confirmation cohort starting at 5,900,000,000 is reserved for a frozen selected candidate and will not be used for pilot selection. The prior 5e9 and 6e9 cohorts remain historical development data. The old curriculum final cohorts remain closed.

Initial OOC recipe: frozen combat, OOC macro credit, gamma 1, lambda 1, 768 games/update, learning rate 3e-4, three epochs, 4,096-row minibatches, 1,024-row microbatches, zero entropy bonus, KL guard .05, full Acts 1–3, progress reward +.05/floor and terminal +/-1. Each new condition gets a fresh output and declared optimizer initialization. Keep learned-only policy evaluation exact argmax. Evaluate meaningful choices and whole-game outcomes, not only proxy losses.

## 14 October 01:48 London Corrected shop backend accepted and learning controls declared

The full shop/exploration acceptance suite passes 505 tests (two declared skips), and bounded core smoke passes. The regenerated native patch initially introduced standalone blank context markers that fail repository whitespace checks. They were normalized without changing C++ source; the new isolated `build013_clean` has exactly the same native binary SHA256 as the first 0.1.3 build (`6fd3533107e9ad61...`). Eleven focused mechanics/sampling checks pass again, and fresh/reverse patch checks pass. Both raw and normalized patch identities are preserved in the format receipt.

New full 4,096-world matched baselines for updates 12 and 20 are queued on 0.1.3 / public v2. The planned first learning pilot starts from update 20, uses a fresh AdamW optimizer, exact frozen-combat argmax, lambda 1 macro OOC credit and new TRAIN worlds. It is bounded to eight iterations; a second matched seed-7 pilot will use card-only temperature 4 if the baseline/control are healthy. Neither is a resume of missing historical update-20 optimizer state. The pilot evaluation uses the 5.8e9 DEV cohort; the 5.9e9 confirmation cohort remains closed.

## 14 October 01:42 London Fresh checkpoint result and shop defect boundary

On the same 4,096 fresh DEV worlds under 0.1.2, update 20 wins 194 (4.74%) versus update 12 winning 139 (3.39%), with zero faults in either. Paired mean floor improves .525 with 95% bootstrap interval [.25,.80]; the paired win comparison has 85 losses/140 gains among discordant worlds. Update 20 is the provisional starting checkpoint.

The 64-world audit completed without faults. It found 54 reached states containing same-named cards with different costs. Constructed contrasts on reached observations show enemy max HP absent from both encodings in all 1,564 probes, and enemy block absent from state encoding (it affects action encoding in 887 probes). These prove representation omissions, not that both counterfactual states are reachable under identical histories. Card-reward top-two logit gaps have median 23.77, motivating a card-only exploration pilot.

A reached Courier shop buys repeatedly at 1 gold. Source review proves `Shop::getNewPrice` overwrites its base-price argument with a rounded random factor, ignores discount assignments, and `buyRelic` ignores the returned restock price. Card restocks apply Courier twice instead of Membership Card. Full/Sozu inventories still expose potion purchases that spend gold without giving a potion. These flaws qualify the 0.1.2 scores as simulator-condition results, not vanilla competence. No learning job has been launched on that flawed shop condition.

An isolated 0.1.3 build corrects restock arithmetic, assigns relic prices, applies purchased Courier discounts, uses the actual purchased egg relic for offer upgrades, and rejects futile potion buys without charging gold. The public observation label advances to `sts_env_public_v2` because legal shop menus change. Five native shop mechanics regressions and the initial 503-test suite pass. Native patch changes are tracked; the 0.1.2 binary and source snapshots remain preserved. All later comparisons will rebaseline on 0.1.3.

Exact frozen-combat argmax is now an explicit new training mode, leaving historical scaled-greedy behaviour readable. A card-only temperature knob will test exploration without changing rest/map/shop sampling or deployed argmax. Both changes have bounded regressions; the new pilot recipes and optimizer resets will be recorded separately.

## 14 October 01:16 London Bounded diagnosis and search preparation

The kernel memory-pressure query reports ample reclaimable headroom and swap use remains unchanged at 20.9 GiB; no unrelated process is altered. A 64-world, two-worker diagnostic is running on new TRAIN worlds from 40e9, saving strategic choice statistics and explicitly labelled synthetic feature contrasts on reached observations. It does not train a model.

The public-search helper now writes per-world progress and uses a finite total budget plus a 60-second query deadline. This avoids a healthy multi-hour search evaluation being mistaken for a stalled job. A small new-backend smoke will precede the 512-world incumbent search pilot; its role is diagnostic, separate from checkpoint selection.

## 14 October 01:10 London Comparison resource check

Both fresh comparison parent processes are live (PIDs 68412 and 68414), with six active collectors and 18 native workers. Collector RSS is approximately 1.4 GiB each; total comparison RSS is roughly 12 GiB. Existing system swap use is about 20.9 GiB, so no additional large collection job is launched until current memory pressure is checked. The models are being evaluated, not trained; no checkpoint selection has occurred.

## 14 October 01:02 London Backend accepted and checkpoint confirmation queued

The full 0.1.2 core plus public-search build passes 498 tests with two declared optional/historical-artifact skips. The bounded smoke has no error, known Discovery prefixes pass their old failure point, and the ordered fresh/reverse patch checks pass. Compiler/build identities are saved in `backend012_identity.json`. No clean release tag is created; this is the new experimental condition. Local default build aliases now point at the owned new build, with legacy aliases retained.

Fresh 4,096-world development comparisons of updates 12 and 20 use identical worlds from 5.8e9, full Act 3, exact argmax and the same corrected backend. Both are queued with three collectors/nine workers each; this preserves a matched comparison while fitting the laptop resource envelope. The 5.9e9 confirmation cohort remains closed.

## 14 October 01:00 London Discovery acceptance expectation repairs

The first full suite ran 498 tests and found two historical expectations that treated world 2005 as an invalid native trajectory. Its archived 0.1.1 fault packet confirms that the rejected action was Discovery. On 0.1.2 the world completes correctly in both cache paths. Tests now require that healthy parity and preserve fail-closed slow/fast prefix coverage using an explicit illegal index. The child error is the existing `IndexError: list index out of range`, identical in both paths. The acceptance rerun and compiler identity are preserved under `data/overnight_20261014/`. No native guard was removed to satisfy these tests.

## 14 October 00:57 London Backend build and ordered patch checks

The new module built successfully with SHA256 `9aeac602912d3f455b17b34e9a549a10b544893936087e9ce2d0bce077a87d41`. The actual captured Discovery sequence passed its former rejection point with the new core and search bindings. Bounded core smoke passed. A direct reverse check of the environment patch while the search overlay remained applied failed at the shared binding context; the proper ordered check applied environment then overlay to a fresh upstream tree, reversed overlay then environment, and passed. The experiment binary was not altered by these checks. Full native suite acceptance is in `data/overnight_20261014/backend012_tests.log`.

## 14 October 00:53 London Preparation and authority

The previous round is complete and the new overnight authorization supersedes the earlier review-only state. The user explicitly asked to continue until stopped in the morning; no automatic overall stop time or extension of an old allocation has been inferred. The queue runner is live and idle before preparation. AC power and approximately 22 GiB free disk were checked.

The 0.1.2 Discovery change and native mechanics regressions were imported from the separate clean-base maintenance branch. Its public schema remains unchanged, while package/native/interface identities change. A new local native checkout was cloned from existing pinned upstream history, then received the tracked environment and public-search patches. Read-only pinned dependency sources are reused; CMake 3.31.6 was copied into an owned build-only runtime, without modifying the frozen training runtime.

Queued `night012_build` builds into this worktree's owned `external/sts_lightspeed_rl/build`. Existing live/historical build directories are not overwritten. The build job's bootstrap identity points at the already verified maintenance candidate; the produced module will have its own manifest and acceptance record before any simulation/training job uses it.
