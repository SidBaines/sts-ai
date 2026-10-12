# Combat versus out-of-combat regression diagnosis

Study: paused full-game RL v2; diagnostic evidence gathered 12 October 2026. Sources are the morning original checkpoint `57f1e961…` and the first confirmed regressed checkpoint, CP54 `b7985c17…`. This is a causal phase crossover of existing weights, not training or checkpoint selection. Native binary `40ab5056…` and Python interface `1fa75a1d…` stay fixed. New final data is unopened.

## Matched TRAIN crossover

Whole combat and OOC models, including encoders, actor scorer and critic, were transplanted independently. All source tensors and hashes were independently verified. The two 32-world panels expanded to 64 solely because every first-panel evaluation completed within the prespecified 45-second limit. All starts are fresh Act 1, all horizons are 4096 and goals Act 3; deterministic policy seed 777. Four CPU workers, one thread, no learning or GPU updates.

Five worlds faulted under at least one combination, leaving **59 common-valid worlds**, with no cutoffs or unstarted tasks. The following table uses those same 59 worlds for every row:

| Combat model | OOC model | Mean ending floor | Act 1 clears | Act 2 clears | Full-game wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Original | 30.88 | 51 | 14 | 0 |
| Regressed | Original | 29.90 | 51 | 11 | 0 |
| Original | Regressed | 26.44 | 47 | 4 | 0 |
| Regressed | Regressed | 25.93 | 39 | 7 | 1 |

Changing OOC alone costs 4.44 mean floors and ten Act 2 clears; changing combat alone costs 0.98 floors and three Act 2 clears. With original combat fixed, new OOC ends lower on 31 worlds, higher on 11 and ties on 17. Combat-only replacement produces 12 lower, 12 higher and 35 ties.

The largest progression loss comes from OOC. There is also a phase interaction: new combat paired with original OOC keeps 51 Act 1 clears, but pairing it with new OOC falls to 39; original combat with new OOC retains 47. This does not identify a particular card/pathing mistake or prove combat is safe on every learner-induced state.

Raw fault counts over all 64 requested worlds are original/original 4, new combat/original OOC 3, original combat/new OOC 1, and new/new 1. All are reported rather than treated as ordinary defeats. These are small, exploratory TRAIN diagnostics. The one new/new victory versus zero original victories does not establish a better population win rate, and means/floor comparisons alone do not answer the full-game objective. The exact DEV crossover below verifies the previously observed regression separately.

## Isolated combat retention

On 32 matched saved TRAIN battles (12 Act 1, 10 Act 2, 10 Act 3), original combat won 24 and regressed combat won 25: one gain, zero losses, no faults, cutoffs or unstarted tasks. Act-specific wins were 10→11/12, 8→8/10 and 6→6/10. Mean HP cost improved slightly, 22.06→21.75. Action traces changed in 31 battles, so retained results are not merely an unchanged policy.

This argues against a broad combat-retention failure on native-produced saved states. It does not cover all states induced by changed deck/pathing decisions. Exact tasks, public start states, model/native/interface identities and metrics are in `data/long_rl_v2/diagnosis/combat/acceptance.json`.

## Exact DEV regression crossover

All four combinations were evaluated on the unchanged 192-world full3 confirmation panel, with four CPU workers and deterministic seed 777. Original and regressed references reproduced **all 190 and 192 valid action-trace hashes**, respectively, from their original evaluations. Every requested task completed; no cutoffs or unstarted tasks. Original/original had two simulator faults, new combat/original OOC one, and both remaining combinations zero. The same **190 common-valid worlds** underpin every row:

| Combat model | OOC model | Mean ending floor | Act 1 clears | Act 2 clears | Full-game wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Original | 27.63 | 151 | 13 | 2 |
| Regressed | Original | 28.48 | 151 | 22 | 0 |
| Original | Regressed | 24.72 | 130 | 5 | 0 |
| Regressed | Regressed | 25.24 | 130 | 12 | 0 |

**OOC drift causes the broad early survival/progression loss on this panel.** Combat-only replacement preserves the net Act 1 count and improves Act 2 clears and mean floor, while still losing the two original winning cases. Thus average floor and Act 2 progression cannot certify full-game retention, and freezing one phase alone is not a sufficient protection for wins.

The two original winning worlds are valid ordinary defeats under every changed combination:

| DEV world | Original/original | Original combat/new OOC | New combat/original OOC | New/new |
| --- | --- | --- | --- | --- |
| 401610 | Victory, floor 51 | Defeat, floor 33 | Defeat, floor 50 | Defeat, floor 33 |
| 401056 | Victory, floor 51 | Defeat, floor 16 | Defeat, floor 33 | Defeat, floor 46 |

These rare two-win outcomes establish exact losses on these worlds, not a precise population win-rate estimate or an identified wrong action. The TRAIN probe even has one new/new win and zero original wins; broader competence remains uncertain. DEV crossover is explicitly diagnostic-only: no gradients, anchor data or champion-selection change. Full outputs, command/worker proofs, checksums and repeated-reference traces are retained in `data/long_rl_v2/diagnosis/dev_crossover`. The independent audit passed and is recorded in `independent_acceptance.json`, with its executable audit script alongside it.

## Why the existing safeguards missed forgetting

Combat and OOC networks are independent; there is no cross-phase shared encoder. Within each network, actor and critic share both state and action encoders. Critic gradients can therefore alter action probabilities, even with PPO actor clipping. Current scalar losses do not establish that critic interference caused the loss.

The PPO KL/ratio guard compares a new update with the immediately preceding rollout policy. It does not constrain accumulated change from the original. Median local KL was about .0049 across 129 chunks; the independent fixed TRAIN probe measured original-to-learner KL up to .79 in combat and .29 in OOC. Global advantage normalization, minibatch means and gradient clipping are weighted by actual decision counts, rather than curriculum episode quotas. Current logs do not quantify phase gradient balance.

Gamma 1 preserves nominal future reward. GAE lambda .95 still attenuates the direct terminal residual across individual external decisions: .95^100 ≈ .0059 and .95^200 ≈ .000035. Correct bootstrapping can propagate credit indirectly; these numbers do not imply all eventual reward disappears. However, changing every goal to Act 3 alone does not establish useful early OOC credit. No obvious terminal/truncation bug was found.

## Proposed protected next pilot

Keep v2 paused. Its 14,252.174 remaining active seconds are unspent during diagnosis. A new pilot must have its own source/condition identity and start from the original checkpoint, not the regressed learner.

1. Initially freeze the entire original combat subtree and exclude it from the optimizer, isolating the phase with the broad observed loss. This preserves combat parameters, not complete-run outcomes; OOC updates still need anchoring and direct win-retention gates. Hash its complete parameters before/after every chunk. Merely freezing the final scorer or zeroing gradients is insufficient: shared encoder updates and AdamW state/decay can still change behavior.
2. Update OOC conservatively: one PPO epoch, learning rate 1e-5 as an initial bounded test, with a frozen-original distribution KL/rehearsal loss on fixed TRAIN public observations stratified by act. Anchor to the original legal-action distribution, not arbitrary teacher labels. No DEV observation enters gradients. Small LR alone cannot ensure retention or learning.
3. Make original-policy drift an explicit stop condition, with phase/act bucket means and tails, total variation and changed greedy choices. Calibrate numerical limits before the pilot; report them as engineering controls, not guarantees.
4. Log actual phase/act transition and minibatch counts, raw/normalized advantages, actor and critic encoder-gradient norms/cosines, return/value scales, explained variance and empirical terminal-return calibration. Inspect critic drift rather than interpreting low self-bootstrapped value loss as accuracy.
5. Check matched full-game DEV and retention frequently in a short pilot, plus untouched TRAIN phase probes. Retained updates, lower losses and continuation wins are operational evidence only. Do not consume the remaining hours without demonstrable retention and an improving full-run signal. A confirmed full-run win decline stops the pilot even if mean floor or Act 2 clears improve; Act 1 retention decline is checked independently. Failure preserves outputs and stops for diagnosis.

Longer-term, test OOC combat-macro transitions or a higher GAE lambda as a separately identified credit-assignment experiment. With frozen combat, macro transitions can sum the same combat rewards and bootstrap at the next OOC decision without changing game trajectories; this requires deterministic equivalence and terminal/truncation tests. Removing critic gradients from actor encoders is another independent ablation, not a proven fix. Do not bundle all these changes and then claim their individual causes are known.

## Operational recovery and reproducibility

The user pause raced child creation/ownership publication, causing a cleanup error. All recorded controller/workers subsequently exited; the original journal/error was backed up and operator pause made durable. Next-condition operational repair should defer signal handling across child ownership publication, terminate/reap the direct child, then verify descendants; regressions must cover signals before spawn, during publication and during worker startup.

Retain immutable model snapshots for every full confirmation, not only their metrics: the second confirmed CP114 was deleted by ordinary retention, preventing exact replay. CP54 remains available and was used throughout the primary diagnosis. Record phase state hashes, exact Task IDs, action traces, faults/censoring and source/binary identities with every evaluation.

See [latest-first summary](long_curriculum_v2_summary.md), [detailed progress](long_curriculum_v2_progress.md), and [live dashboard](http://127.0.0.1:8786/). Primary generated evidence: `data/long_rl_v2/diagnosis/phase_crossover/{results.json,independent_acceptance.json}` and `data/long_rl_v2/preflight/policy_drift_probe.json`.

Diagnostic verification: **471 tests passed** (one optional artifact skip), plus the required bounded seed-3/max-Act-1/50-simulation core smoke, 21 decisions and ordinary defeat. Four new unit regressions verify whole-phase swaps, reject shared/altered tensors, and cover fault/censoring/common-valid accounting. Pinned learner sources and active budget remained unchanged.
