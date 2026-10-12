# Slay the Spire environment and RL experiments — agent handoff

Written 12 October 2026 after the user explicitly paused work for a critical review. This document consolidates the environment extraction, initial overnight learning campaign, two longer continuations, simulator fixes, optimization, dashboard, causal diagnosis, and unresolved questions. Dates below are UTC; London was UTC+1.

## Start here: current authority and state

**Training is stopped. Both v1 and v2 are operator-paused; the scheduled monitor is PAUSED.** The user said: “pause here and think carefully … good reasons to think the next run will go better … write up everything … then commit … a new agent.” No protected next pilot has been implemented, launched or demonstrated to work. Do not interpret a proposal, retained main command, or unused budget as authorization to resume. Future user instructions may authorize a new experiment; until then, review and planning are the task.

Use `/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI`, branch `codex/long-curriculum-v2`. Read this document, [the critical diagnosis](long_curriculum_v2_forgetting_diagnosis.md), and [the latest-first summary](long_curriculum_v2_summary.md) before opening large journals. Source-of-truth research status is [research_plan.md](research_plan.md). The original `/Users/sidbaines/Documents/SlayTheSpireAI` checkout is dirty with unrelated Gemma/fight-lab research; no handoff change stages or resets that work.

| Allocation ledger | Active seconds |
| --- | ---: |
| Original longer-study authorization | 36,000.000000000000 |
| Used by v1 | 17,575.872087707976 |
| Transferred to v2, including its pilot | 18,424.127912292024 |
| Used by v2 | 4,171.954112750012 |
| **Remaining, once only** | **14,252.173799542012 (3.959 hours)** |

Offline/manual/resource pauses and stopped-controller gaps do not spend active time. There was no extension. V1's old journal still contains its original allocation: its historical remainder was transferred, not a second usable allocation. Confirmed old learner/controller processes exited; the read-only dashboard remains available at [localhost:8786](http://127.0.0.1:8786/) and shows operator-paused.

A verified, committed [machine snapshot](handoff/state_20261012.json) records worktrees, exact files, hashes, runtime, budgets and proposed-versus-implemented status. The [frozen generated report](handoff/v2_live_snapshot_20261012.md) preserves the detailed controller output. Its historical `attention_required` line predates verified pause reconciliation; journal/progress and the pause acceptance proof are authoritative.

## Worktrees, commits and dependencies

All paths below share one Git repository; each checkout has its own branch and local outputs. HEAD values are the observations before this handoff documentation commit.

| Purpose | Branch / source commit | Checkout under `/Users/sidbaines/.codex/worktrees/` |
| --- | --- | --- |
| Clean environment | `environment-base`, `b31a898` | `environment-base/SlayTheSpireAI` |
| Completed overnight study | `codex/nonllm-curriculum`, `c8c76bc` | `nonllm-curriculum/SlayTheSpireAI` |
| Paused longer v1 | `codex/long-curriculum`, `1693f1c` | `long-curriculum/SlayTheSpireAI` |
| Paused longer v2 and diagnosis | `codex/long-curriculum-v2`, `b902020` | `long-curriculum-v2/SlayTheSpireAI` |
| Separate Discovery fix PR | `codex/fix-act1-boss-undefined-behavior`, `54df6ef` | `environment-boss-ub/SlayTheSpireAI` |

Important implementation commits: `f23cad1` v1 controller preflight; `9c7b649` frozen v2 scientific/controller source; `9fb2e05` main launch documentation/static dashboard follow-window fix; `c6835cf` standalone phase diagnostic runner and four regressions; `b902020` completed attribution report. Later docs commits do not change the frozen learner condition.

Immutable releases: `env-v0.1.0` → `969fc484762c22480e90dc5bf0e16e140e380adf`; `env-v0.1.1` → `b31a898fef9fc748ae9f155909f5f58ae40b3300`. Never move these tags.

**Weights, task banks, native builds and runtimes are gitignored local artifacts. A commit is not a backup of them.** V2 `.venv` resolves to the frozen overnight worktree's `.venv`; its native module and teacher pools also live there. Preserve those worktrees and files. The original study's [artifact index](nonllm_artifacts.json), final identities/verification/campaign reports, and the new handoff manifest are essential alongside source. A fresh clone needs a native build and restored/copied local artifacts; it cannot reproduce this run from Git alone.

## 1. Clean environment base

The user requested a reusable Gym-style base and a branch literally named `environment-base`, preserving existing experiments. We extracted package `sts_env`, dependency-free core, pinned/patched C++ Lightspeed backend, versioned public observations/replay, CLI/custom-agent factories and optional pinned Gymnasium adapter. Model clients, scientific training and reward recipes stay in adapters/experiment branches. The original LLM/framing research was preserved, not reworked into adapters during this campaign; this trained policy is a purpose-built non-language network.

| Control mode | External agent | Automatic policy |
| --- | --- | --- |
| `full` | Combat and OOC | Internal transitions |
| `ooc` | OOC | Native combat search |
| `combat` | Combat | Scripted/seeded native OOC |

The flipped hybrid exists, but native OOC is scripted/random pathing, not a strategic OOC search. Native search is a privileged reference. CLI supports doctor, run, play, evaluate, inspect, replay, imported factories/config, separate world/policy seeds, trace identity, deadlines and output-overwrite refusal. Public API reset/step follows Gym-style termination/truncation. Invalid core actions raise; optional padded Gym actions are explicit bounded no-ops.

The compact Gym adapter has 15 numeric features and an action mask; serious training uses the richer adapter below. The core's default reward is +1 victory and 0 otherwise, distinct from training rewards. Raw native action bits/hidden positions stay out of observations; diagnostic `info` must not be fed to policies. Ironclad ascension0, ordinary Acts1–3 is the trained scope. Act4 is experimental, other characters need representation/coverage work. This is not complete vanilla-game or all-state simulator certification.

Clean fixes included native reward sampler/event fallthrough, chest-policy recovery, visible routes/card-skip/event descriptions, guards and provenance. Some intentional trajectories differ from the old harness. `env-v0.1.0` passed 191 release tests and CLI/replay/vectorization/package checks. [Environment contract](environment_base.md), [release 0.1.0](release_v0.1.0.md), [release 0.1.1](release_v0.1.1.md).

## 2. Initial overnight campaign: what actually learned

Started 11 October around20:08 UTC; final selection froze 12 October05:58:19, after the learner stopped, before the user's06:30 deadline. Work used BC warm starts, combat DAgger, masked PPO, corrected OOC training, and integrated Act1→Act3 full-control PPO. Six size/context architectures and multiple LR/initialization/hyperparameter controls were executed. DQN, SAC, IQL/CQL, MuZero, recurrence, entity attention and factorized-action alternatives were considered in the plan, not trained. MLX candidate operations were benchmarked/parity-checked, but no online MLX learner was run.

Chosen candidate model: large/no-menu-context MLP, **2,298,882 parameters per phase**. The final composite has two independent combat/OOC models (~4.6M total). The historical role name `shared` does not mean a shared encoder. Within each selected no-context phase, actor and critic share the state encoder; the critic does not consume the action encoder. The optional mean-context architecture also lets critic gradients reach action embeddings. Load checkpoint model_config rather than assuming current CLI defaults.

### How state and actions are understood

Encoder `public_candidate_v2`: 256 state features and128 features per current legal candidate. It uses allowlisted public scalars, fixed Blake2b signed hash bags, public card/potion rules, targets, costs/damage/block, powers, offers/deck/relics and reachable map structure. Candidate scores depend on state/action encodings and their interactions. Masked categorical actions reference the current menu; there is no stable global “action 7”, and enumerated legal subsets need no global exponential output head. Menu permutation permutes logits.

Inputs exclude seeds, raw native action bits, hidden draw order, teacher/search values and diagnostic info. Static glossary rules are public domain knowledge, not a privileged future query. This is not a language model/general text comprehension or a complete Markov encoding: hashes collide, complex rules are approximate/conditional, missing numeric annotations can be zero, and strategic relations/history may be missing. Increasing model size does not automatically repair those limitations. [Encoding detail](nonllm_encoding.md).

### Learning and outcome evidence

Combat BC/DAgger/PPO produced competent isolated fighters. Corrected OOC DEV improved12/32→29/32→31/32. Joint Act1 selection reached45/48. Health-weighted combat alternatives worsened integration (21/32 reference versus11/32 and13/32); their histories differ, so this did not causally isolate HP reward. Short OOC bridges and60-second GAE.95/1 pilots produced no full-run DEV gains and were rejected; they do not rule out those ideas at a properly tested scale.

Selected named jobs demonstrably changed weights and consumed data; for example initial BC had1,993 updates/~2.00M exposures, combat DAgger576/~576k, corrected OOC BC2,973/~2.83M. These are job examples, not whole-stage totals or unique labels. Successful constituent training did not establish successful long-run play. BC agreement on forced single-option menus is distinct from agreement on meaningful choices; neither is a competence endpoint.

| Frozen first-study held-out assessment | Learned / requested | Native reference | Learned faults |
| --- | ---: | ---: | ---: |
| Isolated Act1 fights | 145/160 | 149/160 | 0 |
| Isolated later-act fights | 25/32 | 26/32 | 0 |
| OOC Act1, native combat | 17/21 | 6/21 | 0 |
| Act1 full learned control | 20/21 | 6/21 | 0 |
| Three-act full control | 0/21 | 0/21 | 1 |

Three-act learned outcome:20 defeats and1 simulator fault, no truncations; specialist composition0/21 with21 defeats/no faults. Twenty-two final panels were checked, duplicate combat panels counted once,362 tests and bounded smoke passed, and four DEV full3 CLI runs/two replays had matching traces and zero native-controlled decisions. **Supported conclusion: strong Act1 competence; full-game competence unestablished.** Isolated later-act battles do not mean the model can reach those acts from scratch.

Final selection preceded this study's final evaluation. Its final data is now historical/exposed and cannot be reused as an untouched final for future tuning. A disclosed learned seed3 integration/replay exception touched the broader hash-final partition before freeze, outside the designated quantitative cohort; exclude it from future final inference and do not claim the whole partition was untouched. The first study used OOC DEV horizon1024 and full3 DEV4096, versus generated OOC/full final2048. Longer-study full3 DEV also uses4096; world/cohort/horizon differences prevent pooling0/21 with2/192. [Final evaluation](nonllm_final_evaluation.md), [completion audit](nonllm_completion_audit.md), [protocol exception](nonllm_protocol_audit.md), [initial summary](nonllm_summary.md).

## 3. Throughput work and resource decisions

Mac M5 Pro,15CPU cores,48GB unified memory; optional Python3.12.13/Torch2.14/NumPy2.5.4. Small batched rollout inference was faster on CPU; large updates used MPS. Eight persistent isolated simulator workers and central batched inference avoid per-worker GPU copies. Initial campaign used two Torch threads; longer studies later used one. Dedicated GPU authorization is permission, not evidence GPU inference is faster.

| Measurement/optimization | Evidence | Limit |
| --- | --- | --- |
| Real large model CPU vs MPS inference | Matched ABBA collection0.962s vs1.395s; identical45/48wins/687transitions | Inference only; updates stay MPS |
| Persistent cached teacher workers | Eight~5,002labels/s vs12~5,021; one~1,343 | Specific collection workload |
| Fast prefix replay | Twenty paired starts4.118s→1.579s | Same identities/states, not learning improvement |
| Reuse collection encodings in PPO | Preparation39–40s→0.755s; features/actions/loss/gradients/Adam/weights parity | Numeric cache~71MB; evolving-policy speed comparisons are descriptive |
| Ordered-token encoding cache | ~1.28–1.31× encoding, bit-identical results | Workload dependent |
| Live optimized collection | Adjacent~1,157–1,225→1,460, later~1,630 transitions/s | Policies/worlds change; not a causal competence comparison |
| Longer-study4→8 CPU worker evaluation | ~1,477→1,768transitions/s, exact traces | One matched evaluation, not training gain guarantee |

Also tested CPU/MPS/MLX compute, CPU thread counts, metric/CE amortization, deferred bootstrap, phase dispatch and opt-in action buckets. Synthetic device examples/s are not end-to-end throughput; mixed-menu bucketing was not uniformly faster. Checkpoints (~55MB including optimizer) are frequent; CPU serialization probes~43–69ms exclude MPS sync/transfer. [Throughput](nonllm_throughput.md), [encoding reuse](nonllm_encoding_reuse.md), [bag cache](nonllm_bag_cache.md), [thread screen](nonllm_cpu_threads.md).

## 4. Simulator changes and quarantined evidence

During the first campaign, automatic combat used an uninitialized character class in Player/BattleContext setup. `env-v0.1.1` fixes it and sticky reported-error propagation. Old native-auto OOC/lookahead outcomes, labels and training were quarantined; retained legacy full-control Ironclad combat had scoped checks. Do not resurrect old auto-combat conclusions or assume all undefined memory/phantom powers are eliminated. Release0.1.1 clean tests195; later research suites had additional tests.

Study native module SHA: `40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`.
Study patch SHA: `77fddbd67c82bebcc4ff31eea6bb0f6d25e8216e44825d6459f184f47756ce16`.
Study Python interface SHA: `1fa75a1d3d7de01ceae2ba0461f8ce3cae4051e8e6ba94ff3ff1897a9af02205`.
Native build: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/external/sts_lightspeed_v011/build`.

Later DEV faults401491/400594 were reproduced: Discovery unconditionally flagged undefined behavior despite implemented Ironclad generation/selection. Isolated fix validates supported class/offers, preserves unsupported/error checks, and tests ordinary/upgraded/X-cost semantics, overflow/full-hand/replay/determinism. [Draft PR1](https://github.com/SidBaines/sts-ai/pull/1) is OPEN/draft, target `environment-base`, head `54df6ef`;202 tests and bounded smoke/replay passed. Separate0.1.2 candidate binary SHA`645acbf2cfdf8055f59f379bdb0d2d2e3a06589809f6252f4bef3c50c3a0c198`. It is **not merged/tagged or substituted into the study**. Fix details live in the separate worktree's `docs/discovery_fix.md`.

A build invocation mistakenly installed CMake into the borrowed shared runtime; it was uninstalled to restore prior module absence and Torch/NumPy/native/checkpoint identities reverified. The fix worktree then used its own venv. Build scripts may install CMake into `PYTHON`; use an isolated interpreter for future native experiments. Native patch changes must be regenerated/applied against pinned source, scoped without pybind submodule changes. Never hot-rebuild a frozen study binary; a backend change requires new identity/rebaseline and trajectory checks.

## 5. Longer v1: why accumulated compute misled us

New10-active-hour continuation began with a15-minute pilot around10:37 UTC; main launched10:53:40. It used rotating fresh TRAIN starts plus native/learner continuations and Act1 rehearsal, planned Act2→Act3 stages, fixed reward coefficients, dedicated MPS updates/eight CPU workers/one thread, resource/process/checkpoint guards. Pilot27chunks/~736ktransitions/1,785updates showed small Act2 DEV8/48→9/48 and Act1 retention42/48. That was operational acceptance and weak subgoal evidence, not proof of full3 learning.

All rewards: +.05 per positive floor advance, +1 task success,−1 ordinary defeat; gamma1, GAE.95, terminal HP weight0. V1 task success could be Act1/Act2 clear. Goal/horizon were not encoder inputs, so similar states could have heterogeneous terminal targets; this was a plausible representation/critic mismatch, not a proven cause.

An independent monitoring-only full3 observer evaluated the same48 fresh DEV worlds every30minutes. Candidates fluctuated0,1,0,1,2,1,1wins; selected CP236 stayed0/48. Only two worlds supplied all observed wins. CP405 had2/48 but was rejected: its primary Act2 clears8/48 versus selected11/48, despite passing Act1 guard41/48. Observer wins were deliberately absent from checkpoint selection. Thus the objective could reject better rare full-game outcomes.

The15:25 bounded audit found526completed jobs,36,217accepted updates and14.86Mtransitions, but selected-model ancestry retained only **six chunks,511updates,163,186transitions**. About98.6% of counted updates were absent from that selected parameter lineage. Healthy generated continuations still survived and affected later data, so the compute was not wholly without indirect effect. Hold/rollback restored the incumbent every three-chunk branch; repeated return also restarted Adam for the changed goal mixture. Fresh-start windows were flat: goalAct3 mean floor25.86→25.80, wins0/1300→2/1300; Act1 rehearsal74.0%→71.85%.

These are15:25 audit counts, not final539-round counts. V1 paused15:31:31 UTC; its controller/observer stopped, outputs and two-win snapshot preserved. It used17,575.872seconds; remainder transferred to v2. [Selection audit](long_curriculum_learning_audit.md), [v1 plan](long_curriculum_plan.md), [v1 summary](long_curriculum_summary.md).

## 6. Longer v2: what was fixed, and what still failed

The user authorized a new original-checkpoint restart with only the remainder. V2 frozen source`9c7b649` passed467 tests, JS checks, original/bank/native identity proofs and exact CPU evaluation replay. New condition: continuous learner separate from evaluation champion; all TRAIN goals Act3; warm optimizer survives chunks; ordinary DEV non-improvement does not reset the learner. Retain current/champion/original/hourly/recent/pending recovery checkpoints. Preserve fault/cut/common-valid accounting and plateau/catastrophic guards.

Fixed128-task mix:90fresh Act1 starts,25Act2 continuations,13Act3 continuations. All maxAct3/horizon4096. Continuations reuse native/healthy learner worlds; never add overlapping pool counts as unique seeds. Native pool634Act2/86Act3 starts is not720 independent worlds. Fresh seeds rotate each chunk from100M namespace, disjoint from earlier v1 fresh ranges including its100-billion-offset stream. V1's12:04 audit10,320unique TRAIN worlds/21,376attempts is a bounded snapshot, not final pool size. [Seed audit](long_curriculum_seed_audit.md).

Coefficients/architecture remain unchanged: floor+.05/success+1/defeat−1, gamma1/GAE.95/HP0, LR1e-4,clip.1,entropy.01,batch1024,fourPPOepochs. No reward taper. CPU rollouts/MPS updates. Fixed new48screen routes evaluation; separate192confirmation selects on full-game wins with paired-floor secondary evidence; same192world Act1 retention checks competence. Selection uses common-valid≥187/192, ≤3faults/panel, exact tasks and no cuts. All new finals remain embargoed.

Pilot16:23:40 completed26chunks/862,697transitions/3,086retainedupdates in887.108active seconds; main resumed16:44:51. One cold start followed by warm Adam restoration, exact parent lineage, no rollback. V2 really accumulates learning updates: **final retained CP128 has129chunks,13,514updates,4,030,949validtransitions,128warm restores,zero rollbacks**. Interrupted next round129 has no completed checkpoint and must not be counted. Champion never promoted from the morning original.

| 192-world panels, raw requested denominators | Original | First confirmed CP54 | Second confirmed CP114 |
| --- | ---: | ---: | ---: |
| Full3 wins | 2/192 | 0/192 | 0/192 |
| Full3 simulator faults | 2 | 0 | 0 |
| Mean floor on own valid worlds | 27.63 | 25.18 | 25.86 |
| Act1 retention clears | 151/192 | 131/192 | 139/192 |
| Retention faults | 2 | 0 | 0 |

No cutoffs/unstarted cases. First matched190-world retention fell151→130 (−21); second151→138 (−13). Latest CP128 was not confirmed on192worlds: do not assign either earlier result to its bytes. CP54's evaluation is global round55; CP114's global round115. SHA, not an ambiguous round label, defines the model. First screen baseline1/48 differs from v1 zero/48 and first-study0/21; never pool these cohorts.

The catastrophic guard required a≥29 paired Act1 clear drop or≥8floor drop without win gain on two distinct confirmed snapshots. Actual declines did not cross those deliberately severe thresholds. Thirty-minute plateau warning/one-hour inspection appeared, but neither provided proof of useful learning. User halted v2 around17:39. **Fixing lineage/selection was necessary, but insufficient; another long unchanged run has no supporting evidence.** [V2 protocol](long_curriculum_v2_protocol.md), [latest summary](long_curriculum_v2_summary.md), [detailed progress](long_curriculum_v2_progress.md).

## 7. Causal diagnosis: OOC loss, plus fragile full-game wins

Static384-state TRAIN probe found accumulated original-to-new KL up to~.79 combat/~.29OOC despite local PPO KL~.005. This showed drift, not its cause. Whole-phase parameter transplants then tested original/new combat × original/new OOC. Native/interface/weights/task IDs/action traces were independently audited. No training or champion selection used diagnostic outcomes.

The fresh TRAIN diagnostic expanded32→64worlds solely on prespecified runtime/completeness, not outcomes. Five worlds faulted somewhere;59common-valid, no cuts. Original floor30.88/Act1clears51/Act2clears14; newcombat-originalOOC29.90/51/11; originalcombat-newOOC26.44/47/4; newboth25.93/39/7. Newboth had one win versus zero original; it cannot establish population improvement. Isolated32native-produced TRAIN combats improved24→25wins, no losses/faults/cuts. This does not certify learner-induced states.

Because TRAIN progression alone did not explain known win loss, all four models replayed the **exact192DEV confirmation panel**. Original190 and regressed192 valid action traces exactly matched original evaluations. Fault union2 gives the same190validworlds below:

| Combat / OOC | Mean floor | Act1 clears | Act2 clears | Full wins |
| --- | ---: | ---: | ---: | ---: |
| Original / original | 27.63 | 151 | 13 | 2 |
| New / original | 28.48 | 151 | 22 | 0 |
| Original / new | 24.72 | 130 | 5 | 0 |
| New / new | 25.24 | 130 | 12 | 0 |

OOC replacement reproduces the broad early loss. Combat replacement improves aggregate later progression but erases both rare original wins. Both winningworlds401610/401056 become ordinary defeats under either mixed model, not fault exclusions. NewOOC kills them at floors33/16; newcombat-originalOOC at50/33. Do not call combat harmless or assume freezing one phase alone preserves complete runs.

This identifies phase effects on these panels, not a particular wrong action, gradient mechanism or population win-rate decline. Two rare victories are weak evidence for broader superiority; adaptive repeated DEV selection is descriptive. [Full diagnosis](long_curriculum_v2_forgetting_diagnosis.md) and local `data/long_rl_v2/diagnosis/{combat,phase_crossover,dev_crossover}` retain commands, metrics, proofs and audit scripts. Standalone `scripts/diagnose_phase_forgetting.py` plus four regressions committed`c6835cf`;471tests(one optional skip) and bounded21-step smoke passed. No source/condition change occurred in the paused learner.

## 8. What we know, what remains hypothesis

**Established:** updates really happened in v2; original policy drifted; OOC changes cause observed broad progression loss; combat changes also fail two rare winning trajectories; proxy floor/Act2 gains can coexist with fewer complete wins. The first study learned Act1 but had no established full-game competence.

**Not established:** actor-versus-critic causality, which OOC categories/actions fail, value calibration, a representation bottleneck versus optimization/credit failure, best exploration/reward/architecture, or an intervention that improves complete runs. No anti-forgetting pilot has been tried. No obvious GAE terminal/truncation bug was found; correct finite code does not imply useful targets.

PPO constrains against the immediately previous rollout policy, not the morning original. Small steps can accumulate. Each selected no-context phase's critic shares the state encoder with its actor, so critic gradients can change logits through state features even when the actor surrogate is clipped; critic dominance is unmeasured. Action-encoder sharing with the critic occurs only in the optional mean-context variant, which was not selected. Global normalization/loss/clipping are weighted by transitions, not curriculum episode quotas. Tiny self-bootstrapped value loss is not Monte Carlo calibration.

Gamma1 keeps nominal long-run reward; lambda.95 attenuates the direct terminal TD residual to~.0059after100decisions/~.000035after200. Bootstrapping can transmit credit indirectly; these are not “all future reward disappears” claims. FullAct3 task labels alone do not ensure useful early OOC credit. Positive-floor shaping explicitly rewards progression as well as wins; the current reward should not be called policy-invariant potential shaping without proving terminal handling.

Retention controls are not improvement mechanisms. Original-policy anchoring can preserve a weak strategy without discovering better play. Lower LR/fewer epochs can merely slow changes. Frozen combat prevents weight drift, not failure on changed decks/states, and removes the beneficial combat progression seen above. More seeds/time or a larger model are not demonstrated remedies. Sparse baseline wins make small benchmarks poor detectors of population win-rate gains.

## 9. Before another run: evidence the next agent should demand

First review the scientific question and identify one falsifiable mechanism. Do not launch another multi-hour PPO allocation on the strength of an engineering pilot or a single floor curve. Obtain separate user authorization for a new named condition; preserve v1/v2 and the remaining budget.

A documented candidate—not implemented—is an original restart with conservative OOC updates (one epoch/LR1e-5 starting test), initially frozen original combat/excluded optimizer, and original legal-action distribution rehearsal/KL on fixed TRAIN-only phase/act-stratified public states. Any trainable head needs protection; DEV cases must never enter gradients. Use exact model hashes for freezing, because zeroing gradients or freezing only the scorer does not stop encoder/AdamW changes. Fix operational races and snapshot retention before sustained unattended work.

Predeclare short pilot duration, comparator, cohort identity, thresholds and stop rule. Measure safety AND positive learning. Stop if full-win or Act1 retention regresses, even when floor/Act2 improve; stop/inspect stagnation before spending hours. More accepted updates or lower PPO loss are not acceptance criteria. Repeated small DEV panels do not provide final superiority evidence. Expand/freshen development evaluation thoughtfully while keeping new final data unopened.

Useful diagnostic questions before choosing the remedy: phase/act-wise Monte Carlo return versus critic prediction; actor and critic encoder-gradient norms/cosine; raw/normalized advantage scale and actual transition/minibatch composition; OOC decision categories/state/deck shifts; original-policy KL/TV/argmax drift on fixed TRAIN rehearsal and learner-visited states; paired per-world clear/win gains/losses plus faults/cuts and endpoint HP distributions. Record frozen references/current/champion and exact evaluation versus capture times.

Independent future ablations: detach/separate critic encoders; phase-balanced updates; OOC combat-macro transitions/higher GAE lambda; reward objective/taper; DAgger/counterfactual strategic labels; richer relational/history/action representations; alternative RL/replay methods. Macro transitions with fixed combat must preserve trajectories/reward sums and termination/truncation semantics. Earlier30/60second rejections are not definitive negative studies. Avoid bundling all changes, then claiming individual causes are known.

**There is presently no demonstrated reason to promise the next run will improve full-game play.** The diagnosis narrows the failure and supplies testable controls; measured short comparisons must earn permission to scale.

## 10. Operational failures and monitoring lessons

Pause handler raises KeyboardInterrupt immediately; a signal can race Popen→PID/birth/group ownership publication. V2 cleanup reported “Pause cleanup denied while verified owned process remains alive”; all recorded processes subsequently exited. Original journal/error backed up; root independently verified births absent and atomically recorded operator-paused. No restart occurred. Proposed repair (unimplemented): flag-only/deferred handling across publication; terminate/reap direct child then verify descendants; bounded cleanup; real-process tests at prelaunch, postspawn/prepublication and worker startup.

Retention deleted CP114 model bytes after its second confirmation; its metrics remain. CP54 survives and defines every causal diagnostic, not latest CP128. Preserve immutable bytes for every future confirmation, original/current/champion/hourly recovery and accepted pilot, with deletion/ancestry ledger. Original failure evidence and partial interrupted outputs stay intact.

Dashboard: `scripts/long_rl_dashboard.py`, `[v1|v2]` selector, read-only cached `/api/telemetry` and bounded `/api/episodes`. It includes zoom/brush/pan, rolling averages, dual axes, source/start/goal/horizon/condition filters, histograms/scatter/CSV, coverage/HP/faults, initial/current/champion hashes and evaluated-versus-newer-live weights. Follow-newest startup range bug was fixed in static JS with regressions. Reused sameSHA results keep original evaluation times. Observer monitoring never silently changed v1 selection. See [dashboard guide](long_rl_dashboard.md).

Dashboard launch identity is `data/long_rl_v2/dashboard/launch.json`; learner recovery argv `campaign/main_command.json` is historical, **not a command to execute while paused**. The monitor `slay-the-spire-local-rl-progress` isPAUSED; do not reenable automatically. Status changes or a live PID are not proof of useful learning. No further automatic diagnostic runs are needed.

## 11. Read-only reproduction and document index

From the v2 worktree use `.venv/bin/python` and `PYTHONPATH=src`; plain system `python3` may differ (tomllib was unavailable in the current system interpreter). Explicitly set the frozen `STS_LIGHTSPEED_BUILD` if later authorized evaluations need native imports. Do not rebuild/install into the shared venv or archive dependency worktrees.

Read only selected journal/progress fields; verify hashes via the committed manifest. CPU diagnostic/audit scripts and exact command records are available but **rerunning them is additional work, not part of this paused handoff**. Existing471-test/21-step-smoke evidence remains the last code QA. This handoff changes documentation only; whitespace, relative links, artifact hashes and pause state are checked. No new rollouts/training/final evaluation were run for it.

| Question | Read first |
| --- | --- |
| Full experiment chronology | [Initial summary](nonllm_summary.md), [initial progress](nonllm_progress.md), [v1 summary](long_curriculum_summary.md), [v2 summary](long_curriculum_v2_summary.md) |
| Why v1 selection/compute misled us | [Learning/selection audit](long_curriculum_learning_audit.md) |
| Exact v2 design and gates | [Protocol](long_curriculum_v2_protocol.md), `configs/long_curriculum_v2_protocol.json` |
| Phase attribution and prevention proposals | [Forgetting diagnosis](long_curriculum_v2_forgetting_diagnosis.md) |
| First-study final boundaries | [Final report](nonllm_final_evaluation.md), [protocol audit](nonllm_protocol_audit.md), [completion audit](nonllm_completion_audit.md) |
| Encoder/model semantics | [Encoding](nonllm_encoding.md), `src/sts_policy/{encoding,model,model_agent,losses,rollout_training}.py` |
| Native correction/quarantine | [Native revision](nonllm_native_revision.md), [release](release_v0.1.1.md), separate PR worktree's `docs/discovery_fix.md` |
| Frozen artifacts and exact current status | [Initial artifact index](nonllm_artifacts.json), [handoff state/hash manifest](handoff/state_20261012.json), [controller snapshot](handoff/v2_live_snapshot_20261012.md) |
| Operational/candidate/evaluation evidence | v2 `data/long_rl_v2/{campaign,preflight,diagnosis}`; v1 `data/long_rl_v1/{campaign,preflight,full_run_eval}` |

Suggested message for a fresh agent: “Work in `/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI`. Read `docs/experiment_handoff.md` and its linked evidence. Both longer runs and the monitor are paused. Critically review the failed learning mechanisms and design bounded comparisons that would justify a next run; do not resume training or use final data until I explicitly authorize the new condition. Preserve all worktrees, artifacts and unused budget.”
