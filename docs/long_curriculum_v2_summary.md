# Full-game RL v2 — latest first

## 2026-10-12 17:42 UTC — user paused training; causal diagnosis underway

The user requested combat versus out-of-combat attribution and a way to prevent forgetting rather than spending the remaining allocation on an unproductive setup. V2 is now **operator-paused**, all recorded controller/workers verified absent. The final retained CP128 SHA is `afc854b4…`; **14,252.174 active seconds (3.959 hours)** remain. The budget is stationary; no automatic resumption or extension is allowed.

A shutdown race initially produced an attention error while workers were exiting. The untouched journal was backed up, all recorded process identities independently checked absent, and the operator pause made durable. This operational error is preserved separately from scientific regression evidence.

Parallel diagnosis will compare original and confirmed regressed models on matched TRAIN worlds with four combat/OOC combinations, plus isolated combat probes if feasible. The first confirmed CP54 is retained for attribution; the second confirmed CP114 model had been deleted by retention, so its metrics cannot substitute for that model. No diagnostic will use final data or train on DEV observations. Code review confirms separate phase models but actor/critic shared encoders within each phase, global phase weighting/clipping, and no fixed-original policy anchor. These are plausible mechanisms pending causal evidence. [Detailed progress](long_curriculum_v2_progress.md) · [Dashboard](http://127.0.0.1:8786/).

## 2026-10-12 17:33 UTC — second confirmation still below baseline

A second, distinct learner snapshot completed the same 192-world confirmation: **0/192 full-run wins**, versus the original **2/192**, with zero faults, cutoffs or unstarted tasks. Act 1 clears recovered from 131 to **139/192**, still below the initial **151/192**. Full-run mean ending floor rose from 25.18 to 25.86, still below the original 27.63 (valid episodes).

On the 190 common-valid worlds, full-run floor delta versus initial was -1.73 and net wins −2; paired Act 1 clear delta was -13. The original remains champion. No severe-decline threshold has been crossed; the run continues with approximately 4.07 active hours left. **Two confirmations now show no useful full-run improvement.** Policy-drift diagnosis is documented below. [Dashboard](http://127.0.0.1:8786/) · [Detailed log](long_curriculum_v2_progress.md).

## 2026-10-12 17:32 UTC — one-hour stagnation check; policy drift measured

No confirmed full-game improvement has appeared. The one-hour inspection warning is active; a second confirmation cycle is underway. The last completed 192-world result remains 0 wins against the original 2, with 131 Act 1 clears against 151. The original remains champion. Later screens returned 1/48, matching the screen baseline win count while ending around floor 25.5 versus 32.48 initially; this is not confirmation evidence.

A bounded, read-only probe compared the original and first confirmed learner on 384 fixed TRAIN public states. Both policy heads changed materially:

| Diagnostic across act-specific state buckets | Combat | Out-of-combat |
| --- | ---: | ---: |
| Changed greedy choice | 18.8–26.6% | 12.5–14.1% |
| Original-to-learner distribution KL | 0.177–0.791 | 0.217–0.289 |

Small local PPO KL (~0.005) did not prevent cumulative drift from the original policy. This makes forgetting a plausible explanation, but does not distinguish combat errors from deck/pathing changes or prove an algorithm bug. Low bootstrapped value loss is not evidence of calibrated long-run predictions. The probe used two CPU threads for 2.35 seconds and no new environment trajectories.

At this check, 115 completed chunks retained 12,281 updates, with one initial cold start, 114 warm restores and zero rollbacks; about 4.08 active hours remained. Exact source, protocol, DEV, original/checkpoint ancestry and native binary hashes matched; owned controller/dashboard identities and dashboard HTTP response were verified. No condition changes, attention acknowledgement or budget extension occurred. Evidence is retained in the run's preflight directory. [Dashboard](http://127.0.0.1:8786/) · [Detailed progress](long_curriculum_v2_progress.md) · [Generated detail](long_curriculum_v2_live.md).

## 2026-10-12 17:12 UTC — first confirmation declines; champion remains original

The new run is operationally healthy, but its first larger evaluation gives **no evidence of useful improvement**. The round 55 snapshot lost both baseline winning worlds and had lower mean ending floor. Act 1 retention also declined. These are descriptive results from a small, repeatedly inspected DEV set, not a final generalization estimate.

| Fixed DEV panel | Original baseline | Round 55 learner | Faults: original / learner |
| --- | ---: | ---: | ---: |
| Three-act victories, 192 requested | 2/192 | 0/192 | 2 / 0 |
| Mean ending floor, common-valid 190 worlds | 27.63 | 25.24 | — |
| Act 1 clears, 192 requested | 151/192 | 131/192 | 2 / 0 |

All evaluations completed without cutoffs or unstarted tasks. On the 190 common-valid retention worlds, clears fell from 151 to 130: a paired loss of 21. The original checkpoint remains champion. Neither prespecified severe-decline threshold has been crossed; the 30-minute no-win-improvement warning is active and a read-only diagnostic investigation is underway. No running scientific settings have changed.

At 17:10 UTC, the learner had retained 8,091 optimizer updates across 73 chunks and 2,320,273 valid transitions, with one initial cold optimizer start, 72 warm restores and zero rollbacks. Approximately 4.43 active hours remained. These counters demonstrate continued training, not better play. The dashboard distinguishes newer unevaluated weights from this evaluated snapshot.

[Live dashboard](http://127.0.0.1:8786/) · [Detailed progress](long_curriculum_v2_progress.md) · [Generated live record](long_curriculum_v2_live.md). Simulator [PR #1](https://github.com/SidBaines/sts-ai/pull/1) now passes 202 tests, including Discovery-generated Whirlwind energy semantics; v2 retains its frozen backend.

## 2026-10-12 16:49 UTC — pilot accepted; longer continuous run is live

The pilot started at16:23:40 UTC and completed26chunks,862,697valid transitions and3,086optimizer updates in887.108active seconds. Every completed update remains in the learner's exact checkpoint ancestry; the saved optimizer step is also3,086. One initial cold optimizer start was followed by25warm restores, withzero learner rollbacks. All owned pilot processes exited and source/protocol hashes remained fixed.

Main resumed at **16:44:51 UTC (17:44 BST)**, PID90942, with the same optimizer and remaining17,537.020seconds. Six completed main chunks were verified by16:48 UTC; total retained updates reached3,729 and valid transitions1,058,916. About4.81active hours remained at that snapshot. This uses the transferred remainder, not a fresh allocation.

The first learner48-world screen scored1/48, no faults/cuts, meanfloor31.50 versusinitial1/48,floor32.48. The192-world champion remains the original2/192reference until a valid confirmation promotes a challenger. Retaining updates is an operational improvement; useful game-performance improvement is not yet established.

[Live dashboard](http://127.0.0.1:8786/) defaults to v2 and retains a selectable pausedv1 view. It separates192confirmation/48screen/192retention, original/learner/champion, retained/discarded updates, warm/cold resets, plateau age, faults and public endpoint health. Live browser verification passed. A frontend-only follow-window fix now expands toward the selected history length; model sources were unaffected.

The independently reviewed [Discovery fix PR#1](https://github.com/SidBaines/sts-ai/pull/1) targetsenvironment-base with201tests, exact-prefix replays, deterministic fresh-process checks andfresh/reverse patch application. It is a separate0.1.2candidate; v2continues on the frozen0.1.1backend. Human summary and detailed evidence remain current; the scheduled monitor follows v2 and respects its original allocation and pauses. [Protocol](long_curriculum_v2_protocol.md), [generated detail](long_curriculum_v2_live.md).

## 2026-10-12 16:21 UTC — preflight accepted; pilot launch next

The revised setup passed **467 host/native tests** (one optional artifact skip), JavaScript regressions, independent review and the bounded core smoke. A real controller CPU evaluation reproduced every screening action-trace hash from the initial baseline. Sources and protocol are now frozen for the budget-counted pilot.

| Original checkpoint, new fixed DEV cohort | Successes / requested | Defeats | Faults / cutoffs | Mean ending floor |
| --- | ---: | ---: | --- | ---: |
| 48-world full3 screening | 1/48 | 47 | 0 / 0 | 32.48 |
| 192-world full3 confirmation | 2/192 | 188 | 2 / 0 | 27.63 |
| Paired192-world Act1 retention | 151/192 | 39 | 2 / 0 | 16.77 |

All requested task IDs and checkpoint/native/interface hashes matched. These different-cohort references are not interchangeable with v1's0/48 baseline. The pending pilot will prove more than three continuous chunks, warm optimizer carry and lineage conservation before the longer continuation.

A separate [Discovery simulator fix PR](https://github.com/SidBaines/sts-ai/pull/1) targets `environment-base`: the two observed faults came from an unconditional unsupported-behaviour marker on Discovery. Its isolated build and201tests do not replace v2's pinned0.1.1 binary. The draft awaits independent review/merge; historical releases remain unchanged.

## 2026-10-12 15:40 UTC — revised study being built; v1 paused

The user authorized a new setup from **this morning's original checkpoint**, keeping the remaining allocation. V1 is stopped with all owned learner/native processes absent; its results, two-win snapshot and selected checkpoint are preserved. Its signal cleanup recorded a PermissionError, but independent process checks proved complete exit; the original journal was backed up before recording the verified operator pause. The exact transfer is **18,424.128 active seconds (5.118 hours)**; v1 used17,575.872seconds, so the combined authorization remains10hours.

V2 separates the continuing learner from the evaluation champion, uses full-game victories for selection, and sets all training tasks to the same three-act objective. The fixed128-task mix is90fresh Act1 starts,25Act2 continuations and13Act3 continuations. Expanded192-world DEV confirmations are separate from48-world screening; faults remain explicit and paired comparisons use common-valid worlds. New TRAIN worlds begin at100million, disjoint from v1. Final data stays embargoed.

The controller, sampler, logging and dashboard are under parallel implementation and independent review. The next gates are exact original-checkpoint/baseline proofs, full tests, a budget-counted pilot and verified continued warm-optimizer progress. New training has not started. [Protocol](long_curriculum_v2_protocol.md), [detailed log](long_curriculum_v2_progress.md).
