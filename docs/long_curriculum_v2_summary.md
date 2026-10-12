# Full-game RL v2 — latest first

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
