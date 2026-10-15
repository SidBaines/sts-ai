# Fresh-start progress — newest first

## 15 October 2026, 21:52 UTC — launch decisions, before experiments

Study branch `nonllm/fresh-start`, worktree `/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start`; initial source `2ed7f41`, based on `ws/nonllm` `31b6379`.

Sid authorized autonomous research judgment and modifications within this study. Steps A–C and monitoring only; stop at Gate C before step D. Hard deadline **16 October 2026 08:00 UTC (09:00 BST)**; at most about 10 active wall-clock hours of laptop compute, excluding coding and battery pauses. Preserve evaluations if constrained; combat label floor 50,000, collected in interleaved act/room shards; below that stop at a documented checkpoint. Never delete historical artifacts/worktrees. Local commits and fast-forward milestone integration only; no push.

**Predeclared G2 rule (before G2 evaluation):** paired whole-world bootstrap two-sided 95% interval for student minus champion win rate; non-inferiority margin −0.01 (one absolute percentage point). Pass if lower bound > −0.01. If not passed, fail if upper bound < 0; otherwise inconclusive. These rules overlap for a small precisely measured decline; pass takes precedence for non-inferiority, and any interval wholly below zero is separately identified as evidence of decline. Other metrics have paired intervals but no margins. G2 informs Sid's step-D decision; never opens confirmation or starts step D automatically.

**OOC potion protocol:** hide only out-of-combat drink/discard actions from champion and all students through Gate C. Preserve native/display indices of ordinary actions; leave all in-combat potions alone. Record protocol/filter identity in every collection/evaluation receipt. Decide whether to expose the new choices in step D.

Read plan, rationale, handoff, completed results, speedups, canonical teacher, old/archived entity encoders, workers, panel and provenance code. Champion SHA256 verified as `3d135d9bd53b92b5a5eb3dcc8bf26458140e282ee8ea63d9aaaf6ca3c2f222bb`; frozen runtime verified Python 3.12.13 / Torch 2.14.0 / NumPy 2.5.4. About 45 GiB free; retain 6 GiB reserve. Existing separate native-build runtime will supply CMake; the frozen runtime has no CMake, and the core build script would otherwise try to install it. No pip into frozen runtime.

Mechanical implementation: written task specs, fresh Codex CLI builders, separate Claude specification and quality reviews, then orchestrator commits. Claude login verified outside filesystem sandbox. Encoder/model design stays under orchestrator review. The two installed skill copies disagree about reviewer family; follow the approved plan's explicit Claude requirement.
