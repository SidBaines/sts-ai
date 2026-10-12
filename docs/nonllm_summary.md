# Non-LLM Slay the Spire — experiment summary

Newest updates first. [Detailed log](nonllm_progress.md) · [Decision history](nonllm_progress_notes.md) · [Plan](nonllm_curriculum_plan.md) · [Agent usage](nonllm_agent_usage.md) · [Completion audit](nonllm_completion_audit.md).

## 2026-10-12 06:20 UTC — first campaign complete

The first local curriculum campaign finished within the **06:30 UTC** cutoff. Selection froze at **05:58:19**, after the last GPU learner ended. All **22 final panels** passed independent cohort/hash/source/backend checks. The frozen composite also passed four fresh full-control three-act CLI runs and two replays on verified development worlds, with identical traces and zero native-controlled decisions. **362 tests** and the bounded core smoke passed.

| Held-out assessment | Learned successes / requested | Native reference | Learned faults |
| --- | ---: | ---: | ---: |
| Isolated Act 1 fights | 145/160 | 149/160 | 0 |
| Isolated Acts 2–3 fights | 25/32 | 26/32 | 0 |
| Act 1 OOC, native combat | 17/21 | 6/21 | 0 |
| Act 1, full learned control | **20/21** | 6/21 | **0** |
| Three-act full control | **0/21** | 0/21 | **1** |

Three-act outcomes are **20 ordinary defeats plus one simulator fault**, with no truncations. The separate specialist composition had 21 defeats and no faults. **Conclusion:** strong Act 1 competence; full-game competence is not established. Native is a privileged combat/scripted-OOC reference. Isolated fights are different cohorts from whole runs; duplicate combat panels are counted once. [Full results and world-level uncertainty](nonllm_final_evaluation.md).

**Use:** [frozen joint model](../data/nonllm_native_revision/campaign/joint_act3_0030.pt), [commands](nonllm_agent_usage.md), [artifact hashes](nonllm_artifacts.json). Weights/banks are local, gitignored outputs: preserve them before archiving this worktree. Original work and signed `env-v0.1.0`/`env-v0.1.1` remain intact. Next research should target later-act strategy, learner-induced states and longer credit assignment; no further tuning used final results.

## 2026-10-12 05:50 UTC — restore the anchor; optimize public encoding

Later checkpoints lost Act 1 competence and fell below the best mean reached floor. Two independent 60-second full-run PPO pilots from identical weights/fresh optimizers compared GAE 0.95 versus 1.0: **0 victories each**, with **1/0 faults**. Neither improved the fixed three-act development panel. Reject both and restore the anchor; subsequent training selected round 30.

A bounded ordered-token cache improved full encoding **1.31×**, with bit-identical features, logits, values and seeded choices. A fresh live chunk reached **1,630 accepted transitions/s**; evolving policies/worlds prevent a causal learning claim. CPU thread comparisons did not repeat a batch-eight benefit, so retain two threads. [Cache evidence](nonllm_bag_cache.md), [thread screen](nonllm_cpu_threads.md).

## 2026-10-12 05:03 UTC — extend training using measured evaluation costs

48-world three-act development checks took **20 s**, native control with two workers **12.3 s**, and 32 later-act fights **4.5 s**. Extend training to approximately **06:00**, retaining a 30-minute final reserve and the unchanged cutoff. This preceded all final results; cohorts, seeds, rewards and horizons stayed fixed.

## 2026-10-12 04:50 UTC — later-act diagnosis; bridges rejected

Fresh combat retention: **27/32** for the old specialist, **28/32** for both joint checkpoints, zero faults. Two 30-second OOC/native-combat bridges through Acts 2/3 accepted **22/37 updates**, then each had **0/48 full-control three-act victories** and **0/1 faults**. Reject both; this does not rule out longer training or continuation curricula.

## 2026-10-12 04:19 UTC — Act 1 improves; later acts remain difficult

Act 1 selection reached **45/48**, with zero faults/truncations. Initial three-act development ended in **3 Act 1, 43 Act 2 and 2 Act 3 defeats**, with no victories/faults/truncations.

![Corrected full-control Act 1 learning](assets/nonllm/native011_act1_development.png)

Matched 48-world development cohort; red markers show separate fault/censoring cases. [Data](assets/nonllm/campaign_panels.csv). Keep the best checkpoint, rather than the last.

## 2026-10-12 03:39 UTC — remove repeated PPO encoding

Reuse the exact public features computed during collection. CPU actions, losses, gradients, optimizer states and updated weights match. Preparation falls **39–40 s → 0.75 s** per chunk; observed throughput rises **1,157–1,225 → 1,460 transitions/s**. Peak numeric cache is 71 MB. [Evidence](nonllm_encoding_reuse.md). Earlier matched joint milestones were 29/48 → 37/48 → 38/48 clears.

## 2026-10-12 02:12 UTC — OOC selected

Corrected OOC improves **12/32 → 29/32**. A matched PPO stopping correction reaches **31/32 with zero faults**, versus 29/32/one fault under the old guard. Twenty scripted/lookahead label updates give 19/32 and 16/32; reject both. Random/native/heuristic references clear **3/15/22 of 32**. [OOC evidence](nonllm_ooc_corrected.md).

## 2026-10-12 01:12 UTC — repair native class initialization

Automatic combat indexed potion/card pools with an uninitialized character class. Quarantine old auto-combat OOC/lookahead evidence; retain old full-control Ironclad combat only after scoped checks. Signed **env-v0.1.1** passes release gates. Preserve old binary/tag; regenerate **49,944 Act 1 / 560,620 later-act labels**. [Release](release_v0.1.1.md). Flags detect reported errors, not complete simulator fidelity.

## 2026-10-12 00:07 UTC — combat specialist selected

Earlier panels: **44/48 Act 1, 30/32 later-act wins**. Higher-HP alternatives worsen integrated Act 1 results: **21/32 versus 11/32 and 13/32**. Keep combat17; histories differ. [Coupling evidence](nonllm_health_coupling_corrected.md). Endpoint HP includes healing, not cumulative damage.

## 2026-10-11 22:25 UTC — hyperparameters selected

Ten combat controls cover LR, clip, entropy, discount, rewards, HP and scratch learning. Select **clip0.1, LR1e-4, gamma1, entropy0.01**. Act 3 portions of the panels cover 15 development/12 final worlds.

## 2026-10-11 21:28 UTC — recover after laptop closure

Preserve checkpoints; reconnect power. Deadline extended to **06:30 UTC**.

## 2026-10-11 20:50 UTC — architecture and execution choices

Six size/context screens select a **2.3M candidate MLP**. Final joint model has separate combat/OOC heads. Use eight persistent CPU simulators, CPU inference and MPS updates; measured leading-model collection is 0.96 s CPU/1.39 s MPS.

## 2026-10-11 20:08 UTC — scope

Create `codex/nonllm-curriculum` from immutable `env-v0.1.0`, preserving original work. Curriculum: battles → OOC → full control. **Ironclad, ascension0, ordinary Acts1–3**; no Act4 or mastery claim. A separate early learned seed-3 smoke touched the broader hash-final bucket, outside the designated benchmark; [protocol exception](nonllm_protocol_audit.md).
