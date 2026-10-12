# Non-LLM Slay the Spire — experiment summary

Latest updates first. [Automatic log](nonllm_progress.md) · [Detailed decisions](nonllm_progress_notes.md) · [Plan](nonllm_curriculum_plan.md).
Deadline: **12 October, 06:30 UTC**. Results below are **development selection results**; final worlds remain embargoed. Old native-auto evidence is quarantined.

## 2026-10-12 02:32 UTC — full-control learning is improving

A composed policy now controls **both combat and OOC**. Separate specialists outperform a shared warm start; joint PPO then improves the composed policy:

| Matched full-control panel | Before joint training | Latest selected |
| --- | ---: | ---: |
| Initial 32-world comparison: shared / composed | 7 / 21 clears | — |
| Fixed 48-world long-training panel | 29 clears | **37 clears** |

The 37/48 checkpoint completed with **zero faults and truncations**. Its latest chunk performed 364 optimizer updates over 207,730 accepted transitions. The 32- and 48-world panels are different cohorts and are not one improvement curve.

**Decision:** continue composed full-control PPO. Act 3 follows at 04:08 UTC, with matching three-act development evaluation. Training all stages and final evaluation are still pending. [Composite checkpoint format](nonllm_composite_checkpoints.md).

## 2026-10-12 02:12 UTC — OOC selected; short corrections rejected

Clean OOC training improved from **12/32 to 29/32** development clears. Four matched follow-ups started from the same frozen policy:

| Follow-up on 32 requested worlds | Clears | Faults |
| --- | ---: | ---: |
| Original PPO stopping behavior | 29 | 1 |
| Stop before an over-target KL update | **31** | **0** |
| Twenty scripted-label updates | 19 | 1 |
| Twenty fresh lookahead-label updates | 16 | 0 |

**Decision:** keep the earlier PPO guard; reject both label corrections. Fresh lookahead used 256 clean frontiers, four candidates, horizon 30 and native200. These short proxy/fit results do not establish that lookahead is generally ineffective.

Matched corrected references cleared **3/32 random, 15/32 native, 22/32 heuristic**, with **1/1/0** faults. Faults remain separate from defeats. [Corrected OOC audit](nonllm_ooc_corrected.md).

## 2026-10-12 01:12 UTC — native bug repaired and results isolated

Automatic combat had an **uninitialized character class**, used to index potion/card pools. This caused hangs and inconsistent traces. Old native-auto OOC/lookahead evidence is quarantined; the full-control Ironclad combat policy was retained after source and paired-trace checks.

Released signed **env-v0.1.1** after **195 base / 307 research tests**, bounded smoke and three identical OOC repeats. Old binary/tag remain intact. Fresh banks contain **49,944 Act1 / 560,620 later-act labels**. OOC restarted separately from supervised weights.

Recent throughput increased **353 → 1,074 accepted transitions/s**; failed-call waits fell **66.5% → 0.096%** of job time. Backend, checkpoint and seed differ, so this is operational evidence rather than a controlled learning comparison. Flagged unsupported combats now fail explicitly. [Release evidence](release_v0.1.1.md).

## 2026-10-12 00:07 UTC — combat specialist selected

Selected combat17 won **44/48 Act1 and 30/32 later-act fights**, matching native reference win counts. Higher isolated wins did not uniformly preserve HP. Rechecking health-bonus alternatives with the improved OOC head later produced **21/32 clears for combat17**, versus **11/32 and 13/32** for the alternatives; keep combat17. Their training histories differ. [Coupling evidence](nonllm_health_coupling_corrected.md).

![Matched combat selection scores](assets/nonllm/combat_development.png)

This full-control development snapshot ends at round18. [Chart data](assets/nonllm/combat_development.csv). Scores include floor/HP terms and are not win percentages.

## 2026-10-11 22:25 UTC — later-act evidence chooses tighter clipping

Ten combat controls tested learning rate, clipping, entropy, discount, rewards, HP preservation and scratch learning. Expanded later-act evidence selected **clip0.1, LR1e-4, gamma1, entropy0.01**. Act3 development/final fight panels cover **15/12 worlds**. Final data stays embargoed.

## 2026-10-11 21:28 UTC — laptop recovery and deadline extension

Completed checkpoints survived laptop closure. Power was reconnected and Sid extended the deadline to **06:30 UTC**. Recovery and failed invocations remain in the detailed log.

## 2026-10-11 20:50 UTC — architecture and execution choices

Six size/context screens selected a **2.3M candidate MLP**, not an LLM. Public legal candidates are scored without positional/seed leakage. Use **eight persistent CPU simulator workers, CPU rollout inference and MPS updates**. Actual leading-model collection measured **0.96s CPU / 1.39s MPS**. Replay and phase-dispatch optimizations preserved checked semantics.

## 2026-10-11 20:08 UTC — scope

Created `codex/nonllm-curriculum` from immutable `env-v0.1.0`, preserving the original work. Curriculum: battles → OOC with native combat → full-control Act1–3. Scope: **Ironclad, ascension0**. This first window is not a claim of game mastery or Act4 competence.
