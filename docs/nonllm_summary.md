# Non-LLM Slay the Spire — experiment summary

Latest updates come first. This is the short, manually maintained account of
state and decisions. [Detailed timeline](nonllm_progress.md) ·
[Decision/recovery notes](nonllm_progress_notes.md) · [Experiment plan](nonllm_curriculum_plan.md).

## 2026-10-11 21:41 UTC — architecture comparisons running; execution path verified

**State:** six size/context policies are trained; learning-rate and initialization
comparisons are running. The current best pilot controls combat successfully on
the development panel. OOC learning, joint full-run training and final evaluation
remain ahead. No final-world results have been used for selection.

| Policy on the same 48 development fights / 21 worlds | Wins | Surfaced faults / truncations |
| --- | ---: | ---: |
| Random | 28/48 | 0/0 |
| Simple public-state heuristic | 42/48 | 0/0 |
| Privileged native search, 200 simulations | 44/48 | 0/0 |
| Best learned pilot: large candidate MLP, LR 1e-4 | 45/48 | 0/0 |

This is a small panel repeatedly used for selection, so one extra win is not
evidence that the learner is better than native search. Remaining comparisons
and the frozen final panel will decide what generalizes.

**Decision:** use eight persistent simulator workers, CPU rollout inference and
MPS training updates. The real leading-model benchmark supports this choice:

| Matched measurement | Slower path | Selected path | Evidence |
| --- | ---: | ---: | --- |
| Warm 48-fight collection, 687 decisions | MPS: 1.39s | CPU: 0.96s | Same win and decision counts |
| Twenty later-act prefix resets | Standard: 4.12s | Fast replay: 1.58s | Observations, labels, rewards, diagnostics and fault rejection matched |
| Eight-worker native label collection | Uncached: 3,638/s | Cached: 5,002/s | Same labels and surfaced fault |

GPU updates and saved AdamW-moment restoration passed live checks. Compatible
PPO chunks retain optimizer state; changing method/reward/task resets it with a
logged reason. Tests and the detailed log cover recovery failures as well.

## 2026-10-11 21:28 UTC — recovered after laptop closure; deadline extended

All six architecture checkpoints survived the interruption. Completed jobs were
recovered from saved artifacts and reused. AC power is connected. Sid explicitly
extended the deadline to **12 October, 06:30 UTC**; training and final evaluation
must finish within it. The detailed log records the original deadline and recovery.

## 2026-10-11 20:50 UTC — first combat policies and later-stage data ready

The first small policies each won 42/48 development fights, providing a usable
starting point for combat improvement. Added a separate later-act bank and OOC
lookahead labels so the curriculum can progress beyond starter fights.

| Dataset | Valid worlds / attempted | Teacher decisions | Replayable fights |
| --- | ---: | ---: | ---: |
| Act 1 bank | 252/256 | 49,944 | 2,245 |
| Act 3 enrichment | 246/256 | 68,520 | 2,983 |

The enrichment includes 718 Act 2 and 123 Act 3 fights. Act 3 boss coverage is
thin: seven training starts, one development, zero final. Full-run testing is
essential; this bank cannot establish boss generalization.

The separate train-only OOC teacher evaluates four short privileged continuations
at each of 256 frontiers. Its proxy strictly improves over scripted continuation
on 89 frontiers. We will compare actual held-out outcomes before promoting it.
Auto-native combat currently loses a backend UB flag when its temporary battle
state is discarded; zero surfaced hybrid faults does not certify fault-free
playouts. This limitation is recorded and the frozen binary remains unchanged.

## 2026-10-11 20:08 UTC — scope and experiment design

Created `codex/nonllm-curriculum` from immutable release `env-v0.1.0`, preserving
the original experiments. The plan is combat imitation/DAgger/PPO → separate
OOC improvement with native combat → full-control traces and Act 1-to-3 expansion.
Compare shared and specialized policies; reserve final evaluation time.

The policy scores public legal-action candidates rather than treating changing
menu indices as permanent action meanings. It uses no LLM, hidden game seed or
native action bits. Native teacher labels are explicitly privileged. World-level
train/dev/final splits prevent fights from the same world leaking across sets;
simulator faults are quarantined separately from ordinary defeats.
