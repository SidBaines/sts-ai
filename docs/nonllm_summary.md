# Non-LLM Slay the Spire — experiment summary

Latest milestones first. [Detailed log](nonllm_progress.md) ·
[Decision/recovery notes](nonllm_progress_notes.md) · [Plan](nonllm_curriculum_plan.md).
Deadline: **12 October, 06:30 UTC**. All results below are development results;
final evaluation remains embargoed.

## 2026-10-11 23:25 UTC — health bonuses tested; main combat training continues

The learned combat policy loses materially more HP than native search. Two
matched three-minute reward pilots improved isolated battles, but reduced
full Act 1 clears with the same OOC policy:

| Terminal HP bonus | Battle wins / 80 | Mean battle HP cost | Act 1 clears / 24 |
| --- | ---: | ---: | ---: |
| Reference: none | 70 | 14.91 | 5 |
| 0.25 | 72 | 13.66 | 2 |
| 1.0 | 72 | 14.06 | 3 |

**Decision:** retain the reference under the prespecified selection rule. Keep
both alternatives for later coupling tests; this small panel does not prove
health bonuses are generally harmful. [Study detail](nonllm_health_progress.md).

Training now avoids duplicate inference by reusing the next actor value. Paired
tests preserved trajectories/targets within 7.5e-9 and measured **1.35–1.37×**
faster collection. Direct sticky/live UB checks remove unnecessary replay
deepcopies: **1.613s → 0.915s** over twenty later-act resets. Fault detection,
observations and rewards matched; the native build is unchanged.

## 2026-10-11 22:25 UTC — later-act evidence chooses tighter clipping

Ten combat controls covered learning rate, clipping, entropy, discount,
sparse/potential rewards, HP preservation and scratch learning. Expanded
later-act evidence changed the winner:

| Promoted PPO recipe | Act 1 wins / 64 | Later-act wins / 32 |
| --- | ---: | ---: |
| LR 3e-4 | 59 | 24 |
| Gamma 0.99 | 58 | 26 |
| Clip 0.1 | 55 | 29 |

**Decision:** clip 0.1, LR 1e-4, gamma 1, entropy 0.01, potential shaping.
Matched later-act references: native **30/32**, public heuristic **28/32**,
random **13/32**. The learner's win-rate advantage is uncertain, and its
world-mean HP cost is **11.02 higher than native** (95% paired CI 6.52–16.40).

A CLI flag initially skipped scratch PPO; it was repaired and rerun on the
original matched cohorts. Scratch won **54/64** Act 1 and **20/26** later-act
fights after ninety seconds. It is a useful baseline, not a failed approach.

## 2026-10-11 22:06 UTC — expanded data and fixed comparison cohorts

Added 2,048 native-teacher worlds: **560,620 labels / 24,276 fights**, with
23 flagged worlds quarantined. Act 3 development/final panels now cover
**15/12 worlds**, replacing the earlier single-world limitation. Comparisons
keep fixed cohorts; promoted policies are reevaluated after a cohort change.

Teacher budgets 50/200/1,000 won **42/44/43 of 48** matched fights. Retain
200: mean query time **1.06ms**, versus **4.81ms** at 1,000. Larger search
improved episode-weighted HP, but not world-weighted HP or wins on this panel.

Corrective learning mixes a fixed 20k native rehearsal reservoir with all prior
learner-state labels. OOC teacher comparisons use identical frontiers, initial
weights and time; composite deployment is assessed before final selection.

## 2026-10-11 21:28 UTC — laptop recovery and extended deadline

Six trained architecture checkpoints survived laptop closure. Saved completed
jobs were reused, power was reconnected, and Sid extended the deadline to
**06:30 UTC**. Recovery and failed commands remain in the detailed log.

## 2026-10-11 20:50 UTC — first policies and execution measurements

Initial small models won **42/48** development fights; size/LR selection chose
a large candidate MLP at LR 1e-4: **45/48**, with two seeds each **44/48**.
Use eight persistent CPU simulator workers, CPU rollout inference and MPS
updates. The leading-model warm collection benchmark was **0.96s CPU versus
1.39s MPS**. Phase-routed composite evaluation is **1.57–2.20×** faster.

## 2026-10-11 20:08 UTC — scope and plan

Created `codex/nonllm-curriculum` from immutable `env-v0.1.0`, preserving the
original experiments. Curriculum: isolated combat → OOC with native combat →
full-control Act 1-to-3 traces. Compare shared/specialized policies and reserve
final evaluation time. Inputs are public legal-action candidates and game state;
teacher labels are privileged, and worlds remain in one train/dev/final split.

Known limitation: auto-native combat discards a temporary UB flag. Zero surfaced
hybrid faults cannot certify clean native playouts; explicit full-control faults
are quarantined. The frozen native build has not been silently changed.
