# Joint loop: out-of-combat PPO and teacher-anchored combat, moving together

**Status:** finished 22 October. **Verdict:** better than its control under the rule fixed in advance (945 against 877 of 4,096 unused worlds, p = 0.029), though the loop managed only one combat refit; the gain splits into a combat-refit effect and a borderline co-adaptation effect.

## Question

So far the two halves of the agent improve in turn: out-of-combat PPO with
the combat head frozen, then a combat student refitted to search-teacher
labels and spliced in. Alternation is coordinate ascent and can miss
improvements that need both halves to move at once (a deck that only pays
off once combat learns to play it). Does a loop in which both move every
few iterations, on the same games, beat alternation at equal wall-clock?
This is a first route to joint training, not the only one (Sid, 22
October). Plain PPO on the combat head is not the route: it failed in
October ([combat PPO](2026-10-13-combat-ppo.md)) and would erode the
teacher-distilled combat head.

## Design

- **Joint run** (`joint_20261022`): out-of-combat PPO from the champion
  (`champion_85ea9019ff95.pt`) with the recipe of the continued trainings
  (24 iterations of 3,072 games, potion filter, hygiene options,
  exploration statistics), training seed 4, checkpoints every 2
  iterations, `--combat-follow follow_combat.pt` (starting as the
  champion's own combat student).
- **Orchestrator** (`experiments/2026-10-22-joint-loop/joint_loop.py`), in
  cycles of about two hours: (1) the trainer's newest checkpoint plays
  1,000 new training worlds from 90,500,500,000 + 10,000 per cycle and the
  search teacher labels its combat decisions (standard room
  probabilities); (2) two students (seeds 1 and 2) are fitted from scratch
  on every corpus collected so far plus the round 2 and round 3 corpora
  (`--mixed-sources`; the 15 October corpus is left out after it hurt the
  all-corpora fit); (3) gate: the newest checkpoint with each student
  spliced in, and as it is, play 2,048 gate worlds from 101,000,000,000;
  the better candidate is adopted only if it wins at least as many games
  and its mean floor is at least as high. The gate is needed because one
  fit in four collapsed ([combat variance](2026-10-22-combat-variance.md)).
- **Control:** the [third continued training](2026-10-22-continued-training-3.md),
  the same start and recipe with the combat head frozen (training seed 3).
- **Decision rule, fixed in advance:** the joint run's final checkpoint
  and the control's final checkpoint both play 4,096 unused worlds from
  103,000,000,000 (with the champion); the joint loop is *Better* if it
  wins more than the control with exact McNemar p < 0.05. If the gate
  never adopts a student, the run is a second out-of-combat run with a
  different seed, and the comparison only measures seed variation; say so.
- **Smoke test** (22 October, 24 worlds, one seed, 256 gate worlds from
  102,000,000,000): every step ran; the 24-world student (9 wins against
  54) was rejected.
- **Approval:** Sid, 22 October ("Building the joint loop as the main
  engineering item ... Let's do it"; "not locking in the exact proposal
  ... but a good first thing to start with").

## Results

**Main comparison (rule fixed in advance): the joint run beats its control.**
Final checkpoints on 4,096 unused worlds from 103,000,000,000, environment
0.1.5.1, study protocol, paired by world:

| Player | Wins | Paired difference vs control | Mean floor |
|---|---:|---|---|
| Control: third continued training (combat head = round 2 student) | 877 (21.4%) | | 40.59 |
| Joint run (combat head = cycle 0 student from iteration 10) | 945 (23.1%) | +1.7 points (discordant 436/504, p = 0.029) | +0.07 [−0.31, +0.45] |
| Starting point (champion of 22 October, midday) | 713 (17.4%) | −4.0 points (p < 0.001) | |

**What the loop actually did: one refit.** Cycle 0 labelled 1,000 worlds
from the starting player, fitted two students on those labels plus the
round 2 and round 3 corpora (306,436 rows), and adopted seed 2 (429 against
386 wins of 2,048 gate worlds, p = 0.028; the better of two seeds on reused
gate worlds, so optimistic). The training log shows the new combat head
from iteration 10. Cycle 1's fit stopped at start-up because free disk
space was below 6 GiB; it exited cleanly with status "interrupted", which
the orchestrator did not recognise, so it waited without acting until the
run ended. Iterations 10 to 24 therefore ran with one fixed new combat head.

**Decomposition (chosen after seeing the main result, so exploratory):** the
two final out-of-combat heads crossed with the two combat heads, same
worlds (wins of 4,096):

| | Round 2 combat head | Cycle 0 combat head |
|---|---:|---:|
| Control's out-of-combat head | 877 | 900 |
| Joint run's out-of-combat head | 847 | 945 |

- The new combat head helps on average: +1.5 points [+0.5, +2.5] averaged
  over the two out-of-combat heads.
- The out-of-combat heads differ little on average: +0.2 [−0.9, +1.2].
- Interaction, the joint-training hypothesis (an out-of-combat head trained
  alongside a combat head plays better with it): +1.8 points [0.0, +3.7].
  The joint run's head is worse than the control's with the old combat head
  (847 against 877, floor −0.47 [−0.82, −0.11]) and better with its own
  (945 against 900, p = 0.14): borderline evidence of co-adaptation.
- By boss: the joint run passes the Act 3 boss 44.6% of the times it gets
  there, against 41.8 to 42.6% for the other three combinations; the Act 1
  and Act 2 differences are within a point or two.

**Training curves (same 1,024 evaluation worlds, paired):** before the swap
(iteration 8) the two runs differ only by training seed: 186 against 175
(p = 0.47). After it the joint run led at every evaluation (211–192,
202–182, 209–186; pooled over iterations 12 to 24, +1.7 points [−0.1, +3.5])
but by only 208 to 200 at iteration 24. Collection win rates over
iterations 10 to 24 were equal (19.22% against 19.29%, 46,080 games each).
Seed alone moves training visibly: policy entropy rose from 0.16 to 0.19 in
the joint run and fell to 0.15 in the control from iteration 2, before any
swap. The joint run took 5.8 hours against 4.4, sharing the machine with
its labelling and with other jobs.

## Conclusion

The joint run is the strongest player so far on these worlds (23.1%),
and it beat its control under the rule fixed in advance. Most of the
evidence points to the refitted combat head plus an out-of-combat head
that adapted to it; the co-adaptation term is borderline (z = 1.96) and
was tested after the fact. This was a weak dose of joint training (one
swap), so it says little yet about frequent co-adaptation.

Caveats:
- One run per arm, with different training seeds; seed changes training
  visibly, and the out-of-combat comparison includes it.
- The adopted student was chosen as the better of two seeds on reused gate
  worlds.
- The decomposition was chosen after the main result.
- The orchestrator bug and the shared machine made the run unlike its
  design (one cycle instead of about three).

## Next

- A proper joint run: a robust orchestrator (treat any unfinished fit as a
  failure, check disk before each step, retry once, skip rather than hang,
  heartbeat), the same training seed and configuration as its control so
  the two differ only through the combat swaps, and enough iterations for
  several refits.
- Promotion of the joint run's final checkpoint is Sid's decision; it could
  first be confirmed on another unused world set.

## Log

- 22 Oct, 22:55: two-by-two evaluated; written up.
- 22 Oct, 22:43: final comparison: 945 against 877 (p = 0.029).
- 22 Oct, 18:37: cycle 1's fit stopped on low disk; the orchestrator did not
  notice (fixed in the next version).
- 22 Oct, 13:43: smoke test passed; joint run queued behind the control;
  a detached script starts the orchestrator with the trainer and runs the
  final comparison when both finish.
- 22 Oct, 12:44: design drafted; Codex building `--combat-follow` for the
  trainer.

Code: `src/sts_rl/ooc_trainer.py` (`combat_follow`), `scripts/rl/fresh_collect.py`,
`scripts/rl/fresh_train.py`; orchestration in
`experiments/2026-10-22-joint-loop/` at commit 067d3da (`nonllm-joint-frozen`). Data: run
`joint_20261022` (local, worktree `nonllm-large-batch`), including
`final103/` and `factorial/`.
