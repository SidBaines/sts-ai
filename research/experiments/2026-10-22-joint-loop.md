# Joint loop: out-of-combat PPO and teacher-anchored combat, moving together

**Status:** queued (starts when the third continued training frees the trainer, about 16:25). **Verdict:** pending.

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

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 22 Oct, 13:43: smoke test passed; joint run queued behind the control;
  a detached script starts the orchestrator with the trainer and runs the
  final comparison when both finish.
- 22 Oct, 12:44: design drafted; Codex building `--combat-follow` for the
  trainer.

Code: `src/sts_rl/ooc_trainer.py` (`combat_follow`), `scripts/rl/fresh_collect.py`,
`scripts/rl/fresh_train.py`; orchestration in
`experiments/2026-10-22-joint-loop/` (to come).
