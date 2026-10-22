# Continued out-of-combat training, third pass, from the combination

**Status:** finished 22 October. **Verdict:** better, confirmed: 21.1% against the combination's 17.3% on unused worlds (+3.9 points), gained mostly at the Act 3 boss.

## Question

Each continued-training round has gained 2 to 4.5 points of win rate on
unused worlds, and the trainer's evaluation was still rising at the end of
the second. The best policy is now the second round's out-of-combat head
with the round 2 combat student (18.4% on unused worlds). Does another
24 iterations, with that combat head, keep improving it?

## Design

- **Start:** the combination (`combination_85ea9019ff95.pt`, in run
  `continued3_20261022/inputs`), combat head frozen and played greedily.
- **Training:** as the first and second continued trainings (24
  iterations of 3,072 games, lambda 1, weight decay 0, learning rate
  0.0003, potion filter, detached critic, normalised value targets,
  entropy bonus 0.01 on strategic menus, exploration statistics), training
  seed 3 (new training worlds), from the frozen checkout `ws/nonllm`
  a33bd2b.
- **Evaluation:** final and best-by-trainer checkpoints on second DEV,
  paired with the combination's 701; the final checkpoint on 4,096 unused
  worlds from 100,000,000,000, paired with the combination's own games
  there.
- **Decision rule, fixed in advance:** *Better* if the final checkpoint
  wins more than the combination on second DEV with exact McNemar p <
  0.05; confirmed if it also wins more on the worlds from 100,000,000,000
  with p < 0.05.
- **Approval:** Sid's standing instruction of 21 October to keep doing
  research until stopped. The combination's promotion is Sid's decision;
  it is used here only as a starting point.

## Results

The final checkpoint (iteration 24, also the trainer's best) beats the
combination on both world sets (environment 0.1.5.1, study protocol, 4,096
worlds each, paired):

| Worlds | Combination | Third continued training | Paired difference in win rate | McNemar | Mean floor |
|---|---|---|---|---|---|
| Second DEV | 701 (17.1%) | 899 (21.9%) | +4.8 points [+3.5, +6.2] | p < 0.001 (317/515 discordant) | +0.62 [+0.29, +0.95] |
| Unused, from 100,000,000,000 | 707 (17.3%) | 865 (21.1%) | +3.9 points [+2.5, +5.2] | p < 0.001 (311/469) | +0.18 [−0.16, +0.53] |

- **The gain is at the end of the run.** Act 3 boss pass rate, among runs
  that reach it, rose from 35.5% to 42.0% on second DEV and from 35.4% to
  42.1% on the unused worlds; Act 3 boss reach rose by 1 to 4 points; Act 1
  and Act 2 changed little (Act 2 boss pass +3.7 points [+2.0, +5.4] on
  DEV, +1.1 [−0.6, +2.8] on the unused worlds). The per-act rates are
  descriptive, conditional on reaching the boss.
- During training, the trainer's own evaluation (1,024 worlds) rose from 168
  to 192 wins by iteration 12 while its mean floor stayed near 39.7, which
  already pointed to more wins rather than deeper runs.

## Conclusion

A third round of the same out-of-combat PPO recipe gained as much as the
earlier rounds: about 4 points of wins, confirmed on unused worlds. Its
mean floor barely moved on those worlds; the gain is in converting runs
that reach Act 3 into wins, presumably through deck-building and path
choices that prepare for the Act 3 boss (not yet analysed). The recipe has
not plateaued after 72 iterations across three rounds.

Caveats:
- One training seed; earlier rounds also gained, so the direction is not in
  doubt, but the size may vary by seed.
- The per-act rates condition on reaching each boss and are not paired
  tests.
- Promotion to champion is Sid's decision.

## Next

- If promoted, continue the same recipe from this checkpoint (a fourth
  round), since it has not plateaued.
- It is the control for the [joint loop](2026-10-22-joint-loop.md), which
  started at 16:11 from the combination.
- Find out what changed in its Act 3 preparation (cards taken, upgrades,
  potions held at the Act 3 boss).

## Log

- 22 Oct, 11:37: queued; a detached watcher (`eval_continued3.sh`)
  evaluates it when it finishes.
- 22 Oct, 16:25: finished; 899 against 701 on second DEV and 865 against
  707 on the unused worlds.

Code: `scripts/rl/train_ooc.py` from `nonllm-continued-frozen`. Data: run
`continued3_20261022` (local, worktree `nonllm-large-batch`).
