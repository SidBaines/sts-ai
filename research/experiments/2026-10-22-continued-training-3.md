# Continued out-of-combat training, third pass, from the combination

**Status:** running. **Verdict:** pending.

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

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 22 Oct, 11:37: queued; a detached watcher (`eval_continued3.sh`)
  evaluates it when it finishes.

Code: `scripts/rl/train_ooc.py` from `nonllm-continued-frozen`. Data: run
`continued3_20261022` (local, worktree `nonllm-large-batch`).
