# Continued out-of-combat training, fourth pass, from the 22 October champion

**Status:** queued (starts when the joint run frees the trainer, expected
late evening on 22 October). **Verdict:** pending.

## Question

Three passes of the same out-of-combat PPO recipe have each gained about 2
to 4.5 points of wins on unused worlds; the third reached 21.1%, mostly by
winning more Act 3 boss fights, with no sign of a plateau. Does a fourth
pass from it keep improving? Sid asked for it on 22 October, when he
promoted the third pass.

## Design

- **Start:** the champion (`champion_af2fb9f1f260.pt`, in run
  `continued4_20261022/inputs`), combat head frozen and played greedily.
- **Training:** as the earlier continued trainings (24 iterations of 3,072
  games, lambda 1, weight decay 0, learning rate 0.0003, potion filter,
  detached critic, normalised value targets, entropy bonus 0.01 on
  strategic menus, exploration statistics), training seed 5 (new training
  worlds), from the frozen checkout `ws/nonllm` a33bd2b.
- **Evaluation:** final and best-by-trainer checkpoints on second DEV,
  paired with the champion's 899; the final checkpoint on 4,096 unused
  worlds from 104,000,000,000, paired with the champion's own games there
  (played now, before training).
- **Decision rule, fixed in advance:** *Better* if the final checkpoint
  wins more than the champion on second DEV with exact McNemar p < 0.05;
  confirmed if it also wins more on the worlds from 104,000,000,000 with p
  < 0.05. Promotion is Sid's decision.
- **Approval:** Sid, 22 October, 16:32.

## Results

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 22 Oct, 16:32: queued on the trainer lane after the joint run; a detached
  watcher (`eval_continued4.sh`) evaluates it when it finishes.

Code: `scripts/rl/train_ooc.py` from `nonllm-continued-frozen`. Data: run
`continued4_20261022` (local, worktree `nonllm-large-batch`).
