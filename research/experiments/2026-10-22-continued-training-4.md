# Continued out-of-combat training, fourth pass, from the 22 October champion

**Status:** finished 23 October. **Verdict:** better, confirmed: 23.7% against the champion's 21.7% on unused worlds (971 against 887, p = 0.005).
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

| Worlds | Champion | Fourth continued training | Paired difference in wins | McNemar | Mean floor |
|---|---|---|---|---|---|
| Second DEV | 899 (21.9%) | 980 (23.9%) | +2.0 points [+0.5, +3.4] | p = 0.008 (412/493 discordant) | −0.06 [−0.40, +0.27] |
| Unused, from 104,000,000,000 | 887 (21.7%) | 971 (23.7%) | +2.1 points [+0.6, +3.5] | p = 0.005 (396/480) | +0.11 [−0.24, +0.45] |

- The trainer's best checkpoint (iteration 12) won 934 on second DEV (p =
  0.24): the final checkpoint is the one that improved.
- Act 2 boss pass +2.4 points [+0.7, +4.1] on the unused worlds.
- Training is deterministic: the joint run 2 trainer (same start, seed and
  settings) reproduced this run's first iteration exactly.

## Conclusion

A fourth pass of the recipe gained again, about 2 points of wins,
confirmed on unused worlds; smaller than the earlier passes (3.9 to 4.5
points), so gains may be starting to shrink. Caveat: one seed. Promotion is
Sid's decision.

## Next

- If promoted, a fifth pass, or the joint run 2 recipe if it wins.
- It is the control for [joint run 2](2026-10-22-joint-run-2.md).

## Log

- 23 Oct, 04:15: evaluated; better on both world sets.
- 22 Oct, 16:32: queued on the trainer lane after the joint run; a detached
  watcher (`eval_continued4.sh`) evaluates it when it finishes.

Code: `scripts/rl/train_ooc.py` from `nonllm-continued-frozen`. Data: run
`continued4_20261022` (local, worktree `nonllm-large-batch`).
