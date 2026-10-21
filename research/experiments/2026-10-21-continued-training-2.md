# Continued out-of-combat training, second pass, with exploration statistics

**Status:** running. **Verdict:** pending.

## Question

The [first continued-training run](2026-10-21-round-4.md) raised the
champion from 9.3% to 11.6% on new worlds and its own evaluation was still
rising. Does another 24 iterations keep improving it, and what do the new
exploration statistics show about saturated menus (for example, the policy
has never bought a shop card removal)? Separately: does the Demon Form rule
still add anything on the new champion? (Approach:
[out-of-combat PPO](../strategies.md).)

## Design

- **Start:** the champion of 21 October, afternoon (the first continued
  training's final checkpoint, `champion_2835f40fa439.pt`), combat head
  frozen and played greedily.
- **Training** (`continued2_20261021/train`): the same settings as the
  first continued training (24 iterations of 3,072 games, lambda 1, weight
  decay 0, learning rate 0.0003, potion filter, detached critic, normalised
  value targets, entropy bonus 0.01 on strategic menus), training seed 2
  (new training worlds), from a frozen checkout of `ws/nonllm` at a33bd2b,
  which records the exploration statistics every iteration.
- **Demon Form retest:** the champion with the rule as a menu overlay, on
  second DEV (paired with the champion's 527) and on the worlds from
  95,000,000,000 (paired with its 476).
- **Evaluation:** `fast_eval.py`, study protocol, 8 processes of 32 games:
  final and best-by-trainer checkpoints on second DEV, paired with the
  champion's 527; final checkpoint on 4,096 unused worlds from
  96,000,000,000, paired with the champion's own games there.
- **Decision rules, fixed in advance:** the trained checkpoint, or the
  Demon Form overlay, is *Better* if it wins more than the champion on
  second DEV with exact McNemar p < 0.05; confirmed if it also wins more
  with p < 0.05 on the new worlds. A confirmed Demon Form gain is distilled.
- **Approval:** Sid, 21 October ("make the continued-training final
  checkpoint the new champion, then retest Demon Form on it and start the
  next round of training with the exploration statistics").

## Results

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 21 Oct, 15:40: champion promoted; training and evaluations queued.

Code: `scripts/rl/train_ooc.py` from a frozen checkout of `ws/nonllm` at
a33bd2b (worktree `nonllm-continued-frozen`). Data: run `continued2_20261021`
(local, worktree `nonllm-large-batch`).
