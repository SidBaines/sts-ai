# Large-batch out-of-combat PPO: was gradient noise the limit?

**Status:** running. **Verdict:** pending.

## Question

The champion's out-of-combat PPO ran at 768 games per iteration with
half-batch gradient cosines of 0.09 to 0.21, and adjacent checkpoints
swung between 35 and 57 wins on the same worlds: the 13 October diagnosis
was that updates were a noise-driven random walk. Collection is now 3.2
times faster, so a four-times-larger batch costs what the old batch did.
Does continuing the champion's recipe at 3,072 games per iteration improve
it, and do the PPO hygiene options (detached critic, normalised returns, a
small entropy bonus on strategic menus) add anything on top? (Approach:
[out-of-combat PPO](../strategies.md).)

## Design

- **Start:** the champion checkpoint (round-2 out-of-combat PPO update 20),
  combat head frozen and played greedily, as in its own recipe.
- **Arm 1, large-batch recipe** (`large_batch_20261020/recipe`): the
  champion's recipe with 3,072 games per iteration instead of 768; learning
  rate 0.0003, 3 epochs, minibatch 4,096, clip 0.2, KL guard 0.05, macro
  credit, Monte-Carlo returns (lambda 1), weight decay 0 (the lineage had a
  silent 0.01), 24 iterations, training seed 0.
- **Arm 2, large-batch hygiene** (`large_batch_20261020/hygiene`): arm 1
  plus a detached critic (value loss reaches only the critic head),
  normalised value targets, and an entropy bonus of 0.01 on strategic menus
  (rest, shop, card reward, map and removal). Same seed, so the two arms
  see the same training worlds. Cancelled before it started, pending Sid's
  decision: it should run with `--hide-ooc-potions` (see
  [boss rares](2026-10-20-boss-rares.md)), as should a second recipe arm, so
  that training matches the deployment protocol.
- **Collection:** 12 in-process collectors of 32 games each
  (`--in-process-games`), the first use of in-process collection in
  training; the games are identical in distribution to the subprocess path.
- **Monitoring:** the trainer's own evaluation every 4 iterations on 1,024
  DEV worlds from seed 91,000,000,000, under the per-decision evaluator
  protocol (potions visible out of combat), with the champion's own score
  at iteration 0 as the reference. Health signals per iteration: half-batch
  gradient cosine, KL, clip fraction, value loss, displacement ratios.
- **Decision, fixed in advance:** the final checkpoint of each arm, and the
  best by the trainer's evaluation, are evaluated with `fast_eval.py` on
  the 4,096 second-DEV worlds under the study protocol, paired with the
  champion (156 wins). An arm is *Better* if it wins more with exact McNemar
  p < 0.05; the best-by-evaluation checkpoint is a selection and must be
  confirmed on the confirmation worlds before promotion.
- **Budget:** about 2 hours per arm on the laptop (estimated 5 minutes per
  iteration), run one after the other on the queue's `train` lane alongside
  the day's evaluations.
- **Evidence level:** Planned for the final checkpoints; Exploratory for
  any checkpoint chosen by its evaluation.
- **Approval:** Sid, 20 October 2026 ("could we bump 4a up to being kicked
  off now").

## Results

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 20 Oct, 12:14: queued both arms; the hygiene arm cancelled at 12:16 after
  the potion-protocol finding, to be re-queued with the potion filter.
- 20 Oct, 12:34: iterations 1 and 2 of the recipe arm: 3,072 games, about
  259,000 out-of-combat choice transitions, 192 optimizer steps and 9
  minutes per iteration beside evaluations; half-batch gradient cosines
  −0.12 and −0.17; entropy 0.19; KL 0.012 and 0.011.

Code: `scripts/rl/train_ooc.py` (library `src/sts_rl/ooc_trainer.py`).
Data: run `large_batch_20261020` (local).
