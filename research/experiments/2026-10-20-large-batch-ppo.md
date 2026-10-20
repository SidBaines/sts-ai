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
  see the same training worlds. Cancelled before it started: after the
  potion-protocol finding ([boss rares](2026-10-20-boss-rares.md)) it was
  replaced by the two arms below.
- **Arm 3, recipe with the potion filter** (`large_batch_20261020/recipe_hidden`):
  arm 1 with `--hide-ooc-potions`, so collection and the trainer's own
  evaluation use the deployment protocol (no potion drinking or discarding
  out of combat). Against arm 1 it measures what the filter alone does to
  training.
- **Arm 4, hygiene with the potion filter** (`large_batch_20261020/hygiene_hidden`):
  arm 3 plus the hygiene options of arm 2. Arms 3 and 4 run from a frozen
  checkout of `ws/nonllm` at `8aca4e4`, queued after arm 1 on the same lane;
  approved by Sid on 20 October ("queue the recipe and hygiene arms with
  --hide-ooc-potions after lb-recipe finishes").
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

**Arm 1 (recipe at 3,072 games) wins 276 of 4,096 against the champion's
156 under the study protocol.** Final checkpoint, iteration 24, on the
second-DEV worlds, paired:

| Checkpoint | Wins (rate) | Discordant, champion only / arm only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Champion | 156 (3.8%) | | | 35.08 | |
| Iteration 24 (final, pre-registered) | 276 (6.7%) | 109 / 229 | < 0.001 | +1.11 [+0.78, +1.44] | +5.2 points [+3.1, +7.3] (2,661 worlds) |
| Iteration 20 (best by the trainer's evaluation) | 224 (5.5%) | 109 / 177 | < 0.001 | +1.24 [+0.90, +1.57] | +5.3 points [+3.1, +7.4] (2,652 worlds) |

**Confirmed on fresh worlds.** On 4,096 worlds from seed 94,000,000,000,
never used before, played by both policies:

| Policy | Wins (rate) | Discordant, champion only / arm only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Champion | 152 (3.7%) | | | 35.09 | |
| Iteration 24 | 232 (5.7%) | 106 / 186 | < 0.001 | +1.24 [+0.91, +1.58] | +5.1 points [+3.0, +7.3] (2,670 worlds) |

The win gain on fresh worlds (+2.0 points) is smaller than on the DEV
worlds (+2.9), as expected from noise in a single 4,096-world estimate; the
floor and boss-pass gains are the same size on both.

- Run facts: 24 iterations of 3,072 games (about 259,000 out-of-combat
  choice transitions and 192 optimizer steps each), 8 to 19 minutes per
  iteration beside evaluations, no faults or collector restarts.
- The trainer's own evaluation (1,024 worlds from seed 91,000,000,000, on
  the potions-visible protocol) went 17, 13, 9, 9, 24, 31, 25 wins at
  iterations 0 to 24 and mean floor 30.8 to 32.2: it understated the gain
  by an order of magnitude and would have called the run a failure at
  iteration 12.
- Half-batch gradient cosines ranged from −0.43 to +0.54 with no trend;
  entropy stayed near 0.18; KL per iteration 0.006 to 0.012.
- Training behaviour (sampled): smith rate 0.25 to 0.28 and card skip rate
  0.02 throughout.

**On top of the new champion, the day's other gains stack** (second-DEV
worlds, paired with its 276 wins; Planned): the Demon Form rule 292 (p =
0.076, floor +0.20 [+0.09, +0.31]); the old-encoder combat student 316
(p = 0.050, floor +1.11 [+0.76, +1.46]); both 333 (8.1%, p = 0.006, floor
+1.29 [+0.95, +1.65]). Confirmed on the fresh worlds from 94,000,000,000:
both together 346 against the new champion's 232 (8.4% against 5.7%, p <
0.001, floor +1.09 [+0.74, +1.44]) and the previous champion's 152.

**The potion habit survived.** With out-of-combat potion actions visible
(the `eval_policy.py` protocol), the new champion wins 143 of 4,096 against
the previous champion's 70 (floor +1.17 [+0.84, +1.51]), about half its
study-protocol rate: 24 iterations with the actions visible improved play
in general but did not teach it to keep its potions. In 64 games with the
actions offered it discarded 588 potions and drank 31 out of combat,
against the previous champion's 565 and 32: the habit did not move at all.

Arms 3 and 4: pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 20 Oct, 16:00: confirmed on the fresh worlds: 232 against 152 wins.
- 20 Oct, 15:49: confirmation queued on 4,096 unused worlds from seed
  94,000,000,000, with the champion on the same worlds. Two earlier
  attempts were abandoned: the reserved set from 91,200,000,000 is
  embargoed in code and Sid chose to keep it unopened, and the
  93,000,000,000 namespace turned out to be reserved for probe states (its
  partial output under `confirm93/` is void).
- 20 Oct, 15:32: the recipe arm's final checkpoint wins 276 of 4,096 under
  the study protocol.
- 20 Oct, 15:20: the recipe arm finished.
- 20 Oct, 13:25: arms 3 and 4 queued after the recipe arm.
- 20 Oct, 12:14: queued both arms; the hygiene arm cancelled at 12:16 after
  the potion-protocol finding, to be re-queued with the potion filter.
- 20 Oct, 12:56: iteration 4 of the recipe arm evaluates at 13 wins of 1,024
  (17 at the start), mean floor 31.3 (30.8); cosines −0.43 and +0.48 at
  iterations 3 and 4.
- 20 Oct, 12:34: iterations 1 and 2 of the recipe arm: 3,072 games, about
  259,000 out-of-combat choice transitions, 192 optimizer steps and 9
  minutes per iteration beside evaluations; half-batch gradient cosines
  −0.12 and −0.17; entropy 0.19; KL 0.012 and 0.011.

Code: `scripts/rl/train_ooc.py` (library `src/sts_rl/ooc_trainer.py`).
Data: run `large_batch_20261020` (local).
