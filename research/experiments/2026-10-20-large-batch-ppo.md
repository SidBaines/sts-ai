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
- **Overnight additions (20 October, 23:00), queued after arm 4:**
  - **Arm 5, round 3** (`large_batch_20261020/round3_student`): arm 3's
    settings from a new start, the new champion's out-of-combat head with
    the old-encoder combat student spliced in as the frozen combat head
    (`inputs/champion_ba2fa6a2987d_student_combat.pt`, SHA-256 prefix
    `9eddab12139f`; `combine_heads.py`; it plays bit-identically to the
    champion with the student as a replacement on 8 checked worlds). Its
    baselines are that start's games: 316 wins on the second-DEV worlds and
    its fresh-world run.
  - **Arm 6, batch-size control** (`large_batch_20261020/small_hidden`):
    arm 3 with 768 games per iteration for 96 iterations, the same number
    of games as 24 iterations of 3,072, from the previous champion. Against
    arm 3 it isolates batch size at equal compute.
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
- Training behaviour: the smith rate fell from 0.28 to 0.20 by iteration 9
  and then rose to 0.41 by iteration 24, the one clear behaviour change of
  the run; the card skip rate stayed near 0.02.

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

**Arm 3 (the recipe with the potion filter) wins 208 of 4,096 on the
second-DEV worlds: more than the previous champion's 156, fewer than arm
1's 276, with no clear difference in mean floor from arm 1.** Final
checkpoint, iteration 24 (also the best by the trainer's evaluation),
paired:

| Baseline, worlds | Wins | Discordant, baseline only / arm 3 only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Previous champion, second DEV | 156 → 208 (3.8% → 5.1%) | 101 / 153 | 0.001 | +0.98 [+0.66, +1.30] | +4.6 points [+2.5, +6.6] (2,701 worlds) |
| Arm 1, second DEV | 276 → 208 (6.7% → 5.1%) | 207 / 139 | < 0.001 | −0.13 [−0.48, +0.22] | −1.3 points [−3.6, +1.0] (2,724 worlds) |
| Previous champion, fresh worlds | 152 → 197 (3.7% → 4.8%) | 104 / 149 | 0.006 | +1.13 [+0.81, +1.45] | +4.7 points [+2.6, +6.7] (2,692 worlds) |
| Arm 1, fresh worlds | 232 → 197 (5.7% → 4.8%) | 178 / 143 | 0.058 | −0.11 [−0.45, +0.23] | −0.4 points [−2.5, +1.8] (2,719 worlds) |

- The filter fixed the monitoring: the trainer's own evaluation, now under
  the study protocol, went 35, 32, 46, 31, 44, 38, 49 wins of 1,024 at
  iterations 0 to 24 (3.4% to 4.8%), in line with the 5.1% the final
  checkpoint scores on 4,096 worlds; arm 1's monitoring had been an order
  of magnitude off.
- Health signals matched arm 1: half-batch cosines −0.40 to +0.32 with no
  trend, KL 0.004 to 0.019, clip fraction 0.02 to 0.05, no faults or
  collector restarts. Entropy was lower throughout (0.13 to 0.16 against
  arm 1's 0.16 to 0.19) and fell over the run. Collection wins rose from
  109 to about 125 of 3,072 (arm 1, with potions visible: 44 to 103).
  Median 6.1 minutes per iteration; one iteration took 153 minutes during
  a battery pause.
- Training behaviour: the smith rate went 0.31, 0.22 (iteration 4), 0.34
  (iteration 12), 0.29 (iteration 24), where arm 1 ended at 0.41; the card
  skip rate rose from 0.023 to 0.030.
- Reading, with care: each arm is one run, and arm 1's own checkpoints
  swing between 224 (iteration 20) and 276 (iteration 24) wins, so the
  68-win gap on DEV and the borderline gap on fresh worlds may be
  checkpoint noise rather than an effect of the filter. What is
  established is that hiding the potion actions during training added
  nothing: it gave fewer wins at no clear difference in floor. The filter
  is a monitoring and deployment fix, not a training improvement.

**Arm 4 (arm 3 plus the hygiene options) wins 256 of 4,096 on the
second-DEV worlds, more than arm 3's 208, and is level with arm 1, the new
champion.** Final checkpoint, iteration 24 (also the best by the trainer's
evaluation), paired:

| Baseline, worlds | Wins | Discordant, baseline only / arm 4 only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Arm 3, second DEV | 208 → 256 (5.1% → 6.3%) | 137 / 185 | 0.009 | +0.21 [−0.12, +0.55] | +1.4 points [−0.6, +3.5] (2,717 worlds) |
| Arm 3, fresh worlds | 197 → 228 (4.8% → 5.6%) | 139 / 170 | 0.088 | −0.08 [−0.43, +0.26] | +1.3 points [−0.9, +3.5] (2,703 worlds) |
| Arm 1 (new champion), second DEV | 276 → 256 | 206 / 186 | 0.34 | +0.08 [−0.27, +0.42] | +0.3 points [−1.9, +2.5] (2,701 worlds) |
| Arm 1 (new champion), fresh worlds | 232 → 228 | 177 / 173 | 0.87 | −0.19 [−0.54, +0.16] | +2.0 points [−0.2, +4.2] (2,687 worlds) |
| Previous champion, second DEV | 156 → 256 | 101 / 201 | < 0.001 | +1.19 [+0.86, +1.51] | +5.0 points [+2.8, +7.1] (2,667 worlds) |
| Previous champion, fresh worlds | 152 → 228 | 97 / 173 | < 0.001 | +1.05 [+0.73, +1.38] | +6.1 points [+3.9, +8.2] (2,637 worlds) |

- The entropy bonus did what it was for: out-of-combat entropy rose from
  0.153 to 0.192 over the run, where arm 3, without it, fell from 0.152 to
  0.129. Wins in the sampled training games rose from 109 to 188 of 3,072
  (arm 3: 109 to about 125), a larger gain than greedy play shows.
- The trainer's own evaluation (study protocol) went 35, 32, 37, 34, 45,
  47, 52 wins of 1,024 at iterations 0 to 24, still rising at the end.
- Health: half-batch cosines −0.37 to +0.71, KL 0.005 to 0.012, clip
  fraction 0.04 to 0.05, no faults or collector restarts; value loss is on
  normalised targets and not comparable with arms 1 and 3. Median 6.1
  minutes per iteration, 2 hours 25 minutes in all.
- Training behaviour: the smith rate went 0.31, 0.42 (iteration 4), 0.28
  (iteration 9), 0.33 (iteration 24); the card skip rate 0.023 to 0.020.
- Reading: the hygiene options recover the wins that arm 3 lacked, clearly
  on the DEV worlds and in the same direction on the fresh worlds, at no
  clear difference in floor. They do not beat the new champion, which was
  trained with potions visible and no hygiene options. One run per arm.

**Arm 5, round 3, wins 413 of 4,096 on the second-DEV worlds (10.1%) and
394 on the fresh worlds (9.6%), the best pure policy so far.** It starts
from the new champion with the old-encoder combat student spliced in (316
and 315 wins), trains its out-of-combat head for 24 iterations of 3,072
games with the potion filter, and uses no Demon Form rule. Final
checkpoint, iteration 24, paired:

| Baseline, worlds | Wins | Discordant, baseline only / arm 5 only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Its start, second DEV | 316 → 413 (7.7% → 10.1%) | 153 / 250 | < 0.001 | +0.71 [+0.39, +1.05] | +4.0 points [+2.0, +5.9] (3,050 worlds) |
| Its start, fresh worlds | 315 → 394 (7.7% → 9.6%) | 192 / 271 | < 0.001 | +1.18 [+0.86, +1.50] | +5.6 points [+3.7, +7.5] (3,104 worlds) |
| Harvest stack (start plus the Demon Form rule), second DEV | 333 → 413 | 169 / 249 | < 0.001 | +0.52 [+0.20, +0.85] | +2.2 points [+0.3, +4.1] (3,037 worlds) |
| Harvest stack, fresh worlds | 346 → 394 | 218 / 266 | 0.033 | +1.03 [+0.70, +1.35] | +3.9 points [+2.0, +5.8] (3,091 worlds) |
| New champion, second DEV | 276 → 413 | 175 / 312 | < 0.001 | +1.82 [+1.45, +2.19] | +6.7 points [+4.5, +8.9] (2,861 worlds) |
| New champion, fresh worlds | 232 → 394 | 139 / 301 | < 0.001 | +2.12 [+1.76, +2.49] | +7.4 points [+5.2, +9.6] (2,907 worlds) |
| Previous champion, second DEV | 156 → 413 | 98 / 355 | < 0.001 | +2.93 [+2.56, +3.30] | +11.5 points [+9.2, +13.8] (2,712 worlds) |
| Previous champion, fresh worlds | 152 → 394 | 99 / 341 | < 0.001 | +3.36 [+2.99, +3.74] | +11.1 points [+8.9, +13.3] (2,759 worlds) |

- Iteration 20, the best by the trainer's evaluation, wins 403 on the
  second-DEV worlds (against its start: discordant 167/254, p < 0.001;
  floor +0.76 [+0.44, +1.09]); it is a selection, so the final checkpoint
  is the candidate.
- The trainer's own evaluation went 67, 66, 74, 67, 78, 85, 83 wins of
  1,024 at iterations 0 to 24, mean floor 36.7 to 37.6.
- Health: half-batch cosines −0.53 to +0.40, KL 0.005 to 0.010, clip
  fraction 0.03 to 0.04, no faults or collector restarts; entropy fell
  from 0.139 to 0.119, the lowest of any arm. Median 6.3 minutes per
  iteration, 2 hours 28 minutes in all.
- Training behaviour: the smith rate fell steadily from 0.49 to 0.38; the
  card skip rate stayed near 0.02. Wins in the sampled training games
  rose from 234 to 297 of 3,072.
- Reading: out-of-combat PPO gained more on top of the better combat head
  (+97 wins on DEV, +79 on fresh worlds) than any other arm did from the
  previous champion, and the trained policy beats the start plus the Demon
  Form rule without the rule. Whether the rule still adds on top of it has
  not been tested.

Arm 6: pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 21 Oct, 04:37: round 3's final checkpoint evaluated: 413 of 4,096 on the
  second-DEV worlds and 394 on the fresh worlds (start 316 and 315).
- 21 Oct, 04:21: round 3 finished; the batch-size control started.
- 21 Oct, 01:53: arm 4's final checkpoint evaluated: 256 of 4,096 on the
  second-DEV worlds and 228 on the fresh worlds; more wins than arm 3, no
  clear difference from arm 1.
- 21 Oct, 01:42: arm 4 finished; round 3 started.
- 20 Oct, 23:23: arm 3's final checkpoint evaluated: 208 of 4,096 on the
  second-DEV worlds and 197 on the fresh worlds, better than the previous
  champion but fewer wins than arm 1 with no clear difference in floor.
- 20 Oct, 23:09: arm 3 finished (2 hours 20 minutes paused on battery).
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
