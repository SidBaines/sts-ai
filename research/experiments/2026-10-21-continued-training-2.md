# Continued out-of-combat training, second pass, with exploration statistics

**Status:** finished 22 October 2026. **Verdict:** a second 24 iterations raised the champion from 12.4% to 16.9% on unused worlds; with the round 2 combat student spliced in it reaches 701 of 4,096 on DEV (17.1%). The Demon Form rule, distilled correctly, matches its overlay but adds little.

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
- **Demon Form distillation** (added after the retest, by Sid's decision
  although the retest missed confirmation at p = 0.054): the champion plays
  6,144 training worlds from 8,500,000,000 under the study protocol,
  following "take Demon Form whenever offered"; every card reward offering
  Demon Form is a label, and 2% of other out-of-combat decisions are kept
  as rehearsal (a divergence penalty of weight 1 keeps those choices
  unchanged). Distilled into the out-of-combat head for 12 epochs at
  learning rate 0.0003, batch 256, as for the smithing rule on 13 October;
  combat head frozen. The distilled checkpoint is evaluated on second DEV
  and the worlds from 95,000,000,000, paired with the champion and with the
  champion plus the overlay; it should match the overlay and beat the
  champion.
- **Approval:** Sid, 21 October ("make the continued-training final
  checkpoint the new champion, then retest Demon Form on it and start the
  next round of training with the exploration statistics").

## Results

**The Demon Form rule passes on second DEV and narrowly misses
confirmation on the new worlds.** Champion forced to take Demon Form
whenever offered, paired with the champion:

| Worlds | Wins | Discordant, champion only / rule only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Second DEV | 527 → 550 (12.9% → 13.4%) | 17 / 40 | 0.003 | +0.06 [−0.01, +0.13] | +0.6 points [+0.1, +1.1] (3,396 worlds) |
| From 95,000,000,000 | 476 → 490 (11.6% → 12.0%) | 16 / 30 | 0.054 | +0.01 [−0.06, +0.08] | +0.3 points [−0.2, +0.7] (3,358 worlds) |

- Under the rule fixed in advance this is *Better* on DEV but not
  confirmed (p = 0.054 against the 0.05 threshold), so it is not distilled
  automatically.
- The rule changes the outcome of only about 1% of games, and in each of
  the four tests on the last two champions the changed games favoured the
  rule (55 to 34, 40 to 28, 40 to 17, 30 to 16).

**The first Demon Form distillation lost the overlay's gain, because its
labels sat at the wrong step.** Distilled checkpoint, second DEV, paired:

| Comparison | Wins | Discordant | p | Mean floor, paired difference |
|---|---:|---:|---:|---|
| Distilled vs champion | 527 → 512 | 245 / 230 | 0.52 | −0.11 [−0.37, +0.13] |
| Distilled vs champion plus overlay | 550 → 512 | 254 / 216 | 0.088 | −0.18 [−0.42, +0.07] |
| Distilled vs champion, worlds from 95,000,000,000 | 476 → 485 | 218 / 227 | 0.71 | +0.17 [−0.10, +0.43] |
| Distilled vs champion plus overlay, same worlds | 490 → 485 | 223 / 218 | 0.85 | +0.16 [−0.11, +0.42] |

- The labels fitted perfectly (held-out label agreement 100%), but 2.4% of
  the champion's other out-of-combat choices changed, and about 470 games
  changed outcome against 57 for the overlay.
- Diagnosis: a reward screen is visited several times (gold, potion, card).
  The collector forced Demon Form at the first visit, where the champion's
  own choice was nearly always to take the gold first; only 10 of 2,080
  labels were at a step where the champion was choosing a card. The
  student learnt "take a card before the gold", not "take Demon Form".
- Fix: the collector now intervenes only where the policy itself would take
  a card, skip, or proceed (in a 48-game smoke test, 16 labels, 6 of them
  changing the choice). Labels recollected on new training worlds from
  8,800,000,000 and 8,900,000,000, then distilled with the same settings.

**The corrected distillation reproduces the overlay.** 6,143 games gave
2,048 labels at real card decisions, 543 of which changed the champion's
choice (it takes Demon Form at 74% of offers on its own). After 12 epochs
the student took Demon Form at every held-out label and kept 99.2% of the
champion's other choices (97.6% in the first attempt).

| Comparison | Wins | Discordant | p | Mean floor, paired difference |
|---|---:|---:|---:|---|
| Distilled vs champion, second DEV | 527 → 562 | 169 / 204 | 0.078 | −0.09 [−0.29, +0.12] |
| Distilled vs champion plus overlay, second DEV | 550 → 562 | 168 / 180 | 0.56 | −0.15 [−0.35, +0.06] |
| Distilled vs champion, worlds from 95,000,000,000 | 476 → 489 | 170 / 183 | 0.52 | +0.01 [−0.20, +0.24] |
| Distilled vs champion plus overlay, same worlds | 490 → 489 | 165 / 164 | 1.00 | +0.01 [−0.21, +0.22] |

- Against the pre-set rule it is not *Better* than the champion (p =
  0.078 on DEV); it is indistinguishable from the champion plus the overlay
  on both world sets, so it is a pure-policy version of the rule.
- About 370 games still change outcome against 57 for the overlay: the
  remaining 0.8% drift in other choices reshuffles games without a net
  cost that these samples can see.

**The second continued training gains as much as the first.** Final
checkpoint, iteration 24 (also the best by the trainer's evaluation),
paired with the champion:

| Worlds | Wins | Discordant, champion only / trained only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Second DEV | 527 → 672 (12.9% → 16.4%) | 263 / 408 | < 0.001 | +0.67 [+0.33, +1.02] | +4.2 points [+2.4, +6.0] (3,132 worlds) |
| From 96,000,000,000 (unused before) | 507 → 691 (12.4% → 16.9%) | 253 / 437 | < 0.001 | +1.25 [+0.89, +1.62] | +5.7 points [+3.9, +7.7] (3,084 worlds) |

**Combined with the round 2 combat student** (exploratory, second DEV):
the trained out-of-combat head with the [round 2
student](2026-10-21-combat-student-round-2.md) wins 701 (17.1%): against the
trained checkpoint with the old student, 701 against 672 (discordant
342/313, p = 0.27; floor +0.71 [+0.35, +1.05]); against the round 2
candidate, 701 against 576 (p < 0.001).

- The trainer's evaluation went 118, 121, 127, 142, 146, 139, 152 wins of
  1,024 at iterations 0 to 24.
- Exploration statistics, iteration 1 to 24: card rewards saturated 78% to
  76% (entropy 0.124 to 0.141), map 58% to 54%, shop 36% to 29% (entropy
  0.34 to 0.41), rest sites 53% to 59%, other menus 81% to 83%; a below-5%
  action was sampled in about 0.5% of decisions throughout (1.8% at
  shops). Overall entropy 0.154 to 0.165 under the entropy bonus.
- Behaviour: smith rate 0.40 to 0.45; shop card removals still almost
  never bought (0.1%); elite paths 7%.
- Health: KL 0.004 to 0.008, clip fraction 0.03 to 0.04, cosines −0.39 to
  +0.18, no faults; median 11.6 minutes per iteration under load (the
  exploration statistics took 18 to 63 seconds of each).

## Conclusion

- Continued out-of-combat PPO has still not plateaued: three rounds from
  the 20 October champion gained 9.3% → 11.6% → 16.9% on unused worlds.
- The combat and out-of-combat improvements look additive: the round 2
  student lifts the newly trained head as it lifted the old one (+29 wins
  on DEV, not significant alone, floor +0.71).
- Saturation moved little: the entropy bonus loosened shops and maps a
  little, card rewards hardly at all, yet play improved a lot.
- The Demon Form rule, distilled at real card decisions, matches its
  overlay; its gain over the champion is small and not significant.

Caveats:
- One training run; the combination with the round 2 student is a single
  exploratory DEV comparison.
- The run paused and slowed under other jobs; neither changes its training
  worlds or updates.

## Next

1. Sid to decide the champion: the trained checkpoint (691 of 4,096 on
   unused worlds) or, after confirmation, its combination with the round 2
   combat student.
2. Confirm the combination on unused worlds; then continue both loops
   (out-of-combat PPO and combat expert iteration) from it.

## Log

- 22 Oct, 11:26: trained checkpoint 672 against 527 on DEV and 691 against
  507 on unused worlds from 96,000,000,000; with the round 2 combat student
  701 on DEV. The laptop ran on battery overnight, so the evaluations
  finished at 11:19.
- 21 Oct, 18:52: corrected distillation matches the overlay (562 and 489
  wins against the overlay's 550 and 490); not significantly better than
  the champion alone.
- 21 Oct, 17:31: first distillation lost the overlay's gain (512 against
  527); labels were at the wrong reward-screen step. Collector fixed and
  labels recollected.
- 21 Oct, 16:09: Sid chose to distil the Demon Form rule despite the
  narrow miss; labels, distillation and evaluations queued. (A note here
  first said the champion declined Demon Form at all 16 offers of a smoke
  test; that counted reward-screen steps where it was about to take the
  gold first. Measured properly it takes Demon Form at 74% of 365 offers;
  see the [card rule grid](2026-10-21-card-rule-grid.md).) Labels are
  written at every reward-screen step offering Demon Form, so some teach
  taking it before the gold rather than taking it at all.
- 21 Oct, 15:51: Demon Form retest: 550 against 527 on second DEV (p =
  0.003), 490 against 476 on the new worlds (p = 0.054): not confirmed by
  the rule fixed in advance.
- 21 Oct, 15:40: champion promoted; training and evaluations queued.

Code: `scripts/rl/train_ooc.py` from a frozen checkout of `ws/nonllm` at
a33bd2b (worktree `nonllm-continued-frozen`). Data: run `continued2_20261021`
(local, worktree `nonllm-large-batch`).
