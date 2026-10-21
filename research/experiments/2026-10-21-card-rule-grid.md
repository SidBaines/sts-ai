# Card rule grid: which declined cards should the champion always take?

**Status:** finished 21 October 2026. **Verdict:** none of the eight cards the champion avoids helps when forced; three hurt. Its avoidance is mostly right, and Demon Form remains the only card rule that helps.

## Question

Forcing the champion to take Demon Form whenever offered has helped every
champion so far, yet the champion declines it at almost every offer: PPO
cannot raise near-zero probabilities (see [strategies](../strategies.md)).
Which other strong cards does the champion systematically decline, and
does taking them whenever offered win more games? Each winning rule is a
candidate for distillation. (Approach: distilling strategy rules.)

## Design

- **Player:** the champion of 21 October, afternoon
  (`champion_2835f40fa439.pt`), study protocol.
- **Step 1, measurement:** 1,024 training worlds from 8,700,000,000; log
  every card-reward decision (the offered cards, the policy's probability
  for each, the choice). Take rate per card = takes / offers.
- **Step 2, candidate list, fixed before the measurement was read:** cards
  generally rated strong for Ironclad at Ascension 0: Offering, Shockwave,
  Feed, Reaper, Barricade, Impervious, Corruption, Limit Break, Disarm,
  Uppercut, Inflame, Battle Trance, Feel No Pain, Carnage, Whirlwind, Flame
  Barrier, Shrug It Off, Pommel Strike. Demon Form is excluded (tested
  separately). A card enters the grid if it was offered at least 30 times
  and taken in fewer than half of its offers; the grid tests up to 8 such
  cards, those with the most declines (offers minus takes).
- **Step 3, tests:** for each card, `fast_eval.py --take-card NAME` on the
  4,096 second-DEV worlds (8 processes of 32 games), paired with the
  champion's 527 wins on the same worlds.
- **Decision rule, fixed in advance:** a rule is *Better* if it wins more
  than the champion with exact McNemar p below 0.05 after Holm correction
  across the cards tested. Each Better rule is then confirmed on 4,096
  unused worlds from 97,000,000,000 (champion and rule both played there)
  at p < 0.05 before distillation.
- **Approval:** within Sid's standing instruction of 21 October to keep
  doing research until stopped; extends the Demon Form test that Sid
  approved.

## Results

**The champion never takes five strong cards.** In 1,024 games (16,871
card-reward decisions; card rewards skipped 10.9% of the time), take rates
of the pre-registered candidates:

| Card | Offers | Taken | In the grid |
|---|---:|---:|---|
| Feel No Pain | 361 | 0% | yes |
| Disarm | 361 | 0.3% | yes |
| Shockwave | 343 | 0% | yes |
| Limit Break | 342 | 0% | yes |
| Offering | 326 | 0% | yes |
| Barricade | 321 | 1.6% | yes |
| Whirlwind | 361 | 23.5% | yes |
| Battle Trance | 368 | 28.0% | yes |
| Corruption | 340 | 24.4% | no (9th by declines) |
| Feed | 317 | 28.7% | no |
| Impervious | 345 | 37.1% | no |
| Carnage | 351 | 42.7% | no |
| Reaper | 346 | 49.4% | no |
| Uppercut | 382 | 75.4% | no (taken over half the time) |
| Inflame | 351 | 86.0% | no |
| Flame Barrier | 331 | 75.2% | no |
| Shrug It Off | 1,057 | 76.6% | no |
| Pommel Strike | 1,053 | 73.0% | no |
| Demon Form | 365 | 74.0% | excluded (tested separately) |

Every card's rate is in `figures/2026-10-21-card-rule-grid/take_rates.csv`.
**No rule passes; three are worse.** Each card forced whenever offered,
paired with the champion's 527 wins of 4,096 on second DEV, Holm-corrected
across the eight:

| Card forced | Wins | Discordant, champion only / rule only | p | Holm p | Mean floor, paired difference | Verdict |
|---|---:|---:|---:|---:|---|---|
| Shockwave | 561 | 79 / 113 | 0.017 | 0.085 | +0.16 [+0.00, +0.32] | No clear difference |
| Barricade | 517 | 92 / 82 | 0.50 | 0.96 | −0.17 [−0.30, −0.06] | No clear difference |
| Disarm | 516 | 105 / 94 | 0.48 | 0.96 | −0.13 [−0.28, +0.03] | No clear difference |
| Limit Break | 502 | 94 / 69 | 0.060 | 0.18 | −0.31 [−0.43, −0.19] | No clear difference |
| Battle Trance | 501 | 81 / 55 | 0.032 | 0.13 | −0.22 [−0.35, −0.08] | No clear difference |
| Feel No Pain | 482 | 114 / 69 | 0.001 | 0.007 | −0.51 [−0.66, −0.35] | Worse |
| Offering | 472 | 117 / 62 | < 0.001 | < 0.001 | −0.23 [−0.36, −0.09] | Worse |
| Whirlwind | 463 | 123 / 59 | < 0.001 | < 0.001 | −0.11 [−0.25, +0.02] | Worse |

- Seven of the eight rules lower mean floor; forcing a card also forces
  every later copy and removes the option to skip, so a deck-dilution cost
  is likely part of every result.

## Conclusion

- The champion's near-zero take rates for these cards are not a
  saturation bug worth fixing with a blunt rule: forcing them never helped
  after correction, and three cost 1 to 1.6 points of wins.
- Demon Form is the exception (it helped every champion), which fits its
  reputation as a single card that wins boss fights on its own.
- Saturation is therefore not always a problem: a near-zero probability can
  be correct. Rules need measuring, as on 13 October.

Caveats:
- "Take whenever offered" is crude: it forces duplicates and removes
  skipping. A softer rule (first copy only, or only instead of another
  card) might reverse some results, notably Shockwave.
- One pass on reused DEV worlds; the candidate list came from general
  Ironclad knowledge, not from this agent.

## Next

1. If card rules are pursued further, test "first copy only" for Shockwave,
   the one positive direction, as a new exploratory test.
2. Otherwise rely on the Demon Form distillation and continued training.

## Log

- 21 Oct, 17:25: grid finished: no rule passes, three are worse.
- 21 Oct, 16:22: measurement done; the eight cards above queued for the
  overlay tests. An 8-game smoke test had been printed before the list was
  fixed; none of its rows were candidate cards.
- 21 Oct, 16:13: page written and the candidate list fixed while the
  measurement ran, before any take rate was read.

Code: `experiments/2026-10-21-card-rule-grid/`. Data: run
`card_rule_grid_20261021` (local, worktree `nonllm-large-batch`).
