# Card rule grid: which declined cards should the champion always take?

**Status:** running. **Verdict:** pending.

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
Overlay tests: pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 21 Oct, 16:22: measurement done; the eight cards above queued for the
  overlay tests. An 8-game smoke test had been printed before the list was
  fixed; none of its rows were candidate cards.
- 21 Oct, 16:13: page written and the candidate list fixed while the
  measurement ran, before any take rate was read.

Code: `experiments/2026-10-21-card-rule-grid/`. Data: run
`card_rule_grid_20261021` (local, worktree `nonllm-large-batch`).
