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

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 21 Oct, 16:13: page written and the candidate list fixed while the
  measurement ran, before any take rate was read.

Code: `experiments/2026-10-21-card-rule-grid/`. Data: run
`card_rule_grid_20261021` (local, worktree `nonllm-large-batch`).
