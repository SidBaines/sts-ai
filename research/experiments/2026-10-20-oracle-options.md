# Oracle options: can the deck oracle choose between card rewards?

**Status:** running 20 October 2026. **Verdict:** none yet.

## Question

The [deck oracle](2026-10-20-deck-oracle.md) measures how strong a deck is
against the act boss. To teach deck-building it must also separate the
options at a single decision, where differences are smaller. At Act 2 card
rewards, does the oracle find options clearly better than the champion's
choice, reliably enough to learn from? (Approach:
[deck-strength oracle](../strategies.md#deck-strength-oracle).)

## Design

- **Decisions:** card rewards in Act 2 before the boss, from the champion's
  recorded games on the second DEV worlds (environment 0.1.5), in worlds not
  used by the deck-oracle spike (seeds from 91,001,002,048), at most two per
  world, about 160 in all.
- **Options:** each offered card ("take card") and "skip card reward". The
  champion's choice is what it did with that reward in its recorded game
  (proceeding without a card counts as skipping).
- **Oracle:** for each option, take it in a copy of the game and play this
  act's boss fight 32 times at full HP with the champion's combat network,
  using the same 32 battle seeds for every option. Score: boss damage (share
  of the boss's HP removed; 1 for a win), as in the spike; win share also
  reported.
- **Measures:** for each decision, the best option's score minus the
  champion's choice, with a paired standard error over the 32 seeds; how
  often the oracle's choice is stable (best option on even seeds equals best
  on odd seeds); and the gain from following the oracle measured without
  selection bias (choose on the even seeds, score the choice on the odd
  ones).
- **Decision rule, fixed in advance:** worth distilling if in at least 20% of
  decisions the best option beats the champion's choice by more than two
  paired standard errors, and the even-odd choice agrees in at least 60% of
  those decisions.
- **Compute:** about 10 minutes on 4 laptop cores.
- **Approval:** follows the spike's "Next" step, within the plan Sid approved
  on 20 October.

## Results

## Conclusion

## Next

## Log

Code: `experiments/2026-10-20-oracle-options/`, using `src/sts_rl/deck_oracle.py`.
Data: run `oracle_options_20261020` (local).
