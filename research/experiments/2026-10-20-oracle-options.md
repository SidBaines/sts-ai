# Oracle options: can the deck oracle choose between card rewards?

**Status:** finished 20 October 2026. **Verdict:** no: at a single card
reward the options differ by less than 32 simulated boss fights can
resolve, and following the oracle's choice gains nothing measurable.

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

**The oracle does not pass its decision rule.** 160 Act 2 card rewards from
85 worlds (floors 18 to 32, mean 24), 4 options each, 32 common-seed fights
per option; no faults.

| Measure (boss damage) | Result | Rule |
|---|---:|---:|
| Decisions where the best option beats the champion's by more than 2 paired standard errors | 11.9% | at least 20% |
| ...of which the even-seed and odd-seed choices agree | 68% | at least 60% |
| Gain from following the oracle, chosen on even seeds, scored on odd seeds | +0.002 [−0.005, +0.009] | |
| Even-odd agreement on the best option, all decisions | 43% | |

- The typical gap between the best and worst of the four options is 0.045
  boss damage, against a paired standard error of 0.030 for one difference:
  the options are mostly indistinguishable at this sample size.
- Win share gave the same answer (4% of decisions clearly better; gain
  +0.001 [−0.013, +0.016]).
- The oracle mostly agrees with the champion's habits: it rates Perfected
  Strike, Clothesline, Pommel Strike and Twin Strike above skipping (the
  champion takes them 62% to 100% of the time) and True Grit, Heavy Blade
  and Wild Strike below (taken 0% to 27%). It would skip more often (29% of
  decisions against the champion's 8%), by margins within noise.
- Cost: 7.3 s per decision on one core.

## Conclusion

One card in a deck of about 25 changes the boss fight too little to measure
decision by decision with tens of simulated fights. The oracle measures
whole decks well (its spike) but is not a practical per-decision teacher for
card rewards. The champion's card picks also look reasonable to it, so its
weak decks may come mostly from other choices or from luck: removals,
upgrades, relics, shop purchases or path.

Caveats:
- The oracle is myopic: it values a card for today's boss fight, not for the
  deck it will become part of.
- Fights are played by the champion's combat network.
- Only card rewards in Act 2 were tested.

## Next

- Pool information across decisions instead: train a deck-value network on
  oracle scores from many states, then use it to score options without
  simulation.
- Find where deck strength comes from: regress the spike's before-boss
  oracle scores on deck composition, relics, upgrades and removals.

## Log

- 20 Oct: run finished in 5 minutes on 4 cores; decision rule not met.

Code: `experiments/2026-10-20-oracle-options/`, using `src/sts_rl/deck_oracle.py`.
Data: run `oracle_options_20261020` (local).
