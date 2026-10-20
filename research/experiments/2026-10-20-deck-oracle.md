# Deck-strength oracle: can simulated boss fights measure a deck?

**Status:** running 20 October 2026. **Verdict:** none yet.

## Question

The champion loses most games at the Act 2 boss, and its decks look the same
whether it wins that fight or not ([19 October brainstorm](../brainstorms/2026-10-19-brainstorming.md)).
Full-game reward therefore says little about deck-building. A cheaper signal:
copy the game at any out-of-combat moment, start this act's boss fight with
the current deck, relics and potions, and play it many times with different
random seeds. If that "oracle" score measures deck strength, it could teach
deck-building directly or serve as a dense reward.

This feasibility spike asks: does the oracle score, measured before the boss,
predict whether the champion actually beats the Act 2 boss, beyond the HP it
enters with? And how noisy is it, so what would it cost per decision?

## Design

- **Games:** the champion's recorded games on the second DEV worlds
  (environment 0.1.5), replayed exactly from their action records; the first
  300 worlds that reach the Act 2 boss, in seed order.
- **Three moments per game:** the first out-of-combat decision of Act 2, the
  first at floor 25 or later, and the last before the Act 2 boss fight.
- **Oracle:** from a copy of the game at that moment, enter the act's boss
  fight and play it with the champion's combat network, 16 times with
  different battle seeds. Two variants: HP set to maximum (deck strength
  alone) and actual HP. Scores: share of fights won, HP lost, and boss damage
  (share of the boss's HP removed; 1 for a win), which still varies when the
  deck is too weak to win.
- **Measures:** how well each score predicts the real Act 2 boss outcome
  (area under the ROC curve, AUC, and cross-validated log loss of a logistic
  model with entry HP alone against entry HP plus the oracle); the score's
  split-half reliability (8 seeds against the other 8); cost in seconds.
- **Decision rule, fixed in advance:** worth building on if the full-HP oracle
  at the last decision before the boss adds predictive power beyond entry HP
  (cross-validated log loss improves and the AUC of the combined model rises
  by at least 0.03) and its split-half reliability is at least 0.5. The
  score for this rule is boss damage, the mean over the 16 fights.
- **Compute:** a few laptop cores for under an hour, alongside the Codex
  throughput work.
- **Approval:** Sid approved the spike on 20 October as part of the plan in
  the brainstorm.

## Results

## Conclusion

## Next

## Log

- 20 Oct: a two-world smoke test showed early-Act-2 decks never win the boss,
  so the boss-damage score and the mid-act moment were added before the
  full run.
- 20 Oct: boss damage chosen as the decision rule's score before any full-run
  results were seen.

Code: `experiments/2026-10-20-deck-oracle/`. Data: run `deck_oracle_20261020`
(local).
