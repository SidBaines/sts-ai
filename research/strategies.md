# Strategies

Each approach this workstream has tried: the idea, its status, the key
evidence and our best explanation. Statuses are *working*, *promising*,
*stalled* or *abandoned*. Numbers are in the [results ledger](results.md).

## Play-time search

**Working, but privileged.** Let the search teacher choose the moves in hard
fights while the learned networks play everything else.

- Search at Act 2 and 3 bosses and elites won 333 of 4,096 fresh worlds
  against 213 without it (environment 0.1.4). This is the largest gain
  measured so far. Adding ordinary hallway fights won 98 of 512 DEV worlds
  against 42, but that is not yet confirmed on fresh worlds.
- The gain comes mostly from getting more worlds through Act 2.
- The search runs on copies of the simulator and can see hidden state, so it
  is not a public-information player. It is best seen as a teacher and a
  measure of how much combat headroom there is.
- [Play-time search page](experiments/2026-10-14-play-time-search.md).

## Distilling the search teacher

**Promising.** Train the combat network to copy the search teacher's choices,
so the network plays well without search.

- Early attempts failed. Privileged labels on learner states, resampled boss
  labels and short pilots of 20 to 32 updates on teacher-played fights all
  improved agreement with the teacher while making play worse.
- The fresh-start study trained the old encoder from scratch on 113,042
  teacher labels for about 4,000 updates. Paired with champion out-of-combat
  play, it won 222 of 4,096 DEV worlds against the champion's 143
  (environment 0.1.5). This is exploratory and not yet confirmed.
- Best explanation: the earlier failures were budget-starved. A full training
  run on many labels works, though agreement with the teacher is a poor
  predictor of playing strength.
- Next: confirm on fresh worlds, then run expert iteration (label the states
  the student itself visits and retrain).
- Pages: [combat PPO and early imitation](experiments/2026-10-13-combat-ppo.md),
  [overnight tests](experiments/2026-10-14-overnight-tests.md),
  [fresh start](experiments/2026-10-15-fresh-start.md).

## Out-of-combat PPO with macro credit

**Working, now plateaued.** Freeze the combat network, treat everything
between two out-of-combat (OOC) decisions as one transition, and train OOC
choices with large batches of whole games.

- Update 36 gained 2.99 mean floors [2.39, 3.57] over the original
  composite on fresh worlds; a second training seed reproduced early gains.
- With lambda 1 and distilled strategy rules, this lineage produced the
  champion: round-2 update 20, 213 wins of 4,096 fresh worlds on 0.1.4.
- Continuing the same recipe from the champion gave 58 and 57 wins against
  57 on 1,024 DEV worlds: a plateau at this checkpoint, not proof that OOC
  learning is exhausted.
- Why it worked: it removed combat noise and cross-phase critic bias from
  strategic credit. Macro credit, frozen combat and larger batches changed
  together, so their separate contributions are unknown.
- [OOC PPO page](experiments/2026-10-13-ooc-macro-credit-ppo.md).

## Distilling strategy rules

**Working, used selectively.** Measure a strategic habit with a menu overlay
on paired worlds, and distil only rules that help.

- PPO almost never tried smithing, elites or card removal: their
  probabilities were too small to sample. Smith at HP ≥ 85% (+1.35 mean
  floors) and take an offered elite at HP ≥ 90% were measured, distilled and
  kept.
- Many plausible rules hurt: smithing at lower HP, skipping card rewards,
  hiding the Neow boss-relic swap (29 against 57 wins), and forced removals.
- Lesson: never adopt a strategy rule because it sounds right; measure it.
- Pages: [OOC PPO](experiments/2026-10-13-ooc-macro-credit-ppo.md),
  [overnight tests](experiments/2026-10-14-overnight-tests.md).

## Curriculum imitation and PPO

**Working for Act 1; superseded.** Imitate teachers, correct learner-state
errors with DAgger, then train progressively longer tasks with PPO, from
single fights to full games.

- Full learned control cleared 20 of 21 frozen final Act 1 worlds but won
  none of 21 full games.
- Its combat network is still the champion's combat network.
- [First curriculum page](experiments/2026-10-11-first-curriculum.md).

## Combat PPO

**Stalled.** Improve the combat network directly with PPO on a bank of fight
starts the learner reached.

- Sparse rewards gave no reliable full-game gain; a more aggressive learning
  rate made play worse; dense HP shaping did not help.
- Likely reasons, none isolated: sparse credit, forgetting outside the
  emphasised fights, and representation limits.
- [Combat PPO page](experiments/2026-10-13-combat-ppo.md).

## Joint sparse-win PPO

**Stalled.** Train both networks together from the champion on the full-game
win signal alone.

- After repairing a numerical bug and underfitted critics, a two-hour run
  gave 62 against 58 wins on 1,024 DEV worlds (+0.39 points, interval −0.98
  to +1.76): no clear difference.
- The policy moved about 30 times less per update than in the OOC recipe that
  worked, and critics were poorly calibrated in Act 3.
- [Joint sparse-win page](experiments/2026-10-14-joint-sparse-win-ppo.md).

## Long joint curricula

**Abandoned.** Train both networks together on a mix of fresh games and
saved later-act starts.

- v1 discarded almost all of its training through incumbent restoration; v2
  kept its updates but lost both original wins and many Act 1 clears.
- Swapping networks showed that out-of-combat changes caused the broad
  regression, and that either change could destroy rare wins.
- Replaced by the OOC-only recipe above.
- [Long curricula page](experiments/2026-10-12-long-curricula.md).

## Structured entity encoder

**Stalled, under diagnosis.** Represent the game as separate entities
(player, enemies, cards, relics, potions) for a small transformer, instead of
the old hand-built feature vector.

- It copied the search teacher as well as the old encoder (58.7% against
  58.4% agreement) but played worse: more HP lost in Act 2 boss fights, and
  131 against the old encoder's 222 wins when paired with champion OOC play.
- It lacked the old encoder's hand-made rule features, was smaller, and had
  less learning-rate tuning. The
  [representation controls](experiments/2026-10-16-representation-controls.md)
  study is testing which of these matters.
- [Fresh-start page](experiments/2026-10-15-fresh-start.md).

## Fight-value prediction

**Stalled.** Predict whether the current player wins a fight, to guide
out-of-combat credit or search.

- Classifiers predicted Act 2 boss outcomes only slightly better than a
  constant at fight entry, and worse than a constant for Act 3 bosses. They
  were much better at later states within a fight.
- A proposed variant, estimating boss-pass chances from repeated fights at the
  same deck and HP, has not been tried.
- [Overnight tests page](experiments/2026-10-14-overnight-tests.md).
