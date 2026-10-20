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
- On 4,096 worlds it had never played, the same model won 197 against the
  champion's 156 (+1.0 points [+0.2, +1.8]), about half its first margin; a
  second training seed of the same recipe won 221 (+1.6 [+0.8, +2.4]).
- The gain comes from hallway fights: the students reach the Act 3 boss far
  more often, but every one of 28 students wins fewer Act 2 boss fights than
  the champion.
- Best explanation: the earlier failures were budget-starved. A full training
  run on many labels works, though agreement with the teacher is a poor
  predictor of playing strength. The teacher itself is weak at bosses (see
  the [19 October brainstorm](brainstorms/2026-10-19-brainstorming.md)).
- Next: confirm one chosen student on the confirmation worlds; improve the
  teacher before expert iteration.
- Pages: [combat PPO and early imitation](experiments/2026-10-13-combat-ppo.md),
  [overnight tests](experiments/2026-10-14-overnight-tests.md),
  [fresh start](experiments/2026-10-15-fresh-start.md),
  [representation controls](experiments/2026-10-16-representation-controls.md).

## Deck-strength oracle

**Promising.** Score a deck by copying the game, entering the act's boss
fight and playing it many times in the simulator. Use the score to teach
deck-building, which full-game reward cannot.

- Just before the Act 2 boss, 16 simulated full-HP fights predict the real
  outcome with AUC 0.78 (entry HP: 0.68), independent of HP, with
  reliability 0.91, at about 1.1 s per state.
- The champion's decks range from never to always beating the boss at full
  HP, though they look alike in simple counts.
- It cannot choose between card rewards: one card moves the boss fight by
  less than 32 simulated fights can resolve, and following its choice gained
  nothing measurable ([oracle options](experiments/2026-10-20-oracle-options.md)).
- Powers, which go with strong decks, are not under-picked as a class: the
  champion nearly always takes Juggernaut, Inflame and Metallicize, nearly
  never the other eleven, and skips about two thirds of boss card rewards
  ([power picks](experiments/2026-10-20-power-picks.md)).
- Forcing Demon Form whenever offered wins 179 against 156 of 4,096 games
  (p = 0.010), the first single deck-building habit shown to change wins;
  forcing any rare at boss rewards helps the boss fight but not clearly
  wins ([boss rares](experiments/2026-10-20-boss-rares.md)).
- Next: distil the Demon Form rule and confirm it; test the rules with the
  better combat head; learn a deck-value network from oracle scores to pool
  information across decisions.
- [Deck oracle page](experiments/2026-10-20-deck-oracle.md).

## Out-of-combat PPO with macro credit

**Working.** Freeze the combat network, treat everything between two
out-of-combat (OOC) decisions as one transition, and train OOC choices with
large batches of whole games.

- At 3,072 games per iteration (four times the recipe), 24 iterations from
  the champion won 276 of 4,096 second-DEV worlds against 156 under the
  study protocol, and 232 against 152 on 4,096 fresh worlds (+2.0 points,
  p < 0.001; floor +1.24 [+0.91, +1.58]), the largest confirmed pure-policy
  gain so far ([large-batch PPO](experiments/2026-10-20-large-batch-ppo.md)). The arm
  also set lambda 1 and weight decay 0 explicitly, so batch size alone is
  not isolated. The trainer's own 1,024-world evaluation, on the
  potions-visible protocol, showed only 17 to 25 wins and hid the gain.

- Update 36 gained 2.99 mean floors [2.39, 3.57] over the original
  composite on fresh worlds; a second training seed reproduced early gains.
- With lambda 1 and distilled strategy rules, this lineage produced the
  champion: round-2 update 20, 213 wins of 4,096 fresh worlds on 0.1.4.
- Continuing the same recipe from the champion at 768 games gave 58 and 57
  wins against 57 on 1,024 DEV worlds: the plateau was the batch size and
  the evaluation, not the method.
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

**Abandoned for now.** Represent the game as separate entities (player,
enemies, cards, relics, potions) for a small transformer, instead of the old
hand-built feature vector.

- It copied the search teacher as well as the old encoder (58.7% against
  58.4% agreement) but played worse: 131 against the old encoder's 222 wins
  when paired with champion OOC play.
- The [representation controls](experiments/2026-10-16-representation-controls.md)
  study showed the gap is real at two training seeds (+2.4 and +3.1 points
  for the old encoder). Adding the old encoder's rule features recovered about
  a point; matching its size recovered nothing; about 1.4 points remain.
- Best explanation: the structured model has no card semantics beyond
  identity and cost, and lacks the old encoder's state features (incoming
  and unblocked damage, pile summaries). Untested.
- Wide structured models need a learning rate of 0.0003; at 0.001 they train
  badly.

## Fight-value prediction

**Stalled.** Predict whether the current player wins a fight, to guide
out-of-combat credit or search.

- Classifiers predicted Act 2 boss outcomes only slightly better than a
  constant at fight entry, and worse than a constant for Act 3 bosses. They
  were much better at later states within a fight.
- A proposed variant, estimating boss-pass chances from repeated fights at the
  same deck and HP, has not been tried.
- [Overnight tests page](experiments/2026-10-14-overnight-tests.md).
