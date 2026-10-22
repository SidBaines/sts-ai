# A better search teacher for combat

**Status:** running (step 0 benchmark started 22 October, 17:33). **Verdict:** pending.

## Question

Expert iteration distils the search teacher into the combat network, so the
teacher caps how good the student can get
([expert iteration](../strategies.md#expert-iteration-for-combat)). The
champion still loses about 75% of its games at the Act 2 and Act 3 bosses,
and every distilled student has won fewer boss fights than the search.
Reading the native source (22 October) found four weaknesses in how the
teacher plays and labels:

1. **It plans in one fixed future.** Each of the 24 searches reshuffles the
   draw pile and re-seeds the random streams once, then searches that single
   sampled future, so it can plan around draws and enemy rolls it could not
   know.
2. **It votes with its luckiest line.** Each search's vote is the first move
   of the single best-scoring random playout if any playout won, otherwise
   its most-visited move. Votes are very split: only 4.6% of round 2 labels
   are unanimous, and in 57% the top move has 12 or fewer of 24 votes.
3. **Its score is not calibrated for our game.** A win scores 100 × (35 + HP
   + 4 × potions − 0.01 × turns). At Ascension 0 with three acts, HP after a
   boss fight is worth nothing (full heal after Acts 1 and 2; victory after
   Act 3), yet the score ranks winning lines by HP. The constant 35 sets an
   uncalibrated trade between win chance and HP, potions count a flat 4 HP
   whatever they are, and the loss credit rewards keeping potions and
   surviving more turns. Because the score's range jumps to thousands once
   any playout wins, the tree's value term (about 0.07 for a 10% win rate)
   is probably swamped by its exploration term (about 1), so it spreads
   simulations almost evenly; this is arithmetic from the code, not yet
   measured.
4. **It forgets cards known to be on top of the draw pile** (Headbutt,
   Warcry, the Frozen Eye relic), because each sample shuffles the whole
   pile. The network's observation shows the draw pile only as unordered
   counts, so the student cannot see them either.

Does a teacher that fixes these play boss fights better, and does a student
distilled from it make a better player?

## Design

Each step is a teacher variant, compared with the current teacher on the
same benchmark before any labelling.

**0. Measure, and build the benchmark (first).**
- On about 500 boss positions from the boss-only corpus, separate search
  noise (the same sampled future, different search seeds) from disagreement
  between sampled futures (different futures, same search seed). Also record
  how each search spreads its visits over root moves, and how often its vote
  came from a winning playout, to test point 3's arithmetic.
- Count how often Headbutt, Warcry or Frozen Eye appear in labelled combat
  decisions, and how often a mid-turn draw follows a card being put on top,
  to size point 4.
- **Teacher benchmark:** full games on the second-DEV worlds under the
  study protocol, in which the champion plays everything except boss
  fights and the teacher variant plays every boss fight decision (public
  information only). Metrics: wins and boss pass rates by act, paired by
  world with the champion's own games (899 wins) and with the current
  teacher. A cheaper variant reuses the label collector, so it shares the
  champion's protocol exactly. The old boss-search evaluator
  (`eval_search_boss.py`) is not used: it keeps out-of-combat potion
  actions visible, so its numbers are not comparable. The first question
  it answers: does the current teacher still beat the champion's own
  combat network at bosses? If not, distilling it cannot help there.
- Steps 1 and 2's vote rules (most visited, best mean result, pooled
  searches) can be computed from the root statistics each search already
  returns, so they are benchmarked before any native change. Score fixes
  need the native change.

**1. Vote and score fixes (cheap, together).** These have to go together:
averaging results under the current score would trade win chance for HP.
- The vote becomes the move with the best mean result, not the first move
  of the luckiest playout.
- Score fixes: (A) in boss fights, drop the HP term and keep potions for
  Acts 1 and 2; (B) score a win by estimated run-win probability, from
  exchange rates for HP, maximum HP and potions fitted to games we already
  have, by act and floor; (D) drop the potion and turn terms from the loss
  credit, and scale the value term so that it still matters once a win
  exists.

**2. Pooled searches (cheap).** Add up visits and values per root move
across the 24 searches instead of one vote each.

**3. Information-set tree search (main build).** One tree with a freshly
sampled future on every simulation, at the same total budget (24,000
simulations). Moves are identified by the card they play, not its hand
position, and the selection rule counts how often each move was
available.

**4. Network playouts and value (follow-on).** Replace random playouts with
the student network and score states by the out-of-combat critic's
run-win estimate. This needs network calls inside the native search; it is
the larger project, and step 0 should show whether playouts or
determinisation matter more at bosses.

**Distillation with the pooled teacher (added 23 October, 00:40, before
any labels existed).** The benchmark shows the pooled teacher plays bosses
far better; the question for the agent is whether a student learns more
from it.
- Corpus (`combat_pooled_20261023`): the champion (`af2fb9f1f260`) plays
  2,000 new worlds from 90,501,000,000; the teacher labels with round 2's
  room probabilities, 24 searches of 1,000 simulations, votes by pooled
  searches, with per-search details recorded so that today's luckiest-line
  labels can be derived from the same searches (matched control, no extra
  search). Probes: 8 unlabelled worlds from 93,000,000,300.
- Fits: two seeds per label rule, each on this corpus only (as round 2),
  spliced into the champion.
- Decision rule: a pooled-label student is *Better* if it wins more than
  the champion on second DEV (paired with its 899 wins) with exact McNemar
  p < 0.05, and is then confirmed on 4,096 unused worlds from
  106,000,000,000 before any promotion (Sid's decision). Secondary: pooled
  against luckiest-line students from the same searches, same seeds,
  paired on second DEV, reported with intervals; and Act 2 and Act 3 boss
  pass rates.

**Known top cards (after the vote, score and budget fixes; Sid, 22 October).**
- The simulator tracks how many cards on top of the draw pile are known:
  Headbutt and Warcry add one, a draw removes one, a shuffle resets it, and
  with Frozen Eye the whole order is known. Teacher samples shuffle only the
  unknown cards.
- The public observation gains the known top cards. This changes the
  environment's observation, so it goes to main as its own pull request
  with an observation version bump, regression tests and a new
  environment tag, which the workstream then merges. The new field is
  additive, so existing encoders and checkpoints see the same input.
- A new combat encoder version reads it. Existing encoder files are
  hash-bound and are not edited; the new version is a new file with tests.
  Students are fitted from scratch each round, so no checkpoint migration
  is needed.

**Where the code goes.** Teacher changes are teacher-only, so they go in the
workstream's native patch (`patches/sts_lightspeed_rl_panel.patch`) as new
options beside the current search, leaving the environment's search
untouched. Benchmark and measurement scripts live in
`experiments/2026-10-22-better-teacher/`, and anything reused moves to
`src/sts_rl/` with tests.

**Decision rules.**
- A teacher variant replaces the current teacher if it wins more benchmark
  boss fights (McNemar p < 0.05) and no fewer Act 1 boss fights.
- A new teacher's labels are then used for an expert-iteration round: two
  students per teacher, because fits vary, each spliced into the champion
  and compared with the champion on second DEV (4,096 worlds), then
  confirmed on unused worlds before promotion. Promotion is Sid's decision.
- The known-top-cards encoder is adopted if a student using it is no worse
  than one fitted on the same labels with the current encoder, and better
  in decks containing Headbutt, Warcry or Frozen Eye.

**Compute.** Step 0 takes about an hour of measurement plus building the
benchmark. Labelling with the new teacher costs the same as now (about 1.7
hours per round on 8 lanes). The native work in steps 1 to 3 and the
known-top-cards change is code, written by Codex and reviewed by Claude.

## Results

**Step 0, noise (finished): today's vote rule is mostly search noise, and
more search does not fix it; voting by best mean result does improve with
more search, after which disagreement between sampled futures remains.**
Two searches at the same boss decision choose the same move this often
(134 decisions from the benchmark below, about 45 per act; 8 searches per
condition; "same future" = one sampled future, different search seeds;
"different futures" = different sampled futures, one search seed; chance
is about 23% with 5 to 6 legal moves):

| Act | Vote rule | 1,000 simulations | 4,000 | 16,000 |
|---|---|---|---|---|
| 1 | Luckiest line (today) | 46% same future / 43% different | 50 / 45 | 50 / 48 |
| 1 | Best mean result | 70 / 59 | 78 / 70 | 91 / 74 |
| 2 | Luckiest line | 37 / 35 | 37 / 35 | 38 / 34 |
| 2 | Best mean result | 53 / 43 | 64 / 50 | 76 / 56 |
| 3 | Luckiest line | 50 / 43 | 53 / 44 | 55 / 46 |
| 3 | Best mean result | 58 / 49 | 66 / 56 | 71 / 60 |

- With today's rule, different sampled futures add only 2 to 7 points of
  disagreement to the search's own noise, at every budget.
- With the best-mean rule, the gap between one future and many grows with
  budget (Act 2: 10 points at 1,000 simulations, 20 at 16,000): once search
  noise falls, hidden information is the main remaining source, which is
  what pooling or information-set search addresses.
- The search spreads its simulations almost evenly over root moves (visit
  entropy 0.91 to 0.94 of the maximum), and at 1,000 simulations a winning
  playout is found in about 95% of Act 1 and 2 searches and 60% of Act 3
  ones, so today's vote nearly always comes from a single lucky playout.
- Agreement measures consistency, not strength; the benchmark below decides.
  Agreement also counts two equivalent orderings of moves as different.

**Step 0, how often known top cards matter: rarely, for today's decks.**
Among the 174,862 labelled combat decisions of the round 2 corpus, the deck
holds Headbutt or Warcry in 3.7% of hallway, 2.9% of elite and 4.2% of boss
decisions; Frozen Eye is present in 0.1 to 0.2%. The case the teacher gets
wrong, a card just put on top this turn with a draw card in hand, occurs
in about 0.03% of decisions (approximate: it relies on the observation's
recent-action list). The champion rarely takes these cards, so the fix
would change few labels now; it could matter more if a better teacher made
the cards worth taking.

**Step 0, benchmark: the teacher wins far more boss fights than the
champion's own combat network, and the two vote rules play about equally
well.** 1,024 second-DEV worlds (the first quarter of second DEV); the
champion plays everything except boss fights, which the teacher plays
(24 searches of 1,000 simulations per decision); paired with the
champion's own games on the same worlds:

| Boss fights played by | Wins | Paired difference in wins | Act 1 / Act 2 / Act 3 boss pass, paired difference |
|---|---:|---|---|
| Champion's combat network | 205 (20.0%) | | 95.1% / 67.9% / 38.4% |
| Teacher, today's vote (luckiest line) | 333 (32.5%) | +12.5 points (discordant 44/172, p < 0.001) | +0.3 [−0.9, +1.5] / **+12.9 [+9.9, +15.8]** / **+16.2 [+12.0, +20.4]** |
| Teacher, best mean result | 345 (33.7%) | +13.7 points (38/178, p < 0.001) | +0.2 [−1.0, +1.4] / **+14.7 [+12.0, +17.5]** / **+16.2 [+11.8, +20.5]** |

- Best mean result against today's vote, paired: 345 against 333 wins
  (+1.2 points [−1.0, +3.4], discordant 61/73, p = 0.34): no clear
  difference as players. Aggregated over 24 searches, the luckiest-line
  noise largely averages out in play; it remains in the vote shares the
  student is trained on.
- The gap at the Act 2 and Act 3 bosses is the room that distillation has
  not captured: students trained on these teachers' labels, including 4,000
  worlds of boss-only labels, came out level with the champion at bosses
  ([combat variance](2026-10-22-combat-variance.md)).
- Pairing check: 23 of 1,024 worlds left the champion's own game before the
  teacher's first move, at out-of-combat choices where the network's top two
  scores differ by less than 0.000004 (equal cards to upgrade, equal map
  nodes); batched and single-game inference break such ties differently.
  This adds noise to the pairing, not bias.
- **All six vote rules, full reruns (finished 23 October, 00:21).** Each
  arm plays the same first 1,024 second-DEV worlds (1,020 to 1,024
  completed); paired with today's teacher on the same worlds:

| Vote rule (24 searches × 1,000 simulations) | Wins | Paired difference vs today's vote | Act 1 / 2 / 3 boss pass vs today's vote |
|---|---:|---|---|
| Today's: first move of each search's luckiest playout | 331–333 | | |
| **Pooled searches: add up root visits and values across the 24 searches, take the best mean** | **429** | **+9.6 points (discordant 42/140, p < 0.001)** | +1.9 [+0.9, +3.0] / **+5.3 [+3.1, +7.6]** / **+11.1 [+7.5, +14.7]** |
| Most visited move per search | 361 | +2.9 (50/80, p = 0.011) | −0.5 / +0.6 / +4.0 [+1.0, +7.1] |
| Best mean per search, win-first score | 362 | +3.0 (68/98, p = 0.024) | −0.1 / +2.1 / +5.1 [+1.4, +8.8] |
| Best mean per search | 345 | +1.2 (61/73, p = 0.34) | −0.1 / +1.9 / −0.2 |
| Best mean per search, win-first score, exploration 1.0 | 329 | −0.4 (81/77, p = 0.81) | −0.9 / −3.3 [−5.8, −0.8] / +1.9 |

  - With pooled searches playing the boss fights, the champion's games win
    429 of 1,020 (42%), against 203 for the champion alone (20%).
  - Five rules are compared with today's; with a Holm correction, pooled
    searches and most visited stay significant, the win-first arm does not.
  - Pooling uses exactly the same searches as today's teacher; it only
    reads them differently. A single search's vote is noisy (step 0), so
    combining the evidence of all 24 before choosing beats letting each
    cast one noisy vote.
  - The two arms stopped by low disk space (215 and 152 worlds) are
    superseded by these full reruns.

## Conclusion

Pending.

## Next

Pending.

## Log

- 2026-10-22: planned with Sid after reviewing the teacher's source; he
  asked for steps 0 to 3, the score fixes and the known-top-cards change
  including the encoder, with step 4 as a follow-on. Secret Technique and
  Secret Weapon still get no labels (1 abstention in 35,054 round 2
  labels); no fix is planned.
- 2026-10-22: the search already returns per-move visits and values, so
  step 0 and the vote rules need no native work; the benchmark runs in the
  label collector (teacher-played decisions are marked never trainable).
- 2026-10-22, 17:43: benchmark arms queued for four vote rules on the same
  1,024 second-DEV worlds. In the first 110 worlds, every trajectory matched
  the champion's own game up to the teacher's first move, as designed.
- 2026-10-22: known top cards occur in about 0.03% of decisions; Sid agreed
  to do that work after the vote rule, score and budget fixes.
- 2026-10-22, 17:51: noise measurement done (table above). It changes the
  order of work: fix the vote rule and raise the budget before building
  information-set search.
- 2026-10-22, 19:35: benchmark: the teacher at bosses wins 32.5 to 33.7%
  against the champion's 20.0%; the student has not learned the teacher's
  boss play.
- 2026-10-23, 00:21: full reruns: pooled searches win 429 against 331
  (p < 0.001), the clear best rule.
- 2026-10-22, 22:16: six arms analysed; pooled and most-visited arms
  stopped by low disk space (9.3 GiB free); rerun once space is freed.

Code: `experiments/2026-10-22-better-teacher/` (not yet written).
