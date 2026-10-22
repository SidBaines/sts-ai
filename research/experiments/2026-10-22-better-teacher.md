# A better search teacher for combat

**Status:** planned (approved by Sid on 22 October; starts as compute frees
up from the joint run). **Verdict:** pending.

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

**Known top cards (alongside).**
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

Not started.

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

Code: `experiments/2026-10-22-better-teacher/` (not yet written).
