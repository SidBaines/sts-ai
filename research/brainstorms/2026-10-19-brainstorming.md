# Brainstorm, 19 October 2026: where the agent stands and what to try next

Written by Claude on 19 and 20 October 2026 after an independent look at the
code, the run data and the three audits below, at Sid's request. It is kept
for reference. Section 7 records what Sid and Claude then decided.

Everything in sections 1 to 2 marked *verified* was checked directly in code
or data. Claims marked *inferred* are interpretations. The analyses behind
the numbers are listed in the appendix so that they can be re-run.

## 1. Summary

- The agent loses its games at bosses, above all the Act 2 boss. The reason
  is not mainly how it plays the fights: it is the decks it brings. They are
  uniformly weak, in the same way every game, so the full-game reward carries
  almost no information about deck-building choices. This is the bottleneck.
- The "search teacher" used for distillation is weaker and noisier than the
  notes suggest. It is a Monte Carlo tree search with uniformly random
  playouts, scored by a hand-made HP formula, whose answer is the first move
  of its single best rollout. Its labels have no majority 43% of the time.
- The teacher is, in effect, a public-information player: it reshuffles the
  hidden piles and reseeds the random streams for every sample. The notes
  call it privileged because this has not been certified, not because it
  reads the true draw order.
- The champion's whole out-of-combat training amounted to about 52,000 games
  and 2,600 optimiser steps, with noise-dominated gradients and no
  exploration mechanism. The learning recipe has never been tried at a
  budget where it could work.
- Throughput is about ten times below what the simulator allows: the
  per-decision cost is Python observation building and a cross-process
  round trip, not the engine.

## 2. What the data says

### 2.1 Where games end

Champion on the 4,096 second-DEV worlds, environment 0.1.5 (`verified`):

| | Act 1 | Act 2 | Act 3 |
|---|---:|---:|---:|
| Reach the boss | 4,017 (98.1%) | 3,033 (74.0%) | 810 (19.8%) |
| Beat the boss, of those reaching it | 3,783 (94.2%) | 1,346 (44.4%) | 156 (19.3%) |
| HP on entering the boss, fraction of max | 0.90 | 0.84 | 0.76 |
| Deaths in the act | 313 | 2,437 | 1,190 |

Of 3,940 losses, 2,575 (65%) are boss fights: 1,687 at the Act 2 boss (43%
of all losses), 636 at the Act 3 boss, 234 at the Act 1 boss. The remaining
1,365 are hallway, elite and event deaths spread over floors 23 to 31 and
40 to 48.

The three finished representation-controls arms show the same shape. The
old-encoder distilled combat models win more (197 and 221) by reaching the
Act 3 boss more often (1,098 and 1,186 worlds against 810) with more HP
(0.88 against 0.84 at the Act 2 boss), while passing the Act 3 boss slightly
*less* often (17.9% and 18.6% against 19.3%). The plain structured model at
seed 2020 (93 wins) is worse everywhere.

Search at Act 2 and 3 bosses and elites, environment 0.1.4 (`verified` from
the run logs): Act 2 boss pass rate 44.8% → 53.7%, Act 3 boss pass rate
25.8% → 29.2%, wins 213 → 333 of 4,096. So even a searcher that plays each
boss fight with a simulator loses about half of them. Better play of a weak
deck is worth about nine points at the Act 2 boss; the rest is the deck.

### 2.2 The decks

Replay of 300 champion games to the moment the Act 2 boss fight starts
(`verified`; script in the appendix):

| | All 300 | Passed (127) | Failed (173) |
|---|---:|---:|---:|
| Deck size | 25.2 | 25.5 | 25.0 |
| Strikes / Defends | 4.6 / 3.8 | 4.7 / 3.8 | 4.5 / 3.8 |
| Upgraded cards | 6.4 | 6.5 | 6.4 |
| Attacks / skills / powers | 15.9 / 7.1 / 1.6 | 16.0 / 7.2 / 1.8 | 15.8 / 7.1 / 1.5 |
| Curses | 0.41 | 0.41 | 0.41 |
| HP, fraction of max | 0.86 | **0.94** | **0.80** |
| Potions held | 1.35 | **1.61** | **1.17** |
| Relics | 7.8 | 8.2 | 7.6 |
| Energy per turn | 4.45 | 4.50 | 4.42 |

- The starter cards are all still there: the champion removed a card 0 times
  in 22,205 opportunities and skipped 2.3% of card rewards (3,618 of
  154,325 chances). 30% of its upgrades are on Strikes and Defends.
- Its favourite cards (share of decks): Shrug It Off 61%, Clothesline 58%,
  Perfected Strike 52%, Twin Strike 51%, Iron Wave 45%, Pommel Strike+ 42%,
  Anger+ 36%, Armaments 34%, Metallicize 33%, Inflame 32%. Only 90 distinct
  cards appear in 300 decks. Demon Form is in 8% of decks, Barricade in 1%,
  Feel No Pain, Limit Break and Offering in about 1%, Disarm in 0.3%.
- Boss relics held: Cursed Key 21%, Velvet Choker 20%, Fusion Hammer 19%,
  Busted Crown 17%, Snecko Eye 17%, Sozu 14%, Ectoplasm 14%, Runic Dome 10%,
  Mark of Pain 10%.
- Pass rate by boss: Collector 48%, Champ 41%, Bronze Automaton 38%.
- Decks that passed and decks that failed are indistinguishable. What
  separates them is HP on entry and potions held, both outcomes of the
  hallway fights before the boss.

Inference: because every game builds the same kind of deck, the variance in
outcomes is almost all fight randomness and HP, so a reinforcement signal at
the end of the game says nearly nothing about card choices. This explains
the plateau of out-of-combat PPO and why the distilled combat model's gain
came from hallway fights rather than bosses.

Other behaviour counters (`verified`): smithing at HP ≥ 85% in 78% of
opportunities, at HP 50 to 85% in 14%, below 50% never; spends gold in 83%
of shop visits; takes an offered elite at HP ≥ 90% in 87% of cases; drinks a
potion in 9.9% of combat decisions where one is available.

### 2.3 The search teacher

`verified` in the C++ source and the native patch:

- `BattleScumSearcher2` is UCT with uniformly random playouts to the end of
  the fight (exploration constant 3√2), 1,000 iterations per sample.
- The end-state score is `100 × (35 + HP + 4 × potions − 0.01 × turn)` for a
  win, otherwise a small formula of enemy HP ratio, enemies alive, energy
  wasted, cards drawn and potions. It does not estimate win probability.
- The chosen action is the first move of the single best-scoring playout if
  any playout won; only otherwise the most-visited root move.
- The research teacher runs 24 such searches, each on a copy of the fight
  with every random stream reseeded, the hidden piles sorted into a
  canonical order and the draw pile reshuffled with a seed derived from the
  public observation, then votes by plurality. That is a standard
  determinised search over public information. The docs decline to certify
  it as public-only because card-instance payloads and tie-breaks have not
  been checked, not because it reads the true draw order.

Label statistics on 400 label worlds (32,936 combat labels, `verified`):
the median plurality is 12 of 24 votes; the quartiles are 9 and 17; 43% of
labels have no majority; mean vote entropy 1.02 nats; mean 5.4 legal
actions. 34% of labels are Act 2 boss states, 17% Act 2 elites, 13% Act 2
hallway fights. The champion agrees with the plurality 44% of the time; the
distilled students reach 58%; the teacher repeats its own answer 71% of the
time. Teacher and champion end the turn equally often (17.5% and 17.3%).

### 2.4 The distilled students

Act 2 boss fight panel, 400 champion boss starts × 8 reseeds (`verified`):
the champion wins 44.2% of these fights and loses 58.8 HP per fight (a death
counts as all the entry HP). Every one of the 28 representation-controls
students wins *fewer* of these fights (33.2% to 42.3%), while HP lost ranges
from 1.4 better to 3.4 worse. In Act 2 hallway fights (champion: 11.1 HP
lost, 96% won) the students lose 0.5 to 3.0 HP less. So the students are
better in hallways and no better at bosses, which matches the full-game
decomposition above. The panel's primary metric, HP lost, is dominated by
deaths and pointed the wrong way for bosses; fight win rate is the better
measure there.

### 2.5 The training recipe and budget

From the trainer audit and the run logs (`verified` unless marked):

- Champion lineage: first curriculum (imitation, DAgger, PPO) → 36 updates
  of out-of-combat PPO → smith distillation → 12 updates → smith and elite
  distillation → round 2, update 20. In all about 52,000 full games, 3.9
  million out-of-combat decisions, 2,640 PPO minibatch steps and six laptop
  hours.
- Logged half-batch gradient cosines of 0.09 to 0.21, negative in several
  round-2 iterations; adjacent round-2 checkpoints scored 36, 35, 52, 44, 57,
  37, 39 and 47 wins on the same 1,024 worlds, and the champion is the 57.
- Entropy bonus 0; measured out-of-combat policy entropy 0.13 to 0.2 nats;
  the policy never samples removal, elites or smithing often enough to
  learn about them (the authors' own diagnosis).
- The critic shares its trunk with the actor and regresses unnormalised
  λ = 1 returns with weight 0.5; value loss grew from 0.011 to 0.25 over the
  lineage and advantage standard deviation from 0.15 to 0.72. Inferred: the
  trunk is being shaped by a regression it cannot fit.
- Silent defaults: AdamW weight decay 0.01; λ = 0.95 in the config class
  although the documented recipe is λ = 1; rule distillation saves the last
  epoch with no selection (one run peaked at 0.90 validation top-1 and saved
  0.82).
- Reward: +0.05 per floor, +1 for a win, −1 for a defeat. A death at the
  Act 2 boss scores +0.65; a win +3.55.
- Joint sparse-win PPO was built not to move: one epoch, learning rates
  1e-5 and 3e-5, KL anchors to the original policy, rollback on a 20%
  change of top actions. Its null result says little about joint training.
- No correctness bug was found in GAE, masking, probability ratios or
  bootstrapping.

### 2.6 Engineering and throughput

`verified` from the fresh-start benchmarks and the throughput notes:

- One collection process makes about 420 decisions per second. Encoding
  plus network inference is about 15% of that time (0.4 ms per decision on
  fixed observations, 2,000 to 2,800 decisions per second). The rest is the
  native worker process: building the observation in Python and the
  per-decision round trip to the isolated process.
- The engine itself answers a 200-simulation search in about 1.4 ms.
- A full game costs about 2.5 seconds per process; 7 lanes give about 3
  games per second, 11,000 games per hour on this laptop.
- The native Python API already exposes `copy`, `enter_battle`,
  `exit_battle`, `remove_card`, `obtain_relic`, `obtain_potion`,
  `reseed_rngs`, `search_best_action` and `resolve_current_battle`. There
  is no `obtain_card`.

### 2.7 Representation

From the encoder audit (`verified` in code; severity inferred):

- The old encoder's 256 state features are 64 scalars plus a 192-bucket
  signed hash sketch into which every card group, power, enemy, relic,
  potion, map room and offered item is summed. Identities collide heavily.
  Its 128 per-action features carry the hand-made card rules: cost,
  displayed damage, block from a 13-card table, keyword flags from substring
  tests (which misfire: "rest" matches "for the rest of combat", "+"
  matches "gain +2 max HP"), target HP and lethal flags.
- The structured entity encoder has one token per entity with identity,
  type, rarity, cost and counts, and no card semantics at all: the effect of
  each of about 100 relevant cards has to be learned from 113,000 labels.
- Neither network has memory across turns; the "recent actions" buffer is
  cleared at every turn boundary. Enemy move history is not represented.
- Environment gaps: shops hide items the player cannot afford, so saving
  gold for a target cannot be learned; Gambling Chip and exhaust-many
  effects are unplayable; the per-world limit of 4,096 decisions censors a
  looping policy rather than punishing it.

## 3. Problems with the current setup, ranked by likely effect on win rate

1. **No exploration in deck-building**, and a reward that mostly pays for
   surviving hallways. The out-of-combat policy inherited "take every common
   attack, never remove, upgrade whatever" from the scripted agent it was
   cloned from, and PPO has never left that basin.
2. **A weak, high-variance combat teacher**, whose label rule (first move of
   the best single rollout, scored by HP) is nearly the worst available from
   a tree search, and whose labels are two-thirds boss states where the
   students then underperform.
3. **The wrong fight-panel metric for bosses** (HP lost rather than win rate),
   which steered the encoder study and would steer expert iteration.
4. **Throughput** about ten times below what the engine allows, so every PPO
   run is under-powered and every batch noise-dominated.
5. **Shared-trunk critic, unnormalised returns, no entropy or exploration
   temperature**, and checkpoint selection on 1,024 reused worlds (standard
   error about 7 wins).
6. **Representation**: hash collisions in the old encoder, no card semantics
   in the new one, no cross-turn memory, no enemy move history.

## 4. Ideas

### A. A deck-strength oracle from simulated fights

The engine can copy a game state, change the deck, start the stored
encounter and play it out with search in about a millisecond per search
query. So for any deck we can estimate the chance of beating the upcoming
boss, which is public information, by playing that fight natively many
times. Uses:

1. A teacher for out-of-combat choices: score each card reward, shop
   purchase, rest-site option and removal by the oracle, and distil.
2. A dense, low-variance reward for out-of-combat PPO: oracle after the
   choice minus oracle before it.
3. Training data for a learned value network over (deck, relics, HP, boss),
   from millions of synthetic fights. This is the "boss-pass chance from
   repeated fights" variant that the fight-value work proposed and never
   tried.

It attacks the bottleneck directly and sidesteps the full-game credit
problem. Cost: a small native patch to add `obtain_card`, and about a day
of tooling. First test: do oracle scores separate the 127 passing from the
173 failing decks in the replay above?

### B. Search as the backbone

1. Certify the determinised search as a public-information player (the
   open items are card-instance payloads and tie-breaks) and use it at play
   time in every fight: 8.1% confirmed and 19% exploratory win rates exist
   today against 3.5% for the network alone.
2. Make the searcher stronger: replace random playouts with the distilled
   network as rollout and prior policy, and use win probability plus HP as
   the value, in the AlphaZero style. A stronger teacher gives better labels
   and a stronger network each round; this is expert iteration done
   properly, and it compounds.

### C. Better labels from the searcher we already have

Vote by visit counts or per-action values instead of the best rollout; store
per-action values and distil the full ordering; spend more samples on
ambiguous states; weight hallway and elite states, where students actually
gain; label the student's own states (DAgger).

### D. Throughput before more PPO

Put network inference inside the worker processes, removing the per-decision
round trip; then step many games per process; later, compute the numeric
encoding in C++. Expected three to ten times faster. Cores, not a GPU, are
the limit: the engine is CPU-bound and the networks are small.

### E. PPO hygiene, all cheap

Detach the critic from the actor's trunk; normalise returns; add an entropy
bonus or ε-exploration on strategic menus only; use bigger batches at lower
learning rates; remove the silent weight decay; make λ = 1 the default;
select checkpoints on 4,096 worlds or on boss pass rates from panels.

### F. Representation, after the current study reports

Explicit card attributes parsed from the glossary (damage, block, draw,
exhaust, cost) as numeric features for the entity model; enemy intent
history and cards exhausted this fight; a deck representation for
out-of-combat play that is not a hashed bag. A recurrent policy would give
memory across turns; it is more complex and is left for later.

### G. Measurement

Report reach × pass per act for every evaluation (the records already hold
it); use fight win rate at bosses and HP lost in hallways; log deck
statistics at boss entry so that out-of-combat changes can be read directly.

### H. Bigger bets for later

Full-game planning at out-of-combat decisions with the simulator as the
model and the network as prior and value; a learned win-probability model
for the whole run, used for credit, search evaluation and checkpoint
selection; reward redesign around boss-pass milestones once an oracle
exists.

## 5. Suggested order, as first proposed

1. Finish and write up representation controls.
2. Feasibility spike for the oracle (A): patch `obtain_card`, time synthetic
   Act 2 boss fights with search, and test whether oracle scores separate
   passing from failing decks. One core, about half a day.
3. Throughput (D) in parallel, since everything else waits on it.
4. Choose between oracle-driven out-of-combat distillation and
   search-driven expert iteration based on the spike.

## 6. Caveats

- The deck analysis used 300 of one policy's games on one environment
  version.
- "Teacher weakness" is inferred from the code and vote statistics, not
  from a head-to-head against a stronger searcher.
- The gradient-noise and budget figures come from run logs read by the
  trainer audit and were not re-derived.
- Severity of the representation problems (hash collisions, keyword
  misfires) is inferred from code, not measured.

## 7. Discussion with Sid, 20 October

Sid's responses to the first version of this document:

- The deck analysis is the kind of work he wanted; the weak deck is the
  thing to focus on next.
- No search-based agent for now. The goal is a pure policy player. A
  recurrent policy may eventually be needed for memory across turns; that
  is noted under F and left for later.
- Using better search as a better teacher (B.2, C) is welcome.
- Throughput work (D) is important and should start at once, using Codex
  through the codex-driven-development skill.
- PPO hygiene (E) and the other cheap fixes should be kicked off when
  appropriate.
- krill-mill is not to be used for this project for now: it has fewer cores
  than the laptop and holds more critical work.

Decisions and notes added as the discussion continues:

- 20 October: Sid approved carrying out the plan in order: write up
  representation controls, start the throughput work with Codex, run the
  oracle spike, then PPO hygiene. Sid approved the `brainstorms/` document
  type, now in `CLAUDE.md`.
- Correction to section 4D: a profile of champion games puts the realistic
  gain from Python-side changes at about 2 to 3 times per core, not 3 to 10.
  Per decision through the worker process: about 0.58 ms in the environment
  step (over half of it `copy.deepcopy` of the observation and act map, plus
  observation text rendering), 0.32 ms round trip to the worker process,
  0.19 ms encoding and 0.20 ms network forward. On an idle machine one
  evaluator process makes about 635 decisions per second; seven at once fall
  to about 285 each, so contention matters as much as per-process speed.
- Throughput is two Codex tasks: removing redundant copies and text work from
  the environment step (branch `env/fast-step` from `main`, for a pull
  request and a new environment tag), and a batched in-process game runner
  for evaluation (branch `nonllm/throughput`). Claude reviews the diffs and
  re-runs tests and benchmarks instead of using reviewer subagents, following
  Sid's earlier preference.
- The oracle needs no native patch: the game state can be copied, the act's
  boss fight entered and played from any out-of-combat decision. The spike
  is the [deck-oracle experiment](../experiments/2026-10-20-deck-oracle.md).
- Outcomes of the 20 October plan:
  - Representation controls written up: the old encoder's lead is real at
    two seeds; rule features recover about a point, width none.
  - Environment step 3.1x faster with identical output (pull request #8);
    batched evaluator merged; together 3.4x faster full-game evaluation.
  - Deck oracle (idea A): measures deck strength well (AUC 0.78 for the real
    Act 2 boss outcome, independent of HP), correcting section 2.2's reading
    that decks barely differ; but it cannot choose between single card
    rewards, so idea A.1 (per-decision teacher) fails for card rewards and
    A.3 (a learned deck-value network) becomes the next step.
  - PPO hygiene (idea E) merged as options; not yet tested in training.

## Appendix: analyses behind the numbers

All data is local. Paths are relative to the run folders named.

- Where games end, behaviour counters, boss reach and pass: computed from
  `games/<arm>/merged/summary.json` (or the shards, for the seed-2020
  old-encoder arm) in run `representation_controls_20261016`, worktree
  `~/.codex/worktrees/representation-controls/SlayTheSpireAI`. Arms:
  `champion`, `old1919_retry1`, `old2020`, `off128_s2020`.
- Search against no search on 0.1.4: the `done` lines of
  `confirm014_elite_s24_4096/log.jsonl` and `confirm014_u20_single_4096/
  stdout.log` in `~/.codex/worktrees/4a3c/SlayTheSpireAI/data/rl_runs`.
- Decks at the Act 2 boss: `experiments/2026-10-19-brainstorming/
  deck_at_boss.py` replays recorded games with the frozen runtime and dumps
  the public state when the Act 2 boss fight starts; 300 champion worlds.
- Teacher vote statistics: the first 400 world files of shards 00 and 01 in
  `teacher_corpus_2000_v1` of run `fresh_start_20261015`, worktree
  `nonllm-fresh-start`.
- Fight panels: `panels/<model>/summary.json` in run
  `representation_controls_20261016` (two models' valid panels are in the
  `operational_recovery_*` folders).
- Code audits: three Claude subagents read the trainers (`src/sts_rl/*`,
  `src/sts_policy/training.py`, `rollout_training.py`), the encoders and
  models (`src/sts_policy/encoding.py`, `entity_*.py`, `controlled_*.py`,
  `model*.py`) and the environment and native patch (`src/sts_env/*`,
  `patches/*.patch`, the pinned `external/sts_lightspeed` source) at commit
  `d685c49`. Their findings are folded into sections 2.3 to 2.7.
