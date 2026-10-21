# Metrics, evaluation worlds and terms

Every metric, evaluation set and term used in these pages is defined here.
Scope throughout: Ironclad, Ascension 0, Acts 1 to 3; Act 4 is out of scope.

## Game outcomes

- **Win rate:** full-game victories (the Act 3 boss beaten) per world played.
  The headline metric. Unless stated otherwise, play is greedy: the network's
  highest-scoring action every time.
- **Mean floor:** average floor reached per game. A finer-grained secondary
  measure of progress.
- **Boss reach and pass:** for each act, the number of worlds where the player
  reaches the boss, and the number where it beats it. *Marginal* pass rate is
  per world played; *conditional* pass rate is per world that reached the boss.
  Reach depends on the policy, so conditional rates compare different sets of
  worlds.
- **Boss entry HP:** HP on entering a boss fight, averaged over worlds that
  reached it.
- **Act 1 retention:** how many Act 1 clears a changed policy keeps compared
  with its parent. Used to catch forgetting.
- **Simulator fault:** a simulator crash, timeout or undefined behaviour.
  Never counted as a loss; faulted games are reported separately and excluded.

## Fight panels

- **Fight panel:** a fixed set of fight starts (Act 1 boss, Act 2 and 3
  bosses and elites, hallway fights) recorded from a reference player's games.
  Each start is replayed several times with the battle's random seed changed
  (a *reseed*), so two policies can be compared on identical fights.
- **Fight win rate** and **HP lost per fight:** HP is measured before
  end-of-fight relic healing, and a death counts as zero HP left.

## Copying a teacher

- **Teacher agreement:** how often the student's top action matches the
  teacher's plurality choice, on held-out states.
- **Soft cross-entropy:** the student's loss against the teacher's full vote
  distribution. Lower is better.
- **High-confidence states:** states where at least 18 of the teacher's 24
  votes went to one action.
- **Teacher repeat agreement:** how often the teacher repeats its own choice
  when asked again with different reseeds. A reference for how noisy the
  labels are, not a strict ceiling.

## Training health

- **KL per update:** how far one PPO update moves the policy. A KL guard of
  0.05 limits it; the out-of-combat recipe that worked moved about 0.006 to
  0.016 per update.
- **Probe movement:** KL and the share of changed top actions on a fixed set
  of probe states, measured from a reference policy.
- **Critic explained variance** and **Brier score:** how well a value network
  predicts outcomes. For win prediction, beating a constant guess is not
  enough in late acts, where win chances are much higher than average.

## Statistics

- Comparisons are **paired**: both policies play the same worlds (or fight
  starts), and the result is the difference.
- **Intervals** are 95% percentile intervals from a bootstrap over whole
  worlds (10,000 or 20,000 resamples). **Win tests** are exact paired
  (McNemar) tests on the worlds only one policy won.
- **Detectable difference:** about 2.8 times the paired standard error, the
  smallest difference a comparison would detect 80% of the time.
- **Evidence levels and verdicts** follow the vocabulary in `CLAUDE.md`
  (exploratory, planned, confirmed; better, worse, no clear difference,
  inconclusive).

## Evaluation worlds

A world is one game seed. Separate seed ranges keep training, development and
confirmation worlds apart.

- **DEV worlds:** a fixed set reused across many comparisons. Results on them
  are development evidence: after choosing among variants on DEV worlds, the
  winner's advantage is likely to be overstated.
- **Fresh confirmation worlds:** never used before, evaluated once on a
  frozen candidate. The only evidence level that supports promotion.

| Environment | Use | First seed | Worlds |
|---|---|---:|---:|
| Before 0.1.3 | DEV (13 October studies) | 5,000,000,000 | 1,024 |
| Before 0.1.3 | Fresh confirmation (13 October studies) | 5,600,000,000 | 1,024 |
| Before 0.1.3 | "Win set", reused for win-rate claims (13 October studies) | 6,000,000,000 | 4,096 |
| 0.1.3 and 0.1.4 | DEV | 5,800,000,000 | 4,096 (often the first 1,024 or 512) |
| 0.1.3 and 0.1.4 | Fresh confirmation | 5,900,000,000 | 4,096 |
| 0.1.5 | Training rollouts | 90,000,000,000 | |
| 0.1.5 | Teacher labels | 90,500,000,000 | 2,000 used |
| 0.1.5 | DEV | 91,000,000,000 | 4,096 |
| 0.1.5 | Second DEV (representation controls) | 91,001,000,000 | 4,096 |
| 0.1.5 | Confirmation (not yet opened) | 91,200,000,000 | |
| 0.1.5 | Fight-panel source worlds (fresh start; representation controls) | 92,000,000,000; 92,100,000,000 | 1,024 in the second |
| 0.1.5 | Fixed probe states (the 93,000,000,000 to 94,000,000,000 namespace is reserved for them) | 93,000,000,000 | |
| 0.1.5.1 | Confirmation of the large-batch PPO final checkpoint, opened 20 October and reused for every arm of that study; the reserved 91,200,000,000 set stays unopened | 94,000,000,000 | 4,096 |
| 0.1.5.1 | Confirmation for round 4 and the Demon Form test on the 21 October champion, opened 21 October | 95,000,000,000 | 4,096 |
| 0.1.5.1 | Confirmation for the second continued training, opened 21 October afternoon | 96,000,000,000 | 4,096 |
| 0.1.5.1 | Reserved for confirming the combat student round 2 and any card rule that passes | 97,000,000,000 | 4,096 |
| 0.1.5.1 | Teacher labels, second corpus (current champion's states) | 90,500,100,000 | 2,000 |
| All | Out-of-combat PPO training worlds (`train_ooc.py`): 7,000,000,000 plus 500,000,000 per training seed plus 10,000,000 per iteration | 7,000,000,000 | games per iteration |

## Environment versions

Results from different versions are not directly comparable, because each
release changed game behaviour.

- **0.1.1:** fixed an uninitialized combat character class that invalidated
  earlier automatic-combat evidence.
- **0.1.2:** supported Ironclad's Discovery card instead of rejecting it.
- **0.1.3:** fixed shop pricing and purchases; observation format v2.
- **0.1.4:** made seeded native search independent of hidden pile order.
- **0.1.5:** fixed the remaining known faults (among them Curiosity,
  Intangible, Lagavulin waking, Flight, Angry and several events) and offered
  out-of-combat potion actions. The champion's win rate fell from about 5%
  on 0.1.4 to 3.5% on 0.1.5.

## Players, teachers and encoders

- **Champion:** the best retained learned player. From the afternoon of 21
  October it is the first continued training's final checkpoint
  (`champion_2835f40fa439.pt`, SHA-256 prefix `2835f40fa439`, in runs
  `round4_20261021/inputs` and `continued2_20261021/inputs`): the morning
  champion trained for 24 more iterations with the potion filter and the
  hygiene options; confirmed at 476 against 379 wins of 4,096 new worlds
  ([continued training](experiments/2026-10-21-round-4.md)).
- **Champion of 21 October, morning:** round 3 (`champion_7d876c939a74.pt`, SHA-256 prefix `7d876c939a74`, in
  runs `large_batch_20261020/inputs` and `round4_20261021/inputs`): the
  20 October champion's out-of-combat head with the old-encoder combat
  student as its combat head, trained for 24 more iterations with the
  potion filter; confirmed at 394 against 232 wins of 4,096 fresh worlds
  ([large-batch PPO page](experiments/2026-10-20-large-batch-ppo.md)).
- **Champion of 20 October:** the large-batch out-of-combat PPO checkpoint
  at iteration 24 (`champion_ba2fa6a2987d.pt`), trained from the previous
  champion with the same combat weights.
- Comparisons before 20 October use the **previous champion**: round-2
  out-of-combat PPO update 20 (`champion_3d135d9bd53b.pt`), with combat
  weights from the first curriculum; lineage on the
  [out-of-combat PPO page](experiments/2026-10-13-ooc-macro-credit-ppo.md).
- **Search teacher:** the simulator's combat search run on copies of the game,
  24 reseeded samples of 1,000 simulations each, choosing by plurality vote.
  It can see hidden state such as draw order, so it is privileged, not a
  public-information player.
- **Old encoder** (`public_candidate_v2`): 256 state features plus 128
  features per legal action, including hand-made card and potion rule
  features, scored by a large MLP. Used by the champion.
- **Structured entity encoder:** one token per game entity (player, enemies,
  cards, relics, potions, map nodes) fed to a small transformer. Built in the
  [fresh-start study](experiments/2026-10-15-fresh-start.md).

- **Study protocol** (deployment protocol): how every full-game number in
  this ledger from 15 October is measured: exact argmax, out-of-combat
  potion drinking and discarding hidden from the policy, "take key" and
  "recall" hidden, the visible potion inventory recovered from the raw menu
  (`fresh_study.policy_observation`; `fast_eval.py`, `fresh_collect.py`,
  `--hide-ooc-potions` elsewhere). The champion wins 3.8% under it.
- **Potions-visible protocol:** the default of `eval_policy.py` and of the
  trainer's collection and evaluation: the same filters except that
  out-of-combat potion actions stay in the menu, and the champion wastes
  them. It wins 1.7% under it on the same worlds. Numbers from the two
  protocols are not comparable.

## Terms

- **Out-of-combat (OOC):** map, card reward, shop, event and rest-site
  decisions. The combat and OOC networks are separate.
- **Greedy (argmax) evaluation:** always taking the top-scoring action, which
  makes evaluation exactly reproducible.
- **Macro credit:** treating everything between two OOC decisions (including
  whole fights) as one transition, so OOC learning is not drowned in combat
  noise.
- **Distillation:** supervised training of a network to copy a teacher's
  choices. **DAgger** labels the states the student itself visits. **Expert
  iteration** repeats DAgger rounds with retraining.
- **Lambda (GAE):** how far PPO's advantage estimates look ahead before
  relying on the critic. Lambda 1 uses actual returns only.
- **Menu overlay:** a rule that overrides the policy on one kind of menu (for
  example "smith when HP is at least 85%"), used to measure a strategy before
  training it in.
- **Rehearsal:** while fitting new labels, also training on the parent's own
  choices for other states (or penalising divergence from it), so that
  unrelated behaviour is not forgotten.
- **Fight bank:** a saved set of fight starts reached by a learner, used for
  combat training or panels.
- **Continuation:** a game resumed from a saved later-act start rather than
  from the beginning. Its success is conditional on that start, not a
  full-game win rate.
- **Public heuristic:** a simple untrained reference player that uses only
  visible information (damage, block, lethal targets and basic out-of-combat
  rules).
- **Smithing:** upgrading a card at a rest site instead of resting to heal.
- **Hygiene options:** three PPO settings tested together on 20 October: a
  detached critic (the value loss trains only the critic head), normalised
  value targets, and an entropy bonus of 0.01 on strategic menus (rest,
  shop, card reward, map and removal).
- **Harvest stack:** the champion of 20 October with the old-encoder combat
  student as its combat head and the Demon Form rule as a menu overlay.
- **Smith rate, card skip rate:** behaviour rates in the trainer's
  per-iteration record: the share of rest-site choices in that iteration's
  collected games that smith, and the share of card-reward choices that
  skip the card.
- **Phase swap:** combining one checkpoint's combat network with another's
  out-of-combat network, to see which change caused an effect.
- **Plurality vote:** the search teacher's choice: the action proposed by the
  most of its 24 samples.
- **Take rate when offered:** of the times a card appeared as an option at a
  card reward, the share of times it was taken. It depends on what else was
  offered beside it.
- **Boss card reward:** the card reward after a boss fight, where every
  option is a rare card. Measured as the screens whose offered cards are all
  rare, which includes a few rare rewards elsewhere.
