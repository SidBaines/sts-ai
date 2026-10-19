# Combat PPO on a fight bank, and early search-label imitation

**Status:** finished 13 October 2026. **Verdict:** neither fight-bank PPO nor
these early search-imitation recipes established better full-game play; the
more aggressive sparse PPO run and boss-label imitation caused regressions.

## Question

Could replaying difficult fights give proximal policy optimization (PPO) a
cleaner learning signal than whole games? Could copying the search teacher's
choices improve combat without paying for search at play time? These test
[combat PPO](../strategies.md#combat-ppo) and [distilling the search
teacher](../strategies.md#distilling-the-search-teacher).

## Design

Only the combat network was trained; out-of-combat (OOC) weights stayed fixed
within each comparison. The candidate-scoring network assigned a score to each
legal action. The task was Ironclad, Ascension 0, Acts 1 to 3, on the
historical environment 0.1.1 research builds, including added search bindings.

- **Sparse fight-bank PPO:** start from OOC PPO update 8 with its inherited
  combat network. A fight bank, a collection of learner-reached fight starts,
  held 3,968 boss and elite starts from 1,500 training games. Whole worlds
  were held out for evaluation. Each update sampled 1,024 fights: 35% Act 2
  bosses, 35% Act 2 elites, 15% Act 3 and 15% Act 1 for retention. Reward was
  +1 for a win or −1 for a loss, plus 0.3 times the fraction of player hit
  points (HP) retained on wins. Discount factor and generalized advantage
  estimation parameter were both 1, with a policy-divergence guard of 0.05 and
  minibatch 4,096. The first run used learning rate 0.0001 and three passes
  over each batch. The second continued from its update-10 checkpoint at
  learning rate 0.0003 and four passes: two settings changed together.
- **Dense shaping:** start from the policy taught to upgrade cards at rest
  sites at 85% HP or above. Its new bank contained 21,051 fights, including
  16,860 hallway fights. Per-step rewards added the change in a potential: 0.3
  times player HP fraction minus 0.5 times enemy HP fraction. This used
  learning rate 0.0001, generalized advantage estimation parameter 0.95,
  value-loss coefficient 0.5 and no terminal HP bonus.
- **Privileged labels:** approximately 51,000 labels from 600 learner games,
  covering bosses, elites and low-HP Act 2 fights. Dataset aggregation
  (DAgger) here means asking the teacher to label states the learner visits.
  Each label used 2,000 search simulations over three shuffled hidden draw
  orders. Fitting included a policy-divergence penalty on rehearsal states to
  preserve earlier behaviour.
- **RNG-resampled boss labels:** 18,683 labels and 3,166 rehearsal decisions
  from 320 games of round-2 OOC PPO update 4. Random-number-generator (RNG)
  streams and hidden draw orders were resampled for 24 searches of 1,000
  simulations per decision. The network fitted the full vote distribution,
  with rehearsal, for four passes at learning rate 0.0001. The source called
  these labels “public search”.
- **Evaluation and budget:** greedy held-out fight panels every two sparse
  updates, and paired full games every ten, on 1,024 requested DEV worlds
  reused for development (the sources called these TRAIN-split standard
  evaluation worlds). Sparse runs were configured for 40 updates each; the
  second stopped at update 26. Dense shaping was configured for 30 and stopped
  at 20. The original expert-iteration proposal estimated one day.
- **Decision rule and approval:** the prior proposal required at least an
  8-point boss-fight win-rate gain without hallway regression, then full-game
  evaluation. The sources do not document a separate fixed promotion rule or
  Sid's dated approval for this series. These results are exploratory.

## Results

No tested combat replacement established a full-game gain against its own
unchanged-combat baseline. The full games used paired, reused DEV worlds.
Changes are given as the paired difference ± its standard error, as the
sources report them; a 95% interval is about twice as wide.

| Combat replacement | Unchanged-combat baseline | Paired mean-floor change | Evaluation size |
|---|---|---:|---|
| Sparse PPO, update 10 | OOC PPO update 8 | +0.29 ± 0.20 | About 1,016 valid games |
| More aggressive sparse PPO, update 20 | OOC PPO update 8 | −0.87 ± 0.23 | About 1,016 valid games |
| First sparse model paired with later OOC | OOC PPO update 24 | +0.09 ± 0.21 | 1,024 worlds requested |
| Dense shaping, update 20 | Distilled card-upgrade policy | +0.08 ± 0.24 | 1,024 worlds requested |
| Privileged search imitation | Original policy | −0.34 ± 0.22 | Valid count not stated |

The second sparse run was already worse at update 10 (−0.51 ± 0.22 mean
floor). Act 1 boss deaths rose from 98 to 137 in about 1,016 games. Panels of
120 held-out fights per group suggested elite gains but missed this
regression. Replaying 300 bank fights with the generating policy reproduced
all 300 outcomes. Across approximately 39,000 sparse training fights, sampled
fight win rate only moved from 51% to 55%.

Dense shaping also showed no clear mean-floor improvement: its update-10
change was −0.20 ± 0.22, and full-game wins fell from 19 to 15 by update 20 on
the requested 1,024-world panel. The held-out fight panel contained 1,537
fights; Act 2 boss fight win rate moved from 33.3% to 33.8%. Sampled fight win
rate stayed near 64% over about 20,000 training fights.

Imitation improved agreement with labels while weakening play. On held-out
labels, privileged-label agreement rose from 46% to 52%, but the Act 2 boss
pass rate fell from 23.2% to 13.4%. For the RNG-resampled labels, agreement
rose from 44.5% to 50.3%, while the paired Act 2 boss panel fell from 29.5% to
23.3% across 400 fights: −6.2 ± 2.3 percentage points, exploratory. This
student was rejected on the panel; no full-game comparison is reported.

## Conclusion

These particular combat-training recipes failed to improve full-game play.
Increasing the sparse run's learning rate and training passes made it worse;
adding hallway retention and dense HP credit did not produce a useful gain.
Better agreement with search labels did not establish stronger combat.

Sparse delayed credit, forgetting outside the emphasized fights, noisy labels,
missing state information and copying fragments of incompatible plans are
plausible explanations. These tests do not isolate one cause or establish that
combat PPO or search distillation cannot work at a larger scale.

Caveats:

- The worlds were reused for development, with no fresh confirmation worlds or
  second training seed. Some sample sizes are not recorded.
- These are historical simulator conditions. Later fixes require matched new
  baselines. Older labels included partially faulted worlds; simulator faults
  are not losses, and complete fault accounting is unavailable here.
- RNG resampling did not certify public-information-only search. The old
  sampler could depend on true hidden pile order, so hidden information was
  not ruled out as a contributor to imitation failure.
- The implemented terminal potential changes the return; the dense reward must
  not be described as guaranteed to preserve the original optimal policy.

## Next

Keep the retained combat network until a replacement improves paired full
games. Audit whether its representation contains the information needed for
planning. If revisiting imitation, first verify a search teacher that actually
plays coherent fights, then train on those trajectories and fight outcomes
with retention checks. Larger, representative fight panels and full-game win
rate should govern selection. [Play-time
search](2026-10-14-play-time-search.md) is a separate way to use the teacher
without copying isolated decisions.

## Log

- 13 Oct: privileged imitation weakened boss play; focus moved to fight-bank
  PPO.
- 13 Oct: sparse PPO continued with a higher learning rate and more passes,
  then stopped after a full-game regression.
- 13 Oct: dense shaping and hallway retention were added; stopped at update 20
  without a gain.
- 13 Oct: RNG-resampled boss-label imitation worsened the Act 2 boss panel and
  was rejected.
- 14 Oct: a later review qualified the old search labels and found no single
  identified cause of failure.

Code: `src/sts_rl/combat_trainer.py`, `src/sts_rl/distill.py`,
`src/sts_rl/panel.py`, `scripts/rl/train_combat.py` and
`scripts/rl/collect_combat_labels.py` (no single commit recorded). Data: runs
`combat_bossrl_v1`, `combat_bossrl_v2`, `combat_v3_shaped` and label set
`combat_pub_boss_r2i4` (local).
