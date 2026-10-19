# Results ledger

One row per finished comparison, newest first. Intervals are 95% paired
intervals; "±" marks a paired standard error where that is all the source
reported. Every row names its environment version, because results from
different versions are not comparable. Evidence and verdict terms are defined
in `CLAUDE.md`; other terms in [metrics](metrics.md).

| Date | Experiment | Tested | Result | Evidence | Verdict |
|---|---|---|---|---|---|
| 2026-10-16 | [Fresh start](experiments/2026-10-15-fresh-start.md) | Old-encoder combat distilled from search, with champion OOC, vs champion | 0.1.5: DEV wins 222 vs 143 of 4,096 (+1.9 points [+1.2, +2.7]; p = 2.6 × 10⁻⁶) | Exploratory | Better |
| 2026-10-16 | [Fresh start](experiments/2026-10-15-fresh-start.md) | Old-encoder distilled combat with new OOC student vs champion | 0.1.5: DEV wins 196 vs 143 of 4,096 (+1.3 points [+0.5, +2.1]) | Exploratory | Better |
| 2026-10-16 | [Fresh start](experiments/2026-10-15-fresh-start.md) | Fresh agent (structured combat and new OOC student) vs champion | 0.1.5: DEV wins 119 vs 143 of 4,096 (−0.6 points [−1.3, +0.1]); mean floor −0.48 [−0.84, −0.11] | Planned | Inconclusive |
| 2026-10-16 | [Fresh start](experiments/2026-10-15-fresh-start.md) | Structured combat student, with champion OOC, vs champion | 0.1.5: DEV wins 131 vs 143 of 4,096 (−0.3 points [−1.0, +0.4]) | Planned | No clear difference |
| 2026-10-16 | [Fresh start](experiments/2026-10-15-fresh-start.md) | New OOC student, with champion combat, vs champion | 0.1.5: DEV wins 161 vs 143 of 4,096 (+0.4 points [−0.3, +1.2]); mean floor −0.43 [−0.77, −0.09] | Planned | No clear difference |
| 2026-10-16 | [Fresh start](experiments/2026-10-15-fresh-start.md) | Structured vs old encoder, same teacher labels | 0.1.5: test agreement 58.7% vs 58.4% (+0.3 points [−0.3, +0.9]); Act 2 boss panel HP lost +1.05 [+0.26, +1.83] | Planned | Worse |
| 2026-10-15 | [Joint sparse-win PPO](experiments/2026-10-14-joint-sparse-win-ppo.md) | Repaired two-hour joint run vs champion | 0.1.4: DEV wins 62 vs 58 of 1,024 (+0.4 points [−1.0, +1.8]) | Planned | Inconclusive |
| 2026-10-14 | [Joint sparse-win PPO](experiments/2026-10-14-joint-sparse-win-ppo.md) | First joint pilot vs champion | 0.1.4: DEV wins 55 vs 58 of 1,024 (p = 0.78) | Planned | Inconclusive |
| 2026-10-14 | [Play-time search](experiments/2026-10-14-play-time-search.md) | Search at Act 2–3 bosses and elites vs learned-only champion | 0.1.4: fresh wins 333 vs 213 of 4,096 (8.1% vs 5.2%; p = 9 × 10⁻¹³); mean floor +1.85 [+1.61, +2.07] | Confirmed | Better |
| 2026-10-14 | [Play-time search](experiments/2026-10-14-play-time-search.md) | Adding hallway fights to boss and elite search | 0.1.4: DEV wins 98 vs 42 of 512 (19.1% vs 8.2%); mean floor +2.27 [+1.59, +3.00] | Exploratory | Better |
| 2026-10-14 | [Play-time search](experiments/2026-10-14-play-time-search.md) | Boss-only search vs learned-only champion | 0.1.4: DEV wins 251 vs 219 of 4,096 (p = 0.04); mean floor +0.86 [+0.67, +1.05] | Exploratory | Better |
| 2026-10-14 | [Overnight tests](experiments/2026-10-14-overnight-tests.md) | Three short imitation pilots on teacher-played fights vs parent | 0.1.4: DEV wins 50, 52 and 46 vs 57 of 1,024 (p = 0.44, 0.62, 0.22) | Exploratory | Inconclusive |
| 2026-10-14 | [Overnight tests](experiments/2026-10-14-overnight-tests.md) | HP-curated imitation with the parent's final-boss combat restored | 0.1.4: DEV wins 61 vs 46 (curated) and 57 (parent) of 1,024; p = 0.008 and 0.72 | Exploratory | Better than curated; no clear difference from parent |
| 2026-10-14 | [Overnight tests](experiments/2026-10-14-overnight-tests.md) | Eight more OOC PPO iterations from the champion | 0.1.3: DEV wins 58 vs 57 of 1,024; mean floor +0.47 [−0.10, +1.08] | Planned | No clear difference |
| 2026-10-14 | [Overnight tests](experiments/2026-10-14-overnight-tests.md) | Same, with card-reward sampling temperature 4 | 0.1.3: DEV wins 57 vs 57 of 1,024; mean floor −0.72 [−1.39, −0.04] against the plain continuation | Planned | Worse |
| 2026-10-14 | [Overnight tests](experiments/2026-10-14-overnight-tests.md) | Hide Neow's boss-relic swap | 0.1.3: DEV wins 29 vs 57 of 1,024 (p = 0.0008); mean floor −1.35 [−2.05, −0.61] | Exploratory | Worse |
| 2026-10-14 | [Overnight tests](experiments/2026-10-14-overnight-tests.md) | Buy one card removal by floor 33, policy picks the card | 0.1.3: DEV wins 41 vs 57 of 1,024 (p = 0.05); mean floor −1.70 [−2.34, −1.04] | Exploratory | Worse |
| 2026-10-14 | [Overnight tests](experiments/2026-10-14-overnight-tests.md) | Buy one removal by floor 33, unupgraded Strike only | 0.1.3: DEV wins 49 vs 57 of 1,024 (p = 0.37); mean floor −0.13 [−0.70, +0.49] | Exploratory | Inconclusive |
| 2026-10-14 | [OOC PPO](experiments/2026-10-13-ooc-macro-credit-ppo.md) | Round-2 update 20 vs update 12 (choosing the champion) | 0.1.3: DEV wins 220 vs 146 of 4,096 (p = 4 × 10⁻⁶); mean floor +0.64 [+0.35, +0.92] | Exploratory | Better |
| 2026-10-13 | [OOC PPO](experiments/2026-10-13-ooc-macro-credit-ppo.md) | OOC macro-credit PPO update 36 vs original composite | Before 0.1.3: fresh mean floor +2.99 [+2.39, +3.57] on 1,024 worlds | Confirmed | Better |
| 2026-10-13 | [OOC PPO](experiments/2026-10-13-ooc-macro-credit-ppo.md) | Distilled smithing at high HP on top of update 36 | Before 0.1.3: fresh mean floor +1.34 [+0.85, +1.83] on 1,024 worlds | Confirmed | Better |
| 2026-10-13 | [OOC PPO](experiments/2026-10-13-ooc-macro-credit-ppo.md) | Lambda 1 vs the smith-distilled start, update 12 | Before 0.1.3: mean floor +0.87 [+0.59, +1.13] on 4,096 worlds; wins p = 0.58 | Exploratory | Better |
| 2026-10-13 | [OOC PPO](experiments/2026-10-13-ooc-macro-credit-ppo.md) | Lambda 0.95 vs the smith-distilled start, update 16 | Before 0.1.3: mean floor −0.54 [−1.07, −0.01] on 1,024 worlds; smithing mostly unlearned | Exploratory | Worse |
| 2026-10-13 | [OOC PPO](experiments/2026-10-13-ooc-macro-credit-ppo.md) | Take an offered elite at HP ≥ 90% (overlay) | Before 0.1.3: wins 94 vs 77 of 4,096 (p = 0.10); mean floor −0.06 [−0.29, +0.16] | Exploratory | Inconclusive |
| 2026-10-13 | [Play-time search](experiments/2026-10-14-play-time-search.md) | Early boss-only search vs no search | Before 0.1.3: wins 69 vs 53 of 2,019 (p = 0.09); mean floor +0.78 [+0.54, +1.02] | Exploratory | Inconclusive |
| 2026-10-13 | [Combat PPO](experiments/2026-10-13-combat-ppo.md) | Resampled boss-label imitation vs parent | Before 0.1.3: Act 2 boss fight wins 23.3% vs 29.5% of 400 fights (−6.2 ± 2.3 points) | Exploratory | Worse |
| 2026-10-13 | [Combat PPO](experiments/2026-10-13-combat-ppo.md) | Sparse fight-bank PPO, higher learning rate and more passes | Before 0.1.3: mean floor −0.87 ± 0.23 on about 1,016 games | Exploratory | Worse |
| 2026-10-13 | [Combat PPO](experiments/2026-10-13-combat-ppo.md) | Sparse fight-bank PPO, update 10 | Before 0.1.3: mean floor +0.29 ± 0.20 on about 1,016 games | Exploratory | Inconclusive |
| 2026-10-13 | [Combat PPO](experiments/2026-10-13-combat-ppo.md) | Dense HP-shaped fight-bank PPO | Before 0.1.3: mean floor +0.08 ± 0.24; wins 15 vs 19 of 1,024 | Exploratory | Inconclusive |
| 2026-10-12 | [Long curricula](experiments/2026-10-12-long-curricula.md) | Continuous joint v2 learner vs original | 0.1.1: wins 0 vs 2 of 192 in two evaluations; mean floor −2.39 and −1.73; Act 1 clears −21 and −13 | Planned | Worse |
| 2026-10-12 | [Long curricula](experiments/2026-10-12-long-curricula.md) | v1 selected model vs original | 0.1.1: wins 0 vs 0 of 48; mean floor 30.90 vs 28.60; only 511 of 36,217 updates retained | Exploratory | Inconclusive |
| 2026-10-12 | [First curriculum](experiments/2026-10-11-first-curriculum.md) | Full learned Act 1 control vs native search with scripted OOC | 0.1.1: final Act 1 clears 20 vs 6 of 21 (+65 points [+45, +85]) | Confirmed | Better |
| 2026-10-12 | [First curriculum](experiments/2026-10-11-first-curriculum.md) | Full learned control, three acts | 0.1.1: final wins 0 of 21 (one fault) | Confirmed | Inconclusive |
