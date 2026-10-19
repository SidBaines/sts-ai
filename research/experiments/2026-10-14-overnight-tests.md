# Overnight tests: continued OOC training, strategy rules, teacher-played combat imitation and fight value

**Status:** finished 14 October 2026. **Verdict:** continued OOC training, the
proposed strategy rules and three short combat-imitation pilots all failed to
produce a better player. Restoring the parent's final-boss combat network
recovered the imitation pilot's losses, and fight-value prediction was not
good enough to use.

## Question

Could the champion improve through more out-of-combat (OOC) training, measured
strategy rules, imitation of teacher-played fights, or prediction of fight
outcomes? Approaches: [OOC PPO with macro
credit](../strategies.md#out-of-combat-ppo-with-macro-credit), [distilling
strategy rules](../strategies.md#distilling-strategy-rules), [distilling the
search teacher](../strategies.md#distilling-the-search-teacher) and
[fight-value prediction](../strategies.md#fight-value-prediction). The same
night's search results are on the [play-time search
page](2026-10-14-play-time-search.md).

## Design

All training started from the champion (round-2 OOC update 20), with the other
network frozen. Strategy rules changed available choices without training a
network.

- **Environments:** the OOC and rule tests used environment 0.1.3 (corrected
  shops); the combat and value tests used the 0.1.4 condition.
- **Continued OOC PPO:** two pilots of eight iterations of 768 games each, on
  identical worlds, with the recipe from the [OOC PPO
  study](2026-10-13-ooc-macro-credit-ppo.md) (macro credit, lambda 1, learning
  rate 0.0003, KL guard 0.05) and a fresh optimizer. The variant changed only
  the card-reward sampling temperature, to 4.
- **Strategy rules:** hide Neow's option to swap the starter relic for a boss
  relic; or buy one affordable card removal by floor 33, either letting the
  policy pick the card or restricting it to an unupgraded Strike.
- **Teacher-played fights:** the corrected search teacher (24 reseeded samples
  of 1,000 simulations) actually played 128 paired fights, 32 each of Act 2
  and Act 3 bosses and elites. All 3,217 labels and trajectories were
  independently replayed.
- **Combat imitation:** 2,460 labels from Act 2 bosses and Act 2 and 3 elites
  (Act 3 boss targets excluded, as the teacher did not clearly help there),
  plus 1,072 rehearsal states; 349 worlds for fitting, 87 held out. Three
  pilots: soft targets (the full vote distribution), hard targets (the action
  played), and HP-curated hard targets (only the 1,404 labels from fights
  where the teacher finished with more HP). Each used learning rate 0.0001,
  four passes and a KL penalty towards the parent, which amounts to only 20 to
  32 optimizer steps.
- **Final-boss ablation:** the HP-curated model everywhere except the Act 3
  boss fight, where the parent's combat network was restored.
- **Fight-value classifiers:** predict fight victory under the frozen policy,
  from 7,770 fights in 512 new training worlds (409 for fitting, 103 held
  out). One used the old encoder's features; the other added 35 explicit enemy
  and card fields.
- **Evaluation:** greedy full games on 1,024 reused DEV worlds against the
  parent. The OOC pilots were planned in advance; the rule tests, imitation
  pilots and ablation were chosen as the night went on, so they are
  exploratory. None received fresh confirmation.
- **Approval:** Sid approved continued overnight experiments on 14 October,
  then asked for the final-boss ablation and a report before more training.

## Results

No arm established a win-rate gain. Each row is 1,024 reused DEV worlds
against the parent, which won 57; there were no simulator faults. The sources
give paired win tests but no win-rate intervals.

| Arm | Environment | Wins of 1,024 | Mean floor difference [95% interval] | Paired win p | Evidence |
|---|---|---:|---:|---:|---|
| Continued OOC control | 0.1.3 | 58 | +0.472 [−0.104, +1.079] | 1.0 | Planned |
| Card-reward temperature 4 | 0.1.3 | 57 | −0.248 [−0.853, +0.378] | 1.0 | Planned |
| Hide Neow boss-relic swap | 0.1.3 | 29 | −1.346 [−2.046, −0.609] | 0.000756 | Exploratory |
| Buy removal, policy chooses target | 0.1.3 | 41 | −1.695 [−2.340, −1.044] | 0.0519 | Exploratory |
| Buy removal, unupgraded Strike only | 0.1.3 | 49 | −0.126 [−0.704, +0.485] | 0.366 | Exploratory |
| Soft-vote combat imitation | 0.1.4 | 50 | −0.418 [−0.975, +0.186] | 0.443 | Exploratory |
| Hard-action combat imitation | 0.1.4 | 52 | +0.298 [−0.297, +0.915] | 0.615 | Exploratory |
| HP-curated combat imitation | 0.1.4 | 46 | +0.357 [−0.180, +0.940] | 0.215 | Exploratory |
| Curated model with parent final-boss head | 0.1.4 | 61 | +0.372 [−0.170, +0.957] | 0.724 | Exploratory |

The temperature variant lost 0.720 mean floors [−1.386, −0.042] against the
continued control on the same 1,024 worlds, without a clear win difference (57
versus 58; paired p = 1.0). The removal-target audit found 27 non-basic cards
and four upgraded Strikes chosen in 31 eligible shops from 32 new training
worlds, with no unupgraded basic card selected. Restricting removal to an
unupgraded Strike avoided the large progression loss but did not establish a
better strategy.

The corrected teacher improved remaining HP in three fight groups, enough to
motivate imitation, without establishing better Act 3 boss combat. Each
comparison below has 32 paired fights; defeat counts as zero remaining HP.
Evidence is Exploratory and intervals are nominal 95% paired bootstrap
intervals.

| Fight group | Parent wins of 32 | Teacher wins of 32 | Remaining-HP difference [95% interval] |
|---|---:|---:|---:|
| Act 2 bosses | 13 | 18 | +9.91 [+1.94, +17.94] |
| Act 3 bosses | 6 | 6 | +1.41 [−1.53, +4.81] |
| Act 2 elites | 29 | 29 | +7.88 [+1.84, +14.66] |
| Act 3 elites | 21 | 23 | +9.06 [+3.56, +14.59] |

On the 87 held-out worlds, the soft-target model's cross-entropy improved
from 3.934 to 1.663, yet its agreement with the teacher's actual action fell
from 44.4% to 42.5%. Hard targets beat soft targets on mean floor by +0.72
[+0.22, +1.22], with no clear win difference (52 against 50).

Restoring the parent's final-boss combat network gave 61 wins against the
curated model's 46 on the same worlds (22 gains, 7 losses; nominal p = 0.008),
but no clear gain over the parent's 57. Every world that never reached the
final boss played out identically. Five worlds reached the final boss before
floor 50 through Secret Portal; a brief suspicion that these were invalid was
checked against replays and retracted.

Fight-value prediction helped chiefly after a fight had progressed. Entry
Brier scores (mean squared error of predicted win probabilities; lower is
better) were as follows, against the fitting-data fight-group win rate used as
a constant prediction. No paired intervals were reported for these Exploratory
fits.

| Held fight entries | Constant baseline | Original features | Added public fields |
|---|---:|---:|---:|
| Act 2 bosses: 70 fights, 31 wins | 0.2469 | 0.2382 | 0.2348 |
| Act 3 bosses: 18 fights, three wins | 0.1516 | 0.2980 | 0.3041 |

Within the same 70 held Act 2 fights, later states were predicted much
better (0.1502 and 0.1471 against the constant 0.2469). Neither model beat
the constant at Act 3 boss entry, and neither was deployed.

## Conclusion

The successful OOC recipe had plateaued within these bounds. Plausible
strategy rules needed measurement: hiding the opening swap was harmful, and
neither removal strategy justified adoption. None was distilled. Combat
imitation could improve progression while damaging endgame play, even without
Act 3 boss training targets. The head ablation supports endgame retention as a
real problem; it does not establish a player better than the champion or
identify a unique cause of imitation failure. Fight-value fits describe
outcomes under the frozen policy, with weak entry prediction and particularly
poor Act 3 results; they do not establish useful planning values or game
improvement.

Caveats:

- These are reused DEV worlds, with adaptive follow-ups and nominal tests; no
  trained candidate was confirmed on fresh worlds, and win-rate intervals are
  unavailable.
- Each training condition used one seed. Environment 0.1.3 and 0.1.4 are
  separate conditions; the earlier 0.1.3 teacher sampler depended on hidden
  pile order and is distinct from the corrected teacher.
- The search teacher uses simulator copies and is privileged; remaining
  hidden-state dependence was not certified away by the corrected sampler.
- HP curation changed both label selection and the number of optimizer
  steps. Remaining HP ignores potions, maximum HP, deck changes and later
  game value.

## Next

Keep the champion and do not extend these failed recipes unchanged. Further
combat work should test representation and teacher quality, retain direct
full-game gates, and protect endgame performance; the subsequent [fresh-start
study](2026-10-15-fresh-start.md) revisits distillation with a larger training
budget. Do not use the saved value predictors for Act 3 planning on this
evidence. Measure proposed strategy interventions before distilling them.

## Log

- 14 Oct: Sid authorized overnight continuation; corrected shops required
  rebaselining before training.
- 14 Oct: both eight-iteration OOC pilots finished without an established win
  gain; the three strategy rules were rejected.
- 14 Oct: corrected teacher HP gains justified a bounded imitation pilot;
  soft, hard and HP-curated learners all failed their game gates.
- 14 Oct: Sid requested the final-boss ablation and a report before more
  training; the ablation recovered performance, and the Secret Portal concern
  was retracted.

Code: `src/sts_rl/ooc_trainer.py`, `src/sts_rl/distill.py`,
`scripts/rl/train_ooc.py`, `collect_value_bank.py`, `probe_fight_value.py`,
`diagnose_combat_imitation.py` and `scripts/rl/replay_fault.py` (before the
experiment-folder convention), at final main source `2ea61e2` and
corrected-teacher source `d5c56e1`; value-fit implementation at `4b7aa5d`.
Data: runs `overnight_20261014` (local) and `canonical_20261014` (local),
including `expert/late128`, `expert/boss3_fallback_1024.json` and
`value_bank/value_fit512`.
