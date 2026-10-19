# Joint sparse-win PPO

**Status:** finished 15 October 2026. **Verdict:** training both networks on
full-game wins did not demonstrate an improvement over the champion, even
after repairing a numerical failure and underfitted critics.

## Question

The test asked whether training combat and out-of-combat (OOC) decisions
together on final wins could improve the champion. Earlier gains came from OOC
training; combat remained original. This tested [joint sparse-win
PPO](../strategies.md#joint-sparse-win-ppo), or proximal policy optimization.

## Design

Both attempts started from the champion, including the fresh restart after
repairs.

- **Environment:** Ironclad, Ascension 0, Acts 1–3, environment 0.1.4, on the
  laptop. No search teacher was involved.
- **Reward and credit:** 1 for beating the Act 3 boss, 0 for defeat and
  intermediate decisions. Gamma and lambda were both 1: every decision
  received the undiscounted final outcome, with no bootstrap. Simulator faults
  and incomplete worlds were excluded.
- **Networks:** separate combat and OOC actors choose actions; independent
  critics predict eventual wins from public inputs. Critic gradients could not
  change actor encoders; inherited critics were frozen and unused. Each phase
  had its own actor optimizer and normalization. Critics used up to eight
  states per world in each phase, weighting worlds equally.
- **Actor settings:** learning rates 0.00001 combat and 0.00003 OOC, one pass
  per batch, PPO clipping 0.1 and entropy coefficient 0.001. Sampling
  temperatures were 0.25 and 0.5, with uniform exploration mixtures 0.001 and
  0.002, respectively. Rehearsal penalized departure from the original policy
  on states from 128 greedy worlds; evaluation was greedy.
- **Data and evaluation:** 768 new training games per round; the same 1,024
  DEV worlds every second round and at the repaired run's end. Critic warm-up
  used 12,265 fitting and 3,095 held-out worlds from the first pilot. A
  separate 512-world stochastic cohort tested the warmed critics before actor
  updates.
- **Repair:** stable log-space Kullback–Leibler (KL) divergence, a measure of
  policy change, replaced arithmetic that failed on tiny probabilities.
  Critics took a real optimizer step per 256-state minibatch at learning rate
  0.0003, with 384 steps per critic in warm-up. Actor advantages—the final
  outcome minus the prediction made before observing it—were not recomputed
  after fitting.
- **Decision rule, fixed before results:** at least 8 extra wins on 1,024 DEV
  worlds and paired p-value at most 0.05, then a separately gated confirmation
  for promotion. Numerical, drift and retention guards could stop training;
  stagnation did not reset it.
- **Budget and approval:** three active hours for the 14 October pilot. Sid
  authorized the 15 October restart for two to three hours; its allocation was
  fixed at two active hours, including validation and greedy evaluation.

## Results

The repaired run was inconclusive on 1,024 paired DEV worlds in environment
0.1.4: **Planned** evidence on reused worlds.

| Player | Wins of 1,024 | Win rate difference, percentage points [95% interval] |
|---|---:|---:|
| Champion | 58 (5.66%) | — |
| Repaired joint run, final round 19 | 62 (6.05%) | +0.39 [−0.98, +1.76] |

There were 28 newly won and 24 newly lost worlds; paired p = 0.678. Against
the same champion on those worlds, mean floor changed by +0.35 [−0.13, +0.84].
No confirmation or promotion followed.

- **The repaired run completed its two-hour budget:** 19 accepted rounds, no
  rejections and 1,248 retained actor optimizer steps. Training produced 689
  wins in 14,592 games; each critic took 372 steps after warm-up. Training and
  all eleven baseline/evaluation cohorts had zero simulator faults or
  incomplete games. Stochastic wins do not measure greedy play.
- **The first pilot was also inconclusive:** it stopped after about two hours,
  with 16 accepted and 4 rejected rounds, retaining actors from round 18. They
  won 55 of 1,024 DEV worlds against the champion's 58 (paired p = 0.7838).
  Exploratory replacements of combat alone won 48 (p = 0.1433), and OOC alone
  59 (p = 1.0), each against 58 on those same worlds. Paired intervals were
  not reported. Its best intermediate score, 65 of 1,024 against 58, also
  failed promotion (p = 0.435). Training had 708 wins in 15,360 games, with
  zero simulator faults or incomplete games.
- **The numerical failure was repaired:** in the saved round-20 update, a tiny
  positive reference probability and a current probability rounded to zero
  made categorical KL infinite despite finite log probabilities. Log-space
  arithmetic let all four rejected updates pass the original guards in offline
  replays; those actors were never installed.
- **Too few optimizer steps explained the flat critics:** the original
  schedule made only 20 steps per critic. Audits found valid labels, routing
  and gradients. On 3,095 held-out worlds, 384 minibatch steps at the same
  learning rate raised explained variance from 0.00044 to 0.06266 combat and
  from 0.00007 to 0.04133 OOC. These exploratory fitting comparisons have no
  reported intervals. On the separate 512-world cohort (20 wins),
  warm-to-final explained variance was 0.11025 to 0.11461 combat and 0.05913
  to 0.06953 OOC.
- **Calibration remained unstable:** on that unchanged 512-world cohort, OOC
  predictions rose from 4.44% to 7.01% between rounds 12 and 13, against 3.91%
  observed returns. At Act 3 states they rose from 17.49% to 26.53%, against
  17.67% observed, while explained variance stayed nearly unchanged. That
  batch had 39 wins in 615 fitting worlds, versus 1 in 153 critic validation
  worlds. Sensitivity to recent success prevalence is plausible.
- **Warnings did not establish broad forgetting:** on 614 worlds from rounds
  7–10, OOC Brier loss at round 10 minus warmed was +0.000201 (95% interval
  [−0.000256, +0.000640]), an exploratory paired whole-world comparison. On
  the final recent held-out window, however, Act 3 explained variance was
  −0.03571 combat (225 worlds) and −0.00412 OOC (210 worlds): late-act
  predictions remained weak. Beating a global low-win-rate constant predictor
  was insufficient.

Later collection, storage and evaluation speedups made a 512-game benchmark
1.6 times faster, with the scientific recipe unchanged.

## Conclusion

The repairs made optimization work and fitted informative critics, but did not
establish better play. Joint sparse-win learning remains unresolved: the later
analysis found very small policy changes and little OOC exploration under
these conservative settings. The critic is a baseline; calibration problems do
not change the final-outcome targets.

Caveats:

- Both conditions used one training run and repeatedly inspected the same
  1,024 DEV worlds. The reported intervals are nominal; there were no fresh
  confirmation worlds, and the final interval permits useful gains or losses.
- Critic-held worlds could train actors. The fixed 512-world cohort tested
  critics under the original policy, not the trained player's win rate.
- Recent and fixed critic panels differ in seeds and reached states; they do
  not isolate policy changes, missing features or stochastic outcome noise.
- Episode records said environment 0.1.3 despite the pinned 0.1.4 package and
  binary. These results must not be compared directly with environment 0.1.5.

## Next

The later options review prioritized combat learning over extending this run
or making further critic calibration the main study. A uniform prediction
shift cancels when actor advantages are mean-centred, so calibration alone is
a weak lead.

- Collect more search-teacher labels on states the student visits, retain vote
  distributions and train for thousands of steps; use fight panels before full
  games.
- Compare the existing encoder with a structured entity encoder on identical
  labels.
- After combat improves, revisit OOC macro credit with fight-value predictions
  from repeated boss fights, while retaining the full-game objective.
- Adopt environment 0.1.5 and remeasure baselines before the next study; the
  subsequent [fresh-start study](2026-10-15-fresh-start.md) took that
  direction.

## Log

- 14 Oct: persistent numerical rejection stopped the pilot; no actor was
  promoted.
- 15 Oct: log-space KL repaired the rejected updates; critic underfitting was
  diagnosed.
- 15 Oct: Sid authorized a fresh run with warmed minibatch critics and a
  two-hour budget.
- 15 Oct: unchanged-state comparisons exposed calibration swings; settings
  stayed fixed.
- 15 Oct: the budget ended without demonstrated improvement; the champion was
  retained.
- 15 Oct: the options review deprioritized continuation and favored combat
  learning.

Code: `src/sts_rl/joint_sparse.py` and `src/sts_rl/critic_training.py` at
commit `ed7b102` (frozen restart training source); diagnostics:
`scripts/rl/joint_sparse_kl_check.py`, `scripts/rl/joint_critic_diagnosis.py`
and `analysis/critic_trace_diagnosis_20261015.py`. Data: runs
`joint_sparse_win_20261014`, `joint_sparse_win_v2_20261015`,
`joint_critic_warm_v2_20261015` and `joint_critic_diagnosis_repaired_20261015`
(local).
