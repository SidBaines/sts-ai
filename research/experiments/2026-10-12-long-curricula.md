# Long joint curricula: v1 and v2

**Status:** finished 12 October 2026 (both runs paused, never resumed).
**Verdict:** neither curriculum established better full-game play: v1
discarded most updates, while v2 accumulated updates but lost progression and
the original wins.

## Question

The aim was to turn the [first curriculum's](2026-10-11-first-curriculum.md)
strong Act 1 player into a full-game winner. [Long joint
curricula](../strategies.md#long-joint-curricula) combine later-act saved
starts with learning combat and out-of-combat (OOC) decisions together. V2
tested consistent full-game goals and retention of learner progress.

## Design

V2 changed the task and selection rules, retaining v1's starting weights and
learning recipe.

- **Common baseline:** the first campaign's selected joint checkpoint
  (`joint_act3_0030.pt`), restarted for each version. Both used environment
  0.1.1, Ironclad, Ascension 0 and Acts 1–3, with separate combat and OOC
  networks. World splits separated training from evaluation; v2's fresh
  training stream was disjoint from v1's.
- **Learning:** proximal policy optimization (PPO), learning rate 0.0001, four
  passes per batch, minibatch 1,024, clipping 0.1 and entropy coefficient
  0.01. Rewards were +0.05 per positive floor advance, +1 task success and −1
  ordinary defeat, with no terminal hit-point bonus or reward taper. Future
  rewards were undiscounted (gamma 1). Generalised advantage estimation (GAE),
  combining rewards and critic predictions, used lambda 0.95.
- **V1 tasks:** 128 episodes per batch, mixing rotating fresh training worlds,
  saved native and learner continuations, and Act 1 rehearsal. The initial mix
  was 70 Act 2 continuations, 19 fresh Act 2 goals, 26 fresh Act 3 goals and
  13 Act 1 rehearsals. The next stage used 32, 57, 26 and 13 respectively;
  later stages targeted Act 3.
- **V1 decision rule:** every three completed batches, rank the current
  stage's goal clears first and mean floor second on 48 DEV worlds. Act 1
  retention could lose at most two clears against the fixed initial 41/48.
  Rejection or a hold restored the incumbent. Full-game checks on the same 48
  DEV worlds were monitoring only; v1 paused during its second stage, before
  full-game selection began.
- **V2 tasks and selection:** every 128-episode batch contained 90 fresh
  starts, 25 Act 2 continuations and 13 Act 3 continuations, all targeting Act
  3 victory. A fresh Adam optimizer started from the original; subsequent
  batches retained the latest learner and optimizer state. Champion selection
  used full-game DEV wins first, paired mean floor second; non-improvement
  kept the champion without resetting learning.
- **V2 evaluation:** new panels of 48 DEV worlds for screening and 192
  disjoint DEV worlds for larger confirmation; Act 1 retention used those same
  192 worlds. Both versions allowed 4,096 decisions per full game and 2,048
  for Act 1 retention. Valid v2 pairs needed at least 187 shared valid worlds,
  at most three simulator faults per panel, and no truncations or unstarted
  games. Final evaluation worlds stayed unopened.
- **V2 pause rule:** on two distinct evaluated checkpoints, a paired loss of
  at least 29 Act 1 clears against the original, or at least eight mean floors
  against the champion, without a paired win gain, required an attention
  pause.
- **Approval and compute:** Sid approved 10 active hours of laptop compute on
  12 October, then a v2 restart within the remaining time. V1 used about 4.9
  hours and v2 about 1.2 hours before both were paused.

## Results

Neither version established a full-game improvement over its original
baseline. V1's full-game observations are **Exploratory**; v2's larger
comparisons are **Planned**, on reused DEV worlds. The sources report no 95%
intervals.

| Candidate | Original wins | Candidate wins | Mean floor comparison |
|---|---:|---:|---|
| V1 selected model | 0/48 | 0/48 | 28.60 → 30.90, same 48 worlds |
| V1 candidate with most observed wins | 0/48 | 2/48 | 28.60 → 29.73, same 48 worlds |
| V2 first larger evaluation | 2/192 | 0/192 | Paired change −2.39, 190 shared valid worlds |
| V2 second larger evaluation | 2/192 | 0/192 | Paired change −1.73, 190 shared valid worlds |

**V1 improved its selection subgoal but retained little training.** The
selected model cleared Act 2 in 11/48 DEV worlds against the original 8/48 and
retained 42/48 Act 1 clears against 41/48. Its two-win rival was rejected for
8/48 Act 2 clears, despite passing Act 1 retention with 41/48. Seven checks on
48 DEV worlds gave 0, 1, 0, 1, 2, 1 and 1 wins; only two worlds ever won. All
ten audited panels had no faults or truncations.

At the bounded audit, 526 completed batches had performed 36,217 optimizer
updates, but the selected weights retained only six batches and 511 updates in
their ancestry. Restoring the incumbent erased short branches every three
batches; saved learner continuations could still affect later data.
Fresh-start training curves were broadly flat.

**V2 retained updates but regressed.** The final saved learner contained 129
completed batches and 13,514 updates, with no rollbacks. Both larger evaluated
snapshots lost the original wins. Act 1 clears were 131/192 and 139/192
against the original 151/192; on the 190 shared valid worlds, paired losses
were 21 and 13 clears. The original had two simulator faults in each panel,
both candidates none; no games were truncated or unstarted. Neither severe
threshold was crossed; the champion stayed original. The last saved learner
was not evaluated on the 192-world panel.

**Phase swaps attributed the broad progression loss to OOC changes.** This
exploratory diagnosis swapped entire combat and OOC networks between the
original and first evaluated v2 learner on 190 shared valid worlds of the
192-world DEV panel; reference traces reproduced.

| Combat / OOC weights | Mean floor | Act 1 clears /190 | Act 2 clears /190 | Wins /190 |
|---|---:|---:|---:|---:|
| Original / original | 27.63 | 151 | 13 | 2 |
| New / original | 28.48 | 151 | 22 | 0 |
| Original / new | 24.72 | 130 | 5 | 0 |
| New / new | 25.24 | 130 | 12 | 0 |

Changing OOC reproduced the early survival loss; changing combat improved
progression but lost both original wins. On 64 training worlds, 59 valid under
every combination, OOC replacement cost 4.44 mean floors versus 0.98 for
combat replacement. On 32 saved training battles, combat wins rose from 24 to
25, with one gain, no losses or faults; saved native starts did not establish
retention on states reached in full games.

## Conclusion

V1 exposed an objective mismatch and repeated loss of learner progress; fixing
them was insufficient. V2's OOC changes caused broad progression loss, and
either phase change could destroy rare full-game wins. Population win-rate
effects remain uncertain.

PPO's control over individual updates did not prevent cumulative policy drift.
Actor and critic share state features, so critic gradients can change action
choices. That interference, weak long-horizon credit and v1's mixed goals
without goal inputs are plausible mechanisms; this study did not establish
which caused learning to fail.

Caveats:

- DEV worlds were repeatedly inspected. There were no fresh confirmation
  worlds or independent training repeats.
- Continuation success is conditional on a saved start, not full-game win
  rate. The v1 audit counts precede its final pause; they are not final run
  totals.
- These results are on environment 0.1.1, where Discovery could trigger an
  unsupported-behaviour fault marker; the later fix was not adopted. Faults
  were separate from defeats and excluded from paired comparisons. The
  crossover had 2, 1, 0 and 0 faults in table order, with no truncations or
  unstarted games.
- The second evaluated v2 checkpoint's weights were deleted by retention; its
  metrics survive. Phase attribution uses the preserved first evaluated model.

## Next

The next proposal was a short comparison from the original, initially freezing
combat and anchoring conservative OOC updates to the original policy. Direct
win and Act 1 retention checks must accompany progression. OOC credit across
fights and a higher GAE lambda are separate hypotheses; see [OOC PPO with
macro credit](../strategies.md#out-of-combat-ppo-with-macro-credit). No
protected pilot was implemented or authorized by this study's handoff.

## Log

- 12 Oct: Sid approved 10 active hours with fixed rewards and a counted pilot.
- 12 Oct: v1's audit found most updates absent from the selected model and an
  Act 2 selection veto on a two-win candidate.
- 12 Oct: Sid paused v1 and approved an original-checkpoint v2 restart using
  the remainder.
- 12 Oct: two larger v2 DEV evaluations lost both original wins; Sid paused
  training for phase attribution.
- 12 Oct: phase swaps identified OOC progression loss and fragile wins under
  either phase change; both runs remained paused.

Code: `scripts/run_long_curriculum.py`, `src/sts_policy/long_run_control.py`
at `f23cad1`; `scripts/run_long_curriculum_v2.py`,
`src/sts_policy/long_run_v2_tasks.py` at `9c7b649`;
`scripts/diagnose_phase_forgetting.py` at `c6835cf` (before the
experiment-folder convention). Data: runs `long_rl_v1` and `long_rl_v2`
(local).
