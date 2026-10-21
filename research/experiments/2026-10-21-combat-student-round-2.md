# Combat student, round 2: teacher labels on the current champion's states

**Status:** finished 21 October 2026. **Verdict:** confirmed: refitting the combat student to teacher labels from the current champion's own games raises wins from 12.5% to 13.9% on unused worlds (p = 0.017; mean floor +0.84). Expert iteration works for combat.

## Question

The champion's combat network is a student fitted on 15 October to the
search teacher's choices in 159,707 states that the 3.7% champion of the
time reached ([fresh start](2026-10-15-fresh-start.md)). It added about 2
points of win rate, the largest single gain so far. Today's champion wins
about 12% and reaches Act 3 far more often, so it now fights in states the
student never saw. Does labelling the states the current champion reaches
and refitting the student on old plus new labels improve combat again?
This is the planned step D of the fresh-start study (expert iteration).
(Approach: [distilling the search teacher](../strategies.md).)

## Design

- **Labels:** the champion of 21 October, afternoon
  (`champion_2835f40fa439.pt`) plays 2,000 new training worlds from
  90,500,100,000 (eight shards of 250); the search teacher (24 reseeded
  samples of 1,000 simulations, plurality vote; privileged, used only as a
  teacher) labels its combat decisions with the 15 October room-group
  probabilities (every Act 2 and 3 boss and event fight, 80% of Act 2 and 3
  elites, fewer hallway fights). Out-of-combat potion actions hidden.
- **Student:** the old encoder, large size, trained from scratch at learning
  rate 0.001, batch 128, up to 12,000 steps with early stopping on
  validation loss (patience 6, at least 2,000 steps), seed 1919: the 15
  October recipe unchanged. Planned on the old corpus plus the new one;
  run on the new corpus only (see Log), split into fit, validation and
  test worlds by world hash.
- **Evaluation:** the new student replaces the champion's combat head
  (`combine_heads.py`); `fast_eval.py` under the study protocol on the
  4,096 second-DEV worlds, paired with the champion's 527 wins; teacher
  agreement on the held-out test labels of both corpora.
- **Decision rule, fixed in advance:** the new combat head is *Better* if
  it wins more than the champion on second DEV with exact McNemar p < 0.05;
  it is then confirmed on 4,096 unused worlds from 97,000,000,000 (champion
  and candidate both played there) at p < 0.05 before promotion.
- **Budget:** about 3 to 4 hours of labelling on the laptop alongside the
  running training, then about an hour of fitting on the GPU.
- **Approval:** Sid's standing instruction of 21 October to keep doing
  research until stopped; step D was proposed in the fresh-start study.

## Results

**The refitted combat student passes on second DEV and is confirmed on
unused worlds.** The new student replaces the champion's
combat head; out-of-combat head unchanged; paired:

| Worlds | Wins | Discordant, champion only / candidate only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Second DEV | 527 → 576 (12.9% → 14.1%) | 268 / 317 | 0.047 | +0.69 [+0.35, +1.03] | +1.6 points [−0.2, +3.4] (3,159 worlds) |
| From 97,000,000,000 (unused before) | 512 → 571 (12.5% → 13.9%) | 268 / 327 | 0.017 | +0.84 [+0.48, +1.19] | +2.3 points [+0.5, +4.1] (3,116 worlds) |

- Fit: 122,769 fit rows, early-stopped at step 4,900 (best validation loss
  1.41; the 15 October student's was 1.30 on its own, different validation
  worlds, so the two are not comparable); 7 minutes on the GPU.
- The champion wins 512 (12.5%) on the new worlds, in line with 476 and 527 elsewhere.
- Candidate checkpoint: `combat_round2_20261021/champion_with_student_r2_newonly.pt` (SHA-256 prefix `623b01850482`); student `fit_old_combat_r2_newonly_v1/best.pt`.

## Conclusion

- Expert iteration works for combat: labelling the states the current
  player reaches, then refitting from scratch, gained +1.2 to +1.4 points
  of wins and +0.7 to +0.8 floors on two world sets.
- The gain arrived although the out-of-combat head was trained alongside
  the previous combat student; no retraining of the rest was needed.
- The loop is cheap: about 1.7 hours of labelling on the laptop and 7
  minutes of fitting.

Caveats:
- One fit, one seed; the old and new corpora could not be combined (the
  library refuses to mix players' corpora), so whether more data would
  help more is untested.
- The teacher is privileged (it searches copies of the game); the player
  stays a pure policy.

## Next

1. Sid to decide whether the candidate becomes the champion.
2. Splice the new student with the second continued training's
   out-of-combat head, which was trained alongside the old student.
3. Another round of expert iteration from the candidate's own games.

## Log

- 21 Oct, 19:46: confirmed on unused worlds: 571 against 512 (p = 0.017).
- 21 Oct, 19:37: DEV result 576 against 527 (p = 0.047): Better under the
  pre-set rule; confirmation on the worlds from 97,000,000,000 queued.
- 21 Oct, 19:27: deviation from the plan. The training library refuses to
  mix corpora collected by different players (a safeguard of the fresh-start
  pipeline), so the student is fitted on the new corpus alone: 2,000 worlds,
  no faults, 174,859 combat labels (the old corpus had 159,707). Its fixed
  unlabelled probe states (movement monitoring only) must also come from
  the same player, so 8 new probe worlds were collected from
  93,000,000,100. Fitting started at 19:26 on the GPU.
- 21 Oct, 17:31: labelling started at 17:28 (eight shards); a progress-tied
  keep-alive stops the queue's two-hour silence check from killing shards
  that print only at the end.

Code: `scripts/rl/fresh_collect.py`, `scripts/rl/fresh_train.py`,
`scripts/rl/combine_heads.py`, `scripts/rl/fast_eval.py`. Data: run
`combat_round2_20261021` (local, worktree `nonllm-large-batch`).
