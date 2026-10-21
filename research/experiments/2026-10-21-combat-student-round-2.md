# Combat student, round 2: teacher labels on the current champion's states

**Status:** running. **Verdict:** pending.

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

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

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
