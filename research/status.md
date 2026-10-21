# Status

**Updated 21 October 2026, 10:20 BST.** Round 4 trains (ends about 12:00).

- **Running: [round 4](experiments/2026-10-21-round-4.md)** as `r4-train`
  on lane `train` of the queue in worktree `nonllm-large-batch` (24
  iterations of 3,072 games, about 6 minutes each, so about 2.5 hours);
  output `data/rl_runs/round4_20261021/round4/` there. It runs
  `train_ooc.py` from the frozen checkout `nonllm-large-batch-hidden` at
  `ws/nonllm` 8aca4e4.
- **Done:** the champion on the new worlds from 95,000,000,000 (379 wins)
  and the Demon Form test (434 against 413 on DEV, p = 0.033; 391 against
  379 on the new worlds, p = 0.18); outputs in
  `round4_20261021/{study_eval,confirm95}/`. Waiting for Sid: whether to
  distil a rule this small.
- **Landed:** exploration statistics in the trainer (`ws/nonllm` 3ca1562);
  runs started from that code record them each iteration.
- **Watcher:** `eval_round4.sh` in the Claude session's scratchpad (log
  `round4_eval.log` beside it), detached from the app; when round 4
  finishes it queues the final and best-by-trainer evaluations and prints
  the comparisons with the champion.
- **Check or stop:** `scripts/rl/queue_runner.py status` in either
  worktree; `scripts/rl/queue_runner.py cancel NAME`. The queues pause jobs
  on battery and resume on power. Dashboard: http://127.0.0.1:8795
  (detached, from worktree `nonllm-dashboard`).
- **Champion** (from 21 October): `champion_7d876c939a74.pt` in
  `large_batch_20261020/inputs` and `round4_20261021/inputs`, worktree
  `nonllm-large-batch`. Its paired games: second DEV
  `large_batch_20261020/study_eval/round3_student_i0024` (413 wins);
  fresh worlds from 94,000,000,000 `confirm94/round3_student_i0024` (394);
  from 95,000,000,000 `round4_20261021/confirm95/champion` (379).
- **Worktrees to keep:** `nonllm-large-batch` (every champion and arm),
  `nonllm-large-batch-hidden` (the frozen trainer used by round 4),
  `nonllm-fresh-start` (the native build and the combat student) and
  `nonllm-boss-rares` (evaluations). Cut topic branches from
  `origin/ws/nonllm`; the local branch is behind.

## Next

When the evaluations land, apply the decision rules on the round 4 page;
distil the Demon Form rule if it is confirmed.
