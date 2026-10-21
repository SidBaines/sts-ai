# Status

**Updated 21 October 2026, 09:33 BST.** Round 4 trains and the Demon Form test runs; nothing
waits for Sid.

- **Running: [round 4](experiments/2026-10-21-round-4.md)** as `r4-train`
  on lane `train` of the queue in worktree `nonllm-large-batch` (24
  iterations of 3,072 games, about 6 minutes each, so about 2.5 hours);
  output `data/rl_runs/round4_20261021/round4/` there. It runs
  `train_ooc.py` from the frozen checkout `nonllm-large-batch-hidden` at
  `ws/nonllm` 8aca4e4.
- **Running: evaluations** on lane `eval` of the queue in worktree
  `nonllm-boss-rares`, one after another: `r4-champion-fresh95` (the
  champion on the new confirmation worlds from 95,000,000,000),
  `r4-demonform-dev` and `r4-demonform-fresh95` (the champion with the
  Demon Form rule, second DEV and the new worlds); outputs in
  `round4_20261021/{study_eval,confirm95}/` in worktree `nonllm-large-batch`.
- **Being built:** exploration tracking in the trainer (per-menu entropy and
  rare-action sampling), by Codex in worktree `nonllm-exploration-tracking`.
- **Check or stop:** `scripts/rl/queue_runner.py status` in either
  worktree; `scripts/rl/queue_runner.py cancel NAME`. The queues pause jobs
  on battery and resume on power. Dashboard: http://127.0.0.1:8795
  (detached, from worktree `nonllm-dashboard`).
- **Champion** (from 21 October): `champion_7d876c939a74.pt` in
  `large_batch_20261020/inputs` and `round4_20261021/inputs`, worktree
  `nonllm-large-batch`. Its paired games: second DEV
  `large_batch_20261020/study_eval/round3_student_i0024` (413 wins);
  fresh worlds from 94,000,000,000 `confirm94/round3_student_i0024` (394);
  from 95,000,000,000 `round4_20261021/confirm95/champion` (running).
- **Worktrees to keep:** `nonllm-large-batch` (every champion and arm),
  `nonllm-large-batch-hidden` (the frozen trainer used by round 4),
  `nonllm-fresh-start` (the native build and the combat student) and
  `nonllm-boss-rares` (evaluations). Cut topic branches from
  `origin/ws/nonllm`; the local branch is behind.

## Next

When the evaluations land, apply the decision rules on the round 4 page;
distil the Demon Form rule if it is confirmed.
