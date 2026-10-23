# Status

**Updated 23 October 2026, 10:35 BST.** Nothing running.
Champion: `champion_c8553bc33e53.pt`, joint run 2's final checkpoint (26.8%
on unused worlds), promoted by Sid at 09:58. A copy with its record is in
`joint2_analysis_20261023/inputs/` (worktree `nonllm-large-batch`).

- **Joint run 2 analysis:** finished (results on the
  [joint run 2 page](experiments/2026-10-22-joint-run-2.md); data in
  `joint2_analysis_20261023/`).
- **Queues:** both lanes idle (`scripts/rl/queue_runner.py status` in
  `nonllm-large-batch` or `nonllm-boss-rares`).
- **Disk:** 79 GiB free at 10:05. See `scripts/rl/CLAUDE.md` for the
  periodic check.
- **Dashboard:** http://127.0.0.1:8795 (served from `nonllm-dashboard`).
- **Closing the Claude desktop app kills processes this session started**,
  including detached runners and watchers; resume a trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch`, `nonllm-continued-frozen`,
  `nonllm-joint-frozen`, `nonllm-joint2-frozen` (the frozen code of the two
  joint runs; do not edit), `nonllm-better-teacher` (holds the new native
  build), `nonllm-large-batch-hidden`, `nonllm-fresh-start`,
  `nonllm-boss-rares`, `nonllm-dashboard`, `nonllm-disk-hygiene`,
  `nonllm-joint-run-2`.

## Next

Propose to Sid the next training round from the new champion:
out-of-combat PPO with refit-and-gate cycles as standard (shorter cycles),
and a potion-decision distillation.
