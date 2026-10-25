# Status

**Updated 25 October 2026, 09:50 GMT.** Nothing running. Champion:
`champion_c8553bc33e53.pt`, joint run 2's final checkpoint (26.8% on unused
worlds). Promotion candidate: the reward ablation's wins-plus-floors arm at
iteration 24 (29.4% against the champion's 26.0% on 16,384 unused worlds);
Sid's decision.

- **Paused:** [value-target refits](experiments/2026-10-23-value-target-refits.md),
  with 1,753 of 2,000 worlds labelled in `value_refits_20261023/corpus/`
  (worktree `nonllm-large-batch`); next step is the fits.
- **Queues:** idle runners in `nonllm-large-batch` and `nonllm-boss-rares`
  (`scripts/rl/queue_runner.py status`). The runner's stall watchdog stops a
  job whose log is silent for 40 minutes (two hours if it has no log yet);
  jobs that do not print progress, such as `fresh_collect.py`, need a
  keep-alive.
- **Dashboard:** http://127.0.0.1:8795 (served from `nonllm-dashboard`).
- **Disk:** 68 GiB free on 25 October; `scripts/rl/disk_report.py` finds
  nothing to clean under the approved rules.
- **Closing the Claude desktop app kills processes this session started.**
- **Worktrees in use:** `nonllm-large-batch` (data and queue),
  `nonllm-boss-rares` (notes and queue), `nonllm-fresh-start` (native build
  `build015`), `nonllm-better-teacher` (teacher build), `nonllm-dashboard`
  (dashboard), and the frozen code of runs, which must not be edited:
  `nonllm-continued-frozen`, `nonllm-joint-frozen`, `nonllm-joint2-frozen`,
  `nonllm-reward-ablation`, `nonllm-value-targets`.

## Next

Sid's promotion decision; then the value-target refit fits and play tests.
