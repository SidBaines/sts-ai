# Status

**Updated 25 October 2026, 09:45 GMT.** Nothing running. Champion:
`champion_4058cbf0e271.pt` (promoted by Sid on 25 October; 29.4% on unused
worlds), kept with earlier champions in `data/rl_runs/champions/` (worktree
`nonllm-large-batch`).

- **Paused:** [value-target refits](experiments/2026-10-23-value-target-refits.md),
  with 1,753 of 2,000 worlds labelled in `value_refits_20261023/corpus/`
  (worktree `nonllm-large-batch`); next step is the fits.
- **Queues:** idle runners in `nonllm-large-batch` and `nonllm-boss-rares`
  (`scripts/rl/queue_runner.py status`), restarted on 25 October so their
  stall watchdog also counts `fresh_collect.py`'s progress file.
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

The value-target refit fits and play tests, on the 1,753 labelled worlds.
