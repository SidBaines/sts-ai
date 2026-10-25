# Status

**Updated 25 October 2026, 09:53 GMT.** Value-target refits run. Champion:
`champion_4058cbf0e271.pt` (promoted by Sid on 25 October; 29.4% on unused
worlds), kept with earlier champions in `data/rl_runs/champions/` (worktree
`nonllm-large-batch`).

- **[Value-target refits](experiments/2026-10-23-value-target-refits.md):**
  six fits on the 1,752 labelled worlds (`vr-fit-*`, one lane each in the
  `nonllm-boss-rares` queue; output `value_refits_20261023/`), on the GPU.
  The detached watcher `play.sh` (session scratchpad, `value_refits/`, log
  `play.log`) splices each finished fit into the champion, plays 8,192
  worlds from 111,000,000,000 (lane `eval`, `nonllm-large-batch` queue) and
  runs the pre-registered comparison; it also reports failed fits.
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

Record the value-target refit results.
