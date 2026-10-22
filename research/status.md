# Status

**Updated 22 October 2026, 16:53 BST.** Several jobs run; nothing waits for Sid. Champion:
`champion_af2fb9f1f260.pt` (21.1% on unused worlds).

- **Champion promoted 22 October, 16:32:** the third continued training's
  final checkpoint (`champion_af2fb9f1f260.pt`, 21.1% on unused worlds).
- **[Fourth continued training](experiments/2026-10-22-continued-training-4.md)**
  (`c4-train`, queued on lane `train` of the queue in worktree
  `nonllm-large-batch` after the joint run; output
  `data/rl_runs/continued4_20261022/`). The champion's baseline games on
  worlds from 104,000,000,000 (`c4-champion-confirm104`) are queued now in
  the `nonllm-boss-rares` queue. Detached watcher `eval_continued4.sh`
  (log `continued4_eval.log` in the session scratchpad) evaluates it when
  training ends, about 4.5 hours after it starts.
- **[Joint run](experiments/2026-10-22-joint-loop.md)** (`joint-train`,
  running on the `train` lane since 16:11; output `joint_20261022/`). Detached
  `joint_start_and_compare.sh` starts the orchestrator `joint_loop.py`
  (decisions in `joint_20261022/loop/decisions.jsonl`; create
  `joint_20261022/loop/STOP` to stop it between steps) when the trainer
  starts, and runs the final comparison on worlds from 103,000,000,000.
- **Exploratory check** (`boss-seed2-in-newchamp`, `nonllm-boss-rares`
  queue, lane `eval`): the better boss-label student in the new champion,
  on second DEV against its 899. The boss-label fits themselves finished
  (seed 1 worse, seed 2 no clear difference).
- **Finished today:** combat fit variance (one fit in four collapses; round
  3's collapse was a bad fit), all-corpora fit (worse), the lineage plot
  (`scripts/rl/plot_lineage.py`), `--mixed-sources` and `--combat-follow`
  (both merged after review).
- **Keep-alives** in the session scratchpad refresh quiet jobs' queue logs
  only while their own progress files advance.
- **Check or stop:** `scripts/rl/queue_runner.py status` or `cancel NAME`
  in either worktree. Dashboard: http://127.0.0.1:8795.
- **Closing the Claude desktop app kills processes this session started**,
  including detached runners, trainers, watchers, the orchestrator and
  keep-alives; resume a trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch`, `nonllm-continued-frozen`,
  `nonllm-joint-frozen` (the joint run's code; do not edit),
  `nonllm-large-batch-hidden`, `nonllm-fresh-start`, `nonllm-boss-rares`.

## Next

Record the third continued training and the boss-label fits when they
land; follow the joint loop's gate decisions; then start step 0 of the
[better teacher](experiments/2026-10-22-better-teacher.md) plan
(measurement and the teacher benchmark) on free lanes.
