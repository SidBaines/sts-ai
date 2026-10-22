# Status

**Updated 22 October 2026, 13:44 BST.** Several jobs run; nothing waits for Sid. Champion:
`champion_85ea9019ff95.pt` (18.4% on unused worlds).

- **[Third continued training](experiments/2026-10-22-continued-training-3.md)**
  (`c3-train`, lane `train` of the queue in worktree `nonllm-large-batch`;
  output `data/rl_runs/continued3_20261022/train/`), iteration 9 of 24 at
  13:29, about 11.5 minutes each; ends about 16:25. Detached watcher
  `eval_continued3.sh` evaluates it. It is also the control for the joint run.
- **[Joint run](experiments/2026-10-22-joint-loop.md)** (`joint-train`,
  queued on the same lane after it; output `joint_20261022/`). Detached
  `joint_start_and_compare.sh` starts the orchestrator `joint_loop.py`
  (decisions in `joint_20261022/loop/decisions.jsonl`; create
  `joint_20261022/loop/STOP` to stop it between steps) when the trainer
  starts, and runs the final comparison on worlds from 103,000,000,000.
- **Boss-only labelling** (`teacher-boss-shard0` to `7`, queue in worktree
  `nonllm-boss-rares`; output `combat_bosses_20261022/`): about 140 of 500
  worlds per shard at 13:34; ends about 15:50. See the
  [combat variance page](experiments/2026-10-22-combat-variance.md).
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
