# Status

**Updated 22 October 2026, 19:48 BST.** Several jobs run; nothing waits for Sid. Champion:
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
- **[Better teacher](experiments/2026-10-22-better-teacher.md) benchmark**
  (`bench-*` jobs, lanes `bench0` to `bench7` of the `nonllm-boss-rares`
  queue; output `teacher_bench_20261022/<arm>/`): the teacher plays every
  boss fight inside the champion's games on 1,024 second-DEV worlds. Done:
  today's vote and best mean result. Running: pooled searches and most
  visited (to about 21:15), then the new win-first score with exploration
  1.0 and with today's exploration (native build
  `nonllm-better-teacher/external/sts_lightspeed_teacher/build_teacher_v1`,
  to about 23:00). A background watcher runs `analyse_benchmark.py` on all
  arms when they finish; detached keep-alive `keepalive_bench.py`.
- **Finished today:** boss-label combat fits (no student beat the champion), combat fit variance (one fit in four collapses; round
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
  `nonllm-joint-frozen` (the joint run's code; do not edit), `nonllm-better-teacher` (holds the new native build),
  `nonllm-large-batch-hidden`, `nonllm-fresh-start`, `nonllm-boss-rares`.

## Next

Record the third continued training and the boss-label fits when they
land; follow the joint loop's gate decisions; then start step 0 of the
[better teacher](experiments/2026-10-22-better-teacher.md) plan
(measurement and the teacher benchmark) on free lanes.
