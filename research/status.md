# Status

**Updated 23 October 2026, 13:45 BST.** Two experiments run; nothing waits for Sid.
Champion: `champion_c8553bc33e53.pt`, joint run 2's final checkpoint (26.8%
on unused worlds).

- **[Reward ablation](experiments/2026-10-23-reward-ablation.md)**
  (output `reward_ablation_20261023/` in worktree `nonllm-large-batch`):
  trainers `ra-train-floors` (lane `train`) and `ra-train-wins` (lane
  `train2`) of the `nonllm-large-batch` queue, 12 iterations each from the
  champion, both started 12:13; about 28 minutes per iteration with both sharing the laptop, so expected to finish around 18:00. Code runs
  from worktree `nonllm-reward-ablation` (do not edit it). The detached
  watcher `evaluate_arms.sh` (session scratchpad, `reward_ablation/`, log
  `eval108.log`) queues each arm's iteration 4, 8 and 12 checkpoints and the
  champion on 16,384 worlds from 108,000,000,000 (lane `eval`) and runs the
  comparison when they finish.
- **[Value targets](experiments/2026-10-23-value-targets.md), stage 2:**
  seed 2 fits (`vt-votes-seed2`, `vt-values100-seed2`, `nonllm-boss-rares`
  queue) and play tests on 8,192 worlds from 110,000,000,000 (lane `eval` of
  the `nonllm-large-batch` queue), driven by the detached watcher
  `stage2.sh` (session scratchpad, `value_targets/`, log `stage2.log`),
  which ends with the pre-registered comparison. Code runs from worktree
  `nonllm-value-targets` (do not edit it).
- **Next decision:** if iteration 12 does not separate the arms
  (p < 0.025), resume both to 24 iterations: `--resume --iterations 24`
  with the same command and output directory, then
  `evaluate_arms.sh 109000000000 16 20 24`.
- **Check or stop:** `scripts/rl/queue_runner.py status` or `cancel NAME`
  in `nonllm-large-batch`. Dashboard: http://127.0.0.1:8795.
- **Disk:** 79 GiB free at 10:05. See `scripts/rl/CLAUDE.md` for the
  periodic check.
- **Closing the Claude desktop app kills processes this session started**,
  including the watcher; resume a trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch`, `nonllm-continued-frozen`,
  `nonllm-joint-frozen`, `nonllm-joint2-frozen`, `nonllm-reward-ablation`,
  `nonllm-value-targets` (frozen code of runs; do not edit), `nonllm-better-teacher` (holds the new
  native build), `nonllm-large-batch-hidden`, `nonllm-fresh-start`,
  `nonllm-boss-rares`, `nonllm-dashboard`, `nonllm-disk-hygiene`,
  `nonllm-joint-run-2`.

## Next

Record the value-target play test and the reward ablation's iteration-12
comparison (around 18:00).
