# Status

**Updated 23 October 2026, 18:30 BST.** Two experiments run; nothing waits for Sid. Champion: `champion_c8553bc33e53.pt` (26.8% on
unused worlds).

- **[Reward ablation](experiments/2026-10-23-reward-ablation.md), extended
  to 24 iterations:** `ra-train-floors-to24` (lane `train`) and
  `ra-train-wins-to24` (lane `train2`) of the `nonllm-large-batch` queue,
  resumed from iteration 12 at 18:28; about 28 minutes per iteration with
  both running, so done around midnight. The detached watcher
  `evaluate_arms.sh 109000000000 16 20 24` (session scratchpad,
  `reward_ablation/`, log `eval109.log`) plays each checkpoint and the
  champion on 16,384 worlds from 109,000,000,000 and runs the comparison;
  results around 01:00. Code runs from worktree `nonllm-reward-ablation`.
- **[Value-target refits](experiments/2026-10-23-value-target-refits.md):**
  labelling 2,000 of the champion's worlds with search statistics
  (`vr-label-0` to `vr-label-7`, `nonllm-boss-rares` queue; output
  `value_refits_20261023/corpus/`), nearly done; fits and play tests follow.
  Code runs from worktree `nonllm-value-targets`.
- **Promotion candidate:** the wins-only arm's iteration-12 checkpoint
  (28.3% on 16,384 unused worlds against the champion's 26.7%). Sid will
  decide after the 24-iteration comparison.
- **Check or stop:** `scripts/rl/queue_runner.py status` or `cancel NAME`
  in `nonllm-large-batch` or `nonllm-boss-rares`. Dashboard:
  http://127.0.0.1:8795.
- **Disk:** check with `scripts/rl/disk_report.py` (see `scripts/rl/CLAUDE.md`).
- **Closing the Claude desktop app kills processes this session started**,
  including the watchers; resume a trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch`, `nonllm-continued-frozen`,
  `nonllm-joint-frozen`, `nonllm-joint2-frozen`, `nonllm-reward-ablation`,
  `nonllm-value-targets` (frozen code of runs; do not edit),
  `nonllm-better-teacher` (holds the new native build),
  `nonllm-large-batch-hidden`, `nonllm-fresh-start`, `nonllm-boss-rares`,
  `nonllm-dashboard`, `nonllm-disk-hygiene`, `nonllm-joint-run-2`.

## Next

When labelling finishes, fit and play the value-target refits; record the
24-iteration reward comparison (around 01:00).
