# Status

**Updated 23 October 2026, 16:25 BST.** Two experiments run; nothing waits for Sid.
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
- **[Value-target refits](experiments/2026-10-23-value-target-refits.md):**
  labelling 2,000 of the champion's worlds with search statistics
  (`vr-label-0` to `vr-label-7`, one lane each in the `nonllm-boss-rares`
  queue; output `value_refits_20261023/corpus/`), started 16:25; fits and
  play tests follow. Code runs from worktree `nonllm-value-targets`.
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

Record the reward ablation's iteration-12 comparison (around 17:30); when
labelling finishes, fit and play the value-target refits.
