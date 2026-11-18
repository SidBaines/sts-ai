# Status

Nothing training or queued. The [ML-focused pass](experiments/2026-11-18-ppo-audit.md)
finished on 18 November 2026. The proposed next control uses the existing
29.4% champion and fresh training worlds. The GPU probes created no checkpoints.

- The active `ws/nonllm` checkout on Sid's Mac is
  `~/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-large-batch`.
  Its `data/rl_runs/champions/` holds `champion_4058cbf0e271.pt` and the
  promotion record. The full training checkpoint is also preserved in
  `data/rl_runs/reward_ablation_20261023/floors/iter_0024.pt`.
- The active environment is `env-v0.1.5.2`. The research Python is
  `~/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python`.
  Use `PYTHONPATH=src` from the active checkout. Both the environment's
  default build path and the job queue's default now lead to the verified
  `data/environment_builds/env-v0.1.5.2/rl-src/build` under the main
  checkout. Historical research keeps its original binary and source.
- Adoption passed the 1,295-test workstream suite (three skipped). All 224
  champion worlds reproduced the pre-release action, terminal and outcome
  signatures (65 wins, zero faults) at the same seven processes and eight slots.
- The main checkout's `data/worktree-archive/2026-11-18/inventory.json`
  records archived datasets, teacher builds, frozen source and original
  commits. Start new experiment branches from `ws/nonllm`.
