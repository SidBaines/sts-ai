# Status

Nothing training or queued. The [Mac performance study](experiments/2026-11-18-mac-performance.md) finished on 18 November 2026. The champion won 4,822 of
16,384 evaluation worlds (29.4%). [Results and next studies](README.md).

- The active `ws/nonllm` checkout on Sid's Mac is
  `~/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-large-batch`.
  Its `data/rl_runs/champions/` holds `champion_4058cbf0e271.pt` and the
  promotion record. The full training checkpoint is also preserved in
  `data/rl_runs/reward_ablation_20261023/floors/iter_0024.pt`.
- The runtime used for this champion is
  `~/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python`.
  The native build is under
  `../nonllm-fresh-start/external/sts_lightspeed_rl/build015` relative to
  the active checkout. Both historical paths resolve through local links.
- The main checkout's `data/worktree-archive/2026-11-18/inventory.json`
  records archived datasets, teacher builds, frozen source and original
  commits. Start new experiment branches from `ws/nonllm`.
