# Status

**Updated 20 October 2026, 12:15 BST.** Nothing is running. Waiting for Sid:
the choice of the next deck-building study.

- `ws/nonllm` now has in-process PPO collection (`--in-process-games`, see
  `src/sts_rl/README.md`) and the finished
  [power picks](experiments/2026-10-20-power-picks.md) study (data in run
  `power_picks_20261020`, worktree `nonllm-power-picks`).
- Idle: the representation-controls queue runner and two older ones in the
  `4a3c` and `slay-spire-rl-debug` worktrees.
- Topic worktrees `nonllm-repr-finish`, `nonllm-deck-oracle`,
  `nonllm-throughput`, `nonllm-ppo-hygiene`, `nonllm-power-picks`,
  `nonllm-batched-collection` and `env-fast-step` hold only merged work and
  can be removed. The local `ws/nonllm` branch (checked out in the old
  `workstream-integration` worktree under `/private/tmp`) is behind
  `origin/ws/nonllm`; cut topic branches from the origin branch.

## Next

Discuss with Sid: test the boss-reward skips with a paired overlay
evaluation, or build the deck-value network; see the [README](README.md).
