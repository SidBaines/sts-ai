# Status

**Updated 20 October 2026, 11:30 BST.** One build in progress; nothing is
waiting for Sid except the choice of the next study.

- **In-process PPO collection** is being implemented by Codex on branch
  `nonllm/batched-collection` (worktree `nonllm-batched-collection`): an
  in-process game worker with the subprocess worker's interface, a
  `--in-process-games` training option, collector-process restarts, and an
  exactness test against the subprocess path. It merges into `ws/nonllm`
  after review.
- The [power picks](experiments/2026-10-20-power-picks.md) study is finished
  and written up; data in run `power_picks_20261020` (worktree
  `nonllm-power-picks`).
- Idle: the representation-controls queue runner and two older ones in the
  `4a3c` and `slay-spire-rl-debug` worktrees.
- Topic worktrees `nonllm-repr-finish`, `nonllm-deck-oracle`,
  `nonllm-throughput`, `nonllm-ppo-hygiene` and `env-fast-step` hold only
  merged work and can be removed. The local `ws/nonllm` branch (checked out
  in the old `workstream-integration` worktree under `/private/tmp`) is
  behind `origin/ws/nonllm`; cut topic branches from the origin branch.

## Next

Discuss with Sid: test the boss-reward skips with a paired overlay
evaluation, or build the deck-value network; see the [README](README.md).
