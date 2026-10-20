# Status

**Updated 20 October 2026, 02:45 BST.** Nothing is running and nothing is
waiting for Sid.

- `ws/nonllm` has environment 0.1.5.1 (faster step), the batched evaluator
  (`scripts/rl/fast_eval.py`), the PPO hygiene options and the deck oracle
  library (`src/sts_rl/deck_oracle.py`).
- Idle: the representation-controls queue runner and two older ones in the
  `4a3c` and `slay-spire-rl-debug` worktrees.
- Topic worktrees `nonllm-repr-finish`, `nonllm-deck-oracle`,
  `nonllm-throughput`, `nonllm-ppo-hygiene` and `env-fast-step` hold only
  merged work and can be removed.

## Next

Choose the next deck-building study; see the [README](README.md).
