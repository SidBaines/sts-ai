# Status

**Updated 20 October 2026, 11:50 BST.** The boss-rares evaluation is running;
nothing is waiting for Sid.

- **Running:** [boss rares](experiments/2026-10-20-boss-rares.md), four
  arms of the champion on the 4,096 second-DEV worlds, queued on lane `eval`
  in worktree `nonllm-boss-rares` (branch `nonllm/boss-rares`): names
  `bossrares-champion`, `bossrares-noskip`, `bossrares-demonform`,
  `bossrares-both`; outputs in `data/rl_runs/boss_rares_20261020/`. About 3
  minutes each. Check with `scripts/rl/queue_runner.py status`; cancel with
  `scripts/rl/queue_runner.py cancel NAME`. Compare arms with
  `scripts/rl/compare_evals.py champion.json ARM.json`.
- Idle: the representation-controls queue runner and two older ones in the
  `4a3c` and `slay-spire-rl-debug` worktrees.
- Topic worktrees `nonllm-repr-finish`, `nonllm-deck-oracle`,
  `nonllm-throughput`, `nonllm-ppo-hygiene`, `nonllm-power-picks`,
  `nonllm-batched-collection` and `env-fast-step` hold only merged work and
  can be removed. The local `ws/nonllm` branch (checked out in the old
  `workstream-integration` worktree under `/private/tmp`) is behind
  `origin/ws/nonllm`; cut topic branches from the origin branch.

## Next

Write up the boss-rares result; then the next study, chosen with Sid from
the options laid out on 20 October.
