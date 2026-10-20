# Status

**Updated 20 October 2026, 12:37 BST.** Two things are running; waiting for
Sid: whether to queue the potion-hidden large-batch arms, and the next study.

- **Running: large-batch out-of-combat PPO**
  ([page](experiments/2026-10-20-large-batch-ppo.md)), arm `lb-recipe` on
  lane `train` of the queue in worktree `nonllm-large-batch` (branch
  `nonllm/large-batch-ppo`), output `data/rl_runs/large_batch_20261020/recipe`
  (`log.jsonl`, `eval_XXXX.json`, `iter_XXXX.pt`). About 9 minutes per
  iteration while evaluations share the machine; 24 iterations. The
  `lb-hygiene` arm was cancelled before starting: it should run with
  `--hide-ooc-potions`, which that worktree does not yet have. Check with
  `scripts/rl/queue_runner.py status` there; cancel with
  `scripts/rl/queue_runner.py cancel lb-recipe`.
- **Running: boss rares with the better combat head** (the co-adaptation
  test in the boss-rares page's Next), three arms on lane `eval` of the queue
  in worktree `nonllm-boss-rares`: `bossrares-oldcombat-*`, outputs in
  `data/rl_runs/boss_rares_20261020/fast_oldcombat/`. Compare with
  `experiments/2026-10-20-boss-rares/compare_arms.py`.
- Finished today: [boss rares](experiments/2026-10-20-boss-rares.md) and
  [power picks](experiments/2026-10-20-power-picks.md).
- Idle: the representation-controls queue runner and two older ones in the
  `4a3c` and `slay-spire-rl-debug` worktrees.
- Topic worktrees `nonllm-repr-finish`, `nonllm-deck-oracle`,
  `nonllm-throughput`, `nonllm-ppo-hygiene`, `nonllm-power-picks`,
  `nonllm-batched-collection` and `env-fast-step` hold only merged work and
  can be removed. The local `ws/nonllm` branch (checked out in the old
  `workstream-integration` worktree under `/private/tmp`) is behind
  `origin/ws/nonllm`; cut topic branches from the origin branch.

## Next

Read the co-adaptation arms; evaluate the large-batch checkpoints under the
study protocol when the arm finishes; then the next study, chosen with Sid
from the options in the
[19 October brainstorm](brainstorms/2026-10-19-brainstorming.md), section 8,
and the lock-in discussion of 20 October.
