# Status

**Updated 20 October 2026, 13:25 BST.** One training run is running with two
queued behind it; waiting for Sid: the next study.

- **Dashboard:** http://127.0.0.1:8795 shows the large-batch arms live
  (progress, time estimates, health charts). It runs detached from worktree
  `nonllm-dashboard`; restart with `scripts/rl/dashboard.py restart --port
  8795 --root <nonllm-large-batch>/data/rl_runs` there.
- **Running: large-batch out-of-combat PPO**
  ([page](experiments/2026-10-20-large-batch-ppo.md)) on lane `train` of the
  queue in worktree `nonllm-large-batch`: `lb-recipe` finished at 15:20 (24
  iterations); `lb-recipe-hidden` is running and `lb-hygiene-hidden` queued
  (they run `train_ooc.py` from the frozen checkout
  `nonllm-large-batch-hidden` at `ws/nonllm` 8aca4e4, which has
  `--hide-ooc-potions`), about 3 hours each. The recipe arm's final
  checkpoint (`large_batch_20261020/recipe/iter_0024.pt`) is confirmed on
  fresh worlds (232 against 152 of 4,096) and proposed as the new champion;
  evaluations in `large_batch_20261020/{study_eval,confirm94}/`. Outputs in `data/rl_runs/large_batch_20261020/{recipe,recipe_hidden,hygiene_hidden}`
  (`log.jsonl`, `eval_XXXX.json`, `iter_XXXX.pt`). Check with
  `scripts/rl/queue_runner.py status` in `nonllm-large-batch`; cancel with
  `scripts/rl/queue_runner.py cancel NAME`. That worktree also holds
  uncommitted copies of research notes already merged through `ws/nonllm`;
  discard them when the runs are over.
- Finished today: [boss rares](experiments/2026-10-20-boss-rares.md)
  (including the combat-head arms) and
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

Evaluate the large-batch checkpoints under the study protocol when the arm
finishes; then the next study, chosen with Sid
from the options in the
[19 October brainstorm](brainstorms/2026-10-19-brainstorming.md), section 8,
and the lock-in discussion of 20 October.
