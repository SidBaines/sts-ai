# Status

**Updated 20 October 2026, 01:30 BST.**

Running:

- **Codex task 2, batched game runner** (branch `nonllm/throughput`, worktree
  `.claude/worktrees/nonllm-throughput`). Log:
  `codex/task2.log` in Claude's session scratchpad. Stop by ending the
  `codex exec` process (find it with `ps -eo pid,args | grep codex`).

Waiting for review:

- **Codex task 1, environment step speed** (branch `env/fast-step`, worktree
  `.claude/worktrees/env-fast-step`, uncommitted). Claude reviews, re-runs
  tests, then opens a pull request to `main`.

Idle: the representation-controls queue runner and two older ones in the
`4a3c` and `slay-spire-rl-debug` worktrees.

## Next

Review and land both throughput tasks, then the oracle's option test and
PPO hygiene, as listed in the [README](README.md).
