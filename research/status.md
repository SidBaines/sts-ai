# Status

**Updated 19 October 2026, 23:25 BST.** Running: the last full-game arms of
[representation controls](experiments/2026-10-16-representation-controls.md),
on this laptop. Expected finish about 00:30 BST if it stays on mains power.
Three arms are done (old encoder at both seeds, plain structured seed 2020);
the queue was paused on battery from about 18:55 to 23:14.

- **What:** 56 shards (7 arms × 512 worlds), 7 at a time, one lane fewer than
  the study used, to leave a core free for side checks. Arms run in priority
  order: old-encoder seed 1919, old-encoder seed 2020, plain structured seed
  2020, plain structured seed 1919, then the other structured models. A
  watcher queues each arm's merge once its eight shards are complete.
- **Where:** the study's own queue in
  `~/.codex/worktrees/representation-controls/SlayTheSpireAI`. Outputs go to
  run `representation_controls_20261016`, under `games/<arm>/`; the re-run
  old-encoder seed-1919 arm is in `games/old1919_retry1/`.
- **Check:** from that worktree,
  `.venv/bin/python -B scripts/rl/queue_runner.py status` (jobs named
  `rcf_*`), and `finish_games_20261019/merge_watch.log` in the run folder.
- **Stop:** `queue_runner.py cancel <name>` for each running job; create
  `finish_games_20261019/STOP` in the run folder to stop the merge watcher.
  The queue pauses by itself on battery.
- **Repairs:** two shards had one world time out at the moment the queue
  resumed from the battery pause; both are re-run whole into `*_retry1`
  folders, which the merges use. The old-encoder seed-2020 arm has one world
  cut at the 4,096-decision limit, so the strict merge would refuse it; it is
  analysed from its shards (`no_merge` in the config), counting that world as
  a loss (Sid's decision).
- **Launch script:** `experiments/2026-10-16-representation-controls/finish_games.py`.

## Next

When the arms finish: analyse fight panels and games together, write up the
experiment page, and discuss the next study with Sid.
