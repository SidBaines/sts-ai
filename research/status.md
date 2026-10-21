# Status

**Updated 21 October 2026, 15:40 BST.** Training and evaluations run; nothing waits for Sid.

- **Running: [second continued training](experiments/2026-10-21-continued-training-2.md)**
  as `c2-train` on lane `train` of the queue in worktree
  `nonllm-large-batch` (24 iterations of 3,072 games, about 6 minutes each,
  ends about 18:15 if the laptop stays on power); output
  `data/rl_runs/continued2_20261021/train/` there. It runs `train_ooc.py`
  from the frozen worktree `nonllm-continued-frozen` (`ws/nonllm` a33bd2b);
  do not edit that worktree while it runs.
- **Running: evaluations** on lane `eval` of the queue in worktree
  `nonllm-boss-rares`: `c2-demonform-dev`, `c2-demonform-fresh95` (the
  Demon Form retest) and `c2-champion-fresh96` (the champion on the new
  confirmation worlds); outputs in `continued2_20261021/` subfolders.
- **Watcher:** `eval_continued2.sh` in the Claude session's scratchpad (log
  `continued2_eval.log`), detached; it prints the Demon Form comparisons,
  then queues and compares the trained checkpoint's evaluations when
  training finishes.
- **Champion:** `champion_2835f40fa439.pt` in `round4_20261021/inputs` and
  `continued2_20261021/inputs`, worktree `nonllm-large-batch`. Its games:
  second DEV `round4_20261021/study_eval/round4_i0024` (527 wins), new
  worlds from 95,000,000,000 `round4_20261021/confirm95/round4_i0024` (476).
- **Check or stop:** `scripts/rl/queue_runner.py status` or `cancel NAME`
  in either worktree. Dashboard: http://127.0.0.1:8795 (from worktree
  `nonllm-dashboard`).
- **Closing the Claude desktop app kills processes this session started,**
  including detached queue runners, trainers, the watcher and the
  dashboard; resume a trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch`, `nonllm-continued-frozen`,
  `nonllm-large-batch-hidden`, `nonllm-fresh-start` and `nonllm-boss-rares`.
  Cut topic branches from `origin/ws/nonllm`.

## Next

Apply the decision rules on the experiment page as results land.
