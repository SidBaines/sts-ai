# Status

**Updated 22 October 2026, 11:37 BST.** Two studies run; waiting for Sid on one decision:
which checkpoint becomes the champion. The best confirmed policy is the
second continued training's out-of-combat head with the round 2 combat
student (18.4% on unused worlds).

- **[Third continued training](experiments/2026-10-22-continued-training-3.md)**
  (`c3-train`, lane `train` of the queue in worktree `nonllm-large-batch`;
  output `data/rl_runs/continued3_20261022/train/`), from the best policy;
  about 7 to 12 minutes per iteration, so 3 to 5 hours. Detached watcher
  `eval_continued3.sh` (log `continued3_eval.log` in the session
  scratchpad) evaluates it when it finishes.
- **[Combat student round 3](experiments/2026-10-21-combat-student-round-3.md)**
  (`teacher-r3-shard0` to `7`, lanes `teacher0` to `teacher7` of the queue
  in worktree `nonllm-boss-rares`; output `combat_round3_20261021/`):
  labelling about 60% done at 11:25; detached pipeline
  `combat_r3_pipeline.sh` then fits, splices and evaluates.
- **Keep-alives** (`teacher_keepalive_r3.py` and others in the session
  scratchpad) refresh quiet jobs' queue logs only while their own progress
  files advance.
- **Candidates for Sid** (README table): the combination
  `continued2_20261021/c2_ooc_with_student_r2.pt` (18.4%, confirmed); the
  second continued training's checkpoint alone (16.9%); the round 2 combat
  candidate (13.9%). The champion of record is still
  `champion_2835f40fa439.pt`.
- **Check or stop:** `scripts/rl/queue_runner.py status` or `cancel NAME`
  in either worktree. Dashboard: http://127.0.0.1:8795.
- **Closing the Claude desktop app kills processes this session started,**
  including detached queue runners, trainers, watchers and keep-alives;
  resume a trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch`, `nonllm-continued-frozen`,
  `nonllm-large-batch-hidden`, `nonllm-fresh-start` and `nonllm-boss-rares`.

## Next

Record each result as it lands. If Sid promotes the combination, label its
games for the next combat round.
