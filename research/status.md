# Status

**Updated 21 October 2026, 17:32 BST.** Four studies run at once on the laptop; nothing waits
for Sid. Sid's standing instruction (21 October): keep doing research until
stopped.

- **[Second continued training](experiments/2026-10-21-continued-training-2.md)**
  (`c2-train`, lane `train` of the queue in worktree `nonllm-large-batch`;
  output `data/rl_runs/continued2_20261021/train/`): iteration 8 of 24 at
  17:16, about 12 minutes per iteration under load, so it ends about 21:00.
  Runs from the frozen worktree `nonllm-continued-frozen`; do not edit it.
  A detached watcher (`eval_continued2.sh`, log `continued2_eval.log` in
  the session scratchpad) evaluates it when it finishes.
- **Demon Form distillation, second attempt** (`df3-*`, lane `eval` of the
  queue in worktree `nonllm-boss-rares`): label collection on 6,144
  training worlds, merge, distillation, then evaluation on second DEV and
  the worlds from 95,000,000,000. The first attempt failed (512 against
  527) and its last evaluation (`df-distilled-fresh95-v3`) is running.
- **[Combat student round 2](experiments/2026-10-21-combat-student-round-2.md)**
  (`teacher-r2-shard0` to `7`, lanes `teacher0` to `teacher7` of the same
  queue; output `combat_round2_20261021/teacher_corpus_v1/` in worktree
  `nonllm-large-batch`): about 2 worlds per minute per shard, ends about
  19:30. Then the student is refitted and evaluated.
- **Keep-alives:** `teacher_keepalive.py` and `df3_keepalive.py` in the
  session scratchpad refresh those jobs' queue logs only while their own
  progress files advance, so the queue's two-hour silence check still
  catches a real stall.
- **Finished today:** the [card rule grid](experiments/2026-10-21-card-rule-grid.md)
  (no forced card helps; three hurt).
- **Champion:** `champion_2835f40fa439.pt` in `round4_20261021/inputs`,
  `continued2_20261021/inputs` and `combat_round2_20261021/inputs`
  (worktree `nonllm-large-batch`). Paired games: second DEV
  `round4_20261021/study_eval/round4_i0024` (527 wins), worlds from
  95,000,000,000 `round4_20261021/confirm95/round4_i0024` (476), from
  96,000,000,000 `continued2_20261021/confirm96/champion`.
- **Check or stop:** `scripts/rl/queue_runner.py status` or `cancel NAME`
  in either worktree. Dashboard: http://127.0.0.1:8795.
- **Closing the Claude desktop app kills processes this session started,**
  including detached queue runners, trainers, watchers and keep-alives;
  resume a trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch`, `nonllm-continued-frozen`,
  `nonllm-large-batch-hidden`, `nonllm-fresh-start` and `nonllm-boss-rares`.

## Next

Record each result as it lands; refit the combat student when the corpus
is complete.
