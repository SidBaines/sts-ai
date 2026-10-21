# Status

**Updated 21 October 2026, 15:37 BST.** Nothing is training. Waiting for Sid: whether the
continued training's final checkpoint (476 of 4,096 new worlds, 11.6%)
becomes the champion.

- **Finished:** [continued training](experiments/2026-10-21-round-4.md),
  run `round4_20261021/round4` in worktree `nonllm-large-batch`; candidate
  checkpoint `iter_0024.pt` there. Its games: second DEV
  `round4_20261021/study_eval/round4_i0024` (527 wins), new worlds
  `round4_20261021/confirm95/round4_i0024` (476).
- **Champion until Sid decides:** `champion_7d876c939a74.pt` (combat student
  plus retraining), with games `large_batch_20261020/study_eval/round3_student_i0024`
  (413) and `round4_20261021/confirm95/champion` (379).
- **Idle:** both queue runners (`scripts/rl/queue_runner.py status` in
  worktrees `nonllm-large-batch` and `nonllm-boss-rares`) and the
  dashboard at http://127.0.0.1:8795 (restarted at 12:25 from worktree
  `nonllm-dashboard` after the app closure killed it).
- **Closing the Claude desktop app kills processes this session started,**
  including detached queue runners, trainers and the dashboard; resume a
  trainer with `--resume`.
- **Worktrees to keep:** `nonllm-large-batch` (every champion and run),
  `nonllm-large-batch-hidden` (the frozen trainer), `nonllm-fresh-start`
  (the native build and the combat student) and `nonllm-boss-rares`
  (evaluations). Cut topic branches from `origin/ws/nonllm`.

## Next

See the [README](README.md).
