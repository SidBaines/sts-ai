# Status

**Updated 21 October 2026, 06:52 BST.** Nothing is training. Waiting for Sid: whether round 3's
final checkpoint (394 of 4,096 fresh worlds, 9.6%) becomes the champion.

- **Finished overnight:** all arms of
  [large-batch PPO](experiments/2026-10-20-large-batch-ppo.md), evaluated on
  the second-DEV worlds and the fresh worlds from 94,000,000,000. Outputs in
  worktree `nonllm-large-batch`, `data/rl_runs/large_batch_20261020/`:
  one directory per arm (`log.jsonl`, `iter_XXXX.pt`), evaluations under
  `study_eval/` and `confirm94/`.
- **Round 3 candidate:** `large_batch_20261020/round3_student/iter_0024.pt`.
  Its games, for paired comparisons: `study_eval/round3_student_i0024`
  (413 wins) and `confirm94/round3_student_i0024` (394).
- **Champion until Sid decides:** `large_batch_20261020/inputs/champion_ba2fa6a2987d.pt`
  (arm 1, iteration 24), with baselines `study_eval/recipe_i0024` (276)
  and `confirm94/recipe_i0024` (232).
- **Idle:** the queue runners in worktrees `nonllm-large-batch` and
  `nonllm-boss-rares` (check with `scripts/rl/queue_runner.py status`),
  and the dashboard at http://127.0.0.1:8795, which runs detached from
  worktree `nonllm-dashboard`.
- **Worktrees to keep:** `nonllm-large-batch` (both champions' lineage and
  every arm), `nonllm-fresh-start` (the previous champion, the combat
  student and the native build) and `nonllm-boss-rares` (the evaluations).
  `nonllm-large-batch` also holds uncommitted copies of research notes
  already merged through `ws/nonllm`; they can now be discarded. The other
  topic worktrees hold only merged work. Cut topic branches from
  `origin/ws/nonllm`; the local branch is behind.

## Next

See the [README](README.md): decide on round 3, test the Demon Form rule on
top of it, then round 4 with the hygiene options and exploration tracking.
