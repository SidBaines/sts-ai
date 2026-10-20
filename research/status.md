# Status

**Updated 20 October 2026, 23:50 BST.** Three training arms run overnight,
one after another, with watchers and an hourly check-in; nothing waits for
Sid until morning.

- **Running: large-batch out-of-combat PPO**
  ([page](experiments/2026-10-20-large-batch-ppo.md)) on lane `train` of the
  queue in worktree `nonllm-large-batch`: `lb-hygiene-hidden` (iteration 4
  of 24 at 23:36, about 6 minutes per iteration, ends about 01:40), then
  `lb-round3-student` (24 iterations of 3,072 games, about 2.5 hours) and
  `lb-small-hidden` (96 iterations of 768 games, about 2.5 hours), so the
  last ends about 06:40 if the laptop stays on power; the queue pauses jobs
  on battery and resumes on power. All three run `train_ooc.py` from the
  frozen checkout `nonllm-large-batch-hidden` at `ws/nonllm` 8aca4e4. Done
  today: `lb-recipe` (the new champion) and `lb-recipe-hidden` (arm 3: 208
  of 4,096 on second DEV and 197 on the fresh worlds, better than the
  previous champion, fewer wins than arm 1 at no clear difference in floor).
- **Watchers:** two zsh scripts in the Claude session's scratchpad
  (`eval_hidden_arms.sh`, waiting for the hygiene arm, and
  `eval_overnight_arms.sh`, for round 3 then the control; logs beside
  them). When an arm finishes they queue its final and best-by-trainer
  checkpoints with `fast_eval.py` as `lbeval-*` (second DEV) and
  `lbconfirm-*` (fresh worlds from 94,000,000,000) on the `eval` lane of
  the queue in worktree `nonllm-boss-rares`, then print paired comparisons
  (`compare_arms.py`). Outputs: `large_batch_20261020/{study_eval,confirm94}/`.
- **Hourly check-in:** a scheduled job in the Claude session (job
  `1add53ef`, about 20 past each hour) re-checks both queues, the watchers
  and the active arm's progress, records new comparisons in these notes,
  and restarts a dead queue runner or watcher. It lives only while that
  session is open.
- **Dashboard:** http://127.0.0.1:8795 shows the arms live; it runs detached
  from worktree `nonllm-dashboard` (restart with `scripts/rl/dashboard.py
  restart --port 8795 --root <nonllm-large-batch>/data/rl_runs` there).
- **Check or stop:** `scripts/rl/queue_runner.py status` in either worktree;
  `scripts/rl/queue_runner.py cancel NAME` cancels a queued or running job.
- **New champion** (promoted by Sid at 16:05): the recipe arm's iteration
  24, `large_batch_20261020/inputs/champion_ba2fa6a2987d.pt` in worktree
  `nonllm-large-batch`, which must be preserved. Paired baselines: second
  DEV `study_eval/recipe_i0024/summary.json` (276 wins), fresh worlds
  `confirm94/recipe_i0024/summary.json` (232). The harvest stack (plus the
  combat student and the Demon Form rule) is at 333 on second DEV and 346
  on the fresh worlds; round 3's start has its own baselines,
  `harvest/student` (316) and `confirm94/student`.
- Worktrees `nonllm-large-batch`, `nonllm-fresh-start` and
  `nonllm-boss-rares` must stay; `nonllm-large-batch` also holds
  uncommitted copies of research notes already merged, to discard when the
  runs are over. The other topic worktrees hold only merged work. Cut topic
  branches from `origin/ws/nonllm`; the local branch is behind.

## Next

In the morning: read the arms' comparisons from these notes or the watcher
logs, finish the experiment page, and decide with Sid whether round 3's
final checkpoint replaces the champion; then the plan in
[README](README.md): distil the Demon Form rule, add exploration tracking,
then the ε-mixture exploration arm.
