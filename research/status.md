# Status

**Updated 19 October 2026, 17:45 BST.** Nothing is running.

The [representation controls](experiments/2026-10-16-representation-controls.md)
study is paused partway through its full-game arms, after Sid moved the work
from Codex to Claude. All fits, label scores and fight panels are complete. Seven
of its eight full-game arms still need collecting.

- **Outputs:** run `representation_controls_20261016`, in the worktree
  `~/.codex/worktrees/representation-controls/SlayTheSpireAI` under
  `data/rl_runs/`. The study's STOP file is present and its `control.json`
  says stop.
- **Partial and failed outputs** are kept and excluded: the interrupted
  old-encoder seed-1919 game arm (`games/old1919/`), and two earlier fight-panel
  attempts superseded by validated re-runs in the `operational_recovery_*`
  folders.
- **Resuming:** Codex's detailed instructions are in
  `docs/representation_controls_progress.md` on the tag
  `archive/nonllm/representation-controls-2026-10-19`. Each game arm's exact
  command is in `controller_plan_v1.json`, with checkpoints filled in from
  `global_freeze_v1.json` (the plan file itself holds placeholder checkpoints).
- **Idle leftovers:** two old queue runners from 13 to 15 October studies are
  still alive but idle (`queue_runner.py _run` in the `4a3c` and
  `slay-spire-rl-debug` worktrees).

## Next

Decide with Sid whether to finish the remaining game arms, then analyse the
complete fight panels (see the [README](README.md)).
