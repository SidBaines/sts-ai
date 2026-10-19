# Status

**Updated 19 October 2026, 15:20 BST.** One study is running, driven by a
Codex agent: [representation controls](experiments/2026-10-16-representation-controls.md).
Leave it to that agent; do not start, stop or retry its jobs from another
session.

## Running

- **What:** the representation-controls collection. All model fits and the
  label scoring are done. Fight panels are in progress (11 of 28 complete at
  15:05 BST). The eight full-game arms of 4,096 worlds each come after the
  panels, then the analysis.
- **Where:** this laptop, in the worktree
  `~/.codex/worktrees/representation-controls/SlayTheSpireAI`, branch
  `codex/representation-controls`. Outputs are in the run
  `representation_controls_20261016` under that worktree's `data/rl_runs/`.
- **Check progress:** the top of that worktree's
  `docs/representation_controls_progress.md`. Also count the complete panel
  directories in the run's `panels/` folder (28 expected).
- **Stop:** ask the Codex agent. Its queue pauses by itself on battery.
- **Expected finish:** the agent estimated the evening of 19 October; game
  arms may take longer.

## Known issues

- The queue pauses whenever the laptop is on battery. A long pause can make
  simulator calls time out, and the affected panel then has to be re-run.
- This study's own notes predate the writing convention and stay in its
  worktree's `docs/` until it finishes. They will then be condensed into the
  experiment page.

## Next

When collection and analysis finish: write up the experiment page, merge
`codex/representation-controls` and then `nonllm/docs-cleanup` into
`ws/nonllm`, and discuss the next study with Sid (see [README](README.md)).
