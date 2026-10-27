# Local learning trials

From this worktree root:

```sh
python3 scripts/llm/launch_local.py learning_trials_20261027 experiments/2026-10-27-learning-trials/run.sh
```

`config.json` holds the four arms and deadline. The launch loop reads it once,
writes all logs and settings beneath `data/llm_runs/<run>`, continues a selected
arm only within the remaining allocation, and finishes with paired full-panel
evaluations. Use root/STOP to stop between work units; per-arm STOP files stop
training between iterations. The launcher pauses the process group on battery.
No code in the active launch loop should be edited while it runs.
