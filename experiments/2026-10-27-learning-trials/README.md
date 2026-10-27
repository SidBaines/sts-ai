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

Run `PYTHONPATH=src .venv-mlx/bin/python experiments/2026-10-27-learning-trials/summarize.py` to update paired comparisons from every completed evaluation without model work.

After both full evaluations finish, add `--figures research/figures/learning-trials`
to export every evaluation and paired contrast as CSV and a compact outcome plot.
Use `MPLCONFIGDIR=/tmp/sts-learning-trials-matplotlib` if the default cache is not writable.
The full-panel analysis also separates previously screened validation states from
the other validation states; these are exploratory subsets, not fresh-world tests.
