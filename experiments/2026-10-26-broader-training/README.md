# Continuing on broader decisive states

The design is in `research/experiments/2026-10-26-broader-training.md`.
Run from the worktree root using the existing `.venv-mlx` runtime and local
model. Input and parent-run directories are recorded in `run.sh`.

```sh
python3 scripts/llm/launch_local.py broader_training_20261026 experiments/2026-10-26-broader-training/run.sh
```

The launcher detaches the shell and power babysitter. `caffeinate` prevents idle
sleep on mains; on battery the run pauses. Closing the lid can still sleep the Mac.
`run.sh` writes trainer output directly to `data/llm_runs/<run>/lr_3e-5.log`.
The watcher exits on completion or process death, reports stalls and position
bias, and requests a stop for the deterioration rules on the research page.

Create `data/llm_runs/<run>/STOP` to request a stop between iterations. The
trainer saves the final adapter and evaluates it before exiting. Its eight-hour
limit is checked between iterations and allows the final evaluation to finish.
Check `pids.txt`, `watch.log`, `alerts.jsonl`, `lr_3e-5/iterations.jsonl`,
`lr_3e-5/evaluations.jsonl` and `lr_3e-5/run_summary.json` for state and results.
Check both `screen.log` and `lr_3e-5.log` for failures.

The original dataset preparation exhausted the available distinct validation
combat worlds at 50 rather than the requested 64. The labelled candidates were
retained in `broader_states_20261026`; the final selection is in
`broader_states_selected_20261026`. `preparation_config.json` records the
selection command options, and `manifest.json` records its input identity.
`train_eval.jsonl` holds the first 32 selected states of each training phase.
`verification.json` records counts and the zero world overlap with training,
including the parent adapter's training states.

## Live dashboard

Run `python3 -u scripts/llm/dashboard.py --port 8767` and open
http://127.0.0.1:8767/. The defaults select this continuation and the previous
stage-1 screen. See `scripts/llm/README.md` for data-update behavior. Server
metadata and logs are in `data/llm_runs/broader_training_20261026/dashboard/`.

## Completed-run analysis

Run `PYTHONPATH=src .venv-mlx/bin/python experiments/2026-10-26-broader-training/analyze.py`
to verify pairing, compare every saved evaluation, and regenerate run
`analysis.json` plus the figure and CSVs in `research/figures/broader-training/`.
Statistics use the tested `sts_llm.statistics` library. Active wall time is
estimated from file timestamps and power logs; raw monotonic timers include
awake battery pauses and must not be labelled compute time.

The completed run predates evaluation progress counters. New trainer runs
report token and response counts every 30 seconds; this reporting change was
made only after the run finished, without modifying historical outputs.

Verification after adding progress counters: 26 focused MLX/statistics tests passed; JavaScript syntax and completed-run/progress-counter dashboard smoke checks passed.
