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
