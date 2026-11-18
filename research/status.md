# Status

The [Act 3 boss study](experiments/2026-11-18-act3-boss-refits.md) is running locally from champion `0f527a0fbf93`. Both arms share the control trajectory until a boss refit is admitted. The power-aware queue handles all training, search and evaluation jobs.

- [Training dashboard](http://127.0.0.1:8795/) and [live progress](http://127.0.0.1:8795/progress).
- Run directory: `data/rl_runs/joint_a3_from_0f52`. Check `monitor-status.json` for the current stage and `progress.jsonl` for the timeline.
- Create `STOP` in the run directory to stop further submissions. Use the queue runner to cancel an already submitted job.
- The next decisions are the iteration-8 and iteration-16 boss refits, followed by separate endpoint selection and confirmation panels.
