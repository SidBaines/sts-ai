# Status

The authorized four-hour local learning session is running, 27 October 2026.
Read [the trial design](experiments/2026-10-27-learning-trials.md).

- Worktree: `/Users/sidbaines/Documents/SlayTheSpireAI/scratch/worktrees/llm-learning-trials`.
- Branch: `codex/llm-learning-trials`, cut from `ws/llm`.
- Run: `learning_trials_20261027`; inputs: `learning_panels_20261027` and the
  existing broader training states. The four arms start from the passing stage-1 adapter.
- Deadline and arm settings are in `experiments/2026-10-27-learning-trials/config.json`.
- All four screens finished. The smaller-update candidate is continuing after
  its narrow lead on held-out teacher probability; this is an exploratory choice.
- The launch loop reserves time to evaluate the original and selected adapters
  on the full panels before the session ends.
- Check the run's `session.json`, `outcomes.json`, `screen.log` and per-arm logs.
- Stop the whole session: `touch data/llm_runs/learning_trials_20261027/STOP`.
- Scientific changes get separate arm/run names. Do not edit running scripts.

- Dashboard: http://127.0.0.1:8767/ now follows this session.
- Monitor `llm-broader-training-monitor` is active every 10 minutes.
- All four arms use medium effort. The fourth arm has a soft length cost;
  no low-effort condition is allowed.
