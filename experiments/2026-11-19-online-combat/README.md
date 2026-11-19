# Online combat PPO experiment

The study starts from the 19 November champion with independent weights for strategy, normal combat and Act 3 boss combat. The pilot measures real combat exploration, screens learning rates with fresh games, and passes only settings into the 24-update arms. The [study design](../../research/experiments/2026-11-19-online-combat.md) defines the comparisons.

Both worktrees use the same local job queue and released native build. Point this checkout's ignored `data/rl_runs` at the workstream's existing run store. Use the research Python environment with MPS and set `STS_LIGHTSPEED_BUILD` to the workstream's released build. From this checkout, submit the coordinator with the queue tool:

```sh
python scripts/rl/queue_runner.py add online-joint-study --lane online-joint-supervision -- \
  env PYTHONUNBUFFERED=1 STS_LIGHTSPEED_BUILD="$STS_LIGHTSPEED_BUILD" \
  "$VIRTUAL_ENV/bin/python" -u "$PWD/experiments/2026-11-19-online-combat/study.py" \
  --out "$PWD/data/rl_runs/online_combat_from_9700"
```

The coordinator queues the initial calibration, exploration panels, matched wiring check, common fit data, learning-rate pilots, training and independent full-game comparisons. It checks the shared queue and materialized outputs between stages. The two production trainers share the Mac training lane. A frozen endpoint can begin its paired evaluation while the other arm trains.

The run directory contains `status.json`, `exploration-selection.json`, `fit-selection.json`, `primary.json` and `verdict.json` as stages finish. Put a `STOP` file there to prevent further submissions. Already queued jobs can be stopped through the queue. A failed job requires diagnosis. A recovery can map its queue name to a verified replacement in `recovered-jobs.json`. Reuse the same output only where the command supports resuming safely. The code that changes combat weights and computes the screening statistics lives in `src/sts_rl/online_combat.py`.
