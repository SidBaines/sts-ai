# Local long-curriculum dashboard

Start from this study worktree; the service binds only 127.0.0.1. It needs no package installation or third-party assets.

```sh
.venv/bin/python scripts/long_rl_dashboard.py --port 8786
```

Open http://127.0.0.1:8786. Use `--port 0` for an automatically assigned temporary test port, `--campaign PATH` for an explicitly selected campaign, or `--refresh-seconds 10` for a slower telemetry refresh (minimum five seconds). Another process owning the requested port produces a bind failure; choose a free port without stopping unrelated work. Ctrl-C closes only this dashboard. It has no training, pause, deletion or checkpoint-loading controls.

The service exposes only `/api/telemetry`, `/`, `/index.html`, `/app.js`, and `/style.css`. The API is normalized telemetry, not raw file access. Training metrics must be campaign-owned `metrics/*.metrics.json` or `checkpoints/*.metrics.json`; symlinks escaping those directories, final-named files, other paths and checkpoint bytes are rejected. The journal/progress/config are fixed document names; external checkpoint identities are displayed without opening the checkpoint. The dashboard never evaluates policies or reads embargoed final cohorts.

Journal/config/progress refresh at most once every eight seconds. Individual metric files are cached by mtime/size plus relevant journal status/evidence, reduced once on change, and retained as compact rows. JSON input has a 64 MiB per-file bound. Static assets use no CDN. The browser keeps one snapshot and redraws bounded SVG series (at most about 450 points per series), with selectable history and the latest 30 table rows. Missing metrics stay unavailable. Auto-refresh can be disabled; manual refresh does not override the server's read rate limit. Poll failures preserve the last good display; heartbeat age and read errors flag stale or missing data.

The active budget is the authoritative persisted active-time ledger, never extrapolated while paused or offline; wall elapsed includes those gaps. Headline transitions/optimizer steps sum actor evidence on journal-committed complete training jobs. Logged metric history may include partial, interrupted or not-yet-committed job outputs; these are separate from committed totals. Job throughput divides valid transitions by the metric's measured job elapsed time; rollout throughput divides by summed collector elapsed time. Policy/value losses, KL, entropy, clip fraction and ratio are arithmetic means of the logged round update summaries per job. Returns are actual episode rewards from non-faulted episodes, including truncated episodes; quantiles interpolate sorted returns. Configured gradient clipping is not a measured gradient norm and never fills that field.

Training family strata preserve start source, starting act, goal act and horizon from the journal's TaskID-bound `family_strata`. Fresh Act 1 rehearsal is not pooled into Act 2/3 success. Native and learner saved-start success is conditional continuation performance; fresh full-start target success is a distinct rate. Requested denominators retain faults/truncations/unstarted tasks. Families accumulate repeated TRAIN sampling and are not independent benchmark worlds. Coverage diagnostics compare requested TaskID and observed episode/fault multisets; incomplete cohorts and collector-budget cutoffs are visible.

DEV charts show every recorded matched-cohort candidate, including rollback candidates as open markers, alongside fixed 41/48, 8/48 and 0/48 full-start baselines. Selected per-goal leaders are distinct from latest candidates and next training parents. An Act 2 selected model's paired Act 1 retention comes from that model's candidate event, not the `dev_act1` initial leader. Reused 48-world DEV gains are descriptive selection evidence, not significance, mastery or final performance. Current may roll back while last checkpoint remains the latest candidate; published hashes are journal metadata, not a live hash verification. Pruning counts come from the controller deletion ledger.

Edit `scripts/long_rl_dashboard.py` for normalization and `web/long_rl_dashboard/{index.html,app.js,style.css}` for presentation. These new files sit outside the frozen learner/controller source set. Do not modify frozen scientific sources or training outputs to add a dashboard metric during an active run. Add new logged telemetry only as a separately reviewed future source condition.

Run focused checks:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_long_rl_dashboard
```

These include finite/unavailable metrics, matched TaskID coverage, family separation, pause-ledger semantics, curated paths/symlink rejection, cache invalidation after journal completion and a localhost HTTP smoke on an ephemeral port.
