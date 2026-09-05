# `rl_ops` — unattended local RL run operations

Tooling for running a multi-hour GRPO job on a laptop that sleeps, runs out of
battery, gets asked to be quiet during meetings, and hosts a simulator with a
state-dependent hang bug. Built for `ooc_grpo_v1`
(see [`docs/ooc_rl_v1_status.md`](../../docs/ooc_rl_v1_status.md)) and reusable
for any local `run_grpo.py --backend mlx` run.

Nothing here is imported by the harness; these are operational scripts.

## The pieces

| script | role |
|---|---|
| `rl_supervisor.zsh` | Self-healing launcher. Recomputes the resume point from adapters on disk, launches an attempt, watches for stalls, kills and relaunches with a bumped seed salt. Writes `run.status`. |
| `ooc_rl_attempt.zsh` | One attempt: the full `run_grpo.py` argument list for `ooc_grpo_v1`. Launched detached so it is the process-group leader. |
| `power_babysitter.zsh` | SIGSTOPs the run on battery, CONTs it on AC, owns the `caffeinate` assertion. |
| `rl_live_watcher.py` | Read-only monitor: pushes partial-iteration stats and mid-pass training curves to a `<run>-live` wandb run, refreshes the progress plots. |

Per-iteration tables and plots come from
[`scripts/rl_progress.py`](../rl_progress.py), which tolerates in-flight,
partially-written iterations:

```zsh
PYTHONPATH=src .venv/bin/python scripts/rl_progress.py --out-dir data/rl/<run>
```

## Running

Launch **detached** — harness/terminal background jobs get killed on this
machine, and the `DEVNULL` handles are load-bearing (without them the detached
child inherits the command-substitution pipe and `$()` blocks until the whole
run exits):

```zsh
cd /Users/sidbaines/Documents/SlayTheSpireAI
.venv/bin/python -c "import subprocess as sp; p=sp.Popen(['zsh','scripts/rl_ops/rl_supervisor.zsh'],start_new_session=True,stdin=sp.DEVNULL,stdout=sp.DEVNULL,stderr=sp.DEVNULL); print(p.pid)"
```

Then watch the ops journal:

```zsh
tail -f -n 0 data/rl_ops/<run>/run.status
```

### Configuration

All paths derive from the script location, so the scripts work from any
working directory and survive being checked out anywhere. Env overrides:

| variable | default | meaning |
|---|---|---|
| `RL_OUT_DIR` | `data/rl/ooc_grpo_v1` | run output directory |
| `RL_OPS_DIR` | `data/rl_ops/<out-dir basename>` | status journal, run log, DONE file, attempt rc files |
| `RL_NUM_ITERS` | 12 | total iterations before DONE |
| `RL_ATTEMPT_SCRIPT` | `./ooc_rl_attempt.zsh` | per-attempt launcher |
| `RL_WANDB_PROJECT` | `sts-ooc-rl` | watcher's wandb project |

For a different run, **copy** `ooc_rl_attempt.zsh` and point
`RL_ATTEMPT_SCRIPT` at the copy rather than editing it in place — the argument
list in that file is the provenance record of what `ooc_grpo_v1` actually ran.

## Pausing and resuming

To pause (meetings, fan noise, overnight), SIGSTOP the supervisor, the
babysitter **and** the attempt's process group. Stopping the babysitter is not
optional: left running, it sees AC power and CONTs the run straight back.

```zsh
kill -STOP <supervisor-pid> <babysitter-pid>
kill -STOP -<attempt-pgid>
kill -CONT <supervisor-pid> <babysitter-pid>   # to resume
kill -CONT -<attempt-pgid>
```

The live watcher exits on its own after an hour with no change, so relaunch it
after a long pause (or let the supervisor's `ensure_watcher` do it on the next
attempt).

If the process stack is gone entirely — reboot, or a pause you never
resumed — do not try to reattach. Just relaunch the supervisor: the resume
point is recomputed from disk every pass, and `grpo_loop` keeps a partially
finished iteration's completed episodes.

## Why the moving parts exist

- **Stall watchdog.** The simulator's uninitialized-memory UB can hang mid
  combat-search, and the in-process lockstep runner freezes the whole loop when
  it does. The hang is **state-dependent, so seed screening does not transfer
  across policies** — a cohort screened clean under one agent still wedges
  under another. The watchdog kills after 30 minutes of no output-file
  progress, exempting babysitter-paused (`T`-state) runs so a deliberate pause
  never looks like a stall.
- **Seed salt bump per attempt.** A plain restart would deterministically
  replay into the same hang. Each attempt passes
  `--policy-seed-salt attempt*1000`, so the redone iteration re-rolls
  trajectories. Salt 0 is byte-identical to the historical unsalted stream, so
  the reproducibility contract holds for everything that does not opt in.
- **Fast-failure guard.** Two consecutive sub-10-minute failures that produce
  no new adapter abort the loop rather than burning the attempt cap on a
  persistent error that needs a human.
- **Power babysitter debounce.** Two consecutive bad reads before pausing stops
  a momentary `pmset` flap from halting the run; asymmetric charge thresholds
  (resume at ≥10%, do not pause a running job until <8%) stop it oscillating at
  the boundary.

## Known benign failure mode

A machine sleep the babysitter did not cause (lid closed while on AC) looks
like a stall on wake. The kill-and-resume then simply redoes the interrupted
iteration with a fresh salt.
