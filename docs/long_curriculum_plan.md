# Long local curriculum: provisional execution plan

This is a new study in `/Users/sidbaines/.codex/worktrees/long-curriculum/SlayTheSpireAI`. It starts from the first campaign's selected joint composite (`joint_act3_0030.pt`, SHA256 `57f1e9613a69e31230b591ef0ff2cca27e23ddc81c3a24366cee173bc9909e70`). Earlier frozen releases, checkpoints, final cohorts and reports remain unchanged. Scope remains public-state non-LLM Ironclad, ascension 0, ordinary Acts 1–3, corrected environment 0.1.1.

**Confirmed user settings:** a first pass of **10 active hours**, dedicated GPU (`duty_cycle=1`), AC power connected, eight simulator workers and one CPU thread. Active allocation excludes manual/resource pauses and time between stopped invocations. Pause/resume must preserve the same study and permit an explicit extension if progress warrants it; extensions are logged operational amendments, not permission to change rewards or the evaluation protocol. The required 15-minute pilot precedes the main run and counts toward the active allocation. These settings supersede the earlier unanswered-preference fallback.

**Rewards remain fixed:** progress reward `0.05 × positive floor advance`, terminal success `+1`, ordinary terminal defeat `−1`, `gamma=1`, `gae_lambda=0.95`, terminal HP weight `0`. The hypothetical later floor-reward taper is **not implemented in this run**. It would be a separately named condition with critic/optimizer recalibration and fresh DEV comparison. “Current weights remain constant” here refers to reward coefficients; PPO still trains policy/value parameters.

## Data and evaluation boundaries

Fresh training worlds begin at 1,000,000 and are filtered to `world_split(seed) == 'train'`. New development worlds begin at 200,000; embargoed final worlds begin at 300,000, filtered to their respective immutable world splits. Every prefix, encounter, deck and continuation inherits its world's split. Legacy prefix banks may contribute **TRAIN worlds only**; neither old DEV nor old final outcomes may select this study's checkpoints or tune conditions.

Each run persists exact requested task IDs, seed ranges, rolling sampler offsets and input/source hashes. Separate native-generated and learner-generated starts in manifests, with source policy SHA, public observation hash, prefix actions, preview markers, starting act, goal act, horizon and native/interface provenance. Reject replay hash/provenance mismatches and already-ended starts. Preserve all failed collection attempts and counts; do not silently replace difficult worlds until a cohort appears clean.

Native-prefix later decks are a useful rehearsal distribution, not a substitute for the learner's own decks and combat states. Continuation success is conditional on reaching its saved start: it is not an unconditional full-run victory rate. Fresh full starts remain part of every stage and the development controls.

## Stages and promotion evidence

1. **Preflight and 15-minute pilot.** Validate source/backend/checkpoint identity, sampler quotas/splits, CPU fake-worker tests, prefix replay, checkpoint round trip and owned-process cleanup. Establish new, fixed DEV baselines for Act 1 retention, full Act 2 and full three-act runs before an update. Measure actual throughput, GPU duty behavior, battery/disk changes and fault/censoring rates. The pilot must show genuine completed updates and usable metrics, not merely a live process. Persist the confirmed 10-hour active budget, eight-worker/dedicated-GPU settings and recovery state before launch.
2. **Act 2 practice with Act 1 retention.** Mix saved Act 2 continuations, learner-generated later starts and fresh full starts aimed at Act 2 clearance; include legacy TRAIN-only combat rehearsal where useful. Keep start-source strata explicit. Promote only a complete DEV comparison that improves the declared Act 2 objective while passing paired Act 1 retention and fault/censoring gates. Exact quotas, horizons and guard tolerances must be persisted before the pilot; no arbitrary mastery percentage is claimed here.
3. **Act 2-to-3 continuation and full-run practice.** Balance later-start tasks with fresh full three-act tasks on rolling TRAIN worlds. Retain Act 1 controls and periodic whole Act 2 checks. Rank actual victories/clearances first; use reached-act/floor distributions as secondary progress evidence when victories remain zero. Continuation wins and endpoint HP cannot stand in for complete-game wins or run-long health preservation.
4. **Bounded final evaluation.** Select on DEV only. Freeze selected checkpoint bytes, core/interface/native identity, source, reward/task definitions and exact new final requested IDs before any final policy evaluation. Report every requested world, defeat, fault and truncation separately, plus common-valid-world comparisons and duplicate-panel exclusions.

The earlier 60-second GAE and 30-second OOC bridge pilots were rejected under their particular budgets and conditions. They do not establish that those methods cannot work. Further reward, credit-assignment or architecture changes require separately named, bounded DEV comparisons; preserve the anchor and do not mix conditions without recording the change.

## Transfer and drift safeguards

The new anchor was evaluated on the same 48 DEV worlds (minimum seed 200043), optimizer/policy seed 777:

| Full-start target | Clears/victories | Defeats | Faults | Truncated | Mean reached floor | Horizon |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Act 1 | 41/48 | 7 | 0 | 0 | 16.75 | 2048 |
| Act 2 | 8/48 | 40 | 0 | 0 | 27.42 | 4096 |
| Three acts | 0/48 | 48 | 0 | 0 | 28.60 | 4096 |

There were no unstarted tasks. Source metrics/argv are `data/long_rl_v1/anchor_dev_act{1,2,3}.{metrics,command}.json`; root records exact hashes and requested identities. Act 1 retention is relative to this new 41/48 reference, while Act 2 offers a real 8/48 clearance signal. Full three-act progress still needs explicit guards under zero victories. These cohorts differ from the earlier final benchmark and are not pooled with it.

Load the saved actor and critic weights as the explicit initial reference. A change from an Act 3 goal to Act 2 changes terminal targets: copying a critic does not establish calibrated new values. **Reset Adam/RNG training initialization at a goal/stage change; do not use exact-data resume or implicit warm optimizer carry.** The current base warm-carry helper checks task kind and reward configuration but does not detect `max_act` changes by itself. Same-stage recovery must restore a compatible checkpoint, optimizer and recorded RNG/sampler state.

The public encoder includes the current act, but **does not include goal act or task horizon**. Mixed Act 1/2/3 goal batches therefore present heterogeneous terminal targets for similar early public states. This is the explicit first-pilot multitask objective; transferred critic weights are uncalibrated for the changed mixture. Adding goal/horizon features would be an encoder-breaking representation condition requiring separate compatibility/versioning and DEV evidence, not a silent pilot adjustment.

Stage time allocation is exploratory, not a mastery certificate. The initial active allocation is 30% Act 2 continuation practice (3 hours), 25% fuller Act 2 practice (2.5 hours), 20% Act 3 practice (2 hours) and 25% mixed full three-act practice (2.5 hours). Manual/resource pause and stopped time cannot advance those stages. Extensions preserve the stage cursor and completed stages rather than silently rewinding the curriculum. Maintain goal-scoped current/best/quality and the fixed initial Act 1 reference; Act 2 scores cannot rank Act 3 models. A transition on the time schedule does not itself establish an Act 2 improvement or authorize a mastery claim.

The first controller's scalar rollback threshold of 15 points cannot detect serious all-zero-win drift because floor contributions are only a few score points. The new controller must use explicit retention/progress guards relative to a declared DEV anchor: reject incomplete cohorts; retain fault/censoring counts; detect Act 1 loss and reached-act/floor collapse even when both policies have zero wins. Persist goal-scoped current/selected best, fixed initial per-goal references, rollback reason and comparison task IDs. The two-clear Act 1 drop tolerance must remain relative to the fixed 41/48 initial reference as well as the relevant best, never ratchet downward after each promotion. No additional censoring may bypass the promotion guards. A rejected update must not become the next training parent merely because its process exited successfully.

## Background limits, storage and recovery

Confirmed limits are eight simulator workers, one CPU thread and dedicated GPU job duty `1`. Process niceness remains a CPU scheduling control. The matched 48-world CPU-only DEV repeat measured 1,477 transitions/s with four workers versus 1,768 with eight (+19.7%), with exact action traces/outcomes and no faults/truncations. This single sampling comparison supports the chosen worker setting but does not establish an equal training gain or a GPU-utilization guarantee.

Pause new work below 10 GiB free disk or at/below 25% battery while discharging, including weak-AC discharge. Preserve recovery state and release sleep assertions while paused; resource pause is not completion or an evaluation failure. Manual pause persists until an explicit resume. The active-time ledger must exclude pauses and controller downtime, use monotonic elapsed time while running and never reset consumed allocation on restart. An extension is explicit and durably recorded, preserving science/input identities and stage position.

One controller owns one learner process group and its native workers. Use bounded RPC timeout/whole-source-episode quarantine, bounded child runtime and a true no-progress watchdog. Stop only proven owned groups, including orphan descendants after a quick parent exit; preserve unrelated processes. Restart from the atomic journal and verified checkpoint/task artifacts. Distinguish retry, partial-checkpoint recovery and rollback from advancing a completed chunk. Every policy input checkpoint is checked against its published job/recovery SHA before use.

The user's recovery limit is at most 30 minutes of lost work. Save policy/critic/optimizer state atomically after each completed training chunk (planned chunk budget 180 seconds), preserve the last valid partial checkpoint if interrupted and persist current/best plus pending sampler/RNG/goal state. The pilot must verify actual checkpoint age and pause/resume behavior rather than infer them from configuration. A CPU serialization/durable-write probe of the existing 55.24 MB payload took 69.4, 44.0 and 43.1 ms; it excludes MPS synchronization/model transfer and is not an end-to-end checkpoint-time guarantee (`data/long_rl_v1/preflight/checkpoint_write_probe.json`). Shutdown must leave no owned learner/native children.

Checkpoint retention keeps current, selected best, hourly recovery points and the two most recent checkpoints; preserve all metrics and a deletion ledger with removed artifact SHA, reason and time. Cleanup is confined to this run's owned outputs, never symlinked/shared banks or earlier frozen artifacts. Source ancestry hashes remain recorded even when an unselected checkpoint is intentionally removed.

**Current data interface:** `sts_policy.long_run_tasks` supplies immutable `sample_training_tasks`, `write_task_batch`, `fixed_dev_panels` and learner-frontier capture/merge. `train_nonllm.py --preserve-task-prefixes` preserves per-task prefixes/goals/horizons; `--continuation-output` records healthy TRAIN-only learner starts. The sampler/capture 11-test suite passed independently; this does not validate an unreviewed controller.

## Running and recovery interface

Source commit `f23cad1` passed 404 native-host tests (one optional artifact skip), the bounded core smoke, and focused owned-process/active-budget regressions before root launched the 15-minute pilot at 10:37 UTC. The pilot wall cap is 900 seconds; its actual active time consumes the same 10-hour allocation. Root owns pilot acceptance and main launch. Scientific source files remain unchanged during execution.

Read status or request a manual pause from the long worktree:

```sh
PYTHONPATH=src .venv/bin/python scripts/run_long_curriculum.py --output data/long_rl_v1/campaign --action status
PYTHONPATH=src .venv/bin/python scripts/run_long_curriculum.py --output data/long_rl_v1/campaign --action pause
```

Pause checks the recorded controller birth before signaling its owned process. Inspect `journal.json`/`progress.json` and verify that its learner/native descendants have stopped. Do not remove the run directory, selected checkpoint or task artifacts.

Resume the exact saved argv after the controller has stopped; the saved command has no pilot wall cap and does not repeat an extension:

```sh
PYTHONPATH=src .venv/bin/python - <<'PYTHON'
import json, os, subprocess
from pathlib import Path
record = json.loads(Path("data/long_rl_v1/campaign/main_command.json").read_text())
environment = os.environ.copy()
environment.update(record["environment"])
subprocess.run(record["argv"], cwd=record["cwd"], env=environment, check=True)
PYTHON
```

To extend after stopping, use that same saved argv with `--extend-hours N` appended for the explicitly desired positive number of hours, then resume. Each explicit extension is a new ledger entry and adds only final mixed-stage time; it does not reset consumed time, rewind stages or change reward coefficients/protocol. Avoid reissuing an extension unintentionally. Current/best/recovery SHA and immutable input checks must pass before another learner launches.

Useful records: `data/long_rl_v1/campaign/journal.json` (durable state/budget/ownership), `main_command.json` (exact main/resume argv), `progress.json` (compact status), `events.json` (decisions/amendments), and `docs/long_curriculum_live.md` (latest-first controller output). Final worlds remain embargoed until a separate selection freeze. Root coordinates checkpoint selection, resource limits and any later explicitly named scientific condition.
