# Representation controls controller

This controller implements `docs/representation_controls_plan_2026-10-16.md`.
It schedules the fixed study and validates artifacts; root owns source review,
native pilots, monitoring, repairs and scientific analysis. Implementing or
testing it does not enable the experiment. No controller or workload was started
during implementation.

The implementation consists of `scripts/rl/representation_controls.py`,
`src/sts_rl/representation_controls.py` and
`tests/unit/test_representation_controls.py`. It changes no inherited modules.

## Paths and entry points

Defaults are:

```text
ROOT=/Users/sidbaines/.codex/worktrees/representation-controls/SlayTheSpireAI
BASE=ROOT/data/rl_runs/representation_controls_20261016
OLD=/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start/data/rl_runs/fresh_start_20261015
OLDROOT=OLD.parent.parent.parent
NATIVE=OLDROOT/external/sts_lightspeed_rl/build015
Python=ROOT/.venv/bin/python
CACHE_RECEIPT=BASE/controlled_cache_v1/receipt.json
```

`--root PATH` changes ROOT, BASE, script paths and Python together. OLD and NATIVE
stay fixed. Invoke the reviewed script in that ROOT with its frozen interpreter.
Do not install packages, rebuild native code or put the controller in a queue
job: a queued controller can deadlock the work it schedules.

Root can generate the complete plan without accessing source gates, queue state,
checkpoints, outcome files or confirmation data:

```bash
ROOT=/Users/sidbaines/.codex/worktrees/representation-controls/SlayTheSpireAI
BASE="$ROOT/data/rl_runs/representation_controls_20261016"
PYTHONDONTWRITEBYTECODE=1 "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/rl/representation_controls.py" --root "$ROOT" \
  --plan --out "$BASE/controller_plan_v1.json"
```

`--plan` requires `--out`. Its only writes are the requested output and parent
directories, with a temporary sibling used for atomic publication. Existing
output is never overwritten. The resulting JSON is read-only and contains the
18 main fit commands, eight conditional fits, all 28 possible score/panel
commands, both permitted LR variants for each student game arm, all game shards
and all merge commands. Variant inclusion is conditional; it does not authorize
running both full-game LRs. There are no executable placeholder paths.
The plan's `preencoded_cache` field records CACHE_RECEIPT. All 16 controlled main
fits and eight possible secondary fits use `--preencoded-cache` with that same
path. Old fits have no cache argument. Planning does not inspect the cache or
require it to exist; scientific conditions and original raw input paths stay fixed.

The plan records the approved maximum deadline, 2026-10-17T08:00:00Z. Actual
submission deadlines come from control.json and are frozen in the durable job
record and argv before submission. A changed deadline applies only to new jobs.

After root has enabled the gates below, these are the execution entry points:

```bash
PYTHONDONTWRITEBYTECODE=1 "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/rl/representation_controls.py" --root "$ROOT" --tick

PYTHONDONTWRITEBYTECODE=1 "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/rl/representation_controls.py" --root "$ROOT" --daemon --interval 15
```

Root handles detaching/supervising the daemon and starting the reviewed queue
runner separately. The controller never invokes `queue_runner.py start`.
Intervals must be positive and at most 30 seconds. Waiting dependencies are
polled; STOP, deadline, invalid artifacts or changed bindings stop the daemon
with exit code 2. A single waiting tick returns 0; completion returns 0.

## Root-owned gates

Nothing is fabricated when a gate is absent. Root must provide these files:

1. `BASE/controller_plan_v1.json`, generated with `--plan` at that exact path.
2. `BASE/preflight_pass_v1.json`, with the schema below and status PASS.
3. `BASE/frozen_panel_grid_v1.json`, frozen independently after the four bank
   jobs complete. The controller never collects, merges or chooses bank starts.
4. `BASE/control.json`, initially suitable for preparation only:

```json
{
  "allowed_phase": "preparation_only",
  "stop": false,
  "safety_deadline_utc": "2026-10-17T08:00:00Z"
}
```

Allowed phases are `preparation_only`, `fits`, `evaluations` and
`complete_pipeline`. Fits need `fits` or a later phase; score/panel/game/merge
submissions need `evaluations` or `complete_pipeline`. Later phases still honor
every earlier dependency. `fits` permits completing the validation-only freeze,
then waits without evaluation. `stop` must explicitly be false. A deadline must
be UTC, in the future and no later than the approved boundary.

Each file identity has exactly the recording.py convention:
`{"path": "absolute resolved path", "bytes": 123, "sha256": "64 hex digits"}`.
The preflight receipt is an object with these required fields:

| Field | Required meaning |
| --- | --- |
| `schema` | `representation_controls_preflight_v1` |
| `status`, `root` | `PASS`, exact absolute ROOT |
| `checks` | All checks listed below equal the string `PASS` |
| `source_files` | File identities from `source_paths(ROOT)` |
| `manifest` | Identity of OLD/common_manifest_v1.json |
| `data_identity` | Its validated scientific data identity |
| `preencoded_cache` | Exact cache provenance object below, bound to that manifest/data identity |
| `filters`, `vocabulary_sha256` | Validated original filtering/vocabulary contract |
| `label_source_files` | Exact manifest source_files identities, preserving order |
| `probe_source_files` | Exact original probe shard identities, preserving order |
| `champion` | Identity of OLD/inputs/champion_3d135d9bd53b.pt |
| `python` | File identity of ROOT/.venv/bin/python, resolving its symlink |
| `native` | Complete original build015 native provenance object |
| `native_files` | Nonempty identities of native build artifacts under NATIVE, including the module SHA in native |

The required checks are `source_review`, `same_head_native`,
`controlled_child_pilot`, `cpu_mps_parity`, `capacity_width216`,
`fresh_ranges_disjoint` and `no_queue_auto_retry`. These are root attestations
backed by root's actual checks. Root must record the exact actor counts and the
minimax width selection in its preflight evidence, verify matching initial base
weights, dormant OFF projection, dropout zero, frozen critic, fixed optimizer,
canonical labels, and the full 113,042/17,989 FIT/validation contract. Root must
check both new world ranges against known run specifications before enabling.
The controller does not perform those native or scientific preflight experiments.

Before the final gate, root creates the lossless cache once through
`fresh_cache.py`, producing `BASE/controlled_cache_v1/receipt.json` and
`dataset.pkl` from the original canonical preencode output: 113,042 FIT and
17,989 validation examples. Cache creation and consumer encoding checks belong
to the cache implementation. The controller neither creates nor unpickles it.
The required `preencoded_cache` object has exactly these keys:

```text
{
  "schema": "controlled_preencoded_cache_v1",
  "receipt": <file identity of CACHE_RECEIPT>,
  "dataset": <file identity of dataset.pkl>,
  "manifest": <same file identity as preflight.manifest>,
  "data_identity": <same SHA256 as preflight.data_identity>
}
```

The receipt identity must use the canonical resolved CACHE_RECEIPT path. Receipt,
dataset and manifest identities are verified before state pinning and on gate
rechecks, including immediately before each queue submission. Label source files,
manifest, probe identities and the exact source inventory remain required.
Root preserves/replaces any immutable pre-cache plan before a main fit or state
pin; the controller never replaces it or migrates a pinned state automatically.

`source_paths(ROOT)` matches recording.begin_run's archived inventory: all files
under src, scripts, tests and patches with suffix .py/.json/.patch/.md/.sh,
excluding __pycache__, plus pyproject.toml and the approved design document.
Missing, extra or duplicate identities fail. New launches must record this same
inventory (excluding the design document, which begin_run does not archive).
Original old-seed1919 launches retain their historical source identities.

Root has repaired the queue runner's former watchdog auto-resume path; it no
longer automatically retries stalled commands. A PASS receipt still cannot
override the controller's explicit AST check for the known `--resume` path.
Root's review must also rule out other automatic retries and provide the
queue/watchdog deadline policy. This implementation neither modifies the runner
nor starts it. The static check is conservative; even an exact `--resume` string
used only by rejection logic will need root review before that source can pass.

Preflight, grid and plan identities are pinned in state before fitting. Source,
runtime, native, label and probe files are checked against the receipt. The grid
uses the inherited strict filesystem validator, with exact 400/100/100 targets,
unsalted -1 plus salts 0..7, complete bank bindings and bank range
92,100,000,000 through 92,100,001,023. No shortfall is accepted. Changing a pinned
gate, source inventory, checkpoint or selection blocks continuation.

## Scheduling and immutable selection

The only subprocess command issued by the controller is an argv list:

```text
Python ROOT/scripts/rl/queue_runner.py add NAME --lane L --
  /usr/bin/env STS_LIGHTSPEED_BUILD=NATIVE PYTHONDONTWRITEBYTECODE=1
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  Python ROOT/scripts/rl/EXISTING_SCRIPT.py ...
```

There is no shell, cloud call, direct workload subprocess, fit resumption or
automatic operational retry. The explicit native assignment overrides the
queue's stale default. Existing unrelated pending/running/paused jobs cause the
controller to wait before submitting a new phase. Only shards of the current
game arm can coexist under controller scheduling. Root must keep other heavy
work from being added concurrently through independent clients.

The 18 main fits interleave feature, width, LR and training seed. The prescribed
OFF/128/0.0003/1919 plumbing replicate leads. Every fit uses the original eight
teacher shards, common manifest and probe corpus; all supported CLI settings are
explicit. Dropout, AdamW weight_decay and clipping have no existing CLI flags:
the pinned source gate binds those inherited fixed values. Every new fit uses
one lane named `fit`, with at most one pending/running fit. Root can inspect early
metrics and stop before the next fit. Queue status alone never asserts learning
health; accepted state records actual completed steps and example presentations.

A completed fit must have completed/deferred/unscored status, exact FIT and
validation counts, its actual best.pt file identity, actual checkpoint condition,
best validation step/loss, finite weights, from-scratch initialization, matching
data/source/probe receipts, optimizer LR/weight decay and native snapshot. The
controller inspects checkpoint payloads on CPU; it never constructs an actor,
initializes weights, runs inference, or places a controller model on MPS.
Controlled checkpoints must record the exact CACHE_RECEIPT path in their actual
config, the gate's provenance in `preencoded_cache_provenance` in both config and
launch effective_config, and that same object in source_receipt.preencoded_cache.
Launch data_inputs must contain the exact receipt and dataset identities, which
are verified along with all other training inputs. Old1919 references and old2020
fits must have no cache config/provenance fields and retain their original raw
corpus paths.

After all 18 fits, `lower_lr_extension_v1.json` records the four width216
validation comparisons. Only four strict improvements trigger all eight
0.0001 fits. A tie or any non-improvement prevents the extension. Secondary
fits finish before TEST/panel/games and never affect primary LR choices or
full-game arms. Both original seed1919 old-LR validation summaries and best.pt
files are then validated as immutable references; original TEST files are not
used for selection.

`global_freeze_v1.json` exclusively freezes all 20 conditions, or all 28 if the
extension triggers, including checkpoint/summary/config/data/source identities.
It also records the complete gate `preencoded_cache` provenance object; freeze
recovery requires an exact match.
Primary choices minimize best validation CE, breaking exact ties toward lower
LR within {0.0003,0.001}. The eight game arms are fixed at this point. The
controller never chooses an arm from TEST, panel or game outcomes.

Each condition gets `selections/NAME.json` containing selected_checkpoint,
manifest identity, architecture, actual training_condition, selection_criterion
equal to sts_rl.fresh_scoring.SELECTION_CRITERION, and the global freeze file
identity. Original checkpoint paths are used directly. JSON receipts are
published atomically, exclusively and read-only; checkpoints are never copied
or modified. The global file does not contain selection file hashes, avoiding
a circular hash dependency. State binds all selection identities before scoring.
`cohort.json` exclusively freezes the exact 4,096-world list, arms and global
freeze identity.

On resume an existing global freeze is verified and reused; LRs are never chosen
again. A crash between publishing the global freeze and publishing every
selection can finish publication using that existing decision. Missing already
recorded freeze/selection files cause a block rather than replacement.

All conditions receive sequential label scoring first, then sequential panels.
There is no overlap opt-in in this conservative implementation. Scoring verifies
singleton CPU protocol, complete manifest TEST row identities/targets/conditions,
zero weight updates and immutable launch receipts. The manifest rows are streamed
to avoid retaining the full 393MB document. Panel validation delegates to the
existing strict grid/result/receipt validator without calling its statistics
functions. Incomplete, faulted, cut, censored or quarantined cohorts cannot
advance.

After all panels, full games run champion, old1919, OFF128/ON128/OFF216/ON216 at
1919, old2020 and OFF128 at 2020. Each arm gets eight 512-world shards on
game0..game7, one worker per shard, timeout 30, no teacher labels, no bank and no
self consistency. All use 91,001,000,000 through 91,001,004,095. Champion has no
combat replacement. A queued fresh_merge.py merge into that arm's fresh merged
directory must pass validate_cohorts with the exact world range before another
arm starts. Native, runtime, source archive, environment and filtering identities
must agree across game arms. A final filesystem audit revalidates all evaluations
and merged games before state becomes complete. It runs again on process restart.
No scientific statistics, comparisons or report conclusions are computed here.

## Recovery and limits

`controller_state_v1.json` is atomically replaced under a nonblocking process
lock. It records a submission intent and the exact command before queue add.
A crash after successful add is reconciled against that matching queue entry.
An intent with no queue entry is ambiguous and blocks; the controller never
repeats add automatically. Duplicate/unowned names, changed argv/lane, retry
markers, failed/cancelled jobs and invalid artifacts require root diagnosis.
`done` or `exited` with an unknown/zero exit status advances only after strict
artifact validation. No return code converts a fault or interruption into a loss.

Root must preserve failures and reconcile ambiguous submissions deliberately.
There is no automated repair API and no retry counter. A permitted, diagnosed
panel retry remains root work, preserving the old attempt and unchanged grid and
checkpoint. Never delete the ledger to force reselection or rerun a failed fit.
Source changes after pinning require a new reviewed preservation/recovery plan;
the controller will not silently accept a replacement preflight receipt.

`BASE/STOP`, control.stop, preparation_only and deadlines prevent **new**
submissions. They do not cancel or kill queue jobs that have already been added.
Root must use the queue's reviewed external controls and workers' per-run STOP
files for those jobs. Workers receive frozen deadlines; fresh_merge.py has no
deadline CLI flag, so root's queue watchdog must cover queued merges and the hard
operational boundary. Do not mistake a stopped controller for a stopped queue.
Free space is checked before each submission (6GiB minimum); worker controls
remain responsible during execution.

Unchanged identity checks cache file size, inode, device, mtime and ctime to avoid
rehashing large frozen inputs every idle tick. Changed files are rehashed; a new
process starts with an empty cache. Completed-job core receipts are checked on
resume; strict evaluation contents are audited again before completion. This is
operational integrity checking, not a defense against a malicious writer who
can rewrite the ledger and forge filesystem metadata.

Root still monitors validation curves, throughput, sample exposure and faults
every 20–30 minutes and owns interpretation. A live queue entry or controller
heartbeat provides no evidence that a fit is learning. Confirmation stays closed.

## Synthetic verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest tests.unit.test_representation_controls -v
```

Tests use temporary synthetic files, tiny synthetic checkpoint payloads, and a
mock submission function/validator. No queue is started, no fit/inference/native
operation is performed, and no actual study outcomes or confirmation files are
read. They cover the exact factorial and game arms, extension trigger, LR ties,
dependency/phase gates, resume and ambiguous submission behavior, source and
checkpoint binding failures, strict score/grid validation, serial evaluations,
eight-shard arm scheduling and strict merge dependencies. Real root preflight
and native/CPU/MPS checks remain required before enabling the controller.
Cache tests use synthetic JSON bytes and file identities, covering all 24
controlled commands, pure planning, missing/wrong/tampered gate bindings,
submission rechecks, checkpoint/launch cache mismatches, old-fit exclusion and
immutable freeze recovery. They perform no encoding, fitting or scoring.
