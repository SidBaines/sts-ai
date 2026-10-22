# sts_rl: training, evaluation and diagnostics

`sts_rl` trains, evaluates and diagnoses non-language-model policies on top of
[`sts_policy`](../sts_policy/README.md) and the separately built simulator.
See [current research](../../research/README.md),
[strategies](../../research/strategies.md) and [metric
definitions](../../research/metrics.md).

## Module map

Reusable components support several jobs; study-specific adapters belong in
experiment folders under the code-placement convention.

| Reusable components | Purpose |
| --- | --- |
| `common` | Workers, checkpoints, public menu filters and tasks; contains historical study defaults that callers must override. |
| `recording` | Launch snapshots, input identities, resumable optimizer/random state and compressed collection records. |
| `evaluate`, `parallel_collect` | Fixed-world evaluation, paired comparisons and persistent parallel collectors. |
| `batched_runner` | In-process greedy game batches and spawned runners with decision-progress timeouts and worker recovery. |
| `in_process` | Synchronous curriculum game slots for batched PPO collection without native RPC. |
| `panel`, `teacher` | Privileged scratch-fight panels and an out-of-combat afterstate teacher. |
| `ooc_trainer`, `combat_trainer`, `distill` | Proximal policy optimization (PPO) for one phase, or supervised distillation with rehearsal; the other phase stays frozen by default. |

| Study-specific components | Purpose |
| --- | --- |
| `joint_sparse`, `critic_training`, `packed_rows` | Binary whole-game PPO, independent critic fitting and its compact choice-buffer format. |
| `fresh_study`, `fresh_native`, `fresh_collection`, `fresh_cohort` | Fresh-start inference/filtering, native exception boundaries, whole-world collection and strict merging. |
| Other `fresh_*` modules | Supervised fitting, unlabelled probes, deferred test scoring, frozen fight grids, comparisons and world-cluster statistics. |
| `controlled_*`, `representation_controls` | Representation-control policies, cached encodings, scoring, comparisons, factorial analysis and orchestration. |

## Run jobs through the queue

Run from the repository root with policy dependencies installed in an
appropriate runtime; never install into a frozen study runtime.
`scripts/rl/build_native_rl.sh` builds the current environment patch plus the
research panel patch, using separate `.venv-native` build tools. Select the
target interpreter with `PYTHON` and a new empty output with `BUILD_DIR`.

The queue runs one job per lane, pauses its process group on battery, resumes
on mains power and marks stalled jobs failed without automatically retrying
them. Its native path overrides the shell environment: put an explicit `env
STS_LIGHTSPEED_BUILD=...` inside the queued command. `BUILD` below means your
verified compatible build directory; `MODEL.pt` means your chosen checkpoint.

```sh
export PYTHONPATH=src
.venv/bin/python scripts/rl/queue_runner.py add ooc-example --lane train -- \
  env STS_LIGHTSPEED_BUILD=BUILD .venv/bin/python scripts/rl/train_ooc.py \
  --checkpoint MODEL.pt --out data/rl_runs/ooc-example \
  --iterations 24 --games 768 --gae-lambda 1.0 --collectors 4 --workers 8
.venv/bin/python scripts/rl/queue_runner.py start
.venv/bin/python scripts/rl/queue_runner.py status
```

`cancel NAME` cancels a pending job or kills its running process group. Queue
logs live under `data/rl_runs/<queue name>/`; trainer artifacts follow
`--out`. Use the same queue wrapper and explicit build selection for all heavy
commands below.

## Batched greedy evaluation

`fast_eval.py` accepts the same champion and optional `--combat` / `--ooc`
replacements as `controlled_eval.py --mode eval`. Each spawned process owns its
policy and several native games. The output directory must be new or empty;
use an artifact directory distinct from the queue's log directory.

```sh
.venv/bin/python scripts/rl/fast_eval.py --checkpoint MODEL.pt \
  --world-start 91001000000 --n 256 --processes 7 --games-per-process 8 \
  --out data/rl_runs/fast-eval/artifacts
.venv/bin/python scripts/rl/benchmark_batched_runner.py --checkpoint MODEL.pt \
  --world-start 91001000000 --n 256 --processes 7 \
  --reference EXISTING_EVAL/summary.json --out data/rl_runs/fast-benchmark/artifacts
```

The evaluator writes per-world records and original outcome totals in
`summary.json`, plus configuration, git commit, native identity and checkpoint
SHA-256s in `run.json`. The run record includes K (`--games-per-process`), the
process count and the round-robin world assignment to each process. Cuts (4,096
decisions), simulator errors and timeouts never count as losses. A simulator
error faults its world and retires the process; each in-flight peer restarts
once. A world in flight during a second process failure becomes a fault. Each
record and the summary include `requeues` counts. The timeout defaults to 120
seconds without a completed decision per process, excluding long gaps when the
parent itself was paused. Partial snapshots on process faults are diagnostic.

Use `run_in_process(seeds, policy, games_per_process=K)` for direct library use;
it provides no native crash or timeout containment. K=1 reproduces the
per-decision evaluator exactly. Larger batches can change near-tied logits
through floating point rounding. Two evaluations are directly comparable only
with identical K and process count, alongside the same policy and ordered world
cohort: splitting worlds across processes changes which games share a forward
pass. Retries can also change batch composition. The benchmark compares every
K=8/32 trajectory with K=1 and records the first differing decision and both
logit gaps for every difference.
Timing includes diagnostic comparisons. It saves each configuration before
starting the next and reports when the seven-process rate falls below 4.5
games per second (1.5 times the approximate existing seven-lane rate).
Measured on 20 October 2026 on Sid's laptop with environment 0.1.5.1 and the
champion: 7 processes of 32 games made 18.8 games per second, against 5.3
for seven per-decision evaluator processes; one process made 3.8.
To finish a time-limited benchmark without repeating its slow baseline, pass
`--skip-existing --single-records PREVIOUS/batched_p1_k1.json` with the same
policy and world arguments and a fresh output directory.

## Training, evaluation and collection

Choose checkpoints, banks and cohorts explicitly instead of inheriting
historical local defaults. These commands show the inner queued job.

```sh
.venv/bin/python scripts/rl/train_combat.py --checkpoint MODEL.pt \
  --bank data/rl_runs/banks/fights.jsonl --out data/rl_runs/combat-example
.venv/bin/python scripts/rl/eval_policy.py --checkpoint MODEL.pt --mode greedy \
  --n 1024 --world-start 5000000000 --workers 8 --collectors 4 \
  --output data/rl_runs/eval-example.json
.venv/bin/python scripts/rl/compare_evals.py BASELINE.json CANDIDATE.json
.venv/bin/python scripts/rl/collect_fight_bank.py --checkpoint MODEL.pt \
  --n 1000 --threads 8 --rooms BOSS,ELITE --output data/rl_runs/banks/fights.jsonl
.venv/bin/python scripts/rl/collect_rule_labels.py --checkpoint MODEL.pt \
  --n 600 --smith-above 0.85 --output data/rl_runs/rule-labels
```

`distill_ooc.py --checkpoint MODEL.pt --records DIRECTORY --out DIRECTORY`
fits collected label shards. The out-of-combat trainer exposes `--credit
macro|micro`, `--combat greedy|sample`, `--lr`, `--eval-every`, `--eval-n`,
`--seed` and `--resume`. Out-of-combat `--gae-lambda` now defaults to 1.0
(previously 0.95), matching the champion's delayed strategic credit recipe.
Both trainers now pass explicit `--weight-decay` (default 0.0); older runs
silently used AdamW's 0.01. These default changes break direct run comparability.

Out-of-combat `--in-process-games` changes only where games run: inside each
collecting process without native RPC, making many more slots per collector
affordable. Both worker kinds batch one policy forward per tick, and the pool
splits `ceil(workers / collectors)` slots per collector (at least one).
For example, `--collectors 12 --workers 384 --in-process-games` runs 32 games
per collector.
With one collector all slots share the trainer process. The subprocess path
remains the default; `--no-in-process-games` selects it explicitly.

In-process simulator errors fail closed: the whole faulted episode is excluded,
its native episode is discarded, and the next reset constructs a fresh episode.
At the worker a programming error propagates unchanged, but `collect_rollouts`
records every exception from a game slot as a fault and excludes that episode,
so the difference is only that the fault record names the real exception type.
No fault packets are written, and there is no per-step or teacher-operation
timeout: a native hang stalls the process, with containment supplied by the job
queue's stall watchdog. In-process mode gives up crash isolation: a native
segfault or abort kills every game in that collector process, and with
`--collectors 1` the trainer itself. The pool restarts a dead collector once and
replays the identical request (same model, seed, tasks and slot count), so a
deterministic native crash recurs and ends the run with "failed twice", where
the subprocess path would have excluded one episode. Train and eval records
count `collector_restarts` during that collection or evaluation (zero without
a collector pool).

`check_parallel_collect.py --checkpoint MODEL.pt --games 64 --eval-n 32`
measured the pooled collection path (`collect_pool`) on 20 October 2026 on
Sid's laptop (15 logical cores), using environment 0.1.5.1 and the champion
checkpoint. For 64 games with `--eval-n 32`, 4 collectors with 8 subprocess
workers achieved 8.6 games/s and 686 kept out-of-combat transitions/s;
4 collectors with 32 in-process slots achieved 11.9 games/s and 989
transitions/s; and 4 collectors with 128 in-process slots achieved 11.7 games/s
and 942 transitions/s.

For 256 games with `--eval-n 16`, 4 collectors with 8 subprocess workers
achieved 8.6 games/s and 713 kept out-of-combat transitions/s; 12 collectors
with 24 subprocess workers achieved 13.1 games/s and 1,080 transitions/s;
12 collectors with 384 in-process slots (32 per collector) achieved
27.5 games/s and 2,362 transitions/s; and 12 collectors with 96 in-process
slots (8 per collector) achieved 24.1 games/s and 2,054 transitions/s.

With few slots the in-process path is not faster: the exactness test's 8 games
on 4 slots took 1.9 s with subprocess workers and 2.1 s in-process. The gain
comes from running one process per core with many slots each, instead of two
or three processes per collector.

Out-of-combat hygiene options are `--critic-gradient shared|detached`
(default shared), `--value-weight` (0.5), `--normalize-returns` (off), and
`--strategic-entropy` (0.0). Detached mode trains only the critic head from
value loss. Normalization divides targets by the running population standard
deviation of raw OOC choice returns, without centering; collection rescales
OOC baselines to reward units. Statistics persist in trainer state in both
resume and named checkpoints; legacy checkpoints start with scale 1. A
normalized checkpoint requires normalization when continuing training.
Strategic entropy adds to `--entropy` only on rest, shop, card reward, map
and card-removal menus. Updates already log optimizer minibatch steps as
`steps` and KL guard stopping as `early_stop` (including zero-step updates).

`train_ooc.py --combat-follow PATH` checks a checkpoint before each iteration's
collection and adopts its combat head when the file's SHA-256 changes. The file
may hold a composite or a bare phase head; every combat tensor must match the
model's shape and dtype. Its metadata's `encoder_version` must match the starting
checkpoint's, and its combat `model_config` must match that checkpoint's combat
configuration. Bare heads may omit `model_config`; those still undergo encoder,
tensor shape and dtype checks. Publish updates by writing a temporary file and
renaming it. This option cannot be combined with `--train-combat`.
Enabled runs log `combat_sha256` and `combat_hash_kind` on train and evaluation
records (also in evaluation JSON files), plus `combat_swapped` on train records
and `combat_swap` events. `eval_0000` measures the starting head before the first
follow check. Fresh starts ignore inherited combat identity: the initial hash
covers sorted combat tensor names, dtypes, shapes and bytes (`weights`); after
adoption it covers the exact checkpoint bytes (`file`). The hash, kind, source
path and a separate `combat_weights_sha256` fingerprint persist in trainer state;
named checkpoint metadata also records the hash and kind. Resume restores that
head before checking the followed file, trusting a saved file hash only from
`latest.pt` and only if its fingerprint matches the loaded weights. Otherwise it
uses a weights hash until adoption. Frozen combat parameters are excluded from
AdamW, and swaps copy values without replacing parameters or optimizer state.
Without this option, the existing training log schema is unchanged.

Combat training uses `--fights`, `--mix`,
`--holdout-fraction`, `--hp-weight`, `--shaping-hp` and `--shaping-enemy`;
these change the objective or cohort. Resume uses `--out`'s `latest.pt`;
branching requires copying it into a new output directory and passing
`--resume`. The configured learning rate and weight decay override saved
optimizer values in both trainers.

Greedy evaluation uses exact logit argmax, including deterministic tie
handling. `--baseline FILE` adds a paired comparison. `compare_evals.py`
reports candidate minus baseline, a paired floor difference with a 95%
interval, and an exact McNemar test of discordant wins. Intersection pairing
can omit faulted/absent worlds: inspect counts and require the intended full
cohort before claims. Default evaluation worlds are training-split worlds, not
untouched final data. See [evaluation
rules](../sts_policy/README.md#evaluation).

## Fight panels and fresh-start tools

Use `panel.PanelWorker` for privileged arbitrary-encounter deck diagnostics;
it requires the research native build. `scripts/rl/panel_validate.py --probe
FILE --checkpoint MODEL.pt --output FILE` relates panel scores to later boss
outcomes. Scratch fights suppress post-battle callbacks. Native search votes
retain hidden random streams; `search_votes_public` and `search_details_public` resample them.
Fresh-start panels instead replay stored fight entrances and use a frozen
grid:

```sh
.venv/bin/python scripts/rl/fresh_panel.py --bank BANK_DIRECTORY \
  --freeze-grid-only --resamples 8 --out data/rl_runs/grid.json
.venv/bin/python scripts/rl/fresh_panel.py --bank BANK_DIRECTORY \
  --grid data/rl_runs/grid.json --resamples 8 --checkpoint CHAMPION.pt \
  --combat STUDENT.pt --out data/rl_runs/panel-example
```

Freeze before fitting or scoring students. Banks must be completed validated
collection directories; legacy standalone rows reject. The grid chooses one
start per source world per target group independently of outcomes. Reuse exact
bank paths, targets and resample count with `--grid`. Salt -1 is the unsalted
control. Other salts vary battle initialization streams; misc and potion
streams stay fixed. Panel health-point loss is salted entry health minus exit
health before end-of-fight relic healing, with death set to zero; bank
baseline health includes healing. Cluster starts/salts by source world and
weight worlds equally. Incomplete grids are diagnostic only; coverage
shortfalls and missing results are separate checks.

`fresh_collect.py --mode collect|eval --checkpoint CHAMPION.pt --world-start N
--n K --out DIRECTORY` collects labels or evaluates exact seed ranges.
`--bank` saves fight starts; `--combat`/`--ooc` replace heads. Teacher
settings use `--samples`, `--simulations` and `--teacher-probabilities`;
`--max-labels` stops between worlds. Combat labels use the canonical native
search protocol `canonical_rng_resampled_native_v2`, which is not certified
public-only. Inference recovers visible potion slots from raw public menus
before hiding out-of-combat drink/discard choices, and leaves combat intact.
Singleton processor inference uses actor-only exact argmax; forced choices
bypass the model. Load errors never trigger champion fallback. `fresh_merge.py
--inputs A B --world-start N --n K` validates cohorts; `--out` publishes a
merge. Faults/cuts quarantine whole worlds.

`--vote-rule best_line|most_visited|mean_value|pooled_mean` compares rules over
the same searches (default `best_line`); `--record-search` saves per-search
choices and raw root-edge diagnostics, including native action bits. Alternative
rules merge edges with the same displayed action index before scoring them.
`--teacher-plays` lets selected combat labels
choose actions for benchmarks, requires labelled collection, and marks played
samples `use_for_training: false`, with `source_action` recording the champion's
choice and `played_action` the action stepped. Teacher-played cohorts are refused
by fitting, comparison and fight banks; fitting also rejects these marked samples.
The cohort validator and comparison reader permit explicit analysis with the
keyword `allow_teacher_plays=True`. Collection forbids `--bank` with teacher play.
A boss benchmark needs
`--teacher-plays --teacher-probabilities '{"BOSS1":1,"BOSS2":1,"BOSS3":1}' --self-consistency-every 0`;
the default probabilities sample other rooms too. Self-consistency repeats always
use `best_line` votes, regardless of `--vote-rule`. Defaults preserve the existing
search and trajectory behavior.

`--search-options JSON` (config key `search_options`) configures native searches
through the details path and records the selected options on each teacher sample.
Use a flat dict or group dicts, for example
`'{"default":{},"BOSS1":{"evaluation":1,"hp_weight":0},"BOSS2":{"evaluation":1,"hp_weight":0}}'`;
group overrides replace the default dict, and missing groups fall back to `default`,
then no options. `evaluation` defaults to 0 (the existing score); 1 uses `win_first`:
wins score `1 + 0.1 * (hp_weight * HP + potion_weight * potions) / max(1, max_HP)`
and losses score `0.1 * loss_credit * (1 - remaining_enemy_HP_ratio)`.
`exploration` sets the UCT exploration coefficient (default `3 * sqrt(2)`);
`hp_weight`, `potion_weight` and `loss_credit` default to 1, 4 and 1 and affect
evaluation 1 only. Defaults reproduce today's teacher. For bosses, use
`evaluation: 1, hp_weight: 0` to ignore surviving HP; smaller exploration focuses
visits more strongly. Keep `loss_credit < 10` to ensure every win outranks a loss.
Options apply to all queries including self-consistency repeats and reset after
each details call; older native modules support only calls without options.

`fresh_train.py` requires `--data`, `--probe`, `--architecture`, `--phase`,
`--out` and `--deadline-utc`. All models initialize from scratch; the old
control defaults to large/no menu context. `--batch`, `--lr`, `--steps` and
`--patience` control fitting. Combat `--fit-size 5000|20000|full` uses nested
prefixes shared across encoders; out-of-combat fitting is full-only.
`--mixed-sources` opts into fitting multiple collecting champions and Python
source hashes; native identities, collection settings, environment/observation
versions, Python runtime versions and public policy contracts must still match.
Adding `--mixed-sampling` allows different `teacher_probabilities` collection
profiles while every other setting, including teacher samples and simulations,
still matches. The manifest records the flag and distinct profiles as canonical
JSON strings in `teacher_probability_profiles`. Both options default off.
Abstentions are excluded from targets; forced choices stay in evaluation but
not optimization. Fixed unlabelled probes never supply targets or select
weights. Use `--defer-test` until validation-only selection freezes. The
selection file binds checkpoint/manifest file identities and the criterion:
minimum validation soft cross-entropy, with lower learning rate breaking ties.
Then run `fresh_score.py --selection FILE --out DIRECTORY --deadline-utc
TIME`.

`fresh_compare.py games|labels` requires exact `--cohort` and frozen
`--expectations`. `fresh_panel_compare.py --new A --old B --grid FILE
--expectations FILE --out FILE` directly pairs students and requires identical
champion controls. `fresh_gate_report_data.py --check-ready` inventories
inputs; `--out DIRECTORY` exports completed frozen results in the frozen study
runtime. `plot_fresh_gate.py --data FILE --out DIRECTORY` renders separately.
Both output directories must be new; neither tool fits or selects models.

## Speed and memory

Bound memory with out-of-combat `--collect-chunk` and `--micro-batch`;
`evaluate(chunk=256)` reduces chunks before releasing trajectories.
`--collectors` parallelizes encoding/inference across persistent processes.
See [policy caching and replay](../sts_policy/README.md#collection-and-replay)
for worker caches, prefix acceleration and encoding reuse.

The joint JSON config enables compact collection with `collection_backend:
"compact"` and actor-only evaluation with `evaluation_backend: "lean"`; both
default to `"legacy"`. `configs/joint_sparse_win_v2_fast.json` demonstrates
eight training collectors, 24 total training workers and eight evaluation
collectors with one worker each. Launch with `train_joint_sparse.py --config
FILE --checkpoint MODEL.pt --out DIRECTORY`. Adapt the reference config's
study inputs and cohorts deliberately. Compact buffers retain public choice
features and discard observation histories; faulted/censored worlds stay
excluded. Lean evaluation skips critic, likelihood and advantage work while
retaining action/outcome records. Changing layout changes training
random-number assignment and can change trajectories; it is a recorded
execution condition.

## Records, replay and dashboards

Launch records preserve settings, command, source commit/archive, runtime,
actual native binary/manifest, parent checkpoint and input identities.
Trainers save compressed per-iteration task/prefix/outcome records, optimizer
state and random-number-generator state, rather than all raw training tensors.
Inherited metadata stays under `historical_parent_metadata`; current launch
identity is separate. Legacy resumes report when random state is unavailable.
Fault packets retain successful calls, preview markers, periodic public-state
checks, last state, failed call, bounds, native identity and exception
details. Faults go under the launch record, or `data/simulator_faults` for
direct workers.

```sh
.venv/bin/python scripts/rl/replay_fault.py RECORD.json --timeout 10 \
  --seconds 120 --output data/fault_replays/new-case.json
```

Replay checks recorded public states and teacher results before retrying the
failed operation. It distinguishes reproduced/different/earlier failures,
divergence, budget exhaustion and success past the failed call. Identity
changes reject by default; candidate testing requires `--build-dir`,
`--allow-native-change` and, for a candidate core, `--environment-source`.

- Generic replay supports only `CurriculumWorker` and `PanelWorker` records;
  fresh-panel and lookahead worker classes reject. Replay establishes
  agreement at recorded checks, not complete hidden-state equality or
  historical retraining across platforms. A repeated timeout does not
  establish a deterministic native defect.
- Programming and evidence-writing errors must propagate. Simulator faults,
  undefined behavior, timeouts and cuts stay separate from defeats/targets;
  older helper datasets can contain partial faulted-world rows.
- Joint-controller resume omits rolling critic-validation history, so health
  warnings and stop decisions can differ after resume.

`scripts/rl/dashboard.py start|restart` serves queue/run metrics at
http://127.0.0.1:8791. `fresh_dashboard.py --study-dir DIRECTORY --port 8379`
shows fitting and paired reports. `scripts/long_rl_dashboard.py --campaign
DIRECTORY --port 8786` serves read-only long-curriculum telemetry and bounded
episode samples; missing measurements stay unavailable.
`scripts/monitor_full_run_eval.py` plans a separate development observer;
`--execute` starts it and `--once --max-panels 1` bounds it. It measures
selected and latest snapshots without changing training or selection. Reused
worlds and snapshots remain repeated monitoring, not independent final
evidence.
