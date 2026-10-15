# Joint training collection and evaluation speedups

The combined changes reduce the bounded controller fixture from **263.34 to
162.34 seconds**, a **1.62× speedup** including initialization and shutdown.
The joint sparse-win adapter now has opt-in lean evaluation, lossless compact
choice buffers, and a measured collector layout for the 15-core, 48 GiB Mac.
The [fast reference configuration](../configs/joint_sparse_win_v2_fast.json)
changes only execution settings from the completed v2 study. The scientific
recipe, native binary, observation surface, reward, actor/critic updates,
evaluation cadence, and fault/retention guards stay unchanged.

## Implementation

**B, lean evaluation:** `evaluation_backend="lean"` uses the actor logits without
legacy or independent critic forwards, categorical likelihood calculation, GAE,
or retained training trajectories. It still encodes the same policy inputs and
records complete action prefixes, action hashes, outcomes, decision/boss counts,
public endpoints and fault references. Evaluation keeps one simulator worker per
collector, preserving inference batch shape and exact argmax tie-breaking.

**C, compact storage:** `collection_backend="compact"` consumes each temporary
transition immediately. Per-world numeric choice rows remain quarantinable until
a genuine terminal win/loss; censored and faulted worlds contribute no training
rows. Full observation histories are discarded. `PackedRows` stores float32
state/candidate features, float64 likelihood/value/return/advantage scalars,
integer indices/seeds, ragged action offsets, and the existing metadata. Its
Sequence interface supports the current actor, critic and analysis consumers.

Tensor storage supplies shared-memory transport between collectors and owner,
and streaming storage through `torch.save`; this avoids large NumPy pickle
byte-string temporaries. Saved rehearsal anchors copy only selected features so
views cannot pin an entire packed collection. Atomic gzip writes are retained.
The earlier array-only prototype and its measurements remain separately named
under `data/rl_runs/speed_bench_*`.

**A, parallelism:** the selected layout uses **8 training collectors and 24
simulator workers**, three per collector; evaluation uses **8 collectors with
one simulator worker each**. The sweep covered training layouts 2/6, 4/8, 6/12,
8/16, 4/12, 6/18 and 8/24 (collectors/total workers). The request ledger records
actual worker allocations and seed assignments. Each collector uses one Torch
thread; the learner retains two. There is at least one worker per collector.
Training and evaluation pools are not kept active together.

The legacy paths remain the defaults for existing configurations. The fast JSON
adds the two backend settings and changes collectors, workers and evaluation
collectors; all other values match `joint_sparse_win_v2.json`. It names the
original study inputs and cohorts, so future studies should choose their inputs
and cohort identifiers deliberately.

## Correctness

The baseline is commit `e9fc6a9`, archived into an independent import tree. Both
sources use the same frozen parent, warmed critic inputs, Python/Torch/NumPy
runtime and canonical native binary SHA256
`936734a2289e20971abdbe7ae2f703d2d82f4e357a4e2b8590817d9a96f81a04`.
Candidate changes were measured before their final commit; immutable launch
source archives identify each tested working tree. The evidence also checks
that every final tensor-path core matches the current source AST, ignoring
comment-only differences.

At a fixed collector layout, legacy and compact collection match every episode
receipt and every public feature, action, log probability, value, return and
advantage hash on the 64-, 256- and 768-world panels. The 768-world panel has
262,199 choice rows. With retained rows placed in the same stable world order,
actor and critic weights and both optimizer states match bit-for-bit after a
complete native optimizer pass. Fresh processes reproduce both saved list and
packed buffers exactly. Unit tests additionally check two
successive optimizer steps and moments, rollback/RNG restoration, fault/cut
quarantine, unsupported schemas, anchor ownership and discarded observations.

Greedy evaluation equivalence is checked by per-world receipt/action hashes at
fixed inference batch shape. Collector counts change stochastic RNG assignment
and can change floating-point batch shapes during training and anchor
collection. Optimized results merge in collector-index order to remove owner
arrival-order variability; the legacy path keeps its former completion order.
These are explicitly recorded execution conditions, not a promise to recreate a
historical training trajectory under a different layout.

The required `STS_REQUIRE_SIMULATOR=1` suite ran **538 tests with three skips**;
the mandatory bounded random-agent native smoke also passed. All completed timing
cohorts have zero simulator faults or censored games. The timing cohorts use
TRAIN-split worlds and do not open final evaluation data.

## Production batch measurements

The following measurements use actual 768-game collections with the original
4,096-row actor minibatch, 128-row microbatch, and 256-state critic optimizer
minibatches. Each full-size condition is a single bounded pass; the smaller
256-world layout sweep uses three repetitions. Ranges in the evidence are
observed timings on an interactive host, not statistical confidence intervals.

| Measurement | Legacy 2 collectors 6 workers | Compact 2 collectors 6 workers | Compact 8 collectors 24 workers |
| --- | ---: | ---: | ---: |
| Collection seconds | 151.87 | 140.59 | 68.79 |
| Choice rows | 262,199 | 262,199 | 265,251 |
| Exact buffer write seconds | 17.40 | 10.44 | 10.58 |
| Exact buffer size MB | 96.05 | 64.19 | 64.98 |
| Likelihood reconstruction seconds | 13.03 | 14.01 | 14.30 |
| Actor and critic update seconds | 65.89 | 67.95 | 68.13 |
| Peak owner RSS GiB | 8.51 | 2.13 | 2.17 |

The fixed-layout compact path trades slightly slower row access during updates
for much lower memory use and faster persistence. More parallel collectors give
the main collection speedup. Their different training RNG assignment produces
slightly different trajectory lengths; decision throughput is recorded alongside
games per second to keep this comparison interpretable. RSS is the owner-process
high-water mark, not aggregate physical memory across the process tree. The
legacy encoding-byte counters describe encoding work in compact mode; use the
packed numeric-byte metric and RSS for storage/memory comparisons.

## Evaluation and controller measurements

Full 1,024-world greedy evaluation falls from **186.43 to 113.45 seconds** with
lean evaluation and eight collectors, a **1.64× speedup**. Every episode receipt,
including the complete action trace, matches exactly; both executions have zero
faults/cuts. The original and optimized measurements use the same timing cohort
and frozen policy.

The complete controller fixture performs two accepted actor/critic rounds,
**512 training games**, initial and periodic 128-world evaluation, 128-world
rehearsal collection, 64-world fresh critic validation, all buffer/checkpoint
writes and resource/learning guards. It uses separate TRAIN-only namespaces,
unchanged learning coefficients and a two-iteration stop. Both fixtures run
through the power-aware queue. External command timing includes process startup
and shutdown: **263.34 versus 162.34 seconds**, **1.62×**. The fast fixture performs
46 actor optimizer steps versus 45 because its changed RNG assignment produces
more choice rows. These are test checkpoints, not selected study models.

Reported CPU time rises from **702.41 to 1,001.37 CPU-seconds**: the reduced wall
time uses more CPU work and parallelism. For the full-size measurements,
amortizing the 1,024-world evaluation across two 768-game updates gives
**341.40 versus 218.53 seconds/update**, or **1.56×** before pool startup/shutdown.
That is a calculation from separately measured components, not another training
run. The independently timed controller fixture supplies the direct end-to-end
check. These are performance measurements on the specified policy and host;
they do not establish improved game-playing ability.

## Reproduction and compatibility

[Machine-readable evidence](joint_training_speedups_evidence.json) records
measurements, commands, source/runtime/native identities, artifact hashes and
exactness checks. Raw source snapshots, buffers and receipts remain local under
`data/rl_runs/speed_*`; they are not included in Git. The summary is reproducible
with `analysis/training_speedups_20261015.py` after those artifacts exist.

Run heavy checks through `scripts/rl/queue_runner.py`. The benchmark entrypoint is
`scripts/rl/benchmark_joint_collection.py`: use a fresh `--out`, matching
`--checkpoint`/`--critic-init`, explicit native build, `--kind train` or `eval`,
collector/worker counts, and the desired backend flags. `--source-root` selects
the archived baseline. `--persist` measures gzip level 6 buffer writes;
`--update --repeats 1` additionally measures likelihood checks and both optimizer
updates. The numerical comparison uses stable world order through a Sequence
adapter, preserving each source's row-access costs. Its order normalization is
for format comparison, not a replacement for the production collector order.
A `STOP` file in a benchmark output directory stops it between repetitions.

`scripts/rl/replay_speedup_buffer.py` repeats the optimizer comparison in a fresh
process from the saved buffer. The new internal row format is
`joint_choice_columns_v1`; old list-of-dict buffers remain supported. New packed
buffers require this adapter source. No core observation, CLI trace, encoder or
native release version changes. Existing source/config/native resume checks
continue to reject incompatible resumes; historical artifacts remain intact.

## Existing controller recovery limit

The inherited controller checkpoints model, optimizer, RNG and adaptive guard
counters, but not the rolling `critic_history` validation window. Resume starts
that window empty, so its first few health warnings and potentially health-stop
decisions can differ from uninterrupted execution. Numerical buffer replay does
not certify equivalence of that controller history. This limit is independent of
the collection/storage changes and remains in the current controller.
