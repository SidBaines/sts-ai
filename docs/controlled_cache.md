# Optional controlled preencoding cache

This is an operational optimization for the approved representation diagnosis.
It stores exactly the existing
`preencode(sources, manifest, 'combat', 'controlled', 'full', splits=('FIT','VALIDATION'))`
result. One cache serves all controlled widths, feature flags, learning rates and
training seeds. The controlled encoder already computes legacy action features
in both feature conditions. Encoding, targets, arithmetic, buckets, optimization
and selection are unchanged. No model, encoder, vocabulary or glossary was edited.

The cache is opt-in. `fresh_train.py --preencoded-cache RECEIPT.json` requires
`--architecture controlled --phase combat --fit-size full --defer-test` and the
original `--manifest`. Other architectures, OOC, FIT prefixes and nondeferred
TEST are rejected before output creation. The argument uses
`default=argparse.SUPPRESS`; absent flags add no config key or cache provenance.
The original training path remains the default.

## Root commands, after integration and review

These commands are instructions only; no real cache, fit or study run was
started during implementation. Set `REPO`, `FROZEN_PYTHON` (the approved frozen
`.venv/bin/python`), `STS_LIGHTSPEED_BUILD` (the approved native build), `DATA`,
`MANIFEST`, `CACHE_OUT`, `CACHE_JOB`, `LANE` and `DEADLINE_UTC` to the approved
paths and deadline. Repeat source arguments after `--data` for multiple inputs.

Queue creation using the existing runner:

```sh
PYTHONDONTWRITEBYTECODE=1 "$FROZEN_PYTHON" "$REPO/scripts/rl/queue_runner.py" \
  add "$CACHE_JOB" --lane "$LANE" -- \
  /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  STS_LIGHTSPEED_BUILD="$STS_LIGHTSPEED_BUILD" \
  "$FROZEN_PYTHON" "$REPO/scripts/rl/fresh_cache.py" \
  --data "$DATA" --manifest "$MANIFEST" --out "$CACHE_OUT" \
  --deadline-utc "$DEADLINE_UTC"
```

`CACHE_OUT` must not exist, and must differ from the runner's own
`data/rl_runs/$CACHE_JOB` directory (which the runner creates before launching).
The creator accepts only these four required options; it has no count, phase,
split, architecture, fitting, native-policy or teacher-generation switch. It
requires a `.venv` interpreter, disabled bytecode writes and an explicit native
build from the environment. It uses the existing `TrainingControl` for AC,
`CACHE_OUT/STOP`, UTC deadline and the 6 GiB disk floor, including streaming I/O
and validation boundaries. Root queues this work; this helper does not queue it.

After successful creation, append this exact option to every approved controlled
fit command, leaving all other scientific arguments intact:

```sh
--preencoded-cache "$CACHE_OUT/receipt.json"
```

Root must re-freeze the complete pure commands before the main fits. Keep the
technical uncached repeat separate and excluded from the scientific matrix.
That repeat omits the option. No controller was changed here.

## Format and validation

The receipt schema is `controlled_preencoded_cache_v1`. Its keys are:

| Keys | Meaning |
| --- | --- |
| `schema`, `row_contract`, `encoding` | Fixed controlled/combat/full FIT+VALIDATION contract |
| `dataset` | Exact absolute path, byte count and SHA-256 of `dataset.pkl` |
| `manifest`, `data_identity`, `source_files` | Original full manifest file ID, original data identity and original raw-source file IDs |
| `logic_sources`, `runtime` | Exact encoder, trainer/Example, cache helper/CLIs, filter, identity helper, vocabulary and glossary source identities; Python/platform/NumPy/PyTorch runtime |
| `row_counts`, `row_ids` | Exact counts and original per-split `preencode` list order |
| `alignment_sha256`, `array_bytes` | Per-split original row/action-index/target-byte binding and total NumPy array bytes |
| `launch` | Creation `begin_run` identity, binding the original manifest/raw inputs and matching source/native snapshots |
| `timings` | Manifest validation, raw preencoding, streaming write/hash times and size estimates |

`dataset.pkl` is protocol-5 pickle of the existing `dict[str, list[Example]]`
dataclasses and NumPy arrays. No conversion, quantization, relabeling, sorting or
shuffling is performed. It contains exactly `FIT` then `VALIDATION`; TEST is not
encoded or serialized. Only trusted, self-created caches are supported: pickle
is executable serialization, and hashes are integrity checks, not signatures.

Production always requires exactly **113,042 FIT and 17,989 VALIDATION** combat
targets, with the complete original full FIT subset. Missing, empty, extra,
duplicate, TEST and wrong-phase rows fail. Each example's full metadata and
target digest must match the canonical manifest. Dataset order must match the
receipt's original encoding order, which differs from the manifest's sorted
metadata and from its FIT subset ordering. Original action indices and target
bytes must also match the alignment digest. The existing unpadded encoder ABI
is checked for exact classes, fields, array dtypes/shapes, finite numbers, masks,
token/type/reference ranges, target mass and action/target/coverage alignment.
Load performs no corpus reencoding or new raw-label parsing.

Before unpickling, the loader verifies the cache hash, receipt and original
manifest identity, raw file hashes, source/runtime identities and launch source
archive/native snapshot. It checks identities again after loading and immediately
before fitting. Invalid, missing, stale or changed inputs raise; there is no
rebuild or fallback. Small synthetic contracts exist only on internal unit-test
helpers and pure validators; production entry points and the CLI cannot load
them. Loading does not seed, sample or restore any RNG.

The existing training `freeze_manifest`, source-receipt, original manifest
comparison, probe and strict native checks still run. Only the subsequent
FIT/VALIDATION preencode call is substituted. These existing raw validators, and
the creator's unchanged `preencode` source iterator, scan all shards including
TEST; only selected FIT/VALIDATION observations reach the encoder. The cache
helper adds no TEST raw reader. A requirement to skip even that existing raw
scan would require a separate change to the mandated existing call contracts.

Training `data_identity` and `label_source_files` retain their original meaning.
The effective config adds `preencoded_cache_provenance`; the source receipt
adds `preencoded_cache`. Both contain the original manifest, receipt and dataset
IDs. `begin_run.data_inputs` also includes those actual files, hashing the shared
cache without copying it per fit. `cache_usage.json` records load timings and
sizes. Checkpoint metadata carries the same effective config/source receipt.
`controlled_scoring.frozen_selection` and `controlled_factorial._selection` use
the existing scientific condition and source bindings and accept these additional
fields; neither requires an exact training input inventory. Model source hashes
and scientific condition extraction are unchanged.

## Publication, memory and measurements

The original manifest JSON is parsed once by the creator and released before
`freeze_manifest` constructs its canonical manifest. The dataset is encoded once
and streamed directly to `.dataset.pending`, without a corpus-sized serialized
buffer or a second deserialized encoded set. Files are flushed/fsynced, then
published using exclusive hard links and made read-only. `receipt.json` is
published last and is the success marker. No existing directory or historical
cache is replaced. Interrupted/failed output is preserved for diagnosis and is
not resumable; use a new directory for a new attempt. A pending file or an orphan
dataset without a receipt is not a usable cache.

`summary.json` and stdout include the creation timings, serialized byte count and
array-byte inventory. `estimated_sequential_read_seconds` is the measured cache
hash-read time, a warm-filesystem sequential-read estimate only; it excludes
unpickling, validation and raw-input checks. Array bytes exclude Python objects,
manifest metadata and pickle memo overhead. It is not a measured load/RSS claim.
Actual opt-in fits emit separate pre-load verification, deserialization and
validation/recheck times in `cache_usage.json`; existing status reports retain
process-tree RSS and peak RSS. Root must measure actual speed and memory benefit.

## Verification and remaining work

Run the synthetic tests with the frozen runtime and no bytecode writes:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest \
  tests.unit.test_controlled_cache
```

They cover byte-exact arrays/targets/metadata/order, RNG preservation, identical
collation/minibatch order/initial weights, production count enforcement, schema
and provenance rejection, source/cache tampering, post-load rechecks, interrupted
and exclusive writes, unchanged CLI defaults, and the opt-in training hook. The
creator orchestration test mocks heavy boundaries and stops before publication.
No optimizer update or real native execution is needed.

Still for root: independent SPEC then QUALITY review, the queued actual-data
creation/count/order/schema/RSS validation, measured load benefit, and the
separate actual uncached-versus-cached same-seed training-curve comparison before
main fits. No full-corpus validation, actual cache creation, fitting, confirmation
or outcome inspection was performed during this implementation.

## Boundaries of this study cache

The actual frozen corpus completed creation and full independent raw-versus-cache
byte/order/target/metadata/RNG verification. A static review identified a latent
writer limitation for a single protocol-5 PickleBuffer of at least64KiB:
`_ControlledStream.write` assumes `len(data)` is available. That case would raise
instead of corrupting data; it was absent from this successfully serialized
corpus. Before using this helper for a different corpus or larger menus, change
the writer to a byte-cast memoryview using `view.nbytes`, add a large NumPy-buffer
regression test, and create a new source-bound cache. This study uses the
already verified immutable cache.
