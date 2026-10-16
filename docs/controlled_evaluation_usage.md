# Controlled representation evaluation and comparison

This adds inference and development comparison tooling. Root owns selection,
scientific design, and interpretation. Exposed TEST is development data; these
tools provide no fresh confirmation or new gate/margin verdict. They do not fit
or resume models. The model builder's controlled modules and training helper
extensions must be integrated before actual controlled scoring.

`ControlledStudyPolicy` inherits the original `StudyPolicy` filtering,
distribution, and first-menu argmax. It loads the original champion unchanged,
then validates controlled or original phase replacements. Controlled inference
uses one already-filtered public observation, its preserved menu indices, CPU,
and `actor_logits(dict)`. Its identity retains the exact `fresh_start_v1` schema
and absolute checkpoint paths, sizes, and SHA256 values. `fresh_study.py` and
`fresh_collection.py` remain unchanged.

Use the frozen `.venv`; no installation is needed. Example commands below are
templates; root supplies the frozen inputs, cohort, and deadline.

```sh
.venv/bin/python scripts/rl/controlled_eval.py --mode eval \
  --checkpoint CHAMPION --combat COMBAT --ooc OOC --out NEW_DIRECTORY \
  --world-start DEV_START --n N --deadline-utc UTC_DEADLINE
```

Omit either replacement to retain that champion phase. This wraps the existing
collection parser and `run_collection(config, policy)`, including AC/STOP/disk
controls, finite worker RPCs, protected receipts, and completed-cohort validation.
Games are restricted to `[91_000_000_000, 91_200_000_000)`. Interrupted or faulty
runs remain incomplete. Fight panels use the normal `scripts/rl/fresh_panel.py`
CLI and frozen `--grid`: both the panel parent and native-child policy cache
dispatch controlled checkpoints through the same factory. Replay, salting,
healing, and quarantine behavior are unchanged.

Score a frozen validation selection with the normal scorer:

```sh
.venv/bin/python scripts/rl/fresh_score.py --selection SELECTION_JSON \
  --out NEW_SCORE_DIRECTORY --deadline-utc UTC_DEADLINE
```

The selection contains `selected_checkpoint` and `manifest` file identities,
plus `selection_criterion` exactly
`{"metric":"validation_soft_ce","direction":"min","tie_break":"lower_lr"}`.
Controlled selections also require `architecture: "controlled"` and
`training_condition` equal to the model builder's
`fresh_training.checkpoint_training_condition(payload)`, including the boolean
`legacy_action_features`. The scorer checks the actual model/source/vocabulary
metadata, from-scratch/no-parent/deferred state, best validation step, manifest,
training source receipt, and label file identities before TEST encoding. It
calls only the builder's TEST preencoder and singleton scoring helper. Old/new
selections retain the original scoring implementation. New controlled summaries
also pin the resulting `test_rows.jsonl` file identity.

Generic comparisons use a separate schema; the original fresh old/new
expectations are not relaxed. Build JSON bindings using
`recording.file_identity(path)` and actual checkpoint metadata, then freeze them
externally before analysis. A checkpoint specification is:

```python
spec = {
    "checkpoint": file_identity(checkpoint_path),
    "architecture": payload["metadata"]["config"]["architecture"],  # old/new/controlled
    "condition": checkpoint_training_condition(payload),
}
```

Every expectations file has `schema: "controlled_comparison_expectations_v1"`,
`version: 1`, and `mode` matching the command. Its remaining fields are:

| Mode | Required frozen fields |
| --- | --- |
| labels | `left: spec`, `right: spec`, `manifest: file_identity(manifest)`, `cohort: file_identity(exact_row_ID_list_JSON)` |
| panel | `left: policy_spec`, `right: policy_spec`, `grid: file_identity(frozen_grid)` |
| games | `left: policy_spec`, `right: policy_spec`, `cohort: file_identity(exact_seed_list_JSON)` |

`policy_spec` is `{"policy": full_StudyPolicy_identity, "replacements": {...}}`.
`replacements` maps every non-null `combat`/`ooc` replacement to its checkpoint
specification; a champion arm has an empty map. Panels require combat-only
replacements in both arms. Policies share the same original champion and public
inference contract. Matched phase heads share raw data identity, phase, and FIT
size; their other frozen conditions remain explicit and may differ.

```sh
.venv/bin/python scripts/rl/controlled_compare.py labels \
  --left LEFT_SCORE_DIRECTORY --right RIGHT_SCORE_DIRECTORY \
  --cohort ROW_IDS_JSON --expectations EXPECTATIONS_JSON --out NEW_REPORT_JSON

.venv/bin/python scripts/rl/controlled_compare.py panel \
  --left LEFT_PANEL_DIRECTORY --right RIGHT_PANEL_DIRECTORY \
  --grid FROZEN_GRID --expectations EXPECTATIONS_JSON --out NEW_REPORT_JSON

.venv/bin/python scripts/rl/controlled_compare.py games \
  --left LEFT_COMPLETE_COHORT --right RIGHT_COMPLETE_COHORT \
  --cohort SEEDS_JSON --expectations EXPECTATIONS_JSON --out NEW_REPORT_JSON
```

Labels require completed inference-only scoring receipts, actual frozen
validation selections, matching manifests/sources/targets/split and exact full
row sets. Agreement and CE differences are LEFT minus RIGHT, with high-confidence
choice and act/room groups. Labels remain label weighted; resampling retains
every row in each source-world block.

Panels reuse the strict existing grid/run/provenance reader and require all
unsalted and eight salted cells, no faults/cuts, identical starting states/HP,
and exactly matching champion control results. Intervals compare the actual two
students directly, with equal source-world weighting. BOSS2 resampled metrics
remain primary for full fits; pooled/other/unsalted metrics are descriptive.
Games require both full validated cohorts, both frozen policies, and the exact
development seed list. Controlled versus champion can also use the existing
strict `fresh_compare.py games` interface.

All production intervals use the existing 10,000 whole-world bootstrap draws,
seed 1919. Reports bind the expectations and source files and are written
exclusively; missing, partial, changed, swapped, or mismatched inputs fail.

Synthetic verification (no fitting or native execution):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest \
  tests.unit.test_controlled_evaluation tests.integration.test_fresh_policy \
  tests.unit.test_fresh_statistics tests.unit.test_fresh_panel_comparison \
  tests.unit.test_fresh_panel_grid tests.unit.test_fresh_collection \
  tests.unit.test_fresh_cohort -q
```

Unit tests use explicit synthetic builder stubs. Real-module tests with synthetic
checkpoints/examples are in `tests.integration.test_controlled_evaluation`.
Before integration, set `CONTROLLED_MODEL_SOURCE` to root's approved read-only
builder `src` directory when running that test module. It writes only temporary
fixtures and never fits or executes native code. Root still needs fresh-process
determinism/replay and bounded native parent/spawn-child validation after
integration, before interpreting any study output.
