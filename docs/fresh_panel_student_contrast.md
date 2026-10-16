# G1 combat-student contrast

`scripts/rl/fresh_panel_compare.py` compares **new combat student minus old
combat student**, directly paired on the same frozen source world/start/salt.
Each input is a completed C6 panel directory containing both replacement and
champion results. It performs no fitting, inference, or native execution.

Root must freeze the expectations file before either panel run, using the
selected checkpoints. Its fields are:

```json
{
  "schema": "fresh_panel_student_contrast_expectations_v1",
  "version": 1,
  "grid": {"path": "/absolute/frozen-grid.json", "bytes": 123, "sha256": "..."},
  "new_policy": {"...": "full StudyPolicy.identity with new combat CP and ooc=null"},
  "old_policy": {"...": "full StudyPolicy.identity with old combat CP and ooc=null"},
  "new_condition": {"...": "checkpoint_training_condition(new selected payload)"},
  "old_condition": {"...": "checkpoint_training_condition(old selected payload)"}
}
```

The ellipses above are explanatory placeholders. Use `file_identity(grid)` for
`grid`, the complete `StudyPolicy.identity` dictionaries, and the unchanged
`fresh_training.checkpoint_training_condition` outputs. Both checkpoints must
be combat heads initialized from scratch, with equal raw `data_identity` and
equal `fit_size`. Full fits are required for a primary G1 contrast; matched
smaller fits produce descriptive results. New/old roles cannot be reversed.
There is no signing or timestamp scheme: root owns freezing these decisions.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/rl/fresh_panel_compare.py \
  --new /absolute/completed-new-panel --old /absolute/completed-old-panel \
  --grid /absolute/frozen-grid.json --expectations /absolute/frozen-expectations.json \
  --out /absolute/fresh-contrast.json
```

The validator checks actual checkpoint files/configs/weights, canonical grid
content and complete coverage, bank file identities, launch/native snapshots
and runtime, receipt/source bindings, completion-to-summary identity, and every
summary row against its result file and actual receipt. Both launches must bind
the supplied grid through non-null `config.grid_input`. Missing, duplicated,
unexpected, quarantined, faulty, cut, UB, censored, or programming-error results
are rejected. Inputs and result-directory inventories are rechecked before the
report is returned. The output is published atomically and exclusively.

The two independently recorded champion controls must match exactly on every
cell's win, HP after/lost, entry/source-entry HP, decisions, initial combat hash,
cut and UB. References and timing fields may differ. All four rows per cell
must share initial combat hash and entry HP. Control mismatch rejects the whole
comparison before any interval, providing an actual cross-run determinism check.

Win and HP-loss differences use `fresh_statistics.clustered_mean` with
`equal_world=True`, 10,000 bootstrap draws and seed 1919. A source WORLD block
contains all its starts and salts 0–7; salt −1 is a separate analysis. Pooled
means first average within each world, including worlds with unequal start
counts. Output includes new/old means, direct paired difference, percentile
95% CI, bootstrap SE, 2.8×SE approximate 80%-power MDE, rows and worlds. No
separate new-vs-champion and old-vs-champion intervals are subtracted.

Only full-fit `BOSS2:resampled` win and HP loss are primary. Other groups,
unsalted controls and pooled results are descriptive; EVENT2 is explicitly
marked as few-world descriptive. HP after is measured before end-of-fight relic
healing and set to zero on death. Negative new-minus-old HP loss favors new.
Constant observed differences yield degenerate intervals; these do not measure
uncertainty about unobserved discordance. No margin or G1 verdict is generated.

The JSON records input/expectations identities, grid/bank identity, checkpoint
conditions, exact-control status and provenance. `contrast_metrics` is a pure
fixture/diagnostic entry point and always leaves confidence claims disabled;
the CLI always uses the strict filesystem validator. `--bootstraps` allows
bounded test runs; production defaults to 10,000.
