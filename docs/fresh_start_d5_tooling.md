# D5 optimization and frozen comparison inputs

OOC remains full-only. The common manifest, raw targets, source identities,
world splits and FIT ordering are unchanged. Both encoders use that same
manifest. Combat's 5k/20k/full learning-curve x-axis counts informative combat
targets. OOC's manifest count includes all hard targets; its optimization count
includes only targets with more than one filtered legal action.

`fit` builds its optimization rows before creating AdamW and fails clearly when
there are no choices. Forced targets cannot form momentum/weight-decay-only
steps. Updates, examples/exposures, remaining minibatch indices and the minimum
update/patience budget refer only to choice updates. Summaries, metric events and
checkpoint metadata distinguish `manifest_fit_rows`,
`optimization_fit_choice_rows` and `forced_fit_rows_excluded`.
`actual_fit_rows` retains its original meaning: all selected manifest FIT targets.
FIT diagnostics and VALIDATION/TEST keep forced targets and report all-row and
choice-only metrics. Early stopping still uses VALIDATION all-target soft CE.

The optimizer still uses the mean loss over the actual choice batch. This is
recorded as `loss_normalization: actual_choice_batch_mean`; no nominal-batch
renormalization or hyperparameter/budget change was made. Rare leftover batches
therefore retain the existing higher per-row weighting.

Final `fresh_compare.py games|labels` now requires `--expectations FILE` in
addition to the exact `--cohort` and the two arm inputs. The JSON must have
`schema: fresh_comparison_expectations_v1`, integer `version: 1`, and `mode`.
The CLI verifies every referenced checkpoint file identity before reading the
comparison rows, and rechecks the files before writing results. The expectation
file's path, size and SHA256 are recorded under `provenance.expectations_file`
and its path/SHA256 are printed in the human report. The result also includes
the frozen bindings. Panel and consistency modes do not require this file.

For games, include `champion_policy` and `student_policy`, each a complete
`StudyPolicy.identity` (protocol, filter IDs, public context, inference,
champion/combat/OOC checkpoint identities). Both policies must share the same
champion and input/inference protocol. Champion has no replacements; student
has at least one selected replacement, different from the champion. Every row
must match its arm's frozen identity exactly. Reversed arms, champion versus
champion, wrong or mixed policies fail before statistics. Identical observed
outcomes from two different valid policies remain valid.

Games inputs must be completed receipted cohort directories, or their direct
`summary.json`. The CLI lazily imports C4's `fresh_cohort.validate_cohorts`,
validates each parent cohort, and attaches the validated uniform
`science_identity.policy` to every brief record as `source.policy`. C4 summaries
inherently omit source; a field in standalone JSON/JSONL is never sufficient.
Exact requested seeds and native `expected_provenance` checks remain in force.
CLI tests mock C4's validated return structure and also exercise the staged C4
validator on synthetic completed receipts, without native execution. The protected
C4 dependency copies and C3 study/collection files are unchanged. Root must verify
real completed native cohorts after port.

For labels, include `student_checkpoint` and `control_checkpoint` as exact
`sts_rl.recording.file_identity` dictionaries, `student_architecture: new`,
`control_architecture: old`, and `student_condition`/`control_condition` from
`checkpoint_training_condition` on each selected checkpoint. This helper extracts
phase, FIT size, LR, seed, update/evaluation budget, batch, device, deferred TEST
setting and architecture settings from its saved config (using the saved model
config when an optional architecture field is absent). Disagreement between those
configs fails. The CLI checks those
bindings against the actual checkpoint configs/encoder identities. TEST scoring
records the same `architecture` and `training_condition` per row, alongside the
selected checkpoint SHA. One checkpoint/condition per arm, different checkpoint
SHAs and correct architecture roles are required before any CI.

Root freezes expectations from selected checkpoints **before** scoring/evaluation.
For example, after applying the existing D3 validation-only selection procedure:

```python
from sts_policy.training import load_checkpoint
from sts_rl.fresh_training import checkpoint_training_condition
from sts_rl.fresh_study import atomic_json
from sts_rl.recording import file_identity

expectations = dict(schema='fresh_comparison_expectations_v1', version=1,
                    mode='labels', student_architecture='new', control_architecture='old')
for arm, selected_path in [('student', selected_new), ('control', selected_old)]:
    expectations[arm + '_checkpoint'] = file_identity(selected_path)
    expectations[arm + '_condition'] = checkpoint_training_condition(load_checkpoint(selected_path))
atomic_json(expectations_path, expectations)
```

For games, freeze the two actual `StudyPolicy.identity` dictionaries under the
same schema/version with `mode='games'`. No new selection mechanism is added:
the minimal D3 selection receipt stays unchanged, old LR candidates remain
`3e-4`/`1e-3`, and root selects minimum VALIDATION soft CE with lower-LR ties.
TEST never selects a candidate. Bootstrap, CI, exact McNemar, G2 precedence and
panel equal-world arithmetic are unchanged. No scientific fit or native job was
run for D5; verification is bounded synthetic/pure tooling work.
