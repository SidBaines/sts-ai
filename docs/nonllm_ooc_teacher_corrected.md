# Corrected OOC teacher collection and regularized matched pilot

Fresh collection completed 256/256 training frontiers with zero native faults,
using environment 0.1.1, Python interface `1fa75a1d…` and native module
`40ab5056…`. The source is
`data/nonllm_curriculum/teacher_act1_200_native011_seed10000`; outputs are in
`data/nonllm_curriculum/ooc_lookahead_native011/`. CPU collection used two workers
and completed in 27.80 seconds within its 120-second bound. No GPU learning or
final evaluation was launched for this task; all retained frontiers have split
`train`.

Each frontier evaluates up to four candidates for a 30-external-decision native
continuation. Combat continuation now uses **200 simulations**, matching current
OOC evaluation. The quarantined collection used 50 and the old 8dba automatic
combat binary. These are distinguishable experimental conditions; this fresh
dataset is not mixed with or used to overwrite the old labels.

The corrected teacher changes 92/256 labels (35.94%) and all changes strictly
improve its bounded proxy; mean proxy gain is 0.1883. This is a proxy property,
not evidence of improved learned run performance. Short horizon, partial menus,
one hidden future, scripted continuations and automatic act-transition healing
still limit the teacher's strategic meaning.

`scripted_teacher.jsonl` supplies the matched control: identical observations,
source IDs and order, with `teacher_index` replaced by each recorded scripted
index. Both datasets contain the exact same 256 valid frontiers. The encoder
consumes only their public observation, not privileged continuation diagnostics.
Every retained row's native digest and 200-simulation setting were checked.

Failure handling remains whole-frontier: an exception in any candidate branch
propagates out of `score_ooc`, discarding earlier local branch scores. A CPU-only
injected second-branch failure after a successfully scored first branch verified
that no partial teacher result returns. The worker then reports a fault, and the
collector's valid label file excludes the frontier. The actual fresh run needed
no such exclusions.

The initial attempt used a stale build path from the superseded campaign and
failed setup before any teacher query. Its 256 configuration-error journals were
preserved under `ooc_lookahead_native011_setup_failed/` with an explicit
`setup_failure.json`. They are neither simulator defeats nor scientific native
faults and contribute no labels to the corrected dataset.

Two GPU commands are prepared in `corrective_pilot_plan.json` but were not run.
Root must freeze the same best fresh corrected OOC candidate checkpoint after
the current PPO chunk, substitute `SOURCE_CKPT` in both commands, and record its
SHA256. The plan uses batch 256, learning rate 0.0003, optimizer seed 11,
**20 optimizer updates** and a 30-second cap, with fresh optimizer state in each
condition. With 256 states this is 20 dataset passes, rather than the previous
roughly 7,000-pass wall-clock fit. Conditions differ only in matched labels;
model start, order, update count and optimizer settings must be the same.

The prepared commands fit `--phase ooc`, evaluate BC agreement against fresh
development teacher data, and write `ooc_teacher20_{scripted,lookahead}` outputs
under the active corrected campaign. Whole-run development evaluation must use
the same corrected native build, 200 combat simulations, requested world cohort
and fault/censor accounting. Forced versus choice-menu BC accuracy and live
outcomes remain separate; target disagreement alone does not establish quality.

Correct build for every planned command:
`/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/external/sts_lightspeed_v011/build`.
The active campaign is `data/nonllm_native_revision/campaign/`; the old
`data/nonllm_v1/campaign/` automatic-combat evidence is superseded/quarantined.
No source or campaign files were edited by this collection/preparation task.
