# Fresh-start plotting tools

These scripts export plotting data and scientific figures from **completed frozen
results**. They perform no inference, fitting, selection, native work or gate
adjudication. Root reviews and runs the study export, then derives the report and
G1/G2 verdicts separately. No study result or final report is supplied here.

From this reporting worktree, inspect readiness (presence only; exit 2 lists
missing required inputs without opening TEST rows or creating outputs):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/rl/fresh_gate_report_data.py --check-ready
```

After review and after the required results finish, run the analysis in the
existing **frozen study `.venv`**. The script checks its prefix and Python 3.12,
Torch 2.14 and NumPy 2.5. Do not install anything into that runtime.
Use fresh output directories in private temporary storage (or pass another new
directory), outside the read-only study and source checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/rl/fresh_gate_report_data.py \
  --out /private/tmp/fresh-gate-data-v1

PYTHONDONTWRITEBYTECODE=1 /Users/sidbaines/Documents/SlayTheSpireAI/.venv/bin/python \
  scripts/rl/plot_fresh_gate.py \
  --data /private/tmp/fresh-gate-data-v1/plot_data.json \
  --out /private/tmp/fresh-gate-figures-v1
```

`--study` overrides the exporter's default study directory. Both `--out` paths
must not exist; use a new suffix on reruns. The renderer sets `MPLCONFIGDIR` and
`XDG_CACHE_HOME` to a private temporary directory before importing matplotlib.
It uses only the exported JSON, never the source study, torch or statistics code.

The full export requires all five selected TEST scores, six completed FIT runs,
all three G2 comparisons and merged 4,096-world student cohorts, the original
champion merged baseline, frozen expectations, teacher repeat reference,
`comparisons/g1_panel_new_minus_old_v1.json`, and the strict paired TEST label
comparison `comparisons/g1_labels_full_v1.json` with its frozen expectation.
A missing or invalid required input fails the export; no partial final plot
dataset is created. Rendering stages its
entire artifact family privately and publishes the fresh directory on success.
The inventory is not a substitute for the export's scientific validation.

Analysis uses existing `fresh_statistics.clustered_mean` and
`bootstrap_components`: 10,000 percentile WORLD resamples, seed 1919. TEST
agreement and soft CE are label weighted; high-confidence choice agreement
(target maximum ≥0.75) and choice-row act, room, and act-by-room strata have CIs
in the JSON and CSV. Forced rows remain in the separate `all_rows` aggregate.
Combat scores require exactly 28,676 rows and identical row/target/source cohorts.
OOC scores independently require exactly 44,704 rows, even when a shorter summary
and its row list agree with each other.
Checkpoint metadata, selected step, training condition, receipt, launch, split,
target checksum and global selection freeze must agree. The common 393 MB
manifest is stream-hashed once, never parsed into memory or used to retune.
Its binding and the complete scored rows provide the TEST data identity.
The paired G1 label export binds its expectation file, selected checkpoints and
training conditions to these validated scores, checks their data and row-ID
identities, and reproduces the comparison with the existing `compare_labels`
WORLD bootstrap. Every paired group estimate and CI must match the comparison
JSON. Its choice-row new-minus-old point and 95% CI are annotated on the learning
figure; separate arm intervals are not used to compare models.

G2 intervals are read directly from strict comparison JSONs, without subtracting
arm CIs. Behavior statistics use validated completed world summary records via
`fresh_cohort.validate_cohorts`, with exact frozen policies and all 4,096 seeds.
Each arm pools chosen/opportunities; paired differences resample the same WORLD
component vectors. Shop gold pools amount/opportunities, including eligible
zero-spend choices. JSON contains arm estimates/CIs, paired difference CIs,
chosen/opportunity/amount totals, worlds with opportunities and bootstrap-valid
counts. G2 win counts, discordant pairs, denominator, win point and mean-floor
point must match the validated cohort records before any outputs are created. Environment/native
science identities and input file identities remain in the dataset.
Undefined conditional bootstrap draws are omitted and counted, never
turned into zero. Behaviors are observed shares of menus with at least two
legal actions and the matching `WorldMetrics` opportunity; other options may
be unavailable. Smith uses high-HP menus offering smith (normally rest/smith;
rest need not be available). Potions are per combat menu offering drink/discard,
not per game or fight. Shop card-removal purchase excludes event removal.
Explicit card-reward skip counts only the action `skip card reward`, per rewards
menu offering it. Proceeding past an untaken card is not counted; each
gold/potion/relic pick on the same rewards screen adds another opportunity.
Opportunity cohorts depend on policy and are descriptive, not causal or pure
free-choice propensities. Full-game results are greedy evaluation, not stochastic training
win rates. Only the G2 win panel displays the 1 pp NI margin.

Behavior CSVs include student, champion and paired `bootstrap_degenerate`
flags, opportunity definitions and uncertainty notes. Hollow diamonds mark
degenerate empirical bootstrap intervals. Zero observed shop purchases with
nonzero denominators can yield [0,0]; this does not establish population absence
or zero uncertainty about unobserved purchases.

Artifacts are white-background PNGs at 300 dpi plus SVGs with editable text:

- `combat_test_learning`: new 5k/20k/113,042 combat TEST agreement and soft CE;
  selected old full LARGE/NONE reference and independent teacher repeat reference.
  Teacher repeats are a **REFERENCE**, not a strict ceiling.
  Its caption discloses native assistance (not certified public-only), the
  repeat-reference row/world counts, and full-FIT high-confidence TEST agreement
  for both models with their direct paired difference and CI. The selected old
  control was fitted at full FIT only: its horizontal line is not an old learning
  curve and supports no data-efficiency comparison. A shared legend sits below
  the axes, above the caption; displayed paired annotations use two decimals.
- `validation_learning`: combat new three FIT sizes and old two learning rates;
  OOC **choice soft CE** and agreement; VALIDATION solid, fixed 128-row FIT
  diagnostic dashed, frozen selected step marked.
- `fixed93e9_movement_combat`, `fixed93e9_movement_ooc`: pooled and per-act logged
  initial/previous KL, argmax changes, entropy/top probability; global actor L2
  relative weight displacement. Cadence is 100 supervised optimizer steps, not
  PPO updates. Act 3 has only three source worlds. Critic EV and PPO clipping N/A.
- `g2_paired_differences`, `g2_boss_differences`: paired win/floor and descriptive
  boss rates/entry HP, with explicit denominators and units.
- `g1_direct_panel`: direct new-minus-old BOSS2 win/pre-healing HP-loss primary
  contrast plus all available groups, unsalted/resampled separate. WORLD blocks
  retain all salts/starts with equal-world weighting. No margin or verdict.
- `g2_behavior_rates`, `g2_shop_gold`: opportunity ratios and gold per opportunity,
  explicit arm numerators/denominators and paired WORLD CIs.

`test_strata.csv`, `behavior_components.csv`, and `artifact_provenance.json`
accompany the figures. The provenance records the input dataset SHA256, analysis
runtime, bootstrap method and artifact inventory; the dataset records source
file identities. Comparison gate fields are deliberately excluded from exported
metrics. Inputs and historical artifacts remain read-only.

Bounded native-free regression command (only small synthetic inputs):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest tests.unit.test_fresh_gate_report_data
```

The R3 development rendering check uses entirely synthetic JSON at
`/private/tmp/fresh-gate-synthetic-v3/plot_data.json`, with its nine PNG/SVG
families, CSVs and provenance in `/private/tmp/fresh-gate-synthetic-v3/figures-final`,
with every figure prominently labeled `SYNTHETIC FIXTURE — NOT STUDY RESULTS`.
The fixture includes nonzero removal opportunities with zero observed purchases,
degenerate flags, undefined potion-discard ratios and a paired G1 label interval
crossing zero. The 12 small tests include the original seven plus negative
checks for stale G2 counts before output creation and paired-label binding/CI
failures (including changed pairing with unchanged marginal agreement). R3 extends
the existing cases to reject stale floor points and OOC summaries with short
matching row lists. The R2 preview remains intact; R3 renders the final captions.
No full live analysis/export was executed by the reporting builder; root runs it after
final quality review. Figures may still need layout review on actual
values, especially long G1 group inventories or extreme intervals.
