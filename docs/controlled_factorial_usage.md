# Controlled representation factorial analysis

This tool analyzes the predeclared 16 controlled **combat** fits: features
OFF/ON × width 128/216 × LR 0.0003/0.001 × training seed 1919/2020.
It performs no fitting, inference, native execution, or model selection.
Filesystem **panel mode only** is implemented. Label mode is deliberately
omitted; there is no shortcut around immutable manifest/target validation.

Run after root has frozen all checkpoint identities and completed every panel:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  scripts/rl/controlled_factorial.py panels \
  --expectations /absolute/path/factorial-expectations.json \
  --out /absolute/path/factorial-analysis.json
```

The CLI uses 10,000 world bootstrap resamples, RNG seed 1919. The output is
exclusive and atomic; an existing output causes failure. Invalid inputs produce
no report. Standard output contains the three primary effects; the JSON file
contains every endpoint and secondary contrast. Python entry point:
`sts_rl.controlled_factorial.compare_panel_runs(expectations)`.

## Frozen expectations contract

Top-level keys are exactly `schema`, `version`, `mode`, `grid`, `manifest`,
`global_freeze`, and `arms`. All file identities use the existing
`recording.file_identity` shape: absolute canonical `path`, positive integer
`bytes`, and lowercase 64-character `sha256`. No identity is inferred from a
run name. The following shows **one arm**; the actual `arms` object must contain
all 16 canonical keys with no duplicates, omissions, or extra arms.

```json
{
  "schema": "controlled_factorial_expectations_v1",
  "version": 1,
  "mode": "panels",
  "grid": {"path": "/frozen/grid.json", "bytes": 123, "sha256": "<actual SHA256>"},
  "manifest": {"path": "/frozen/manifest.json", "bytes": 123, "sha256": "<actual SHA256>"},
  "global_freeze": {"path": "/frozen/global-freeze.json", "bytes": 123, "sha256": "<actual SHA256>"},
  "arms": {
    "s1919_f0_w128_lr0.0003": {
      "run_directory": "/panels/s1919_f0_w128_lr0.0003",
      "checkpoint": {"path": "/fits/arm/best.pt", "bytes": 123, "sha256": "<actual SHA256>"},
      "architecture": "controlled",
      "condition": {
        "phase": "combat", "fit_size": "full", "seed": 1919,
        "legacy_action_features": false, "width": 128, "lr": 0.0003,
        "steps": 12000, "min_steps": 2000, "eval_every": 100, "patience": 6,
        "device": "mps", "batch": 128, "defer_test": true, "layers": 2, "heads": 4
      },
      "selection": {"path": "/frozen/arm-selection.json", "bytes": 123, "sha256": "<actual SHA256>"},
      "policy": {
        "protocol": "<actual StudyPolicy protocol>",
        "filter_ids": ["<actual filter ID>", "<actual dead filter ID>"],
        "public_context_id": "<actual context ID>",
        "inference": "<actual inference ID>",
        "champion": {"path": "/frozen/champion.pt", "bytes": 123, "sha256": "<actual SHA256>"},
        "combat": {"path": "/fits/arm/best.pt", "bytes": 123, "sha256": "<actual SHA256>"},
        "ooc": null
      }
    }
  }
}
```

Replace placeholders with actual frozen identities; copy the full StudyPolicy
identity rather than inventing protocol strings. Every arm has exactly the six
keys shown. Generate canonical keys from **actual checkpoint metadata**, using
`condition_key(condition)`: `s{seed}_f{0|1}_w{128|216}_lr{0.0003|0.001}`.
Caller labels must agree with those conditions. LR 0.0001 and old references
cannot enter this matrix. Both declared LRs remain included, irrespective of
which LR wins validation within a feature/width/seed profile.

The root-owned `global_freeze` JSON must contain a `checkpoints` mapping from
each of these 16 canonical keys to its complete checkpoint file identity:

```json
{"checkpoints": {"s1919_f0_w128_lr0.0003": {"path": "/fits/arm/best.pt", "bytes": 123, "sha256": "<actual SHA256>"}}}
```

That example also abbreviates 15 required entries. Other global-freeze fields
and checkpoint entries for old references are allowed. All 16 entries must
exactly match expectations and the actual files. The tool validates these
bindings; root must establish the pre-evaluation freeze chronology externally.

Each per-arm `selection` is the existing deferred-selection receipt, including
`selected_checkpoint`, common `manifest`, `architecture: controlled`, exact
`training_condition`, and `selection_criterion` equal to:

```json
{"metric": "validation_soft_ce", "direction": "min", "tie_break": "lower_lr"}
```

Selections here bind the validation-best checkpoint **within each declared
fit**, including both LRs. Actual checkpoints must be from scratch, deferred,
at their recorded best step, source/vocabulary compatible, dropout 0, critic
frozen/untrained, and use actual-choice batch mean loss. The common immutable
manifest must have 113,042 combat FIT targets and 17,989 validation targets;
the full FIT subset and checkpoint optimization count must agree. Fixed
training conditions are enforced exactly as shown above. AdamW weight decay
0.01, gradient clipping 1, and the bucketing implementation remain properties
of the inherited trainer; its source archives are pinned, not reimplemented or
executed by this analysis. Root remains responsible for auditing those source
settings against the predeclared plan.

The existing `_checkpoint`, `_policy`, `frozen_selection`, `_Inputs`, `_grid`,
and `_run` validators are reused. Panels must cover the frozen 92,100,000,000
bank of 1,024 worlds, BOSS2=400/BOSS1=100/MONSTER2=100, all unsalted -1 and
resampled salts 0–7. Every result file, completion, receipt, launch, checkpoint,
archive, native snapshot, manifest and source identity read by those validators
is pinned and rechecked before return. Missing cells, duplicates, wrong grids,
faults, cuts, quarantine, nonfinite values, or incomplete runs fail closed.
All arms must share native/runtime/environment and champion policy identities;
champion `CONTROL_FIELDS` match exactly across all arms, and every cell shares
first-combat hash, entry HP, and source entry HP. Panels are read sequentially;
only compact world means and pairing identities survive each arm. Selection
manifests are validated sequentially and released before the next arm.

## Estimands and report fields

For each source world and group, replacement HP loss and win probability are
averaged across its eight resampled salts. Unsalted outcomes stay separate.
For each training seed, each main effect is the high-minus-low contrast
averaged equally over the four contexts of the other factors:

- Feature: ON minus OFF.
- Capacity: width 216 minus 128.
- LR: 0.001 minus 0.0003.

Negative HP loss favors the higher factor level; positive win differences
favor it. Wins use probability units (multiply by 100 for percentage points).
The three main effects on `groups["BOSS2:resampled"]["hp_lost"]` are primary.
All wins, unsalted outcomes, BOSS1/MONSTER2, and pooled `all` are descriptive.
Pooled outcomes also weight source worlds equally, even when a world contributes
more than one group.

Each `main_effects[factor]` has `seeds["1919"]`, `seeds["2020"]`,
`mean_of_seeds`, `direction_status`, and `direction`. Each interval object
contains `point`, `ci95`, `se`, and `worlds`. The mean first averages the two
**paired per-world seed deltas**, then calls existing `clustered_mean` on those
world deltas with equal-world weighting. Training seeds never become extra
worlds. Intervals are conditional on these two fixed seeds, with no
population-of-training-seeds inference.

`secondary` contains fixed-level factor effects at all four contexts,
two-way interactions averaged equally over the third factor, the three-way
interaction, and `seed_2020_minus_1919` at all eight fixed configurations.
Interactions are differences of factor effects: two-way coefficients are
±1/2 across the eight cells, three-way coefficients are ±1, and main-effect
coefficients are ±1/4. They are not regression coefficients under ±1 coding.
The three-way interaction is the change in a two-way interaction when the
remaining factor moves from low to high. Secondary effect objects also show
both seeds and their mean; fixed-configuration seed differences have one world
interval. `cell_means` exposes equal-world descriptive means for all 16 arms.

Direction is `replicated` only when **both seed-specific 95% world intervals**
strictly exclude zero in the same direction. Otherwise it is `suggestive` if
the mean interval excludes zero, or `inconclusive`. Zero-touching intervals do
not qualify. Secondary classifications retain their secondary status. All
intervals are nominal, without multiplicity adjustment, equivalence claims,
or gates. A constant observed delta can give a degenerate interval and does
not quantify unobserved discordance.

For synthetic tests only, `factorial_metrics(rows_by_key, conditions_by_key,
frozen_grid, resamples=...)` offers the same arithmetic. It explicitly reports
`filesystem_validated: false` and `confidence_claims_allowed: false`.
The filesystem function also accepts cheaper `resamples` for fixtures; the CLI
always uses the predeclared 10,000.

## Validation and root handoff

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest tests.unit.test_controlled_factorial -v
```

Tests use tiny synthetic weights/manifests and scaled cohort constants while
keeping inherited provenance and completed-panel readers active. They cover
known main effects/interactions, salt means and unequal world contributions,
strict condition inventories and metadata bindings, complete panels and exact
pairing, opposing-seed interpretation, and identical seed/world effects with
identical single-seed and mean CIs (proving world n is not doubled).

Remaining root validation: review this schema against the actual global freeze,
audit the frozen trainer settings, run broader integration checks under the
root workstream, and eventually run analysis on authorized completed panels.
This implementation task accessed no study results and performed no native or
training validation. Filesystem label analysis is **not implemented**.
