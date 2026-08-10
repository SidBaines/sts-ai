# COMP-015 paired-report analysis

`scripts/analyze_comp015.py` is the pure-report decision path for COMP-015. It
does not load a model or alter an input artifact. It validates the exact
dataset, manifest, report, reference-report, and checkpoint identities declared
in its config; requires all three paired seeds and both arms; recomputes report
summaries from rows; and refuses partial grids or existing output paths.

Run it from the repository root:

```sh
PYTHONPATH=src .venv/bin/python scripts/analyze_comp015.py \
  configs/competence/comp_015_analysis.json \
  --out data/competence/comp_015/eval/paired_analysis.json
```

Relative artifact paths are resolved from `--root` (the current directory by
default), not from the config's directory. The output is published with an
atomic fail-if-exists link.

## Exact config shape

The top-level keys and frozen thresholds are exact. Each `path`/`sha256`
object names the literal JSON report bytes. A checkpoint declaration contains
its `identity_sha256` and may additionally contain the exact `files` mapping
from adapter provenance.

```json
{
  "kind": "comp015_paired_analysis_config",
  "version": 1,
  "experiment_id": "COMP-015",
  "seeds": [0, 1, 2],
  "arms": ["control", "augmented"],
  "datasets": {
    "train": {
      "sha256": "<train150 dataset sha256>",
      "manifest_sha256": "<train150 manifest sha256>",
      "n_rows": 150
    },
    "development": {
      "sha256": "<development57 dataset sha256>",
      "manifest_sha256": "<development57 manifest sha256>",
      "n_rows": 57
    }
  },
  "thresholds": {
    "primary_step": 1500,
    "checkpoint_grid": [750, 1500, 2250, 3000],
    "sensitivity_invariance_min": 0.75,
    "sensitivity_tv_max": 0.2,
    "strong_stability_invariance_min": 0.9,
    "strong_stability_tv_max": 0.1,
    "minimum_pairs_ceasing_sensitivity": 2,
    "mean_invariance_delta_min": 0.2,
    "mean_tv_delta_max": -0.2,
    "mean_all_variant_macro_agreement_delta_min": 0.1,
    "mean_all_variant_macro_nll_delta_max": -0.25,
    "selection_dev_correct_min": 28,
    "behavior_train_correct_min": 135,
    "behavior_dev_correct_min": 32,
    "behavior_teacher_probability_min": 0.372162,
    "behavior_teacher_nll_max": 1.776904,
    "behavior_qualifying_seeds_min": 2
  },
  "bootstrap": {
    "seed": 0,
    "n_resamples": 10000,
    "alpha": 0.05,
    "expected_window_clusters": 11
  },
  "replicates": {
    "0": {
      "control": {
        "fixed_step1500": {
          "checkpoint": {
            "identity_sha256": "<checkpoint identity>",
            "files": {
              "adapter_config.json": "<sha256>",
              "adapters.safetensors": "<sha256>"
            }
          },
          "train_report": {"path": "<path>", "sha256": "<sha256>"},
          "development_report": {"path": "<path>", "sha256": "<sha256>"},
          "cyclic_report": {"path": "<path>", "sha256": "<sha256>"},
          "cyclic_checkpoint_label": "<label inside cyclic report>"
        }
      },
      "augmented": {
        "fixed_step1500": {
          "checkpoint": {"identity_sha256": "<checkpoint identity>"},
          "train_report": {"path": "<path>", "sha256": "<sha256>"},
          "development_report": {"path": "<path>", "sha256": "<sha256>"},
          "cyclic_report": {"path": "<path>", "sha256": "<sha256>"},
          "cyclic_checkpoint_label": "<label inside cyclic report>"
        }
      }
    },
    "1": {"control": "<same shape>", "augmented": "<same shape>"},
    "2": {"control": "<same shape>", "augmented": "<same shape>"}
  }
}
```

The strings used to abbreviate seeds 1 and 2 above are documentation only:
the real config must contain complete objects.

## Optional checkpoint selection and behavior evidence

Checkpoint selection is all-or-none: add `checkpoint_dev_reports` to every
seed/arm beside `fixed_step1500`. It must contain exactly keys `750`, `1500`,
`2250`, and `3000`, each with:

```json
{
  "checkpoint": {"identity_sha256": "<identity>"},
  "development_report": {"path": "<path>", "sha256": "<sha256>"}
}
```

The step-1500 declaration must exactly reuse the fixed endpoint's checkpoint
and development report. Selection retains checkpoints with at least 28/57
correct, then takes the lowest NLL, then the earlier step.

Behavior qualification is also all-or-none and requires the complete grids.
Add `behavior_step_reports` beside each seed's `control` and `augmented`
objects:

```json
{
  "behavior_step_reports": {
    "step": 1500,
    "control": {
      "checkpoint": {"identity_sha256": "<same-step control identity>"},
      "train_report": {"path": "<path>", "sha256": "<sha256>"},
      "cyclic_report": {"path": "<path>", "sha256": "<sha256>"},
      "cyclic_checkpoint_label": "<label>"
    },
    "augmented": {
      "checkpoint": {"identity_sha256": "<selected treatment identity>"},
      "train_report": {"path": "<path>", "sha256": "<sha256>"},
      "cyclic_report": {"path": "<path>", "sha256": "<sha256>"},
      "cyclic_checkpoint_label": "<label>"
    }
  }
}
```

`step` must equal that seed's prospectively selected treatment step. The
control evidence is deliberately evaluated at the same step, even if the
control run selected a different checkpoint. The analyzer requires at least
two qualifying treatment seeds before returning a deployment adapter; its
deterministic tie-break is lowest development NLL, higher agreement, then lower
training seed.

The hierarchical bootstrap is explicitly secondary. Each resample draws the
three paired training-seed replicates, then the 11 development-window clusters
within each sampled replicate. It never counts rotations, states, or
seed-by-state cells as independent experimental replicates.
