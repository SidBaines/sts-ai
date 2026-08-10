# Local competence tasks

Local tasks cut a finite, replayable combat window out of a full-policy source
trace. They are deliberately separate from the full-run reward: their output is
an adapter that the normal harness can load, not encounter-specific logic in the
main policy loop.

## Registered tasks

| Task ID | Exact encounter | Extra diagnostic metrics |
| --- | --- | --- |
| `gremlin_nob` | one Gremlin Nob | attack/block/skill share |
| `lagavulin` | one Lagavulin | sleeping turns, wake turn, attacks and setup cards played while asleep |
| `sentries` | three Sentries | first-kill slot/turn, outer-vs-middle first kill, actions targeted at each slot |

Lagavulin and Sentries use `EliteFightTask`; the older Gremlin Nob task remains
on its original version-1 implementation so existing Nob manifests and datasets
do not silently change.

## Manifest and replay contract

Splits are deterministic functions of `world_seed`, never of a decision or
individual sampled rollout. Every new elite-task window records:

- the exact encounter composition;
- the source trace and contiguous combat-decision indices;
- all actions before the fight, used to replay from a fresh simulator;
- a fixed start signature and SHA-256 digest covering act, floor, room, HP,
  turn, and the public enemy state;
- entry deck, relics/counters, and potions parsed from the most recent source
  observation;
- source policy/model/sampling/git provenance;
- bounded reward, label, and common plus encounter-specific metrics.

Elite source-manifest replay validates the replayed summary against this legacy
fixed signature before the first model decision. Replay validation then adds a
stronger common `start_state_signature` to every passing task window, including
Gremlin Nob. Signature schema 2 hashes the exact `combat_public_v2` observation text, ordered
displayed legal-action descriptions, and exact encounter composition. This
contains no action bits or hidden draw order. Every later replay through the
shared task entrypoint recomputes it and fails on a mismatch, independent of
whether the evaluation arm itself uses the legacy or public observation.

The parsed `entry_loadout` remains audit metadata rather than a field in the old
source signature: historical structured combat summaries did not carry the full
deck/relic/potion loadout. Historical source traces also omitted `+`
from combat card action labels (and mutable values from a few card names), and
did not disambiguate same-name potion targets with `[enemy N]`. Local-task replay
tolerates only those display migrations, only when the action bits match exactly
and all other description text is identical.

The common terminal reward is competence-only and lies in `[-1, 1]`:

```text
loss, incomplete episode, invalid-format stop, or decision-budget stop: -1
win: 1 - min(entry-to-exit HP loss, 40) / 40
```

This makes stalling unable to beat a completed fight. Reports retain
`survived` for compatibility and also emit explicit `won`, `completed`, and
`completion_reason` fields for the new tasks.

## Commands

Prepare manifests from the same source-policy rollout directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_task_prepare.py \
  --task lagavulin \
  --source-rollout-dir data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192

PYTHONPATH=src .venv/bin/python scripts/local_task_prepare.py \
  --task sentries \
  --source-rollout-dir data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192
```

Replay-validate a source manifest before using it for training or evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_task_validate.py \
  --task lagavulin \
  --manifest data/local_curricula/lagavulin/manifests/source.json \
  --report data/local_curricula/lagavulin/manifests/source.replay_validation_v5.json \
  --out-manifest data/local_curricula/lagavulin/manifests/source.validated_v5.json \
  --combat-observation combat_public_v2 \
  --timeout-seconds 30
```

The validator starts a fresh subprocess for every window, so a native crash or
hang becomes an explicit per-window `failure` or `timeout` rather than losing
the batch. The report accounts for every source window and records its status,
return code, error, settings, source-manifest SHA-256, every uniquely referenced
source JSONL's byte hash (plus a content-set hash), and the actually loaded
simulator extension's byte hash. Mixed simulator builds are rejected. The
optional `validated_v5.json` has the ordinary local-task manifest shape and can be passed
directly to collection/evaluation commands; it contains only successful windows
and embeds each passing window's complete public start signature, every exclusion
reason, and refreshed split and label counts. Never skip a replay failure
dynamically inside an eval.

The existing `local_task_build_sft.py`, `local_task_eval.py`,
`local_task_compare.py`, and `local_task_grpo.py` entry points accept either new
task ID. Use the manifest's `train` split for collection/training and `holdout`
for evaluation. For stochastic evaluation, use temperature greater than zero
and aggregate K samples within each window before paired comparison, as
`local_task_compare.py` does.

Competence training through `local_task_build_sft.py` defaults to
`--loss-mask action`; pass `--loss-mask completion` only to reproduce the old
full-response objective. The Python SFT/PG builders retain `completion` as their
API default for old callers, and every new dataset manifest records the selected
mode (plus token accounting for action-only data).

`local_task_eval.py --combat-observation legacy` is the compatibility default.
Use `--combat-observation combat_public_v2` for the human-public pile/relic/type/
turn-history observation. The selected value is recorded in rollout metadata;
all arms in a matched comparison must pin it explicitly.
The same flag is available on `local_task_grpo.py`; its default is also legacy
for backward compatibility.

## Current and historical source replay audit (2026-07-23)

Against the existing 100-run E4B source directory, the current v2 binary and
signature schema 2 pass all 43 Gremlin Nob windows (32 train/11 holdout), all 43
Lagavulin windows (33/10), and all 36 Sentries windows (27/9). A second complete
run, again using one fresh subprocess per window, produced identical public
start signatures for every window. The exact current manifests are
`source.validated_v5.json`; they embed the binary and source hashes.

On the historical v1 build, strict validation passed 42/43 Nob windows and
40/43 Lagavulin windows. Nob `seed_128_r0_w0` and Lagavulin `seed_53_r0_w0`,
`seed_84_r0_w0`, and `seed_149_r0_w0` diverged before the task start. Those
exclusions remain valid facts about that exact binary and are preserved in
`source.validated_v2.json`; the v5 success does not establish cross-build
determinism. Any future binary change must rerun isolated validation and write a
new versioned manifest rather than silently retaining or dropping windows.
