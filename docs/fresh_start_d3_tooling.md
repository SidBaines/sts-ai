# D3 fair old control and deferred TEST

The old supervised control now defaults to `CandidateActorCritic(size='large',
context='none')`, matching the champion's architecture. All actor and critic
weights initialize from scratch; the critic's own parameters remain frozen.
This does not imply a parameter-count match to the entity transformer. Entity
defaults remain width 128, two layers, four heads, dropout zero.

Training and benchmark accept `--old-size small|medium|large` and
`--old-context none|mean`. Python configs missing these fields use `large/none`.
Checkpoint model config, metadata, launch config and training receipt record the
settings. Both architectures now default to batch 128 (an explicit `--batch`
still overrides it). The benchmark includes the random old control on the same
fixed FIT observations as champion and random entity. Existing single-pass OOC
preprocessing and strict native/probe contracts are preserved.

For the real study, root predeclares two full-label old runs, LR `3e-4` and
`1e-3`, with the same steps and batch 128 budget as new-full. Select using the
minimum `best_validation_loss` (soft CE), breaking an exact tie by lower LR.
Each run already selects its best checkpoint using VALIDATION loss alone.
This tooling does not perform the cross-run LR selection.

Use `fresh_train.py --defer-test` for every candidate and final learning-curve
condition until all hyperparameters/selections are fixed. Config `defer_test`
defaults to `False` for compatibility. Deferred runs save `best.pt`, `latest.pt`,
normal FIT/VALIDATION curves and `best_validation_loss`; the summary includes the
selected checkpoint file identity, `test_deferred: true` and
`test_evaluated: false`. They do not encode TEST, call TEST inference, report
TEST metrics, or create `test_rows.jsonl`. Raw corpus validation/hashing still
includes every split to freeze the same common label manifest; TEST observations
and targets never enter a model during deferred fitting.

After fixing all hyperparameters, freeze one selection JSON per selected
condition. The minimal format is:

```json
{
  "selected_checkpoint": {
    "path": "/absolute/run/best.pt", "bytes": 123, "sha256": "actual file hash"
  },
  "manifest": {
    "path": "/absolute/run/manifest.json", "bytes": 456, "sha256": "actual file hash"
  },
  "selection_criterion": {
    "metric": "validation_soft_ce", "direction": "min", "tie_break": "lower_lr"
  }
}
```

Use `sts_rl.recording.file_identity(path)` to generate each exact identity;
sizes/hashes above are placeholders. The selected run's summary already contains
the checkpoint identity. Root owns the predeclaration and selection freeze; the
scorer verifies its identity and criterion without comparing candidates.

```bash
PYTHONPATH=src .venv/bin/python scripts/rl/fresh_score.py \
  --selection /absolute/selection.json --out /absolute/fresh-score-output \
  --deadline-utc 2026-10-16T07:30:00Z
```

The scorer verifies both file identities, the manifest content digest and its
binding to a compatible deferred checkpoint before opening any raw label shard.
It rebuilds the manifest for exact source agreement, keeps strict native launch
recording, and encodes only TEST. It loads the frozen best checkpoint, disables
gradients/dropout, and produces CPU singleton actor-only `test_rows.jsonl` and
TEST summary metrics with the usual inference/row/checkpoint identities. A fresh
output is mandatory; scoring writes a separate receipt with selection, source,
manifest, checkpoint and launch provenance. It never instantiates an optimizer,
updates weights, saves weights or rewrites the training outputs.

Validation is synthetic only. D3 does not launch study fitting, native jobs,
PPO, or confirmation. Root reruns host native/probe/localhost checks separately.
