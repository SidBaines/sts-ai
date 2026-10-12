# Trainable composite checkpoints

The experiment checkpoint loader now supports a model-config union. Legacy
candidate configs still contain size/context/state_dim/action_dim and need no
`kind` field. Composite configs use `kind: composite` with `combat` and `ooc`
candidate configs. Architectures and encoder features are unchanged.

Create a composed checkpoint atomically:

```bash
PYTHONPATH=src .venv/bin/python scripts/compose_nonllm.py \
  --combat-checkpoint COMBAT.pt --ooc-checkpoint OOC.pt --output COMPOSED.pt
```

Use COMPOSED.pt through the ordinary `train_nonllm.py --checkpoint` path for
PPO, BC, evaluation and compatible resume. ModelAgent also accepts it. The
two-source evaluation flags remain supported, with candidate-only heads.
`policy_from_config` / `policy_from_payload` centrally validate encoder identity,
known model kind, fixed public feature dimensions and exact state-dict structure.
Unsupported/nested composite kinds are rejected. Composition will not overwrite
an input file and verifies each source's SHA256 stayed stable while loading.

Source paths, digests, configurations, original metadata and recorded common
provenance are retained. Any recorded native/interface provenance mismatch is
rejected. Older checkpoints in this study did not record native/interface fields;
composition labels that absence explicitly instead of inventing provenance.
The campaign/data manifests remain the authority for those historical sources.
New training checkpoints preserve composition ancestry and parent-checkpoint
digests. If STS_LIGHTSPEED_BUILD is set, the trainer records its manifest SHA256,
the complete manifest and actual module SHA256, rejecting a binary/manifest
mismatch. It records the complete Python public-interface identity and encoder/
model code digests without initializing the native simulator.

Composition starts at step zero with no optimizer state and stage `composition`;
it does not pretend to combine the old heads' optimizer histories. Subsequent
joint PPO genuinely updates both heads, and normal resume restores the composite
parameter ordering and Adam state. Compatible later PPO chunks can retain their
moments through the existing warm-optimizer mechanism.

This extends model-config semantics while retaining outer checkpoint format 1.
Legacy candidate files remain readable by old and new code. Older candidate-only
readers cannot load the new composite-kind config; use the upgraded loader for
these files. This does not change the environment observation/trace formats.

CPU verification: five new tests check legacy/compound round trips, unchanged
source checkpoints, source-provenance rejection, strict kind/encoder/dimensions,
ModelAgent routing, ordinary training-CLI loading, genuine PPO updates in both
heads and restored optimizer/logits. Together with dispatch/training/encoder/model
tests, 37 tests passed. The optimized phase dispatcher also retains its prior output and
gradient parity coverage. No GPU runs, final evaluations or backend edits were
made for this implementation.

The checkpoint utility does not certify the scientific validity of a source
model's data. In particular, models/labels using the old automatic native combat
path must follow the separately documented backend quarantine and fresh-evaluation
requirements before selection. Recorded digests distinguish those conditions.
