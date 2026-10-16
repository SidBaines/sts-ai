# Controlled representation: model and training

This opt-in implementation adds `architecture='controlled'` and checkpoint kind
`entity_controlled_v1`. Original `new`/`old` defaults, entity/legacy encoders and
models, vocabulary, and `fresh_study.py` remain unchanged. No study fits or
scientific outcomes are part of this implementation.

## Public inputs and arithmetic

Call the archived `policy_observation(raw_observation)` once before
`controlled_encoding.encode_observation(public_obs)`. The controlled encoder
wraps both original public allowlists, requires identical original legal-index
order, and retains only the entity encoding and exact legacy 128 action
features. It discards the legacy 256 state features. It does not filter again,
serialize arbitrary observation fields, or mutate the input.

The encoded wrapper exposes `.entity`, `.entity_ids`, `.action_indices`,
`.coverage` (original entity coverage), and `.legacy_action_features`. Collation
returns the ordinary entity dictionary with an additional zero-padded float32
`legacy_action_features[B,A,128]`. ON and OFF use the same encoder and buckets.

`ControlledEntityActorCritic` inherits the original singleton actor, critic and
parameter APIs. Its independent critic is unchanged. Its actor adds the output
of `actor.legacy_action_projection`, a zero-initialized bias-free
`Linear(128,width)`, to the action embedding before the existing
action/source/target/state scorer. The singleton path strips action and member
padding; the vectorized path uses the existing masks.

Both feature conditions construct the projection and preserve the parent's
shared initialization and RNG stream. OFF feeds zero features and freezes the
dormant projection; ON trains it. Thus ON and OFF start with identical actor
logits at the same seed/shape, and OFF matches the original actor with shared
weights. Trainable projection gradients can distinguish ON after updates.

Metadata reports total parameters, currently trainable parameters, total policy
parameters, active policy parameters, dormant policy parameters, and trainable
policy parameters. At width 128, layers 2, heads 4:

| Feature flag | Policy total | Policy active/trainable during warm-up | Policy dormant |
|---|---:|---:|---:|
| OFF | 874,625 | 858,241 | 16,384 |
| ON | 874,625 | 874,625 | 0 |

The trainer freezes all critic parameters. Dormant parameters are excluded from
its optimizer. Neither actor inference nor training calls the critic.

## Checkpoint contract

Use `sts_policy.controlled_entity.policy_from_payload(payload)` for inference;
it imports neither trainers nor native code. `model_config` has `width`,
`layers`, `heads`, `dropout`, `phase` (`combat` or `ooc`) and the strict boolean
`legacy_action_features` (default OFF). Missing or incompatible source identities
are rejected before weight loading. Original entity source hashes are unchanged.

Identity fields are `encoder_version`/`encoder_sha256` (controlled encoder),
`controlled_model_sha256`, `entity_encoder_version`/`entity_encoder_sha256`,
`entity_model_sha256`, `legacy_encoder_sha256`, `vocabulary_sha256` and
`vocabulary_size`. `model_config_sha256` binds the configuration. The loader
also checks duplicated saved configuration and optional training configuration.
The trainer's checkpoint condition includes the controlled feature boolean.
Total/trainable counts describe the saved model's training state, rather than
changing the inference loader's inherited default critic trainability.

## Fixing capacity without outcomes

The following calls use meta tensors, consume no RNG, read no data and require
no native binary. Root chooses and fixes the width before any fit. Match **active
policy parameters**, keeping the dormant projection count visible. For this study,
one common width of 216 minimizes the worst mismatch across feature flags on
multiples of 8: OFF has 2,056,537 active parameters (+1.02% versus the legacy
2,035,713), and ON has 2,084,185 (+2.38%). Both use two layers and four heads.
The per-flag helper below is illustrative; it does not choose separate study widths.

```python
from sts_policy.controlled_entity import describe_capacity, nearest_policy_width

describe_capacity(width=128, legacy_action_features=False)
nearest_policy_width(target=2_035_713, legacy_action_features=False)
nearest_policy_width(target=2_035_713, legacy_action_features=True)
```

The returned `nearest` and adjacent `neighbors` are counts only. The target is
the existing LARGE/NONE legacy actor's trainable count; this does not establish
performance equivalence or select a model.

## Training and evaluator handoff

Use the existing frozen `.venv`. After Root fixes width, feature condition, LR
and seed, the production invocation is the existing warm-up CLI with:

```text
--architecture controlled --phase combat --width <fixed-width>
--layers 2 --heads 4 --legacy-action-features --defer-test
```

Use `--no-legacy-action-features` for OFF (also the default), and the normal
`--data`, `--manifest`, `--probe`, `--out`, `--lr`, `--seed` and deadline options.
Production controlled runs require `--defer-test`; no TEST rows are encoded or
scored in warm-up. Preserve the validation-only selection freeze before exposed
TEST scoring. `TrainingControl` continues enforcing AC/STOP/deadline/disk checks.
Fixed unlabelled probes use the same CPU singleton actor monitoring and detached
CPU shadow for MPS training. They never supply optimization targets or selection.

Evaluator integration must dispatch controlled policies to the new loader and
encoder. `fresh_training.score_test_checkpoint` supports controlled checkpoints.
The evaluator-owned finalization layer must also extend the architecture
allowlist in `src/sts_rl/fresh_scoring.py`; this model/training change does not
edit it or the protected archived study policy.

## Bounded verification

```bash
PYTHONPATH=src .venv/bin/python -B -m unittest \
  tests.unit.test_controlled_entity tests.unit.test_controlled_training \
  tests.unit.test_fresh_training tests.unit.test_fresh_probes \
  tests.unit.test_fresh_scoring tests.unit.test_entity_model \
  tests.unit.test_entity_encoding
```

These tests use public synthetic fixtures and temporary outputs. Their tiny
optimizer updates and synthetic scoring exercise plumbing only. They do not
access real FIT/VALIDATION/TEST inputs, run native jobs, or choose scientific
architecture settings. MPS shadow verification skips when MPS is unavailable.
