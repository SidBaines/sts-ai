# Public semantics revision: environment 0.3 / entities v4

This unreleased candidate follows the 0.2 investigation. The native binary,
observation identity and typed model inputs change explicitly. Historical 0.2
models, banks and outcomes remain preserved; they require the archived source
profile rather than silently loading into current code.

| Identity | Current value |
| --- | --- |
| Environment | `0.3.0` |
| Observation | `sts_env_public_v3` |
| Native binding API | `3` |
| Encoder | `public_entities_v4` |
| OOC residual model | `typed_residual_ooc_v3` |
| Combat residual model | `typed_combat_residual_v2` |
| Vocabulary | Existing `public_vocabulary_v3.json`; IDs unchanged |

## Reached failures and repairs

The native TRAIN world **320000039** supplies a real rest-room frontier.
After smithing, upgrade candidates 14 and 25 had identical legacy/action/entity
relation inputs, yet upgraded deck slots 14 (bottled Strike) and 28 (ordinary
Strike). API 3 exposes each deck selection's public source and bottled flag;
the encoder joins it to the exact inventory entity. Indices route relations,
never neural ordinals. A source-controlled 404-action prefix regression exercises
both choices with native upgrade transitions.

With Prayer Wheel, six offered cards belong to two mutually exclusive groups.
Choosing one removes three offers and leaves the other three. Earlier metadata
omitted that association. API 3 supplies the current reward group; v4 represents
offered cards as entities and joins each action to its group's alternative bag.
Group renaming does not change encoded tensors. No absolute group ID becomes a
neural feature.

Hand of Greed raised live gold **100→120**, while the earlier top-level field and
encoder remained at 100. Public-v3 top-level gold, HP and max HP now reflect the
active combat player. Explicit `gold_at_battle_entry`, `hp_at_battle_entry` and
`max_hp_at_battle_entry` retain the public entry snapshots. Tests also cover
enemy-turn damage and Fruit Juice's max-HP increase. The historical bridge
restores entry values for the frozen legacy policy without mutating the current
observation. Package and episode version reporting share one environment
constant.

## Acceptance and migration

The full host/native suite passes **536 tests** (one optional generated-bank
skip); the required bounded smoke passes. Fresh and reverse application of the
scoped patch pass. Independent old/new processes match **all 192 canonical
control traces**, reward totals and endpoints: baseline **156/20/0**, mean floor
28.1979, no faults/cuts. Canonical traces contain phase/index/description. Full
action JSON hashes intentionally change when public mechanics are added; two
current-profile processes separately check the new full hashes.

Current artifacts are under `data/nonllm_v4`. The new zero-residual initialization
transfers the immutable original composite, preserving its prior rather than
porting failed learned residuals. Old observation banks are rejected by v4;
new collection records current native/interface provenance. A new TRAIN-only
64-world bank has **23,278 labels**, 970 fight starts and zero faults, collected
at 1,000 search simulations in 31.01 seconds. Native OOC labels remain a weak
scripted policy, and combat labels remain privileged. Neither provenance nor
teacher likelihood establishes useful game learning.

The old source snapshot is `external/reference_code_v020` (Git commit
`e7145f6`), with the original `external/sts_lightspeed_v020/build` binary. For old
scripts, set `PYTHONPATH=external/reference_code_v020/src:external/reference_code_v020`
and `STS_LIGHTSPEED_BUILD` to that old build explicitly. The current default
`external/sts_lightspeed` points to independent `sts_lightspeed_v030`.

Remaining work includes useful-learning evidence, stronger strategic supervision
and simulator/real-game conformance. Outside-combat potion enumeration and its
native execution fallthrough remain a separate action-space issue; this revision
does not silently expand those actions or alter their mechanics.
