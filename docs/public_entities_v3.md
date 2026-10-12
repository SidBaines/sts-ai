# Typed policy inputs and preserved behavior

This candidate uses `public_entities_v3` over the repaired `sts_env_public_v2` observation. The static vocabulary has 1,167 tokens and a recorded SHA-256. It contains public game names, including events, card types, enemy moves and statuses, rather than identities learned from sampled futures. The core remains dependency-free; policy code uses the existing pinned optional Torch/NumPy dependencies.

The encoder retains every legal candidate and every public entity. Four categorical slots identify each entity's role, identity, rarity/intent and card type. Role-specific numeric values preserve cards' upgrades, base/current costs, free play, retention, exhaust, bottled state and meaningful special counters. Enemy values preserve HP, maximum HP, block, damage/hits and associated powers. Timed player powers have separate amount/countdown fields. Draw/discard/exhaust stay unordered bags with multiplicity. Visible hand and enemy indices perform source/target joins only; they never become neural numbers. Unknown identities/statuses, unresolved joins and unsupported damage calculations are measured.

The candidate input includes its typed card or potion/relic identity, energy cost, free/X-cost fields, supported immediate damage, prices, gold/heal amounts, conservative displayed event amounts and source/target memberships. Explicit availability bits distinguish an unknown amount from known zero. Complex event effects still use public prose and the legacy prior; complete typed event semantics and per-target nonlinear AoE outcomes remain open work.

Visible map nodes carry feasible-path minimum/maximum room counts, nearest rest/shop/elite, path count and path lengths. These describe paths through the DAG rather than the union of reachable rooms. Coordinates only join nodes. The implicit act boss is not manufactured as a visible graph node.

| Tensor | Meaning |
| --- | --- |
| Legacy state, 256 values | Unchanged historical public encoding after explicit menu projection |
| Legacy actions, 128 values per candidate | Historical features expanded back to all repaired instances |
| Global values, 48 slots | Phase, act/floor, health, energy/block, inventory/menu/pile counts and turn counters; unused slots are reserved zeros |
| Entities, four IDs + 150 numeric slots | First 32 role-specific values; next 118 fixed status amounts, with unknown-status diagnostics |
| Candidate values, 96 slots | Card fields 0–31; cost/free/X/target damage 32–42; price/gold/unknown 43–47; parsed amounts and availability 48–59; heal/source/target count 60–63; route 64–86; rule/card coverage 87–89; remaining slots reserved |
| Candidate links | Source and target membership masks over public entities, with no ordinal index features |
| Representatives and masks | Exact prior-menu routing and padding; never trainable identity features |

The model keeps the original actor/critic as an immutable prior. A small entity actor adds an OOC logit residual; its final layer starts at zero. A separate entity/menu critic adds a value residual and shares no trainable parameters with the actor. The whole prior is excluded from both optimizers. Combat actions use the greedy original public-menu bridge throughout the protected strategic study. The separate critic may learn combat-state values in a microdecision arm.

This is an explicit behavior transfer, not direct reuse of weights on incompatible inputs. The bridge restores old labels and first-copy routing using public mechanics. The richer environment still exposes all distinct actions. Zero residuals reproduce prior logits/values, including mean-menu contexts with duplicate instances. OOC offers remain one-to-one. Sampling mass over duplicate combat cards is not claimed equivalent; the study uses the deduplicated greedy combat prior in both arms.

Micro and macro collection use the same worker, world/task definitions, game horizon, objective, frozen combat and task-bound OOC random streams. Macro transitions sum rewards until the next strategic state; gamma is currently fixed at one. True terminals have zero bootstrap; cuts use the actual endpoint value. Faults quarantine whole episodes. Completed realized returns remain separate from bootstrapped targets. A partial initial combat before the first OOC state has no macro actor row, while its game outcome/reward is still reported.

Checkpoint identities pin vocabulary, bridge, encoder and model code. Legacy and typed loaders are separate and fail closed. A code/interface migration must be explicit; old checkpoints and banks are not overwritten.

Implementation: `src/sts_policy/entities.py`, `entity_model.py`, `legacy_bridge.py`, `strategic_collection.py`, `strategic_learning.py` and `strategic_data.py`. The numeric layouts are fixed by this encoder identity; adding new fields requires a new identity or an explicitly checked migration.
