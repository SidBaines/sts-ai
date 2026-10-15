# Fresh public entities (B1–B4)

`entity_encoding.py` and `entity_model.py` are independent of the previous policy
encoder/model and initialize all trainable weights from scratch. The contract is
`env-v0.1.5` (`2c83f969d062873a318b30b1914510f7cf96f6bf`): `env.py::_observe`,
`lightspeed.py::summary`, the public serializers/action descriptions in
`patches/sts_lightspeed_python_api.patch`, and the v5 multi-selection presentation.
Tests use fixtures transcribed from those public serializers and static native
sources (`Map.cpp`, `PrintHelpers.cpp`, `GameContext.cpp`, `Cards.h`), plus the
captured raw public observation in `tests/fixtures/native015_boss_map_case.json`,
without loading game worlds or datasets. That fixture preserves its original
public hash and external source metadata; only its `observation` enters the
encoder, so the source seed stays outside policy input. The worktree's
environment stays unchanged.

## Encoding and batch ABI

Call `encode_observation(raw_public_observation)` with the complete dictionary
containing `state`, `text`, `phase`, and `legal_actions`. An empty menu raises
`ValueError`, including terminal observations; collectors should handle terminal
states before encoding. Every current legal action produces exactly one candidate
in input menu order. `action_indices: int64[A]` preserves displayed routing indices
and is excluded from the model batch. It is never a learned identity.

The shared study adapter hides OOC potion drink/discard choices before inference.
Before filtering, it must extract only the original RAW PUBLIC menu's potion
names and `(slot N)` labels into the adapter-only top-level field
`adapter_public_ooc_potions: [{'slot': N, 'name': NAME}, ...]`, identically for all
policy heads. The encoder reads only `slot` and `name` from that field, alongside
public inventory text; extra record fields are ignored. If the field is absent,
direct raw-observation calls recover the same information from the public menu.
If present (even empty), the field takes precedence over menu recovery. This
retains public inventory knowledge without restoring filtered choices. Adapter
extraction/filtering is the D/orchestrator integration responsibility.

`EncodedEntityObservation` contains NumPy arrays. `collate_observations(list,
device='cpu')` returns tensors with an added batch dimension and padding. All
IDs/references/types are int64, numeric arrays float32, and masks bool. No entity,
action, pile, enemy, hand, power, or history cap is applied.

| Field | Unbatched shape | Meaning |
|---|---|---|
| `entity_ids`, `entity_types` | `[N]` | Namespaced identity; fixed entity role |
| `entity_aux` | `[N,3]` | Additional categorical IDs: enemy intent, card type/rarity, stance, boss/phase/room, or screen/purpose/task |
| `entity_numeric` | `[N,32]` | 16 values followed by their 16 availability flags |
| `entity_mask` | `[N]` | Real entities |
| `power_owner`, `power_ids`, `power_amounts` | `[P]` | Entity index, namespaced power ID, normalized amount |
| `power_amount_available`, `power_mask` | `[P]` | Finite amount available; real power record |
| `recent_ids`, `recent_types`, `recent_mask` | `[R]` | Ordered public recent-action sequence |
| `recent_numeric` | `[R,32]` | Played-card properties, when available |
| `action_ids`, `action_types` | `[A]` | Card/relic/potion/room identity or `action:*`; fixed semantic action kind |
| `action_aux` | `[A,3]` | Card type/rarity and selection task |
| `action_numeric`, `action_text` | `[A,32]` | Numeric values/availability; deterministic wording bag |
| `action_source`, `action_target` | `[A]` | Entity index, or `-1` for absent/unresolved |
| `action_source_members`, `action_source_mask` | `[A,M]` | All multi-select members, or the ordinary single source |
| `action_mask` | `[A]` | Real candidates |

Padding uses ID/type zero, numeric zero, masks false and references `-1`. Empty
power/history axes have one masked padding record in batches. The collator also
provides `entity_counts`, `power_counts`, `recent_counts`, `action_counts` `[B]`
and `action_source_counts` `[B,A]`. Coverage and displayed indices stay outside the
neural batch. Models require prefix padding masks and valid public-owner references.

Entities cover run/player, each enemy, hand slots, draw/discard/exhaust multisets,
deck multiset, relics, potion slots, legal map destinations and screen/event context.
Each available pile/deck has a total/unique-count header in addition to grouped
card records. Grouping preserves name, upgrades, public `=N` counter, base/current cost, free-once,
retain, and type/rarity when present. Sorting is canonical and no neural position
embedding is used for unordered inventories. Hand/enemy/potion slots are public
numeric features. Recent plays have card IDs/upgrades/cost flags and use a GRU in
recorded order; history has no cap. Coverage reports its total, zero truncation,
and `recent_unparsed`: the number of recorded entries mapped to `action:unknown`,
counting repeated labels individually. Known plays and `end turn` do not increment
it. HEADBUTT-style selections without `hand [` and keep-hand/no-card labels still
retain their positions as unknown tokens; this counter does not change history
parsing, numeric features, or potion-history identity handling.

## Numeric conventions

For each supplied finite number, the stored value is
`clip(value / scale, -10, 10)` and its availability flag is one. Missing, null,
non-numeric and nonfinite values become zero with availability zero. Boolean scale
is one. Clipping changes magnitudes, never token/card multiplicities; exact raw
pile counts also remain in coverage. X costs remain `-1`. All raw costs below
`-1` normalize to the unplayable class `-2` in entities, matching and labels:
native curses have raw cost `-3`, statuses `-2`, and both print `unplayable`.
Card type remains a separate categorical field. Changed/free costs stay distinct.

Columns below are zero-based value columns; availability is always column `i+16`.
Unused columns and availability flags are zero.

- Run: act/4, floor/60, HP/100, max HP/100, gold/500, ascension/20,
  ruby/emerald/sapphire booleans. Boss identity comes from the public header.
- Player: HP/100, max HP/100, block/100, energy/10, energy-per-turn/10,
  draw-per-turn/10, orb-slots/10, gold/500, turn/20, cards/attacks/skills played
  and cards discarded this turn each /20, named-power count/100.
- Enemy: public slot/5, alive, HP/100, max HP/100, block/100, intent damage/50,
  intent hits/10 (`-1` means nonattacking), named persistent-power count/100.
  Intent is a categorical `intent:*` ID in `entity_aux`, never a float ID.
- Card: count/100, upgrades/10, base cost/5, current cost/5, free-once, retain,
  public hand slot/10 when applicable (columns 0–6); public `=N` amount/100 is
  column 11, with availability in 27. The amount is parsed independently of the
  upgrade; native output orders them `Rampage=8+`, `Ritual Dagger=15+`, and
  `Genetic Algorithm=5+`. Combat public names expose all three; OOC Card/deck
  printing exposes Ritual Dagger's amount. Missing amount differs from zero.
  OOC deck text lacks cost/type fields;
  those remain unavailable. Relic: live public counter/100, counter-is-live flag.
- Potion: public slot/10 and empty flag. A known occupied potion whose identity
  and slot are unobservable has ID PAD, slot availability zero, and empty=false.
  Missing capacity stays unavailable.
- Map: row/15, burning-elite flag, `log1p(feasible_path_count)/10`, then min
  and max path counts /15 for MONSTER, ELITE, REST, SHOP, EVENT, TREASURE. Bounds
  use dynamic programming on the public DAG; alternate branches are never added
  together as a single path. Native row-14 edges to the absent `(3,15)` node
  terminate at the implicit boss. At `cur_y=14`, the native menu instead labels
  its transition `choose map node x=0 advance to boss`: that `x` is an action
  parameter, not a physical column. Explicit `advance to boss` choices reference
  the same BOSS map entity/path at `(3,15)` regardless of their `x` parameter;
  they never create a node at `(x,15)`. Ordinary map choices still resolve their
  physical `(x,cur_y+1)` destination. Missing intermediate destinations or other
  absent terminal coordinates still raise. Only destinations in the current
  legal menu appear; no hidden node or encounter information is inferred.
- Event context: allowed `publicEvent()` fields in order: event_data, phase,
  upgrade_one, cleanup_is_remove, hp_amount0/1/2, gold_loss, gold. Booleans use
  scale one, others /100; chosen/required selection counts /10 follow.
- Inventory context: hand/draw/discard/exhaust/deck totals /100; potion records /10,
  capacity /10, relic count /100, map/history availability flags. Absent inventory
  totals have availability zero, distinct from known empty totals.
- Action cards reuse card columns. Multi-selection uses column 0 = member count/10;
  map actions use column 0 = burning flag. All actions may add price/gold amount/500
  in column 7 and scalar displayed damage total/100, per-hit/100, hits/10 in 8–10.

Each power keeps its name and owner; basic enemy strength/weak/vulnerable/poison
fields are included without double-counting matching names in `powers`. Amounts
use /100 and the same clipping. Invalid amounts retain name/owner with an explicit
unavailable flag and a parser diagnostic. Name, amount and availability are fused
nonlinearly before owner-specific scatter-add, preserving which power has which
amount. Powers from separate enemies never share an owner pool.

## Parsing, coverage and vocabulary

Only enumerated public state fields and the adapter-only public potion list above
are accessed. Inventory text reads anchored
`Deck:`, `Relics:`, `Potions:` lines; context reads the boss header and selection
purpose/count sentence. The full observation text is never hashed. Injected seeds,
bits, native objects, search values and unrelated diagnostic fields are ignored.

Deck `(N): {name, ...}` and relic `{name:counter, ...}` formats are strict when
present, including declared deck-count checks. Empty/none/braced-empty forms are
supported; absence is marked unavailable. Malformed inventory raises `ValueError`.
Unknown but structurally valid names become UNK, rather than guessed known names.
OOC potion text prints slots `0 .. potionCount-1` in actual slot order, including
the `EMPTY_POTION_SLOT` sentinel, which encodes an empty slot. Its list length is
the occupied-potion count, not capacity. Higher occupied slots are recovered from
raw public `(slot N)` drink/discard labels or the retained adapter field. Repeated
labels for the same slot/name merge; conflicting slot names or text/menu claims
raise. Identical names in different slots stay distinct. No slots are filled in
between known positions and no capacity is inferred. `WE_MEET_AGAIN` can suppress
all recovery labels: any remaining occupied count gets records with unavailable
identity/slot, never invented empties. Its event potion name has no slot and is
not used to guess one. Absent text does not imply an empty inventory.

Play/selection matching uses name plus upgrade and every displayed mechanical
attribute, including explicit free-once versus ordinary copies. It selects the
first fully identical copy. OOC deck matching allows the inventory's absent static
type. Combat play labels omit base cost even when it differs: if candidate matches
have distinct underlying public properties, the encoder returns source `-1` and
increments `ambiguous_sources`. Pile selection labels expose base-cost differences
and resolve them. Duplicate enemy targets resolve the explicit `[enemy i]` slot;
an unresolved/ambiguous target raises. Rewards, purchases and discovery offers keep
identity IDs without requiring ownership or fabricating source references.
`Choose a card to OBTAIN` (including The Library's 20 new offers) bypasses deck
source matching and unresolved-source diagnostics; REMOVE/UPGRADE/TRANSFORM and
other operations on owned cards retain their deck joins.

V5 multi-select labels resolve every public `hand [i]` member. Older raw nonempty
native mask labels are known to print incorrect indices: they receive a parser
fallback bag and no invented membership. No bits are consulted. Generic unknown
options remain candidates with a fallback diagnostic. Only event/unknown/legacy
fallback action wording uses a 32-bin signed BLAKE2 word/bigram bag, normalized by
term count. Known action families use categorical/numeric features. Structured
multi-target/random computed-damage annotations are not separately encoded;
scalar damage is. Event option ordinals are not stable semantic IDs.

Coverage reports lookups and UNKs per namespace, parser fallbacks, unresolved and
ambiguous sources, unresolved targets, inventory counts/availability, unresolved
potion slots, history totals/truncation/unparsed entries, and entity/action totals.
`recent_unparsed` is separate from `parser_fallbacks`, which tracks current-menu
parsing fallbacks (and unavailable power amounts), not recent-history parsing.
`action:unknown` is a vocabulary token, so these history entries also do not imply
a namespace UNK. Track these during collection; vocabulary IDs remain frozen
before training.

The JSON embeds static public name lists, aliases, source/version metadata, original
header hashes and bootstrap checksum. The permitted archive vocabulary supplied
names only; no model, behavior, observations or labels were reused. Runtime needs
only the committed JSON. PAD=0 and UNK=1. Names normalize case/spaces/underscores/
punctuation; card `=N` and following `+N` suffixes become numeric public amounts
and upgrades, independently of canonical card identity. Public event aliases map
`BONFIRE_SPIRITS` to Bonfire Elementals and `DESIGNER_IN_SPIRE` to Designer.
Regeneration is deterministic:

```
PYTHONPATH=src .venv/bin/python scripts/rl/build_entity_vocabulary.py --output /tmp/entity_vocabulary.json
```

`--native-source` reads static simulator name arrays plus glossary keys;
`--bootstrap-json` accepts a static name vocabulary. Neither fits observations.
Changing names/IDs changes the file SHA and requires a new compatible checkpoint.
`pyproject.toml` includes `sts_policy/entity_vocabulary.json` as package data;
the focused tests check the TOML declaration and source resource's checksum.
No package build/install is needed for these repairs. The regenerated B2 SHA is
`a61e9c57a3a99d3514b14929897d85956f211564e6aafc8b077ada44afa5f132`.

## Model and validation

`EntityActorCritic(width=128, layers=2, heads=4, dropout=0, phase='combat'|'ooc')`
returns `(logits[B,A], values[B])`; padded logits are negative infinity. Each phase
uses a separately instantiated model. Actor and critic have disjoint embeddings,
numeric MLPs, owner-power fusion, GRUs and transformer towers (feed-forward width
2×width, nested tensors disabled). The actor scores action identity/type/numeric/
text, referenced source-members/target embeddings and pooled state. The critic has
its own menu embedding pool so unowned offers/prices affect value estimates.
`actor_logits()` executes no critic; `critic_values()` executes no actor.

Rows are unpadded and evaluated individually, including trimming membership width.
This intentionally trades fused-batch throughput for exact CPU logits/greedy argmax
independent of batch partners and padding. No process-global thread settings are
changed by the implementation. Use `.eval()` for inference; nonzero configured
dropout is stochastic in training. Accelerator bitwise reproducibility is untested.

Checkpoint format is exactly `{'model_state': model.state_dict(), 'metadata':
model.metadata()}`. Architecture is under `metadata['model_config']`.
`policy_from_payload` rejects missing or mismatched kind, encoder version,
`encoder_sha256`, vocabulary size or vocabulary SHA before loading weights,
strictly loads weights, and returns a CPU model in its default training mode.
`ENCODER_SHA256` is the SHA-256 of the exact installed `entity_encoding.py` source
bytes, computed at import; it includes the semantic `ENTITY_TYPES`/`ACTION_TYPES`
index lists as well as all parsing and collation code. Checkpoints require both
this exact source identity and the existing vocabulary identity. Even a
formatting-only source change invalidates the source SHA; a fresh process reads
the current installed bytes. The encoder version `public_entities_fresh_v1` and
model kind `entity_fresh_v1` remain unchanged because B4 preserves all neural
inputs and model arithmetic. Future semantic encoding changes must update the
version and document migration; the SHA enforces exact source compatibility even
if a version bump is missed. Prefit prototype checkpoints without this identity
are intentionally rejected; no trained student exists and no conversion is provided.
Metadata includes architecture, identities and parameter count; this is not a legacy
checkpoint converter. `actor_parameters()` and `critic_parameters()` are disjoint;
critic-only optimization leaves actor weights/logits unchanged.

Focused validation command:

```
PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_entity_encoding tests.unit.test_entity_model
```

Tests retain the original 27 cases and add native-format regressions for full maps
and boss transitions, OOC potion gaps/raw and filtered menus/unobservable slots,
counter names/grouping, curse/status/X/changed costs, Library offers, event aliases
and package data. B3 checks the captured row-14 observation unchanged against its
original public hash, its raw three-action menu, and its filtered single-choice
menu with `adapter_public_ooc_potions` retained. It verifies BOSS source context,
one candidate per legal action, original routing indices, nonspatial boss action
aliases, and rejection when a normal destination is removed from the real map.
Tests also include fresh-process hash-seed/source-identity determinism, exact CPU
checkpoint round trip, padding/batch invariance, private-field exclusion, parser/multiset/slot tests,
critic-only update isolation, and a tiny soft-target plumbing fit with one CPU thread
and a 30-second bound. B4 adds recent-history coverage regressions, installed-source
SHA checks, missing/mismatched checkpoint identity rejection before weight loading,
and exact batch-padding/save-load checks in `.eval()` under `torch.inference_mode()`
at the default 128×2 size with one CPU thread. The latter verifies actual fused
transformer-kernel calls for both actor and critic; the gradient-enabled checks
and isolated actor/critic tests remain. It checks exactness within inference mode,
not equality with gradient-enabled training arithmetic. The default model has
1,683,714 parameters. A five-iteration
local fixture benchmark (24 entities, 8 actions, one CPU thread) measures actor and
full forward latency in test output; this is a small implementation check, not a
collector-throughput study. No training experiment, simulator rollout, dispatch
integration, environment edit or prior-policy import is part of B1.
