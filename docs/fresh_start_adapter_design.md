# Fresh-start adapter contract

`policy_observation` derives OOC potion inventory from the **original raw public
legal descriptions**, before the potion and legacy dead-action filters run.
`drink potion NAME (slot N): ...` and `discard potion NAME (slot N)` supply only
`{"slot": N, "name": NAME}` to the top-level `adapter_public_ooc_potions` list.
Entries are deduplicated by slot and sorted by ascending slot; conflicting names
for one slot raise an error. Duplicate names in different slots remain distinct.
This recovers occupied slots omitted by the native OOC text after an
`EMPTY_POTION_SLOT` gap. No bits, action indices or diagnostics enter this list.

The adapter always rederives the field, ignoring any pre-injected value. Without
visible slot labels it emits an empty list, including at `WE_MEET_AGAIN` when
potion slots are unavailable. Text and event fields do not justify invented
slots. The supplied B2 encoder consumes this optional context; its source/model/
vocabulary are external test inputs and are not changed by this adapter work.
The context identity is
`PUBLIC_CONTEXT_ID = "ooc_potion_public_menu_inventory_v1"`, recorded as
`StudyPolicy.identity["public_context_id"]` and in launch policy receipts.

Raw observations, nested state, and legal entry objects are immutable. OOC model
input uses a new top-level mapping; retained actions keep their original objects,
order and native indices. The potion filter removes only OOC drink/discard
entries; buy/take/event choices survive. Combat returns the exact original
observation object. World shards, public hashes, teacher inputs and fight-bank
records retain raw observations; augmentation is internal to policy inference.
Call the adapter with raw observations, not a previously filtered model input.

Inference is singleton CPU `eval()` under `torch.inference_mode()`, actor only.
Entity `actor_logits` receives **one dict batch**; legacy candidate and composite
`actor_logits` receive **the unpacked tuple batch**. Greedy selection uses exact
logit argmax with the first tie winning, then maps the position back to the
original, possibly gapped native action index. Forced choices bypass the model.
Checkpoint identity changes, unknown kinds, incompatible encoders/vocabularies,
wrong phase heads, invalid weights and nonfinite logits reject explicitly;
loading errors never trigger champion fallback. The legacy composite champion
supplies only phases with no requested replacement.

Combat labels use `canonical_rng_resampled_native_v2`: canonical RNG native-assisted
search, **not certified public-only**. No native success or throughput claim is
made by the generated-checkpoint tests. Native015 parity and collector throughput
remain pending. D1 training modules are separate and will be combined after review.

Fight-panel salts change only the battle AI, monster HP, shuffle and card-random
streams. `BattleContext::init` initializes those from `gc.seed + gc.floorNum`,
but copies misc and potion streams from the already-entered context. Those two
streams remain fixed across salts. This describes panel start salting, not the
teacher's separate `BattleContext.reseed_rngs` protocol; teacher sample IDs, seeds,
simulations, repeat split and target histograms are unchanged.

Each persistent child retains one prepared start, keyed by the full immutable
bank row (source, provenance, both public hashes, world/start/prefix/action and
task identity included). It replays the prefix and executes the saved entering
action once, saves a pre-initialization `GameContext.copy()`, then verifies the
unsalted first-combat hash once. Scheduling keeps every salt and both policies
for that start on that child. Every fight initializes a new base copy; histories,
context references, episode counters and the native exit callback restore in
`finally`, including on errors. The cache is bounded to one start and is lost on
child replacement after a native fault. Every policy/salt remains a finite RPC.

Scratch exit sets writable `GameContext.outcome` to the battle victory/loss and
leaves BATTLE without running rewards callbacks. A terminal outcome prevents map
move requests at boss/Secret Portal row 15. Canonical read-only native evidence:
`external/sts_lightspeed_rl/src/game/GameContext.cpp` sets row 15 on boss entry;
`src/sim/search/GameAction.cpp:getAllActionsInState` returns immediately for a
decided outcome; `bindings/slaythespire.cpp` exposes writable `outcome` and
`GameOutcome.PLAYER_VICTORY/PLAYER_LOSS`. `src/combat/BattleContext.cpp:65-75`
defines the stream initialization described above. The unit fake now rejects any
undecided row-15 post-fight map lookup, for both victory and defeat. It does not
establish real native parity; root must cover actual boss/Portal/bank starts.

Native exceptions are contained at explicit binding methods in the research
child (`fresh_native.py`), including context construction/copy/init, action
execution/enumeration/descriptions, public observations and search. C-raised
RuntimeError/ValueError/IndexError/OverflowError become `SimulatorError` with the
operation and original cause; the worker records the child traceback and fault
receipt, kills that child, and continues the cohort. Python traceback frames
(including callbacks), TypeError API bugs and inference errors still propagate
as programming errors. The existing draw-order-search ValueError abstention is
preserved. No blanket reset/step/inference exception conversion is used.

A horizon-cut world is censored and quarantined, with partial labels/bank rows
discarded, and the worker moves to the next exact seed. STOP, absolute deadline,
low disk and the existing between-world label budget still stop collection. A
cut leaves the cohort incomplete and prohibits confidence claims, even when
every later requested seed finishes. Drink/discard behavior fields are explicitly
`combat_potion_drink` / `combat_potion_discard`: OOC choices were hidden before
their denominators. EVENT1–3 fights have collection sampling coverage; panel
targets follow the separately frozen world counts below. Existing group probabilities are unchanged; root selects final frozen
probabilities explicitly. Progress reports `parent_peak_rss_bytes`; this is not
an aggregate of native children.

Banks now accept one or more **completed receipted directories** (`--bank A B`).
Optional `--bank-world-start N --bank-n K` pins an externally requested grid.
Unreceipted JSONL/individual shards and legacy runs lacking per-shard identities
reject. New collector summaries bind every immutable gzip shard's SHA/bytes/path;
completion receipts bind the summary. `fresh_cohort.validate_cohorts` checks exact
declared seed ranges, complete statuses, no duplicate/missing/extra/quarantined
worlds, hashes, raw row/source/provenance/prefix alignment and uniform identities.
Only config `out`, `n`, `world_start`, `workers`, `deadline_utc` may differ between
process shards. Source archive SHA/bytes, every other config field, full policy/
filter/inference identity, runtime, environment and native provenance must agree.

Use `scripts/rl/fresh_merge.py --inputs A B --world-start N --n K` to validate;
add `--out MERGED` to publish byte-identical world copies, a merged summary,
receipt and completion. The merged summary keeps completed per-world records for
`fresh_statistics.compare_games` or `fresh_compare.py games --student
MERGED/summary.json --champion ... --cohort exact-seeds.json`. Pair the exact
requested cohort, never intersections. Original launch receipts and source
archive SHA remain in `cohort_sources` / `science_identity`. Merging does not
repair or relabel incomplete/faulted runs.

Independent C3 findings were accepted. Two qualifications: the parent observation
comparison could not validate native nonmutation and was removed in favor of the
existing child hash check; the injected adapter field stays untouched in combat
to preserve exact observation identity, with native structured combat potion
slots taking precedence in B2. The read-only `panel_native_bank_pilot_v1` predates
per-shard hashes and is rejected by the strict loader; root can use its original
rows directly for native parity fixtures. No historical artifact is rewritten.

## C5 panel decision: freeze before fitting or scoring students

The new selection protocol is `one_start_per_world_group_hash_v1`. For each
target group separately, rank eligible **unique source worlds** by SHA-256 of
canonical JSON `{namespace, stage: "world", group, world_seed}`. Choose the first
requested N worlds, then select exactly one start within each chosen world/group
by an independent hash of `{namespace, stage: "start", group, world_seed,
start_id}`. The fixed namespace is
`fresh_panel_one_start_per_world_group_hash_v1`; canonical JSON sorts keys, uses
compact separators and ASCII escaping. Integer world/start keys break hash ties.
No outcomes, baseline HP, model scores or checkpoint keys enter either ranking.
Reversing input rows or target order leaves the manifest unchanged. A source
world may occur in several groups; it remains the uncertainty cluster.

The approved default world targets are BOSS1:100, BOSS2:300, BOSS3:100,
ELITE2:150, ELITE3:100, MONSTER2:100, MONSTER3:100, EVENT1:52, EVENT2:4.
Root's validated `fight_bank_1024_v1_merged` catalog has respectively 1004, 766,
195, 651, 167, 966, 326, 52 and 4 eligible worlds. EVENT3 is absent on those
1,024 source worlds: it is reported in `absent_groups`, with no default target
and no coverage gate. Additional EVENT diagnostic counts must be supplied
explicitly through `--starts-per-group` (this replaces the whole target mapping).
Targets must be positive integer world counts. If a requested count exceeds
eligible worlds, `coverage_sufficient` is false and `coverage_shortfall` reports
the deficit; many starts from one world cannot hide the shortage.

These targets yield **1,006 starts × 9 salts × 2 policies = 18,108 results per
comparison**, including 300 distinct Act 2 boss source worlds. Salt -1 is the
exact unsalted control; salts 0–7 are the eight repeats. Metrics still average
within each source world, then give worlds equal weight within each group.
Uncertainty clusters every salted repeat from a source world together; unsalted
controls are reported separately. Previous whole-world-retention pilot outputs
are archived engineering evidence only and must remain untouched.

Freeze a new manifest **before any student fit or score**; then pass that same
file to every old/new comparison. For example, using the approved defaults:

```sh
PYTHONPATH=src .venv/bin/python scripts/rl/fresh_panel.py \
  --freeze-grid-only --bank /path/to/fight_bank_1024_v1_merged \
  --resamples 8 --out /path/to/new_panel_grid.json

PYTHONPATH=src .venv/bin/python scripts/rl/fresh_panel.py \
  --bank /path/to/fight_bank_1024_v1_merged --grid /path/to/new_panel_grid.json \
  --resamples 8 --checkpoint /path/to/champion.pt --combat /path/to/combat.pt \
  --out /path/to/new_comparison
```

The freeze-only command loads no model, launches no native worker, and writes
one new read-only JSON file, refusing overwrite. It validates the completed bank
receipts/shards. The reusable `freeze_panel_grid`, `save_grid` and `load_grid`
helpers support the same workflow. Custom targets/resamples must be repeated
exactly on comparison commands. `--grid` rejects unknown protocols, noncanonical
hashes, changed targets/resamples, different bank inputs/shards/provenance, changed
start IDs, public/first-combat hashes or replay specs. Recomputed selection is
only a validation check; the supplied artifact is never repaired or replaced.
Paths form part of the bank's file identities, so relocating a bank rejects.
Without `--grid`, the API still computes selection before loading a policy;
root's student comparisons must use the pre-frozen file.

`grid_hash` covers the whole model-free manifest except `grid_hash` itself.
The manifest records all selected starts, worlds and salt cells, the sampling
specification, eligible world counts, targets, coverage and validated bank file
identities. Effective run config and launch receipt bind `grid_hash`,
`selection_protocol`, `bank_identity`, targets, coverage and `hp_definition`
**before `begin_run`**. A supplied grid file's SHA/bytes/path is `grid_input` and
is also a launch data input. Each run writes an identical `grid.json` manifest;
summary and completion expose the same grid hash and coverage fields. No receipt
or summary hash participates in the grid hash, so there is no circular hash.

Each individual result/quarantine JSON and summary record stores only
`source_ref = file_identity(receipt.json)` and `grid_hash`, instead of duplicating
the full launch receipt. The complete source lives at the top of `receipt.json`,
`summary.json` and `completion.json`. World/start/salt/policy/group and first
combat hashes retain the shape consumed by `fresh_compare.py panel --grid ...`.
Individual result files remain sufficient to recover paired statistics with the
manifest and referenced receipt. `results_complete`, `missing_results` and
`invalid_results` distinguish execution gaps from `coverage_sufficient` and
`coverage_shortfall`. Incomplete grids mark every reported group/metric
`diagnostic_only: true` and prohibit confidence claims, even when some groups
have all their pairs.

The machine-readable HP definition is
`salted_entry_minus_pre_end_of_fight_relic_healing_v1`. `entry_hp` is measured
from the initial **salted public combat observation**, including combat-specific
player HP; `source_entry_hp` preserves the bank value for trace inspection only.
`hp_after` is player HP at battle exit **before end-of-fight relic healing**;
death uses HP zero. `hp_lost = entry_hp - hp_after`. The bank's baseline HP is
post-healing and is not expected to match panel HP. Unsalted parity against bank
baselines should compare win/loss, not those HP figures. Combat execution,
salting, callbacks and the prepared-start cache are otherwise unchanged by C5.

C5 builder validation uses pure fakes, synthetic catalogs and receipted tiny
banks only. Root must run native checks after integration; the builder does not
launch a real full panel or edit the bank, running teacher collector, core,
patches, vocabulary or source-model dependencies.
