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
their denominators. EVENT1–3 fights now have default sampling coverage and panel
targets. Existing group probabilities are unchanged; root selects final frozen
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
