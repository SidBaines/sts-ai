# COMP-020 — Semantic retargeting plan (Tier 0 + Tier 1)

**Status:** in progress (2026-08-11)
**Branch:** `feat/semantic-turn-targets`

## Motivating diagnostics (2026-08-11, measured on frozen artifacts)

1. Performance ladder on dev57 (top-1 vs teacher consensus): random ≈12, constant
   index-0 = 21, base E4B = 22–26, best SFT (COMP-012) = 28, a 25-line rule
   policy (lethal > Vulnerable-applier > max-damage > defend) = **32 = the
   frozen gate**, teacher self-agreement ceiling ≈ 46–48 (per-query vote
   agreement with hidden-order consensus = 81.8%).
2. ~16/59 dev rows are near-ties (aggregated visit margin < 0.2) and are mostly
   first-action-of-turn ordering artifacts: the teacher's *turn set* is stable
   across queries while the first action flips. Strict top-1 wastes ~25% of its
   mass on unlearnable order noise. Visit distributions and per-query
   `best_sequence` turn plans are already stored in `teacher_queries`.
3. Phantom-power simulator UB contaminates dev at ~5× the train rate: 17/60 dev
   states across windows `seed_104_r0_w0`, `seed_108_r0_w0`, `seed_120_r0_w0`
   (enemy Metallicize 4, enemy Regen 3, player Thorns with no source) vs 8/150
   train states (windows `seed_103_r0_w0` enemy Regen, `seed_129_r0_w0` player
   Buffer).
4. 16 attack card types never get a `(deal N)` annotation — exactly the
   computed-damage ones (deliberate skip-list in the C++ patch:
   `hasSpecialDamageFormula` + untargeted AoE + X-cost).
5. Targets are bare menu indices (`{"action_index":N}`), which discard all
   pretrained card semantics and make menu position the easiest shortcut
   feature (COMP-014/015/016 index-collapse results).

## Frozen design decisions

- **New observation version `combat_public_v3`** = v2 + computed damage
  annotations + TURN MATH derived lines. v2 output stays byte-identical
  (annotations gated behind a new C++ flag defaulting off).
- **Damage annotations are computed in C++** (patch), using
  `bc.calculateCardDamage` and per-card base formulas mirrored from upstream
  play logic, verified by a sim-oracle integration test (predicted damage ==
  observed HP delta). No Python damage math.
- **TURN MATH lines are computed in Python** from displayed values only
  (aggregations, not new game rules), in a pure unit-testable module.
- **New output contracts** `action_text` and `turn_plan` alongside the legacy
  `action_only`. Menu body format (numbered lines) is unchanged so the target
  change is isolated; only the instruction block and the completion change.
- **First training comparison runs on v2 prompts** (byte-comparable with
  COMP-012's control): isolates the target-representation change. A `+v3`
  prompt arm follows once the target result is known.
- **Soft targets are realized at the dataset level** (per-pass seeded sampling
  from the aggregated visit distribution), NOT via loss weighting (COMP-018
  showed fixed loss reweighting destroys learning).
- **Metrics:** primary = mean visit-share regret + tie-aware top-set agreement
  on dev57, reported for {all, clean} where clean excludes quarantined
  windows. Strict top-1 stays as legacy secondary. Tie ratio τ = 0.8.
- **Frozen artifacts are never mutated** — quarantine and rescoring are
  additive sidecars.
- Rationale distillation (Tier 1.4) is **deferred** until the first
  action_text/turn_plan results are in.

## Repo conventions (include in every implementer prompt)

- Run everything with `PYTHONPATH=src .venv/bin/python`; the package is not
  installed.
- Tests: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`
  (the `-t .` is required). Unit tier (`tests/unit`) must not require the
  built simulator; integration tier (`tests/integration`) is gated by
  `@requires_simulator` (see `tests/CLAUDE.md` and existing tests for the
  decorator/conventions).
- Style: `from __future__ import annotations`, type hints, frozen dataclasses,
  small pure functions; mirror neighbouring code.
- No new dependencies.
- Deterministic, fail-closed CLIs: refuse to overwrite `--out`, hash inputs
  with sha256, stable key order in emitted JSON (see
  `scripts/analyze_comp015.py` for the house style).
- Unit tests must use small synthetic inline fixtures, never files under
  `data/` (gitignored, machine-local).
- Do NOT run `git commit` — the orchestrator commits after review.

---

## Task A — state sanity quarantine

**Files:** `src/sts_ai/state_sanity.py` (new), `scripts/audit_state_sanity.py`
(new), `tests/unit/test_state_sanity.py` (new).

Detect physically impossible ("phantom") powers in serialized combat states,
caused by a known simulator uninitialized-memory bug, and publish a quarantine
sidecar listing affected states/windows.

API (in `state_sanity.py`):

```python
@dataclass(frozen=True)
class SanityFinding:
    side: str        # "enemy" | "player"
    power: str
    amount: int
    reason: str

def phantom_power_findings(state_text: str, *, task_id: str) -> list[SanityFinding]
```

Rules for `task_id == "gremlin_nob"` (any other task_id → `ValueError`, fail
closed):

- Parse the `Enemies:` block (lines between `Enemies:` and the next section
  header, `Hand:` or `Piles:`). Enemy powers appear as `Name N` tokens after
  the intent descriptor (comma-separated; also the `(no attack) Name N` form).
  Whitelist for Gremlin Nob: `{Enrage, Strength, Vulnerable, Weak}`. Any other
  `Name N` power → finding with reason `enemy_power_impossible_for_encounter`.
  Be careful not to mis-parse `HP a/b`, `block N`, intent `(deal N)` text as
  powers.
- Parse the `Player powers:` line (`none` or comma-separated `Name N`):
  - `Thorns` without `Bronze Scales` in the `Relics:` line → reason
    `player_thorns_without_source`.
  - `Buffer` (any) → reason `player_buffer_without_source` (no Ironclad-pool
    source exists in this data).
  - Flag NOTHING else on the player side. In particular player Metallicize,
    Rupture, Vulnerable, Weak, Strength, Flame Barrier are all legitimately
    reachable (played Power cards leave no trace in the listed piles — do not
    infer contamination from missing cards).

CLI `scripts/audit_state_sanity.py`:

```
PYTHONPATH=src .venv/bin/python scripts/audit_state_sanity.py \
  --labels <rows.jsonl> [--labels <more.jsonl> ...] \
  --task gremlin_nob --out <quarantine.json>
```

- Each labels file: JSONL rows containing at least `state_text`, `window_id`,
  `public_state_hash`, `world_seed` (e.g. teacher label / audit files).
- Output JSON (stable key order, fail if `--out` exists):

```json
{
  "kind": "state_sanity_quarantine",
  "version": 1,
  "task_id": "gremlin_nob",
  "inputs": [{"path": "...", "sha256": "...", "n_rows": 0}],
  "findings": [{"public_state_hash": "...", "window_id": "...", "world_seed": 0,
                 "findings": [{"side": "...", "power": "...", "amount": 0, "reason": "..."}]}],
  "quarantined_windows": ["..."],
  "quarantined_state_hashes": ["..."],
  "summary": {"n_rows": 0, "n_states_flagged": 0, "n_windows_flagged": 0}
}
```

- `findings` sorted by (window_id, public_state_hash); list fields sorted;
  dedupe identical rows appearing in multiple inputs by public_state_hash
  (identical findings required, else error).

Tests (synthetic state_texts modelled on the real serializer format — copy the
shape from a real `state_text`, e.g. the block quoted in
`docs/comp020_semantic_retarget_plan.md` motivating section or existing test
fixtures): clean Nob state → no findings; enemy `Metallicize 4` and `Regen 3`
→ enemy findings with correct amounts; player Thorns without Bronze Scales →
flagged; with Bronze Scales → not flagged; player Metallicize → not flagged;
player Buffer → flagged; unknown task → ValueError; CLI: end-to-end tmp-file
run, deterministic byte-identical output across two runs, refuses existing
`--out`, multi-input dedupe.

---

## Task B — regret/turn-set metrics + per-row eval output + rescore CLI

**Files:** `src/sts_ai/teacher_metrics.py` (new),
`tests/unit/test_teacher_metrics.py` (new), modify
`src/sts_ai/teacher_action_eval.py` and `scripts/score_teacher_actions.py`
(additive), `scripts/rescore_teacher_report.py` (new),
`tests/unit/test_rescore_teacher_report_cli.py` (new).

### B1. `teacher_metrics.py`

Consumes audit-style rows (the schema of
`data/competence/comp_008/audit/holdout_turn_first.jsonl`: fields
`public_state_hash`, `window_id`, `turn`, `legal_actions`
(`[{index, bits, description}]`), `teacher_queries` — each query has
`teacher_vote` with `abstained: bool` and `displayed_action_visits:
dict[str,int]`, and `search.best_sequence: [{bits, description, turn}]`), and
model choices.

```python
@dataclass(frozen=True)
class StateVisitStats:
    public_state_hash: str
    window_id: str
    turn: int
    total_visits: int
    visit_share: dict[int, float]          # displayed index -> share of all visits
    consensus_action_index: int            # argmax of aggregated visits (tie -> lowest index)
    top_set: tuple[int, ...]               # indices with share >= tie_ratio * max share, sorted
    turn_set_descriptions: frozenset[str]  # union over non-abstaining queries of
                                           # best_sequence descriptions with entry turn == row turn
    margin: float                          # max share - second share (0.0 if single action)

def state_visit_stats(audit_row: dict, *, tie_ratio: float = 0.8) -> StateVisitStats
def score_choice(stats: StateVisitStats, chosen_index: int,
                 chosen_description: str | None = None) -> dict
def build_metrics_report(per_row_choices: list[dict],
                         audit_rows_by_hash: dict[str, dict], *,
                         quarantined_windows: frozenset[str] = frozenset(),
                         tie_ratio: float = 0.8) -> dict
```

Semantics (all fail closed with ValueError on malformed input):

- Aggregate `displayed_action_visits` over non-abstaining queries only; zero
  total visits → error.
- `score_choice` returns `{strict_top1, in_top_set, in_turn_set,
  regret_visit_share, chosen_share, margin}` where `regret_visit_share =
  max_share - chosen_share`; `in_turn_set` matches the chosen displayed
  action's description (resolved from the audit row's `legal_actions` when
  `chosen_description` is None) against `turn_set_descriptions` by exact
  string equality (`end turn` participates like any description).
- `build_metrics_report`: `per_row_choices` items carry at least
  `public_state_hash` and `chosen_index` (int). Join by hash (missing hash →
  error). Output dict: `{"tie_ratio": ..., "overall": {"all": AGG, "clean":
  AGG}, "per_window": {window: AGG}, "rows": [...]}`. AGG =
  `{n, mean_regret_visit_share, top_set_rate, turn_set_rate, strict_top1_rate,
  mean_margin}`. `clean` excludes quarantined windows. `rows` carries the
  per-row score_choice dicts merged with hash/window/turn/chosen_index.

### B2. Per-row output for the existing scorer (additive only)

- `build_teacher_action_report(...)` in `teacher_action_eval.py` gains
  `include_rows: bool = False`; when true the report dict also contains
  `"rows": [<score_teacher_row results>]`. Default-off path must be
  byte-identical to today (existing report SHAs are frozen artifacts).
- `scripts/score_teacher_actions.py` gains `--per-row-out <path.jsonl>`
  (optional): writes one JSON line per scored row (the `score_teacher_row`
  result). Refuses existing path. No change to the main report when the flag
  is absent.

### B3. `scripts/rescore_teacher_report.py`

```
PYTHONPATH=src .venv/bin/python scripts/rescore_teacher_report.py \
  --per-row <rows.jsonl> --audit <audit.jsonl> \
  [--quarantine <quarantine.json>] [--tie-ratio 0.8] --out <report.json>
```

- Per-row rows: output of B2 (uses `public_state_hash`, `top1_action_index`).
- Emits `{"kind": "teacher_regret_report", "version": 1, "inputs": [{path,
  sha256} for per-row/audit/quarantine], "tie_ratio": ..., <the
  build_metrics_report payload>}`. Deterministic, fail-if-exists.

Tests: synthetic audit rows with hand-computed visit aggregations (include an
abstaining query and assert it is excluded); tie_ratio boundary (share ==
tie_ratio * max is IN the top set); margin with a single legal action; turn-set
matching including `end turn`; quarantine split changes `clean` but not `all`;
join failure on missing hash; CLI determinism + refuse-overwrite; report
without `include_rows` unchanged (compare full dict).

---

## Task C — C++ computed damage annotations (patch + rebuild + oracle test)

**Files:** `external/sts_lightspeed/bindings/slaythespire.cpp` (already
patched working copy), `patches/sts_lightspeed_python_api.patch`
(regenerated), `tests/integration/test_computed_damage_annotations.py` (new).

### What to build

In `describeBattleAction(...)` (bindings/slaythespire.cpp), add a new boolean
parameter `includeComputedDamage` (default `false`) plumbed through the Python
binding as keyword `include_computed_damage=False` on the same `describe`
method that today takes `include_card_type`. With the flag false, output must
be byte-identical to current behaviour for every action.

With the flag true, extend the existing `(deal ...)` annotation:

1. **Special-formula targeted attacks** (currently skipped via
   `hasSpecialDamageFormula`): BODY_SLAM, HEAVY_BLADE, PERFECTED_STRIKE,
   RAMPAGE, SEARING_BLOW. For each, compute the correct `base` by mirroring
   the authoritative upstream play logic (read
   `external/sts_lightspeed/src/combat/BattleContext.cpp` and the card damage
   code — do NOT guess formulas; e.g. Heavy Blade's strength multiplier, which
   piles Perfected Strike counts, where Rampage's growth counter lives,
   Searing Blow's times-upgraded formula, Body Slam = current player block),
   then pass it through `bc.calculateCardDamage(c, target, base)` exactly like
   the existing path so Strength/Weak/Vulnerable handling is identical.
   FIEND_FIRE and X-cost cards remain unannotated.
2. **Untargeted ATTACK cards** (Cleave, Thunderclap, Immolate, Sword
   Boomerang, ...): only when exactly ONE monster is alive, annotate the
   damage against that monster. Multi-hit cards keep the `(deal T = P xH)`
   format (e.g. Sword Boomerang `(deal 9 = 3 x3)`); note its hit targeting is
   random, which is deterministic with a single living enemy. With more than
   one living monster, leave unannotated (unchanged behaviour).
3. The trailing-format contract matters: `src/sts_ai/rollout_view.py`
   (`_PLAY_RE`) and `src/sts_ai/affordances.py` parse `(deal N)` /
   `(deal T = P xH)`. Do not invent a third format.

### Build (no network)

Do NOT run `scripts/build_lightspeed.sh` (it hits the network). Rebuild with:

```
.venv/bin/cmake --build external/sts_lightspeed/build --target slaythespire -j 8
```

### Oracle integration test

`tests/integration/test_computed_damage_annotations.py`, gated with the same
`@requires_simulator` convention as neighbouring integration tests. Strategy:

- Construct or reach battle states through the project's existing machinery
  (inspect `src/sts_ai/local_tasks/start_state.py`,
  `src/sts_ai/local_tasks/elite_fights.py`, `src/sts_ai/lightspeed.py`, and
  existing integration tests for the cheapest way to get a live
  `BattleContext`-backed harness state with known cards; if a configurable
  deck path exists, use it to force Heavy Blade / Perfected Strike / Body Slam
  / Cleave / Thunderclap into hand).
- For each legal attack action carrying a `(deal ...)` annotation under
  `include_computed_damage=True`, when the target enemy has 0 block: record
  enemy HP, execute the action, and assert `hp_before - hp_after ==` the
  annotated total (cap at remaining HP: if the enemy would die, assert
  `hp_after == 0` and annotated total `>= hp_before`).
- The test MUST fail if it never exercised at least one special-formula card
  and at least one untargeted AoE card (no vacuous passes). Also assert the
  flag-off path: with `include_computed_damage=False` (and omitted), the same
  states produce today's descriptions (no new `(deal` on the special cards).

### Patch regeneration (follow exactly; from repo CLAUDE.md)

```
cd external/sts_lightspeed
git diff -- bindings include src > ../../patches/sts_lightspeed_python_api.patch
```

Then: (1) normalize any line that is exactly one space (`^ $`) to an empty
line in the patch file; (2) `grep -c 'pybind11 b/pybind11'
../../patches/sts_lightspeed_python_api.patch` must print 0; (3) from inside
`external/sts_lightspeed`, `git apply --check --reverse
../../patches/sts_lightspeed_python_api.patch` must report clean. Report all
three check results.

Run the full test suite at the end (integration included) and confirm no
regressions.

---

## Task D — combat_public_v3 plumbing + TURN MATH derived lines

**Files:** `src/sts_ai/lightspeed.py`, `src/sts_ai/turn_math.py` (new),
`src/sts_ai/teacher.py` (version parameter only),
`tests/unit/test_turn_math.py` (new),
`tests/integration/test_combat_public_v3.py` (new). Tolerance checks in
`tests/unit/test_rollout_view.py` / affordances tests if their parsers need
new-format cases.

- Accept `combat_observation="combat_public_v3"` everywhere `combat_public_v2`
  is accepted in `lightspeed.py` (update the validation error message). v3
  implies `include_card_type=True` and `include_computed_damage=True` on
  action/state describe calls (Task C flag).
- `teacher.py`: `PUBLIC_OBSERVATION_VERSION` stays `combat_public_v2` as the
  default; add an explicit parameter so future collection can request v3.
- New pure module `src/sts_ai/turn_math.py`:

```python
@dataclass(frozen=True)
class TurnMathInputs:
    incoming_damage: int
    player_block: int
    player_metallicize: int
    energy: int
    hand_attacks: list[HandAttack]   # (label, cost, deal_total or None, copies)
    living_enemies: list[tuple[str, int]]  # (name, hp)

def turn_math_lines(inputs: TurnMathInputs) -> list[str]
```

  Exact line wording (frozen):
  - `End-turn projection: you would take X damage (incoming I - block B - Metallicize M; minimum 0).`
  - `Max attack damage playable this turn (using shown deal values, current modifiers only): X.` —
    append ` Unannotated attacks excluded: NameA, NameB.` when any hand attack
    lacks a deal value.
  - Only when exactly one living enemy:
    `Lethal check vs NAME (HP h): lethal available this turn.` or
    `Lethal check vs NAME (HP h): not lethal this turn.`
- Max-damage computation: bounded knapsack over hand attack cards (integer
  costs; X-cost and unannotated attacks excluded from the sum and listed as
  excluded), respecting card multiplicity in HAND (the legal-action menu is
  deduplicated, so multiplicity must come from the hand contents, matched by
  identical card label; each copy is one item with the same cost/deal).
- `lightspeed.py` composes `TurnMathInputs` from its structured state (hand,
  energy, block, player powers, enemies, incoming damage — all already
  computed for the v2 text) and inserts the TURN MATH lines directly after the
  `Incoming attack damage this turn: ...` line, before the KEY section, only
  for v3.
- Integration test: for a fixed seed/state, v2 text contains no TURN MATH
  lines and no new annotations; v3 text contains the three lines with values
  consistent with a recomputation from the same describe output; determinism
  across two processes for v3 (mirror the existing determinism test pattern).
- Unit tests for `turn_math_lines`: knapsack correctness incl. multiplicity
  (two Strikes count twice), energy binding, exclusion listing, lethal
  boundary (max == hp → lethal), end-turn projection floor at 0, multi-enemy
  omits lethal line.

---

## Task E — semantic output contracts (action_text, turn_plan) + eval

**Files:** `src/sts_ai/train/sft_format.py`, `src/sts_ai/prompting.py` (or
wherever the action_only instruction block is composed — follow
`output_contract` plumbing), `src/sts_ai/teacher_action_eval.py`,
`scripts/score_teacher_actions.py`, tests
(`tests/unit/test_sft_format.py` additions,
`tests/unit/test_teacher_action_eval.py` additions or a new module,
`tests/unit/test_prompting.py` additions).

### Contracts

- `ACTION_TEXT_OUTPUT = "action_text"`: completion is canonical
  `json.dumps({"action": d}, separators=(",", ":"))` where `d` is the exact
  legal-action description text as displayed in the menu.
- `TURN_PLAN_OUTPUT = "turn_plan"`: completion is canonical
  `json.dumps({"plan": [d0, d1, ...], "action": d0}, separators=(",", ":"))`.
- Instruction blocks (frozen wording; keep the surrounding prompt structure
  identical, and keep the numbered LEGAL ACTIONS menu format unchanged):
  - action_text: `Return exactly one JSON object with this schema:
    {"action": "<the exact text of one legal action>"}
    Copy the action text exactly as it appears in LEGAL ACTIONS.`
  - turn_plan: `Return exactly one JSON object with this schema:
    {"plan": ["<action text>", "..."], "action": "<the first entry of plan>"}
    "plan" lists, in order, the exact texts of the actions you intend to take
    this turn (end it with "end turn"). "action" repeats the first entry.
    Copy action texts exactly as they appear in LEGAL ACTIONS.`
  - The `Valid action_index values are: ...` sentence appears ONLY for the
    legacy action_only contract; the new contracts instead say `Choose from
    the LEGAL ACTIONS list below.`
- Loss masking: generalize loss_mask_mode="action" span location so the
  "action token(s)" are the tokens of the JSON string VALUE of the "action"
  key (for turn_plan, the plan array tokens count as format tokens in the
  accounting). `_action_object_span`/`_json_members` already parse JSON
  structure — extend, don't duplicate. If generalization turns out invasive,
  fall back to loss_mask_mode="completion" for the new contracts (with
  token_counts still populated) and say so in the report.
- Everything must round-trip through `tokenize_example` /
  `loss_mask_token_accounting` and the mlx data path
  (`test_train_mlx_data.py` conventions) for at least one example per new
  contract.

### Eval (`teacher_action_eval.py`)

- `declared_action_descriptions(prompt) -> list[str]`: parse the LEGAL ACTIONS
  menu lines (`^<i>: <description>$`, zero-based contiguous), failing closed
  exactly like `declared_action_indices` (count/duplication checks).
- Contract-aware row validation and candidates:
  - action_text rows (`output_contract == "action_text"`): candidates are the
    canonical `{"action": d_k}` strings for every declared description;
    teacher candidate = the row's completion. Primary top-1 by raw sequence
    log-probability (ties → lowest index, as today). Because candidate token
    lengths now differ, ALSO record `top1_by_mean_token_index` on each row
    (argmax of mean token log-probability) — secondary diagnostic only.
  - turn_plan rows: likelihood candidates are NOT enumerable over plans.
    Scoring uses action_text-style candidates against a prompt whose
    instruction block is re-rendered for action_text (add a small helper that
    swaps the instruction block; the GAME STATE and menu must stay
    byte-identical). Record `"scoring_contract": "action_text"` in the row
    result.
- Generative eval mode (new, both new contracts + action_only): greedy
  generation (temperature 0, max ~128 new tokens) from the row prompt, then
  parse: valid JSON object; extract `"action"` (or `"action_index"` for
  action_only); exact-match against declared descriptions (or indices) →
  chosen index, else invalid. Per-row result: `{generated_text, valid_json,
  matched, chosen_index}` + aggregate valid/matched/top1 rates. Implement as a
  separate report kind so candidate reports stay untouched. A greedy
  `generate` capability gets added next to `MlxCandidateScorer` (same
  lazy-load pattern); keep the scorer Protocol for candidates unchanged and
  define a small separate Protocol for generation so fakes stay easy.
- `scripts/score_teacher_actions.py`: `--contract
  {action_only,action_text,turn_plan}` (default action_only — legacy
  behaviour byte-identical) and `--mode {candidates,generate}` (default
  candidates). `--per-row-out` works in every combination.

Tests: parser edge cases (non-contiguous menu, duplicate indices);
canonical-completion validation failures for both contracts; candidate
construction; fake-scorer rows where raw-sum and mean-token rankings disagree
(assert primary uses raw-sum and the secondary field is recorded); turn_plan
instruction re-rendering (menu byte-identity); generative parse edge cases
(invalid JSON, JSON with extra keys, unmatched description, trailing text
around the JSON object); action_only legacy path unchanged.

---

## Task F — dataset builder: contracts, passes, sampled targets, turn plans

**Files:** `scripts/build_teacher_sft.py` (extend; move logic into
`src/sts_ai/teacher_sft.py` if the script gets unwieldy — follow existing
repo balance), `tests/unit/test_teacher_sft_output_contract.py` (extend),
`tests/unit/test_build_teacher_sft_passes.py` (new).

- `--output-contract {action_only,action_text,turn_plan}` (default
  action_only; single-pass legacy output byte-identical — regression-test by
  building a tiny synthetic labels file both ways).
- action_text targets: the consensus displayed action's description (resolved
  from the row's `legal_actions` by the same consensus index used today).
- turn_plan targets, derived per row from `teacher_queries`:
  - Eligible queries: non-abstaining AND `teacher_vote.action_index ==`
    consensus index.
  - Select the eligible query with maximum `search.best_evaluation`; ties →
    lowest `query.search_seed`, then lowest `query.draw_order_seed` with
    `None` ordered first. Document the tie-break in a docstring.
  - `plan` = descriptions of `search.best_sequence` entries whose `turn` ==
    the row's `turn`, in order; append `"end turn"` if not already the final
    entry.
  - HARD requirement: `plan[0]` must byte-match the consensus displayed
    action's description (they come from the same describe call); mismatch →
    raise with the offending hash (fail closed, no skip).
- `--passes N` (default 1) + `--schedule-seed S`: emit an N-pass expanded
  dataset with `pass_index` and `schedule_step` fields. Reuse/mirror the
  identity-arm scheduling semantics from `src/sts_ai/permutation_sft.py`
  (COMP-015 datasets) so the trainer consumes it identically; do not import
  the cyclic-rotation machinery.
- `--target-sampling {consensus,visit_sampled}` (default consensus).
  `visit_sampled` valid only with `--output-contract action_text` and
  `--passes > 1`: for each (state, pass_index), sample the target index from
  the aggregated non-abstaining `displayed_action_visits` distribution with a
  deterministic RNG seeded from `(schedule_seed, public_state_hash,
  pass_index)` (document the exact construction); the row records
  `target_source = {"kind": "visit_sampled", "share": <float>}`; consensus
  rows record `{"kind": "consensus"}`.
- Manifest: add `output_contract`, `passes`, `schedule_seed`,
  `target_sampling` keys (additive; existing keys unchanged).
- Tests: synthetic labels fixture (mirror existing tests) covering: canonical
  action_text completion; turn_plan derivation determinism + tie-breaks +
  trailing `end turn` + plan[0] byte-match enforcement (mismatch raises);
  visit-sampling determinism (same seed → same targets; different pass may
  differ; empirical frequency over many passes approximates shares); legacy
  byte-identity; manifest keys.

---

## Run stages (orchestrator, after code review passes)

- **R0**: `audit_state_sanity` over the three label/audit files → frozen
  quarantine sidecar in `configs/competence/`. Per-row rescoring of existing
  checkpoints (base, COMP-012 step-1500/3000) with `score_teacher_actions
  --per-row-out` + `rescore_teacher_report` → regret baselines.
- **R1**: build datasets (v2 prompts): train150 action_text consensus
  20-pass; train150 turn_plan consensus 20-pass; dev57 single-pass eval sets
  per contract. (Later: action_text visit_sampled arm.)
- **R2**: train E4B LoRA (rank 8, scale 20, dropout 0, seed 0,
  grad_accum 1 — the exact COMP-012 recipe from
  `configs/competence/comp_012.json`), 3000 updates, checkpoints
  {750, 1500, 2250, 3000}, one run per arm.
- **R3**: eval each checkpoint: candidate scoring + generative eval on dev57
  and train150; regret reports (all/clean). Compare against rescored COMP-012.
- **R4**: record COMP-020 result config + registry entry; update
  `docs/research_plan.md` status.

**Decision framing (informal, recorded before runs):** primary outcome = mean
visit-share regret and top-set agreement on dev57-clean at the best
early-stopped checkpoint. The retarget hypothesis is supported if an arm beats
the rescored COMP-012 checkpoints on regret by more than the COMP-015
seed-to-seed spread (~0.05 regret / ~4 rows top-set); one seed only, so any
positive result is "promising, replicate with seeds 1/2", not "confirmed".
