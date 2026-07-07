# Tier-1 serializer fixes + local validation (2026-07-06)

Implements the Tier-1 recommendations from
[`gemma_performance_analysis_2026-07-06.md`](gemma_performance_analysis_2026-07-06.md)
(§3): strategy-neutral serializer/comprehension fixes aimed at the two mechanisms
behind the model's act-1 elite/boss deaths — playing Skills into Gremlin Nob's
Enrage, and double-counting Strength when reading incoming damage.

## What changed

### Fix 1 — "damage shown is final" (Python, `glossary.py`)
The model repeatedly read the modifier-inclusive `(deal N)` and then *added* the
enemy's Strength again (verbatim from a transcript: *"Base damage: 24. Strength:
10. Total = 34"*). Two changes, both comprehension-only:
- The incoming-damage combat note now ends: *"(this total already includes each
  attacker's Strength, Weak, and Vulnerable — do not add those again)"*. Gated by a
  new `augment(..., damage_note=True)` param so an A/B harness can reproduce the
  pre-fix wording.
- The `Strength`/`Weak`/`Vulnerable` KEY definitions now say *"(already reflected in
  the shown deal N)"*.

### Fix 2 — persistent enemy powers are now serialized (binding + `glossary.py`)
Previously only Strength/Vulnerable/Weak/Poison were shown on the enemy line, so a
persistent power like **Enrage was invisible after the Nob's opening Bellow** — the
model watched the Nob's Strength climb with no on-screen cause and kept feeding it
Skills. The binding (`describeBattleState` + `enemies()`) now emits a curated set of
player-relevant persistent monster powers (`kPlayerRelevantEnemyPowers`: Enrage,
Metallicize, Plated Armor, Regen, Thorns, Intangible, Curl Up, Malleable, Mode
Shift, Angry, Flight, Sharp Hide, Asleep, Spore Cloud, Time Warp, Painful Stabs,
Artifact, Barricade), read via the engine's own `hasStatusInternal`/
`getStatusInternal`. Each has a source-grounded KEY definition in
`glossary.STATUS_DB`, so the existing status scanner surfaces it automatically.

Verified end-to-end: replaying a recorded Nob fight, `Enrage 2` now appears on the
enemy line from the turn after Bellow, every turn, with the KEY entry *"whenever you
play a Skill, this enemy gains that much Strength."*

### Fix 3 — subsumed by Fix 2
The analysis proposed a Python-side "keep the intent-effect explanation visible for
the whole fight" as a fallback. Serializing the persistent power (Fix 2) delivers
that goal more robustly: Enrage is shown as a standing power every turn, not
reconstructed from stateless intent history. No separate change needed.

### `structured` combat state
`state["combat"]["enemies"][i]["powers"]` (a `{name: amount}` dict) is added for
downstream analysis (blunder attribution, in-combat risk proxies) — additive, no
schema change.

## Tests & build
- `tests/unit/test_glossary.py`: new `EnemyPowerKeyTest` (Enrage/Metallicize defined
  in the KEY; all emitted powers have STATUS_DB entries) + damage-note-clause tests
  (present by default, disable-able). Full suite **372 tests green**.
- Binding rebuilt; patch regenerated scoped to `bindings include src`, reverse-apply
  clean, no `pybind11` submodule hunk. **`Monster.cpp` is not in the patch** (the
  construct-reset was reverted — see below).

## ⚠ Discovery: phantom enemy-power simulator UB
Fix 2 surfaced a **pre-existing, still-open simulator bug**: monsters can carry
stale/uninitialized statuses (seed 104 Nob had a phantom `Metallicize 4` and gained
4 block/turn; seed 103 Nob `Regen 3`; others clean — deterministic per seed, varies
across seeds → uninitialized memory). It is **real applied combat state** (a sentinel
proved a `buff` runs during `init()`; the recorded CUDA data confirms the block
gain), so it has been silently affecting all full-combat training/eval data, and the
combat logic reads the same `statusBits`. Zeroing `Monster::construct` did **not**
fix it (the buff is applied post-construct from an untraced source), so that change
was reverted to keep the sim identical to the recorded data. Full characterization
and the root-cause plan are in
[`simulator_issue_handoff.md`](simulator_issue_handoff.md) (top section).

**We surface the powers anyway** — the model must play *this* simulator, and the
phantom is that simulator's true state; hiding it would make the model underestimate
in-sim enemy tankiness. The divergence-from-real-StS and cross-build-reproducibility
concerns are logged for root-causing.

## Local validation (A/B)

**Target-model note (updated 2026-07-06):** when this A/B was run, gemma-4-E4B did
not load in mlx-lm 0.31.2 (`ValueError: Missing 54 parameters`), so the probe used
**Qwen3-4B-4bit** as a model-agnostic proxy. That blocker is now **fixed** — it was
a bug in 0.31.2's `gemma4_text` decoder (it instantiated k/v/k_norm for the 18
KV-shared layers the checkpoint omits); **mlx-lm 0.31.3 loads E4B correctly**
(pyproject pins bumped; verified bf16 loads ~3s, ~27 tok/s, full agent path works).
So the probe below can now be **re-run directly on gemma-4-E4B locally**
(`--model mlx-community/gemma-4-e4b-it-bf16`, or the ~2×-faster
`mlx-community/gemma-4-e4b-it-8bit`), which is the right instrument since the
failure mode was diagnosed on E4B. The vLLM/CUDA path remains the option for a
large paired eval.

**Design** (`scratch/nob_ab_experiment.py`): for each eval-base seed with a Gremlin
Nob fight, replay the recorded action prefix to the fight entry (same deck/relics,
no model calls), then play *only that fight* under two serializers on the same model
and sampling seed —
- **NEW**: current glossary (Enrage/Metallicize visible + damage-final clause);
- **OLD**: reconstructed pre-fix view (added enemy powers stripped from the enemy
  line; `damage_note=False`).
Isolates the serializer effect. Metrics per (seed, arm): survived the fight,
HP loss, skill-vs-attack play mix (skills feed Enrage), decisions, and a rough
double-count-in-thinking flag.

### Results on gemma-4-12B (GPU/vLLM, A100)

Ran the same matched-state probe on the larger **`google/gemma-4-12B-it`** (bf16,
full precision) on an A100 via **vLLM** (which loads the `gemma4_unified` arch
natively — no transformers fallback needed), 10 Nob seeds, 50 samples/arm,
**8192-token** thinking budget (the 12B is very verbose — arms ran 3–21 min),
K=5, temp 0.7. `scratch/nob_probe_12b.jsonl`.

| choice at the str≈6 Nob decision | OLD | NEW |
| --- | --- | --- |
| **double-counted Strength in thinking** | 15/50 (30%) | 21/50 (42%) |
| attack (races the Nob down) | 18 (36%) | 11 (22%) |
| block — Defend, feeds Enrage | 10 (20%) | 17 (34%) |
| skill (other, feeds Enrage) | 20 (40%) | 15 (30%) |
| invalid (truncation, even at 8192) | 2 (4%) | 5 (10%) |

**Verdict: no improvement on the 12B — arguably the opposite.**
- **Double-counting is *higher* under NEW (30% → 42%)** — the reverse of E4B. Read
  this cautiously as a **verbosity/proxy confound, not "the fix backfired":** the
  12B thinks far more than E4B (arms took 3–21 min = thousands of tokens vs E4B's
  ~1k), and the richer NEW prompt gives it more numbers to reason over, so it does
  more damage-arithmetic — more chances for the regex "N + M" detector to fire. The
  detector likely measures verbosity here, not comprehension. (A cleaner check would
  be a hand-labeled sample or a stricter detector.)
- **Behaviorally it mirrors E4B**: NEW shifts toward blocking (20% → 34%) and away
  from attacking (36% → 22%); Enrage-feeding plays (skill+block) 60% → 64%. So Fix 2
  (Enrage visibility) again does *not* push toward the Enrage-avoiding "race" play.
- **NEW also raised the invalid rate (4% → 10%)** — the longer NEW prompt tips the
  verbose 12B into truncation on a few decisions even at 8192.

### Cross-model conclusion

On **both** E4B and the 12B, at single matched Nob decisions the Tier-1 fixes do
**not** produce a cleaner tactical choice (both shift toward blocking, not toward
racing the Nob). Fix 1's damage-clause showed a modest comprehension win on E4B
(double-count 10% → 4%) but the 12B's double-count signal is confounded by
verbosity and points the other way. The consistent finding across models: **making
Enrage visible doesn't change the immediate choice** — at a high-Strength state,
blocking-to-survive-this-turn is locally rational regardless, and the Enrage *cost*
is a sequential consideration a single-decision (Markov) probe can't reward. So the
fixes' real value can only be judged by **full-fight outcome** (survival / HP), not
this propensity probe. The fixes remain correct and strategy-neutral (verified);
whether they help play is an open question that needs a fight-level or eval-level
measurement.

### Results on full-size gemma-4-E4B (after the mlx-lm 0.31.3 fix)

Re-ran the matched-state propensity probe on the **actual target model**,
`mlx-community/gemma-4-e4b-it-bf16` (thinking, temp 0.7, max_tokens 4096, 1 retry),
10 Nob seeds, 50 samples/arm at str≈6 Enrage states. **0 invalids** (the 4096
budget removed the truncation confound the Qwen run hit). `scratch/nob_probe.py`,
`scratch/nob_probe_e4b.jsonl`.

| choice at the str≈6 Nob decision | OLD | NEW |
| --- | --- | --- |
| **double-counted Strength in thinking** (Fix 1's direct target) | 5/50 (10%) | **2/50 (4%)** |
| attack (races the Nob down) | 21 (42%) | 19 (38%) |
| block — Defend, itself a Skill that feeds Enrage | 21 (42%) | 25 (50%) |
| skill (other, feeds Enrage) | 8 (16%) | 6 (12%) |
| invalid | 0 | 0 |

**Verdict: Fix 1 shows a real (small) comprehension win; Fix 2 does not move the
single-decision choice.**
- **Double-counting halved (10% → 4%)**, and no seed regressed (NEW improved on
  seeds 103/105/111, tied elsewhere). This is exactly what the "damage already
  includes Strength/Weak/Vulnerable" clause targets — the clearest positive signal,
  though N is small (5 vs 2 samples) and the detector is a regex proxy.
- **The tactical choice mix barely changed** — both arms are ~40% attack / ~45%
  block / ~14% skill; if anything NEW blocks slightly *more* (Enrage-feeding plays
  skill+block: 58% → 62%). So making Enrage visible did **not** make E4B avoid
  Enrage-feeding plays at these decisions.

Why Fix 2 doesn't shift single-decision behavior (an informative finding): at a
str≈6 state incoming damage is already high, so blocking-to-survive-this-turn is
locally rational regardless of Enrage — the Enrage *cost* is a multi-turn,
sequential consideration that a Markov, single-decision policy underweights. The
death spiral is therefore not purely "the model can't see Enrage at one turn"; even
seeing it, the immediate-survival choice looks the same. **Implication:** the
fixes' real payoff (if any) is a fight-level emergent effect on survival/HP, which a
single-decision propensity probe cannot measure — it needs full-fight outcomes or a
paired eval. Fix 1's cleaner damage arithmetic is the more likely lever; Fix 2's
value may be smaller than hoped for the immediate choice.

### Earlier attempt: Qwen3-4B proxy (superseded by the E4B run above)

I switched from full-fight A/B to a cheaper, sharper **matched-state propensity
probe**: replay each Nob-fight seed to a mid-fight decision where the Nob's
Strength has built up (`str≈6`, Enrage active — where both fixes should bite), then
sample K=5× at that fixed state under NEW vs OLD. 9 usable seeds, 45 samples/arm
(`scratch/nob_probe.py`, `scratch/nob_probe_results.jsonl`). Metric: which card
kind it plays (a Skill — incl. Defend — feeds Enrage; an Attack races the Nob down).

| choice | OLD | NEW |
| --- | --- | --- |
| attack (races the Nob) | 4 (9%) | **7 (16%)** |
| block — Defend, itself a Skill that feeds Enrage | 18 (40%) | 18 (40%) |
| skill (other, feeds Enrage) | 2 (4%) | 1 (2%) |
| invalid (2048-token truncation, not retried) | 21 (47%) | 18 (40%) |
| double-counted Strength in thinking (rough regex) | 5 (11%) | 7 (16%) |

**Verdict: inconclusive, weakly NEW-favorable.** Among *valid* samples, NEW shifts
~9pp from blocking toward attacking (attack 26% vs 17%; block 67% vs 75%) — the
direction the fix intends (understand Enrage → race the Nob rather than
block-and-feed it). But the effect is small-N (7 vs 4 attacks) and swamped by
noise, and the double-count proxy shows **no** benefit (if anything the wrong way,
within noise). Three reasons the proxy can't settle it:
1. **Wrong model.** gemma-4-E4B (which exhibited the diagnosed failure) won't load
   locally; Qwen3-4B is a stand-in and it already plays defensively / barely
   double-counts at these states, so there is little failure-mode to remove.
2. **Truncation dominates.** ~45% of samples were invalid because Qwen's thinking
   exceeded the 2048-token probe cap (kept low for speed; `max_retries=0`), so the
   choice distribution rests on ~55% of samples — noisy.
3. **Single-state, K=5** is a low-power probe; the real question is full-fight
   outcome, which needs the (slow, thinking-mode) fights or the GPU path.

The engineering is verified (Enrage now visible every turn, damage clause present,
tests green); the **behavioral benefit is not confirmable on the local proxy** and
should be measured on the vLLM/E4B path below. This is an honest null-to-weak
result, not evidence the fix doesn't work — it is evidence the local proxy is the
wrong instrument for a gemma-specific comprehension fix.

## Next
- **Definitive E4B validation on the GPU/vLLM path** (the recorded-data backend):
  ```
  PYTHONPATH=src python scripts/run_until.py --model google/gemma-4-E4B-it --backend vllm \
    --seeds-config configs/frozen_seeds.json --split eval --rollouts-per-seed 4 \
    --thinking --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 8192 \
    --combat-control llm --max-act 3 --battle-simulations 50 --output-dir data/tier1_eval/new
  ```
  then `compare_paired.py` against a pre-fix baseline (or the existing eval/base),
  reading the death-by-encounter + blunder battery from the analysis doc.
- **Root-cause the phantom-power UB** (simulator_issue_handoff.md) — it corrupts all
  full-combat data and breaks cross-build reproducibility.
- Fix mlx_lm E4B loading (version bump / arch patch) if local E4B iteration is wanted.
