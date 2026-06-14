# Progress Report — 2026-06-14

A working session that took the harness from "still hardening" to **GO for data
collection**, and kicked off the first Stage 5 thinking-mode rollouts. Source of
truth for the plan remains [`research_plan.md`](research_plan.md); this is a
point-in-time summary.

## Headline

Stages 0–4 are effectively complete. The harness is validated end-to-end with a
local Qwen3-4B agent, dev/eval seeds are frozen, risk proxies exist, and the
Stage 5 (fixed neutral rollout) collection path is wired and exercised on one
seed. The first inspected thinking-mode trace exposed a real data-quality issue
(the model misreads bare action labels) that is being addressed before the full
collection.

## What changed this session

### Harness correctness
- **Fixed a false-green regression test.** `tests/integration/test_battle_search.py`
  stopped one decision *before* the floor-12 battle it was meant to guard (its
  `max_decisions` equalled the replayed decision count), so it passed in ~40 ms
  without exercising the bug. Added headroom so the battle is entered; relaxed
  assertions to build-portable containment invariants after confirming seed-2 is
  **non-deterministic** across runs (hang / clean resolve / early divergence).
- **Serializer cleanups** (binding patch, rebuilt): dropped the internal `bits=`
  prefix from action descriptions (kept on `LegalAction.bits`) and render the
  Neow/event `room INVALID` header as `room none`.

### Model I/O
- **Thinking-mode comparison:** at 2048 tokens, thinking was 88.9% valid and
  ~19 s/decision (truncation-limited); no-thinking @256 was 100% valid and sub-2
  s/decision. At **4096 tokens thinking is 100% valid with no truncation** (seen
  in the Stage 5 single-seed run).
- **`<think>` capture:** `AgentDecision.thinking` now stores the full chain-of-
  thought (separate from the brief JSON `reasoning`), including partial text from
  truncated blocks — so reasoning can sit in the training forward context and be
  audited for framing leakage.

### Data + measurement
- **Larger baseline batch** `data/baseline_rollouts_300` (seeds 2-151, three
  agents) under the rebuilt serializer: mean floor 14.94, 100% valid; 142-seed
  clean intersection.
- **Stage 2 risk proxies** (`src/sts_ai/risk_proxies.py` + script + 23 tests):
  deterministic, computed from stored traces. They discriminate policies as
  expected (low-HP campfire rest: `first`/`heuristic` 1.0 vs `random` 0.39).
- **Frozen seeds** in `configs/frozen_seeds.json`: smoke 10 / dev 31 / eval 100.

### Stage 4 evaluation — GO
Qwen no-thinking @256 on the smoke seeds: 100% valid, mean floor 13.3. Versus
baselines it is clearly non-random and **HP-conservative** (rest@lowHP 1.0 vs
random 0.25; final HP 37 vs 16) but **over-rests at high HP** (1.0 vs heuristic
0.0). All go/no-go criteria met.

## Key finding from the first Stage 5 trace (seed 3, thinking @4096)

The capture is perfect (25/25 valid, 0 retries, full thinking, good screen
coverage), and there is **no framing leakage** — but the model **misunderstands
bare action labels**:

- It rested at **full HP (80/80)** because it believed `smith` *fetches a relic*
  (it upgrades a card) and dismissed it.
- It fabricated boss HP ("The Guardian has 80 HP").

This is a measurement confound: part of the "over-rests at high HP" risk signal is
actually "doesn't know what smith does," not genuine risk-aversion. **Action
descriptions are being enriched** (campfire and similar) so the neutral reasoning
is mechanically correct before the full Stage 5 collection. The model's own
vocabulary already leans "safe/cautious", which is a useful neutral baseline for
the later framing comparison.

## Stage status

| Stage | State |
| --- | --- |
| 0 Reproducible harness | Done |
| 1 Baseline dataset | Done (142-seed clean intersection, >100 criterion met) |
| 2 Serialization & risk proxies | Risk proxies done; serializer enrichment in progress |
| 3 Qwen inference loop | Done (no-thinking 100% valid; thinking @4096 100%) |
| 4 Qwen evaluation | Done — **GO** |
| 5 Fixed neutral rollout collection | **In progress** — path wired, 1 seed run + inspected |
| 6–10 | Not started |

## Next steps

1. Enrich action descriptions (campfire etc.); rebuild, regen patch, re-validate.
2. Re-run the single seed at full Act-1 depth (`max-decisions 200`) under the
   enriched serializer; inspect.
3. Scale Stage 5 neutral collection to the smoke → dev → eval splits.
4. Build a reward labeler (final floor/HP/outcome per trajectory) and a
   frame-leakage audit pass over `agent.thinking`.
5. Defer-or-resolve the seed-2-class native UB (currently accepted-and-excluded).

## Reproducibility caveats

- The seed-2-class uninitialized-memory UB is only *contained*, not root-caused;
  cross-machine identical traces are not yet guaranteed.
- Thinking-mode collection at full depth is slow (~1–2 h/seed); use
  `--seed-timeout-seconds`.
