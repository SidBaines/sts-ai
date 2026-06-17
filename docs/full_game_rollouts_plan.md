# Full-Game Rollouts — Implementation Plan

**Created:** 2026-06-17 · **Status:** planning complete, implementation not started.

This is the durable plan for moving the harness from Act-1-capped rollouts to
**full-game** rollouts (beat the whole run), plus the orchestration and data-quality
work that has to land before we spend real GPU time on long, expensive rollouts.
It is written to be resumable: a fresh context should be able to read this file and
continue without re-investigating.

---

## 1. Goal & motivation

So far rollouts are capped at Act 1. We want to run much longer rollouts to see if
models can **beat the whole game**, while keeping Act-1 as a selectable option
(it becomes the non-default). This is both a capability question (can current
models finish a run) and a prerequisite for richer risk-relevant data spanning
all acts.

Two operational problems to solve alongside the cap lift:
1. Long rollouts die at very different points → naive fixed batches waste GPU when
   one long survivor holds up everything.
2. Data quality / comprehension was only ever audited on Act 1; Acts 2/3 introduce
   new enemies, intents, relics, events that the serializer may render poorly.

---

## 2. Decisions taken (2026-06-17, with Sid)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | "Full game" depth | **`max_act=3`** default (standard StS victory = beat Act 3 boss). Act 4 only triggers if the agent collects all 3 keys (rare for a 4B-class model). Keep `max_act=1` and `max_act=4` as flags. | Captures essentially every realistic full run; minimal blast radius. |
| D2 | Combat control for these runs | **Keep LLM combat** (`combat_control="llm"`). | Tests the model end-to-end, deterministic at temp=0, **bypasses the C++ combat-search hang**. Cost: ~5× more decisions/run and low win-rate (4B-class models are weak at combat). |
| D3 | Orchestration / restart semantics | **M total rollouts, N live, NEW seeds, stop launching after the Mth start.** Supervisor draws fresh `(world_seed, rollout_index)` specs from a generator, keeps N in flight, and **launches exactly M rollouts on M predetermined seeds**. Once the Mth has been *started*, no new rollouts are launched; all in-flight runs are allowed to finish. | A dead seed at temp=0 would just die identically, so restarts must use new seeds, not retries. |
| D3a | Stopping rule: starts, **not** completions (2026-06-17 refinement) | Stop on the **Mth start**, then drain. **Do not** stop at M *completions*. | Stopping at M completions biases the sample toward fast rollouts: failures finish quicker than wins, so the slow successful runs would still be in-flight and uncounted. We accept wasted GPU on the tail in exchange for an unbiased sample of exactly M predetermined seeds (all M complete; none discarded). |
| D4 | Serializer/glossary multi-act coverage (Workstream C) | **Include now, in parallel with B.** | Biggest science risk; want good comprehension before costly runs. |

Consequence of D2: the heavy per-rollout **subprocess isolation** work is *not* needed
(LLM combat does not invoke the C++ combat search that hangs). We keep only a **light
watchdog** as insurance against a single stuck generation stalling the in-process pipeline.

---

## 3. Key facts established during investigation (so we don't re-derive them)

**Act-1 cap is purely a Python gate; the C++ sim already plays the whole game.**
- `src/sts_ai/lightspeed.py:19` — `max_act: int = 1` (constructor default).
- `src/sts_ai/lightspeed.py:28` — `self.max_act = max_act`.
- `src/sts_ai/lightspeed.py:50` — `is_terminal()` returns `self.gc.outcome != UNDECIDED or self.gc.act > self.max_act`.
- C++ (`external/sts_lightspeed/src/game/GameContext.cpp`) fully implements Acts 2/3/4:
  boss treasure room → `transitionToAct(act+1)` (auto), per-act `Map::fromSeed(...)` regen,
  Act 3→`enterAct3VictoryRoom()` (victory unless all 3 keys → Act 4), Act 4 boss → `PLAYER_VICTORY`.
  **No C++ changes needed.**

**`--max-act` is only wired into `run_sweep.py:56`.** `run_rollout.py` (builds env at
`scripts/run_rollout.py:60`), `run_batch.py`, and both orchestrators' `make_env` closures
do **not** expose it — they inherit the `lightspeed.py` default. So lifting the cap =
change the default **and** thread the flag through the remaining entry points.

**Decision budget will silently truncate full runs.** The Gemma seed-4 runs that reached
Act 2 used **145–169 decisions for Act 1 alone** (LLM combat = one decision per card play).
A 3-act game ≈ **450–700+** decisions. The `max_decisions=200` default is set in every
entry point: `rollout.py:99/40`, `run_batch.py:51`, `run_sweep.py:53`,
`parallel_rollout.py:135`, `streaming_rollout.py:43`. A run that hits the cap stops with
`stopped_reason="max_decisions"` — indistinguishable at a glance from a real ending.

**Orchestrators already refill finished slots from a queue** (so early death ≠ wasted GPU
*as long as the queue has more specs than concurrency*):
- `src/sts_ai/streaming_rollout.py` — vLLM continuous batching, `concurrency` default 48,
  `queue` + `in_flight` dict, refills via `fill()`. **This is the one the Gemma runs use.**
- `src/sts_ai/parallel_rollout.py` — MLX lockstep, `batch_size` default 8.
- Neither has a "run until M completions from an unbounded generator" supervisor, nor a
  watchdog for stuck requests. The static finite queue is the only gap for D3.

**Resume is not possible; replay is.** No C++ state serialization exists. But the Gemma
seed-4 runs were `temperature=0` + `combat_control="llm"` (fully deterministic),
`stopped_reason="terminal"` purely because `max_act=1`. So bumping `max_act` and rerunning
`--seed 4` replays Act 1 identically, then continues into Act 2/3. This is the acceptance test.
- Reference runs (both reached Act 2 / floor 17, outcome UNDECIDED, temp=0, llm combat, max_tokens 4096, battle_sims 50):
  - `data/rollouts/gemma_bench/vllm_gemma_4_E4B_it_thinking_4096/seed_4_r0.{jsonl,meta.json}` (145 decisions, thinking)
  - `data/rollouts/gemma_bench/vllm_gemma_3_12b_it_nothinking_4096/seed_4_r0.{jsonl,meta.json}` (169 decisions, no-thinking)
  - `data/rollouts/gemma_bench/vllm_gemma_4_E4B_it_thinking_4096/seed_14_r0.{...}` (189 decisions)

**Serializer coverage was only audited on Act 1.** `src/sts_ai/glossary.py` `INTENT_DB` /
`RELIC_DB` / `POTION_DB` are *curated subsets* (potions complete; relics a "confident
curated subset, long tail skipped" per research_plan 2026-06-16). Act 2/3 enemies, intents,
relics, events may render with missing/"unknown" effect text — degrading comprehension
exactly where we have no baseline. The prompt-comprehension invariant (see
`memory: prompt-neutrality-is-the-iv`): all additions are **comprehension only**, never
strategy/risk/objective language.

**Other Act-1-only artifacts:** `configs/frozen_seeds.json` (derived from Act-1 baseline
reliability) and `docs/throughput_benchmarks.md` (Act-1 decisions/run).

---

## 4. Workstreams

Ordering: **A → (B ∥ C) → D.** A unblocks reaching Act 2/3 states (needed by C) and the
full-game runs (B). B and C are independent.

### Workstream A — Lift the cap + decision budget + plumbing (foundational)

**Goal:** a single config can run a full 3-act game from every entry point, and a budget
truncation can never be silently mistaken for a real ending.

**Changes:**
1. `src/sts_ai/lightspeed.py:19` — default `max_act` `1 → 3`.
2. Thread `--max-act` (default 3) through and into the env at:
   - `scripts/run_rollout.py` (add arg; pass to `LightspeedHybridEnv` at line ~60).
   - `scripts/run_batch.py` (add arg; pass through to its rollout calls / subprocess args).
   - `src/sts_ai/streaming_rollout.py` & `src/sts_ai/parallel_rollout.py` — their `make_env`
     closures are supplied by callers; ensure the caller-side closures (in `run_sweep.py:81`
     and the new supervisor) pass `max_act`. (`run_sweep.py:86` already does.)
3. Decision budget (refined during impl, 2026-06-17): since `max_act=3` is now the *default*
   depth, a 200-decision default would silently truncate every full run. Resolution:
   - **CLI `--max-decisions` defaults raised 200 → 1500** in `run_rollout.py`, `run_sweep.py`,
     `run_batch.py` (full-game-appropriate; Act-1 runs terminate at the act boundary well
     before any budget, so unaffected).
   - **Library/orchestrator function defaults stay 200** (`rollout.run_rollout`,
     `parallel_rollout`, `streaming_rollout`) so existing tests/callers are unchanged.
   - Add a visible signal whenever a rollout stops as `max_decisions` while
     `outcome == UNDECIDED` — e.g. a `WARNING` log line and/or a meta field
     (`budget_truncated: true` in `RolloutMeta.extra`) so truncation is never silent.
     ⚠ Adding a top-level `RolloutMeta` field is a schema change — prefer `extra`.
4. Tests:
   - Keep existing `max_act=1` integration tests as **Act-1 regressions** (don't generalize).
   - Add a **fast** progression test: deterministic baseline agent (`heuristic`/`first`) +
     **hybrid** combat (search is fast & strong, no GPU) + `max_act=3` on a seed it survives,
     asserting `final_floor > 17` (i.e. reached Act 2). Gate with `@requires_simulator`.
     Run it in a subprocess with a timeout (matching `run_batch.py` containment) since hybrid
     re-exposes the C++ search path. Pick the seed empirically (try a few clean Act-1 seeds).

**Risks:** changing the `lightspeed.py` default affects every caller that didn't pass
`max_act`; audit that nothing relied on the implicit `=1` for correctness (tests do pass it
explicitly). The progression test needs a seed a baseline agent actually survives to Act 2 —
may take a couple tries.

**Done when:** `run_rollout.py --max-act 3` on a deterministic agent reaches Act 2+;
budget-truncation is visibly flagged; unit+integration tests pass.

### Workstream B — "run until M completions" supervisor (the orchestration ask)

**Goal:** launch exactly M rollouts on M predetermined fresh seeds, keep N live, and drain the
final in-flight batch; resilient to a single stuck generation.

**Changes:**
1. Generalize `streaming_rollout.py` (the vLLM path) so its work source is a **seed-sequence
   generator + a target count M of total rollouts** instead of a finite pre-built queue:
   - Accept either an iterable/generator of `(world_seed, rollout_index)` specs **or** a
     `target_rollouts: int` (M) plus a seed-sequence generator.
   - Keep `concurrency = N` in flight; on each finalize, **start the next spec only while the
     number of rollouts *started* `< M`**. Once M have been started, stop launching and let all
     in-flight runs finish. ⚠ The counter that gates launching is **starts, not completions**
     (D3a) — stopping at M completions biases toward fast-failing runs. End state: exactly M
     completed rollouts, none discarded; some GPU idle on the tail is expected and acceptable.
2. **Light watchdog:** per-request wall-clock cap (configurable, e.g. `--request-timeout-s`);
   if a generation exceeds it, abort that request, finalize the rollout with
   `stopped_reason="watchdog_timeout"`, and free the slot. Prevents one stuck request from
   stalling the in-process pipeline. (Lower priority given LLM combat bypasses the known hang,
   but cheap insurance for long expensive runs.)
3. New CLI: `scripts/run_until.py` (or `--target-rollouts` on `run_sweep.py`) exposing
   `--concurrency N`, `--target M` (total rollouts to launch), `--max-act` (default 3),
   `--max-decisions` (full-game default), model/backend/thinking flags, `--output-dir`, and the
   seed-sequence start/stride.
4. Seed generation: extend `src/sts_ai/seeding.py` usage — draw distinct world seeds in
   sequence (skipping any configured exclusions), `rollout_index` per the K-per-seed policy
   (default 1 for temp=0). Keep it deterministic/reproducible (record the seed range used).
5. Tests: unit test the supervisor loop with a **mock agent + mock env** (no GPU): assert it
   **launches exactly M rollouts** (verify via start-count, including a case where slow "success"
   rollouts are still in-flight when the Mth starts — they must still be launched and counted, not
   pre-empted by fast finishers), keeps ≤ N in flight, draws new seeds on refill, all M finalize,
   and the watchdog finalizes a deliberately-stalled request.

**Risks:** the streaming loop currently assumes a finite queue; refactor carefully to preserve
the existing finite-list behavior (run_sweep) as a special case. Watchdog must abort the vLLM
request cleanly (`llm_engine.abort_request`) without corrupting the engine step loop.

**Done when:** `run_until.py --target M --concurrency N --max-act 3` launches exactly M rollouts
on M fresh seeds, stops launching after the Mth start, drains all in-flight runs (M complete),
a stalled request is recovered, and the supervisor unit test passes.

### Workstream C — multi-act serializer/glossary coverage audit + fill (science-critical)

**Goal:** Act 2/3 (and Act 4 boss) states render with the same comprehension quality as Act 1;
no missing intent/relic/potion/event effect text. **Strategy-neutral** additions only.

**Changes:**
1. **Audit (depends on A):** generate a handful of rollouts into Acts 2/3 (cheap: deterministic
   baseline agent + hybrid combat + `max_act=3`, or replay Gemma seed 4 on GPU) and scan the
   rendered `state_text` across the new decisions for missing-effect markers / "unknown" /
   fallback labels — the same method as the prior ~74k-decision scan. Quantify the gap per
   category (enemies/intents, relics, events, potions, map/shop).
2. **Fill `src/sts_ai/glossary.py`:** add Act 2/3 enemy `INTENT_DB` entries (source-grounded
   from the C++ `MonsterMoves`/intent logic), extend `RELIC_DB` to the Act 2/3 boss & common
   relics that appear, confirm `POTION_DB` completeness. Add event coverage where Act 2/3
   events render poorly.
3. Verify against the comprehension invariant: no objective/risk framing leaks into any new
   text (this is the independent variable; see `docs/research_plan.md` Stage 2 2026-06-16 note
   and `memory: prompt-neutrality-is-the-iv`).
4. Tests: extend glossary unit tests with known Act 2/3 intent/relic labels.

**Risks:** intent text must be **source-grounded** (read the C++ monster move definitions),
not guessed, to stay faithful. Act 2/3 content is large — prioritize what actually appears in
sampled traces over exhaustive coverage; `log()`/note what's deliberately skipped (long tail).

**Done when:** sampled Act 2/3 traces show no missing-effect markers for content that appears;
new label tests pass; a diff review confirms strategy-neutrality.

### Workstream D — multi-act metrics, reward labels & re-benchmark (follow-on)

**Goal:** downstream tooling and docs reflect full-game depth.

**Changes:**
1. Verify `src/sts_ai/risk_proxies.py`, `scripts/compute_risk_proxies.py`,
   `scripts/compare_models.py`, and any summarizer handle `final_act` 1–3 and floors 1–51
   (per-act-keyed proxies, reward = outcome/floor/HP across acts). Fix any Act-1 assumptions.
2. Refresh `docs/throughput_benchmarks.md` with full-game decisions/run and wall-time
   (per model, LLM combat).
3. Note in `configs/frozen_seeds.json` / research_plan that the frozen splits are Act-1-derived;
   **defer** a formal full-game re-freeze (the D3 M/new-seeds model makes it non-blocking for
   initial testing).
4. Update `docs/research_plan.md` status + near-term ordering once A–C land.

**Done when:** metrics compute correctly on a full-game trace; benchmarks doc updated.

---

## 5. Acceptance validation (on the GPU pod)

Primary end-to-end check (the "continue a successful run" test):
- Replay **Gemma seed 4** (`temperature=0`, `combat_control="llm"`, `max_tokens=4096`) with
  `--max-act 3`. Expect: Act 1 reproduces identically to the reference run above, then the
  run continues into Act 2/3 instead of stopping `terminal` at floor 17. Compare the first
  ~145–169 decisions against `seed_4_r0.jsonl` for the chosen model to confirm determinism.
- Then a small `run_until.py` smoke: `--target ~8 --concurrency ~4 --max-act 3` to confirm
  the supervisor, budget, and watchdog all behave on real GPU before any large run.

---

## 6. Orchestration approach

Implement via **codex-driven-development** (Claude orchestrates, Codex `exec` builds, Claude
subagents review) — matches the repo's recent pattern (world-seed split, vLLM backend both
"Implemented via codex-driven-development"). A and B are well-scoped build tasks; C is
audit-then-fill; D is verification. Run B ∥ C after A.

Per `CLAUDE.md`: every change must pass
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .` plus at least one smoke
rollout, and every bug fix gets a regression test. Schema-touching changes
(`schemas.py` / on-disk JSONL) are breaking — prefer `RolloutMeta.extra` over new top-level fields.

---

## 7. Progress tracker

- [x] **A1** `lightspeed.py` default `max_act` 1→3 *(commit e84459c)*
- [x] **A2** Thread `--max-act` through `run_rollout.py`, `run_batch.py`; `run_sweep.py` default 1→3 *(e84459c)*
- [x] **A3** CLI `--max-decisions` 200→1500 (library defaults stay 200) + `extra["budget_truncated"]` flag + warning *(e84459c)*
- [x] **A4** Act-1 regression tests kept; new fast Act-2 progression test (heuristic+hybrid, subprocess+timeout); 163 tests pass *(e84459c)*
- [ ] **B1** ~~Generalize streaming orchestrator~~ — **NOT NEEDED** (impl realization 2026-06-17): `run_streaming_rollouts` already launches all given specs, keeps N in flight, refills on finish, and drains the tail without early-stopping. Passing a finite list of exactly M specs *is* the bias-safe behaviour. No orchestrator change.
- [ ] **B2** ~~Watchdog~~ — **DEFERRED** (impl decision 2026-06-17): with LLM combat (D2) there's no C++ combat-search hang, and vLLM generation is bounded by `max_tokens`, so a per-request watchdog is marginal insurance with real cost (engine `abort_request`, flaky wall-clock tests). Revisit if real runs show stuck requests.
- [x] **B3** `run_until.py` CLI: generate exactly M fresh `(seed,0)` specs (ascending from `--seed-start`, skipping `--exclude-seeds`/config + on-disk), single model, `--target M`/`--concurrency N`/`--max-act 3`/full-game budget, outcome histogram (incl. `budget_truncated`) summary; runs them through the existing streaming/parallel orchestrator *(commit 22a07f6)*
- [x] **B4** Unit test for the pure `generate_specs` helper + integration test (ScriptedStreamingAgent) proving exactly-M-and-all-complete (no drops/extras) at concurrency<M *(22a07f6)*
- [x] **C1** Audited Act 2/3 (source-driven for intents via MonsterMoves.h; trace-driven for relics via hybrid heuristic runs) *(commit 86e37d0)*
- [x] **C2** Filled `glossary.py`: +73 intents, +13 relics, +1 status (Constricted), +Exploder formatter exception; source-grounded, deliberate long-tail skips documented *(86e37d0)*
- [x] **C3** Strategy-neutrality verified (independent review: zero valuation/threat language in new entries); CHOSEN_HEX accuracy fix *(86e37d0)*
- [x] **C4** Glossary unit tests for Act 2/3 intents/relics/status + neutrality + Exploder guards; 173 tests pass *(86e37d0)*
- [x] **D1** Multi-act metrics verified act-agnostic (no code change needed) — `summarize_rollouts`/`compute_risk_proxies`/`compare_models` validated on a real multi-act trace (heuristic+search seed 4 → Act-2 boss, floor 33; 97 risk events across acts)
- [ ] **D2** Refresh throughput benchmarks for full-game — **PENDING GPU** (needs real full-game LLM-combat runs)
- [x] **D3** Updated research_plan.md (2026-06-17 Done block + Stage 5 note) + frozen-seeds Act-1-derived caveat
- [ ] **V1** GPU acceptance: replay Gemma seed 4 with `--max-act 3` continues into Act 2/3 — **PENDING GPU**
- [ ] **V2** GPU smoke: `run_until.py` small target end-to-end — **PENDING GPU**

---

## 8. Open questions / things to revisit

- **Decision-budget value (~1500):** confirm against real Act 2/3 LLM-combat decision counts
  after the first runs; raise if full runs still truncate.
- **Watchdog timeout value:** pick from observed per-decision latency once we have Act 2/3
  numbers (thinking mode is tens of seconds/decision).
- **rollouts-per-seed K:** default 1 at temp=0 (deterministic). If we later want temp>0 GRPO/RLOO
  groups, the supervisor should support K>1 per world seed.
- **Frozen full-game seed set:** deferred; revisit before any full-game *training* data freeze.
- **Hybrid full-game arm:** not chosen now (D2 = LLM combat). If we later want a cheaper/stronger
  full-game arm, hybrid would re-introduce the C++ combat-search hang and require the deferred
  subprocess-isolation work.
