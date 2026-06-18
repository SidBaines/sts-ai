# gemma-4-E4B-it (thinking) — full-game run, throughput, and failure-mode analysis

**Date:** 2026-06-18 · **Model:** `google/gemma-4-E4B-it`, native thinking, temp 1.0 / top_p 0.95 / top_k 64 · **Mode:** full game (`--max-act 3`), full LLM combat (`combat_control="llm"`).

This is a self-contained writeup of a single session: collecting full-game E4B-it·thinking rollouts on an H100, benchmarking vLLM concurrency for long rollouts, and analysing **how the agent dies**. Reproduction pointers in §6.

---

## 1. What & why

Goal: a first empirical read on how far the E4B-it thinking model gets at the *full* game (Acts 1–3, every combat micro-decision played by the LLM), and a characterisation of its failure modes — both as a capability baseline and to confirm the structured-map fix holds for a verbose reasoner at temperature 1.0 (the failure that made gemma-4-12B-thinking unusable; see [`map_representation_handoff.md`](map_representation_handoff.md)).

## 2. Data collected

- **240 completed full-game rollouts**, stored at `data/rollouts/e4b_think_perf/vllm_gemma_4_E4B_it_thinking_8192/` (per-decision JSONL + per-rollout `.meta.json`; **gitignored**, local only). 42,184 total decisions.
- Config: `--temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 8192 --max-retries 3 --max-act 3 --combat-control llm --battle-simulations 50`, distinct world seeds (ascending from 2, frozen exclusions skipped), `rollout_index=0`.
- Hardware/stack: RunPod **H100 SXM 80GB**, CUDA 13.0 driver, **vLLM 0.23.0** (cu130 torch). One gotcha: the cu130/flashinfer-JIT mismatch on the `runpod-torch-v21` base image required `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler; handles top_p/top_k fine, negligible cost). Launched via `scripts/run_until.py` (`run_streaming_rollouts` continuous-batching orchestrator).
- Run mechanics: started at `--concurrency 64`, stopped, then **resumed** at `--concurrency 384` (re-running the same command skips completed seeds via `existing_rollout_seeds`; only the ≤N in-flight at a kill are abandoned — partial-rollout resume is not supported, no sim-state serialisation). Finally stopped with **382 rollouts still in flight (abandoned)** — see the survivorship caveat in §4.

## 3. Throughput / concurrency for full-game rollouts

vLLM concurrency was benchmarked twice. The first (act-1, instant-sample) optimised the *wrong* metric and led to a mistake; the second (full-game, warmed up) gives the real picture.

**Act-1 sweep** (short rollouts, dec/s, 0 preemption everywhere): 64→7.6, 128→12.1, 192→13.8, 256→15.4, 384→**17.1**, 512→16.9. Peak ~384.

**Full-game sweep** (`--max-act 3`, warmed up so prompts had grown, 3-min samples, mean **176 decisions/completed-rollout**):

| `--concurrency` | dec/s | preempt | steady-state completed/min | pipeline fill | abandoned-on-stop |
|---|--:|--:|--:|--:|--:|
| 64  | 7.8  | 0 | 2.7 | ~24 min | 64 |
| 128 | 12.6 | 0 | 4.3 | ~30 min | 128 |
| 256 | 16.7 | 0 | 5.7 | ~45 min | 256 |
| 384 | 19.5 | 0 | 6.7 | ~58 min | 384 |

**Lessons:**
- **No preemption at any concurrency** — full-game contexts never strain the 2.5M-token KV cache (even at 384). Memory is not the limit up to 384.
- **Completions are heavily back-loaded for long rollouts.** Aggregate dec/s rises with concurrency, but per-rollout speed = dec/s ÷ in-flight *falls* (0.12 dec/s/rollout at 64 → 0.05 at 384), and a full game is ~176 decisions, so at high concurrency hundreds of rollouts sit in flight before any finish. Measured `completed/min` during a bounded sample was ≈0 at every arm — the empirical proof.
- **Pick concurrency by use-case, not by peak dec/s:**
  - *Short collect-and-stop:* `--concurrency 96–128` — fills in ~25–30 min so completions actually flow during the window, and far fewer are abandoned on stop.
  - *Long run-to-target (hours):* `--concurrency 256` — ~5.7 completed/min (~340/hr), fill amortised.
- **Planning rule:** full-game E4B-it·thinking yields ~**300–360 completed rollouts/hour** at steady state, with a ~30–45 min fill before completions appear — only efficient in ≥1-hour blocks.

(The earlier act-1 sweep crowned 384 because it measured aggregate dec/s on *short* rollouts; that is the wrong objective for full-game collection.)

## 4. Failure-mode analysis (240 rollouts)

Computed by `scratch/failure_analysis.py` → `scratch/e4b/failure_analysis.json` + plots. All numbers below are over the 240 completed rollouts (42,184 decisions).

### Headline
- **0 wins / 240.** Coherent but weak. Only **5 format failures in 42,184 decisions (0.01%)** → the structured-map fix holds at temp 1.0; essentially every death is a *gameplay* death, not a parse artifact.
- Depth: final floor **mean 12.5 / median 12 / max 33**; **15% reach Act 2 (36/240), 0% reach Act 3**.

### Two death walls, one dominant killer
Death floors spike at **floor 6 (59 deaths)** and **floor 16 (57 deaths)**. Of 235 deaths: **60% die before the Act-1 boss**, 24% **at** it (floor 16/17), 15% in Act 2+.

Terminal-encounter tally (what's on screen at death):

| Encounter | Deaths | Note |
|---|--:|---|
| **GREMLIN_NOB** | **77** | floor-6 elite — **33% of all deaths** |
| LAGAVULIN | 29 | act-1 elite |
| THE_GUARDIAN | 24 | act-1 boss |
| HEXAGHOST | 20 | act-1 boss |
| SENTRY ×2 | 11 | act-2 elite |
| (long tail) | … | Chosen, Book of Stabbing, slimes, … |

**The floor-6 elite Gremlin Nob is the single biggest wall** (a third of all deaths). Gremlin Nob enrages (gains Strength) whenever the player plays a **Skill** card; a Defend-heavy starter feeds it — a *plausible, specific* mechanism, not yet confirmed from the card-play sequences.

### It bleeds out — it isn't one-shot
- **88% attrition vs 12% burst** (burst = a single ≥40%-of-max-HP hit in the last 15 decisions). Largest final drop: attrition median **20%**, burst median **44%** of max HP.
- Dies at a **median of 8 HP** (mean 10.8) — a slow grind to near-zero, then a small final hit.
- **HP erosion curve** (mean player HP by floor): 80 → 64 (f6) → 54 (f12) → 52 (f16); the bottom quartile is already ~28 HP by the Act-1 boss. It arrives at elites/bosses too gutted to survive. (The apparent jump to ~83 HP at floor 17 is survivorship — only boss-beaters reach Act 2.)

### It's *aware* it's in danger — and dies anyway
- **100% of deaths (235/235)** mention danger in their final pre-death reasoning: "incoming" (235), "block" (232), "die"/"death" (179), "survive"/"survival" (105), "lethal" (12), "fatal" (15).
- Final pre-death decision: ~466 thinking tokens, ~548 completion tokens; overall ~791 thinking tokens/decision with no truncation.
- **So deaths are tactical/deckbuilding weakness, not obliviousness** — it knows it needs to block/kill and can't generate enough.

### Behaviour
- Decision mix: mean **176 decisions/rollout** (median 159, max 677); **combat 130/rollout (74%)**, out-of-combat 46/rollout (26%).
- Campfire: **rest 237 : smith 101 : recall 18** (rest:smith ≈ 2.3:1) — **rest-heavy / HP-conservative**, consistent with prior Qwen findings; over-resting forgoes upgrades that might help it survive elites.

### Caveat
This is the **240 completed** rollouts; the **382 in-flight at the stop were abandoned** (those were the longest-running, often deepest/strongest runs). So Act-2 reach is, if anything, **under-counted** here.

## 5. Implications & next steps

- **Clean RL signal.** A strong, consistent competence gradient (survive the floor-6 elite → reach the Act-1 boss → beat it → Act 2) with negligible format noise. Plenty of headroom; nobody wins yet.
- **Trait axis already visible and separable from awareness** (rest-vs-smith, HP-conservatism) — the kind of behaviour the framing experiment aims to move.
- **Concurrency:** future full-game collection should use `--concurrency 96–128` (collect-and-stop) or `256` (long run-to-target), not the act-1-derived 384.
- **Optional deeper dive:** confirm the Gremlin Nob mechanism by pulling card-play sequences in those 77 deaths (is it feeding Enrage with Skills?).

## 6. Reproduction

Analysis + benchmark artifacts (this commit, under `scratch/`):
- `scratch/failure_analysis.py` — the failure-mode analysis (run: `PYTHONPATH=src .venv/bin/python scratch/failure_analysis.py`). Outputs `scratch/e4b/failure_analysis.json` and 3 plots: `hp_erosion_curve.png`, `largest_final_hp_drop_hist.png`, `death_final_floor_distribution.png`.
- `scratch/bench_concurrency.sh`, `scratch/fullgame_bench.sh` — the two concurrency benchmarks (run on the pod).
- `scratch/e4b_progress.py` — live progress/rate monitor used during collection.

Not committed (intentionally): the **240-rollout dataset** (`data/rollouts/e4b_think_perf/…`, ~700 MB, gitignored). Re-collect with `scripts/run_until.py` using the §2 config.

Infra change landed elsewhere this session: `scripts/run_until.py` gained `--top-p` / `--top-k` (threaded into `build_agent`), so the vLLM Gemma sampling params are honoured and recorded in `meta.extra.agent_config`.
