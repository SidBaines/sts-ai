# Fight-focused training on Gremlin Nob — capability assessment & next steps

**Date:** 2026-07-06 · **Status:** parked (candidate work; a side quest may reshuffle the ordering below)
**Goal:** improve gemma-4-E4B (native thinking) at a *single* fight — Gremlin Nob — as a possible first rung of a
fight-by-fight "learning schedule" (curriculum RL). This doc is the writeup of the codebase capability
investigation so we can pick it up without re-deriving.

## Context / why the Nob

- We are **not** in the framing-experiment phase yet — the near-term goal is competence (get E4B to play well).
- From `scratch/nob_fight_analysis.py` over the 100 base E4B rollouts: **43 Nob fights — 23 won, 20 lost,
  9 "convincing" (≤20 HP lost), incl. seed 74 flawless (0 HP, 3 turns).**
- **The winning policy is attack-forward:** won fights attack 31% / block 23%; lost fights attack 23% / block 28%,
  and in wins the attack share *rises* over turns. The model loses by turtling behind Block, which feeds Enrage.
  (See the plots in `scratch/plots/` and the artifact built by `.../scratchpad/build_nob_artifact.py`.)
- Implication: a reward keyed on "convincingly won the Nob fight" (survived, low HP lost) naturally selects
  attack-forward play, and is **trait-neutral** (an outcome, not a risk/strategy word), so it won't contaminate
  the later framing research (`docs/rl_and_framing_design.md` neutrality invariant).

> **Update (2026-07-07):** the "GRPO is CUDA-only / NVIDIA-only" statements below are now **outdated**. A local
> **MLX GRPO path** was built — `src/sts_ai/train/train_pg_mlx.py` (in-process MLX LoRA policy-gradient trainer) +
> `src/sts_ai/train/mlx_grpo.py`, run via `run_grpo.py --backend mlx`: the same `grpo_loop` in one unified-memory
> process, no vLLM/CUDA, validated end-to-end on Qwen3-1.7B and gemma-4-e4b. My earlier reasons #2/#3 (LoRA
> hot-swap into a separate vLLM engine; the CUDA sleep/wake co-residency dance) were **architecture, not
> fundamentals**, and #1 (no MLX PG trainer) is resolved. It's the **cheap-local-iteration** path (lockstep
> generation, mlx LoRA ≠ PEFT bitwise, no LoRA/sleep on Metal); `--backend cuda` stays the defensible pod path.
> See `docs/grpo_dryrun_checklist.md` §5b and `src/sts_ai/CLAUDE.md`.

## Bottom line

The training *machinery* already exists (SFT/RWR/PG/GRPO). What's missing is **fight-scoped reward** and a
**fight-scoped episode** — both small additions. The single enabling fact:

> Every rollout record stores the model-facing prompt (`state_text`) **and** the verbatim completion
> (`agent.raw_response`, incl. E4B's `thought` channel). So offline training examples reconstruct with **zero
> train/inference skew** (`sts_ai.train.sft_format.build_example`) — **no re-sampling required** to start.

Three rungs, increasing cost (details below):
1. **RWR / filtered-SFT on won-Nob combat decisions** — ~a day of glue, **runs on MPS**, uses existing data.
2. **Hinted-rollout SFT** — the existing "teacher" analogue; breaks the thin-positives ceiling; mostly config.
3. **Nob-scoped GRPO** — the real "learning schedule" rung; ~1–2 days new code; **NVIDIA-only**.

---

## Axis A — Do we have the rollouts / deterministic seeds? **Yes.**

- **Data:** `data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192/` — 100 full E4B rollouts
  (native thinking, full-LLM combat, Acts 1–3), per-decision JSONL + `.meta.json`.
- **Nob fights already extracted & HP-labelled:** `scratch/nob_fight_analysis.py` (43 fights; seeds of the 9
  convincing wins: `47,69,74,79,85,88,111,116,128`).
- Each combat record has everything a reward fn *and* a skew-free SFT target need: `state_text`,
  `agent.raw_response`, `agent.valid`/`retries`, `selected_action`, `legal_actions`, `affordances`
  (lethal-available / full-block-possible), and `state`/`after_state` HP for outcome.
- **Deterministic seeds** → we can regenerate/extend cheaply if the positive set is too thin.

**Caveat:** 9 convincing wins is thin for SFT. Cheap fixes: (a) relax the label to "won" (23), or
(b) generate more via *partial rollouts from the Nob entry* (Axis B, replay primitive) — ~5–10× cheaper than
full games because we skip the pre-Nob game.

---

## Axis B — SFT / RWR / RL / logit distillation? + code deltas

| Method | Exists today? | Delta to focus on the Nob |
|---|---|---|
| **Filtered-SFT / RWR** | ✅ full pipeline (`scripts/build_sft_dataset.py` → `scripts/train_policy.py`); RWR = deterministic resampling→SFT (`train/reward.py::rwr_multiplicities`) | **~100–150 LOC:** a "Nob-fight dataset builder" — reuse the fight-window extraction from `scratch/nob_fight_analysis.py`, label won/convincing by HP, emit `sft_format.build_example()` for those combat records (replicate by HP-margin for RWR). **No new trainer.** |
| **Offline signed-advantage PG** | ✅ `scripts/build_pg_dataset.py` + `scripts/train_pg.py` | Nob reward + un-gate thinking (gates below). CUDA-only. |
| **Online GRPO** | ✅ loop/loss/vLLM-hot-swap built & fake-tested (`train/grpo_loop.py`, `scripts/run_grpo.py`, `scripts/runpod/run_grpo.sh`) | Fight-scoped episode + Nob reward + replay-to-Nob env + un-gate thinking. **~1–2 days.** CUDA-only. |
| **Logit distillation** | ❌ not supported | We store completion *text + token counts* but **no teacher logprobs**. Needs a teacher emitting logits + a KL-distillation loss — neither exists. **Skip** — hinting (rung 2) is a cheaper teacher. |

### Two real gates for the RL path
1. **`train/pg_dataset.py` refuses native-thinking traces** — `require_no_thinking=True` default raises on
   `reasoning_mode="native"` (~line 50). E4B *is* native-thinking, so PG/GRPO on it needs this plumbed off +
   validated. The thinking reconstruction machinery already exists (SFT handles `gemma_thought` passthrough,
   `enable_thinking=True`), so this is plumbing + a test, **not new algorithm**.
2. **Reward is whole-run only** — `train/reward.py` = VICTORY / `final_act>min_act` / `final_floor`. **No
   per-fight reward.** Everything to compute "survived the Nob with ≤N HP lost" is in the traces → new pure fn,
   ~30 lines (effectively already written in `nob_fight_analysis.py`).

### The mid-game-start primitive already exists
`src/sts_ai/interactive/replay.py::replay_actions` is a **pure, reusable** helper: the C++ `GameContext`/
`BattleContext` are opaque (no binary snapshot), so replay re-applies a recorded action sequence into a *fresh*
`LightspeedHybridEnv` to rebuild any position. So **"start a rollout at the Nob entry" = replay that seed's
recorded pre-Nob OOC actions, then let the policy play the combat.** It is **not** wired into any training/CLI
runner yet (only the interactive server) — hooking it into a `make_env` is small glue. Curriculum shape:
*pin the pre-Nob path (from the base trace), sample only the combat.*

---

## Axis C — MPS vs NVIDIA?

- **SFT / RWR → MPS ✅.** `train/train_mlx.py` shells to `mlx_lm lora --train --mask-prompt`. Confirmed the
  installed **mlx-lm 0.31.3 ships `gemma4_text` + `linear_to_lora_layers`**, so LoRA can target the E4B arch
  locally (worth one smoke iter to confirm end-to-end). This is the fast local loop.
- **Any advantage-weighted gradient (offline PG *or* online GRPO) → NVIDIA only.** Trainer is TRL/PEFT on
  torch-CUDA (`train/train_pg_trl.py`); online GRPO also needs vLLM (`VllmJsonAgent`, CUDA-only). **There is no
  MLX policy-gradient trainer** — only SFT/LoRA. RWR is the one reward-driven method that runs on the Mac
  (resampling→SFT, not a gradient with advantages).
- **Generating more Nob data:** MPS works for E4B but slow; vLLM (NVIDIA) is the throughput path. Partial
  rollouts from the Nob entry cut generation cost sharply either way.

---

## Recommended rungs (cheapest signal first)

1. **Rung 1 — RWR-SFT on won-Nob combat decisions, on MPS.** Answers "does concentrating supervision on this
   fight move the needle?" for ~a day and no GPU rental. Clean, trait-neutral reward (attack-forward falls out).
   Deliverables: Nob-fight dataset builder (~100–150 LOC) → `train_policy --backend mlx` → eval the adapter on
   the Nob (paired vs base, HP-lost + win-rate).
2. **Rung 1.5 — Hinted-rollout SFT.** `src/sts_ai/hinting.py` (tactical-truth hints → improved play → launder
   the hint out → SFT) is the existing substitute for logit distillation; an "attack-over-block vs Enraging
   enemies" hint fits its tactical-truth constraint. Breaks the thin-positives ceiling without a teacher model.
3. **Rung 2 — Nob-scoped GRPO** (the RL "learning schedule"). New code: replay-to-Nob `make_env`, combat-scoped
   episode termination + Nob reward, un-gate thinking. ~1–2 days; dry-run per `docs/grpo_dryrun_checklist.md`;
   NVIDIA pod.

## Open questions the side quest may inform
- **Positive-set size:** relax "convincing"→"won", or generate more via partial rollouts? (affects rung 1 data)
- **Thinking vs no-thinking for training:** un-gate native thinking (more faithful to how E4B plays) vs train
  no-thinking (simpler, already supported by PG/GRPO). Affects rungs 2/3.
- **Eval metric for "improved":** win-rate on held-out Nob seeds, mean HP lost, or the attack/block action share
  shifting toward the winning profile. Paired base-vs-adapter via `scripts/compare_paired.py`.
- **Curriculum vs single-fight:** is the Nob a one-off competence fix, or genuinely rung 1 of a per-fight
  schedule (Lagavulin, Sentries, act bosses…)? The replay primitive generalizes to any fight.

## Key file references
- Data: `data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192/`
- Analysis/labels/plots: `scratch/nob_fight_analysis.py`, `scratch/plots/`
- SFT: `src/sts_ai/train/sft_format.py` (`build_example`), `scripts/build_sft_dataset.py`, `scripts/train_policy.py`,
  `src/sts_ai/train/train_mlx.py` (MPS), `src/sts_ai/train/train_trl.py` (CUDA)
- RWR: `src/sts_ai/train/reward.py` (`rwr_multiplicities`)
- PG/GRPO: `src/sts_ai/train/pg_dataset.py` (no-thinking gate ~L50), `scripts/build_pg_dataset.py`,
  `scripts/train_pg.py`, `src/sts_ai/train/train_pg_trl.py`, `src/sts_ai/train/grpo_loop.py`,
  `scripts/run_grpo.py`, `scripts/runpod/run_grpo.sh`
- Reward (whole-run only): `src/sts_ai/train/reward.py`
- Mid-game start: `src/sts_ai/interactive/replay.py` (`replay_actions`)
- Teacher analogue: `src/sts_ai/hinting.py`
- Dry-run discipline: `docs/grpo_dryrun_checklist.md`
</content>
</invoke>
