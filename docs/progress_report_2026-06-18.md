# Progress report — 2026-06-18: first offline-BC training run + GPU eval

**Status:** the offline filtered-BC training pipeline is built and ran end-to-end
(generate → filter → LoRA → fuse → convert → vLLM eval). **Results below are
PARTIAL** (trained-arm eval still running at write time) — see the ⚠️ markers and
overwrite once N=60 lands. Branch: `feat/mlx-e4b-local` (built on
`feat/offline-filtered-bc-training`).

## TL;DR

- Built the **Stage-6 offline filtered-BC / expert-iteration pipeline** for
  Gemma-4 **E4B**, via codex-driven-development (Codex builder + Claude
  spec/quality/cross-task review). 14 commits.
- Migrated the project to **Python 3.12** and rebuilt the C++ simulator, because
  Gemma-4 on MLX needs `mlx>=0.30.4` (Python ≥3.10).
- Ran the **first training**: filtered-BC on the 240 E4B *thinking* rollouts →
  36 act-boss-clear positives → 12,630 decision examples → MLX LoRA. Held-out
  validation loss dropped 0.522 → ~0.44 but **plateaued by ~iter 1,000**; we ran
  to 10,000 (overtrained).
- Stood up an **H100 GPU eval** (base vs trained on fresh held-out seeds). Solved
  the anticipated **MLX-LoRA → vLLM** compatibility problems (fuse → text-only
  arch conversion → flashinfer/nvcc workaround).
- **Result (final N=60): the trained adapter plays modestly BETTER than base** —
  mean floor 12.80 → **13.95**, boss-clear 10% → **17%**, max 25 → **33**, invalid
  ~0 in both. So the feasibility question is answered **yes**: filtered-BC made E4B
  play better. (An early *partial* read looked worse — a completion-order artifact;
  see §5/§6.) The improvement is modest and not strongly powered at N=60; the
  **native-thinking-format** training defect (§6) is still worth fixing and may
  unlock more.

## 1. What we built (pipeline)

New `src/sts_ai/train/` package + scripts (all unit-tested; heavy ML libs
lazy-imported so the core stays dependency-free):

- `reward.py` — per-trajectory **act-boss-clear** label (monotonic milestone:
  `final_act > min_act` or VICTORY) + `min_positives` sparsity guardrail with a
  top-floor-quantile fallback.
- `sft_format.py` — **skew-free** reconstruction. Canonical training surface is
  role-based **`messages`** = `[{user: render_action_prompt output}, {assistant:
  raw_response}]`; each backend applies the chat template **once**. Single shared
  `chat_template_probe_hash` guards build↔train tokenizer/template skew.
- `dataset_builder.py` + `scripts/build_sft_dataset.py` — reward-join → filter →
  examples + manifest; derives `enable_thinking`/`induce_reasoning` from the run's
  `reasoning_mode`; **skips** retry-augmented / agent-invalid / terminal records;
  sparsity guardrail refuses a fallback dataset without `--allow-fallback`.
- `train_mlx.py` / `train_trl.py` behind `scripts/train_policy.py --backend {mlx,trl}`.
  MLX uses `mlx_lm lora --mask-prompt` (assistant-only loss); TRL uses conversational
  `messages` + `assistant_only_loss`. wandb + periodic val-loss wired.
- Adapter-aware eval: `adapter_path` plumbed through `agent_factory` + both agents
  + `run_until.py --adapter-path` (no-adapter path unchanged).
- `scripts/train_overnight_e4b.py` — seed-quarantined train/val split + wandb launcher.

**Two correctness bugs found + fixed during the build:**
1. **Build↔train chat-template hash skew** — consumer recomputed the hash without
   `enable_thinking`; factored a single shared helper.
2. **Double chat-templating + missing `--mask-prompt`** — the pre-templated prompt
   double-templated under mlx-lm's `CompletionsDataset`, and without `--mask-prompt`
   mlx trained on prompt+completion. Fixed by the role-based `messages` format +
   `--mask-prompt`.

## 2. Gemma-4 E4B on MLX (environment findings)

- **`mlx-community/gemma-4-e4b-it-bf16`** is the non-quantized local MLX build
  (15 GB; ~15 GB peak RAM). 4-bit exists but bf16 is better for LoRA quality.
- **mlx-lm ≥ 0.31.2** is the first Gemma-4-capable release; it ships a native
  `gemma4_text` decoder, so the **mlx-lm** path runs E4B text (no `mlx-vlm`
  needed — that's only for vision).
- **Python ≥ 3.10 required** — `mlx ≥ 0.30.4` has no cp39 wheels. We migrated
  `.venv` to **python3.12** and rebuilt `sts_lightspeed` against it (pybind 2.13.6
  supports 3.12), so MLX + the simulator now share one interpreter. `requires-python`
  bumped to `>=3.10`.
- Verified end-to-end on the Mac: inference, LoRA training, adapter reload all work.

## 3. First training run

- **Data:** the 240 E4B *thinking* rollouts (`data/rollouts/e4b_think_perf/...`,
  full LLM combat, neutral framing, temp 1.0) → **36 act-boss-clear positives**
  (15%; 0 wins) → **12,630 decision examples**. Seeds 3–596.
- **Quarantine:** 4 of the 36 positive seeds held out for validation (26, 77, 119,
  126) → 11,126 train / 1,504 val examples. Train/val games are disjoint.
- **Config:** MLX LoRA, 8 layers (3.88 M params), `--mask-prompt`, max-seq 4096,
  grad-checkpoint, lr 1e-4, **10,000 iters** (~17 epochs over 32 train trajectories).
- **Loss curve:** val 0.522 → ~0.44 (best 0.411 @ iter 5000), **plateaued by
  ~iter 1,000**; train drifted to ~0.35 → mild overfit after iter ~2,000. wandb
  project `sts-e4b-offline-bc`, run `iter1-thinking-mlx`.
- **Takeaway:** the useful learning happened in the first ~1–2k iters; **the final
  (iter-10000) checkpoint is over-cooked.** Checkpoints saved every 1,000 iters.

## 4. MLX-LoRA → GPU eval: compatibility findings (reusable gotchas)

vLLM (CUDA) cannot load an MLX LoRA adapter, so we **fused** the adapter into a
standalone model (`mlx_lm fuse`) → 14 GB HF model. Getting it to run on vLLM took
three fixes (all the kind of MLX→GPU friction we anticipated):

1. **Multimodal vs text-only loader.** The fused config is
   `Gemma4ForConditionalGeneration` → vLLM routes it through its **multimodal**
   `gemma4_mm` loader, which needs `processor_config.json` (mlx didn't copy it) and
   a `vision_config` + vision weights (the model is text-only). **Fix:** convert to
   vLLM's text-only **`Gemma4ForCausalLM`** — strip the `language_model.` weight
   prefix, flatten `text_config` to top-level, relabel `architectures`. Weights then
   load clean. (Artifact: `data/train_run1/text_iter1_thinking/`.)
2. **flashinfer JIT + no nvcc.** The eval pod's driver had no `nvcc`, so flashinfer
   couldn't JIT its sampling kernel (`ninja … non-zero exit`). **Fix:** run vLLM with
   `VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0`.
3. **CUDA-13 host needed.** vLLM 0.23 pulls cu130 torch; the first H100 host had a
   CUDA-12.8 driver (`torch.cuda.is_available()==False`). **Fix:** redeploy via
   `create-pod-cuda.sh … 13.0` (CUDA-filtered).

**Eval pod:** H100 SXM ($3.29/hr), CUDA-13, `feat/mlx-e4b-local` code rsync'd,
`setup_pod.sh` (vLLM 0.23 + sim). Base = `google/gemma-4-E4B-it` (full MM, loads as
the 240-run did); trained = the text-only converted model.

## 5. Results — final (N=60)

Fresh held-out seeds 600–659 (disjoint from training), full-game thinking, temp 1.0
(matches data-gen). Base re-establishes the baseline on the *same* seeds.

| Arm | n | boss-clear | wins | mean floor | median | max | invalid | stopped |
|---|---|---|---|---|---|---|---|---|
| **base** | 60 | 6 (10%) | 0 | 12.80 | 13.5 | 25 | 0.0002 | 58 terminal, 2 agent_invalid |
| **trained** | 60 | **10 (17%)** | 0 | **13.95** | **14.0** | **33** | **0.0000** | 60 terminal |

**Trained beats base on every metric**: mean floor **+1.15**, boss-clear **+7 pp**
(10→17%), max **+8** (33 = into Act 2), and **zero** invalid (base had 2). Modest but
consistent — a **positive trainability result**: filtered-BC made E4B play better.

- Base mean floor (12.80) ≈ the original baseline (12.5) → fresh seeds comparable.
- invalid ~0 in both → the gain is policy, not format.
- **Caveat (significance):** +1.15 mean floor at N=60 is ~1 standard error; the
  direction is consistent across all metrics but the gain is **not strongly
  powered**. Confirm with the loader-confound control (§7.2) and/or more seeds.

> **⚠️ Partial-read lesson.** At 36/60 the trained arm looked *worse* (mean floor
> 9.94, 0 clears). `run_until` launches all rollouts concurrently and **fast-dying
> (low-floor) runs complete first**, so partial means are biased low until the
> slow/deep runs land (trained climbed 9.94 @36 → 11.22 @46 → 13.95 @60). Base was
> already 60/60 (unbiased) while trained was mid-completion (biased low) → the early
> comparison was apples-to-oranges. **Only compare at full N (or equal completion).**

## 6. Reading the result (and the partial-read reversal)

**The trained model improved modestly** (§5). The early "regression" I reported was a
**completion-order artifact** (fast-dying runs finish first → partial means biased
low), not a real effect — see the partial-read lesson in §5. Compare only at full N.

**On overtraining:** we ran 10,000 iters (~17 epochs) on 32 trajectories, well past
the ~iter-1,000 val-loss plateau, and fused the **final** checkpoint — yet it still
beat base. So the final checkpoint was *not* catastrophically overfit. It may still
be past the optimum: an **earlier checkpoint (iter 1,000–2,000)** could match/beat it
more cleanly and trains far cheaper — worth a quick check.

**It improved *despite* a flawed thinking target.** We trained on a marker-stripped
transcript of the model's reasoning (below) — a known defect; fixing it is a plausible
lever for a *larger* gain. Per this session's decision, the next run trains on the
**correct native thinking format.**

### The native-thinking-format issue (and the fix)

Gemma-4 doesn't write reasoning as plain prose — it emits it through a dedicated
**"thought channel"** marked by special control tokens (`<|channel|>thought … <|channel|>answer …`).
When we generated the 240 rollouts, vLLM's default `skip_special_tokens=True`
**stripped those channel markers**, so each stored `raw_response` is the *flattened*
text:
```
thought
<reasoning as plain text>
...
{"reasoning": "...", "action_index": N}
```
Our SFT then trained the model to reproduce that flattened string **as an ordinary
assistant turn** — i.e. "emit `thought\n…{json}` as plain output tokens", **not**
via its native thought-channel. That creates a train/inference representation
mismatch: at eval the model reasons through the channel, but we fine-tuned it toward
a lossy, marker-stripped transcript of that reasoning. Fine-tuning on a slightly-off
copy of the model's own reasoning can **interfere with the mechanism that produces
good moves** rather than reinforce it — "blunting the reasoning that drives play."

**Decision / fix: train with the correct native thinking format.** Concretely, for
the next data-generation + training run:
1. **At generation**, retain the thinking-channel tokens — generate with
   `skip_special_tokens=False` (or capture `output.outputs[0].token_ids` and decode
   without stripping) so `raw_response` preserves the native `<|channel|>thought…`
   structure. (Touches `VllmJsonAgent`; keep the JSON parser tolerant of the extra
   special tokens.)
2. **In the SFT target**, build the assistant turn from the channel-structured text
   (not the stripped text), so the model is fine-tuned on exactly the representation
   it uses at inference. Add a check/test that the training target round-trips through
   the chat template to the same token stream the model emits.
3. **Alternative (simplest, original recommendation):** train **no-thinking** — the
   target is just the clean JSON action, sidestepping the channel subtlety entirely.
   Good for a clean trainability baseline; the native-thinking path is needed if/when
   we want to train *through* the reasoning (e.g. for the framing study).

## 7. Recommended next steps

1. **Eval an early checkpoint** (iter 1,000 / 2,000) on the same seeds — the final
   ckpt already beat base; an earlier one may match/beat it and trains far cheaper.
   Fuse + text-only convert as in §4.
2. **Kill the loader confound:** also run the **base** through the text-only
   `Gemma4ForCausalLM` path (not `gemma4_mm`), so both arms use the identical vLLM
   code path. Confirms the +1.15-floor improvement is the model, not the path.
3. **Retrain on the correct native thinking format** (§6) — or a clean **no-thinking**
   run — with **far fewer iters** (~1–2 epochs), and re-eval.
4. **The real ceiling-raiser is winning data:** filtered-BC on 0-win rollouts can only
   amplify "play like your boss-clearing self." Consider more rollouts, hybrid mode
   (search agent clears the game → real positives), or expert-iteration rounds.

## 8. Artifacts & repro

- Branch `feat/mlx-e4b-local` (14 commits off `main`); pipeline in `src/sts_ai/train/`.
- Training: `data/train_run1/` — `overnight_summary.json`, `overnight_train.log`,
  `adapter_iter1_thinking/` (final + per-1000 checkpoints), wandb `sts-e4b-offline-bc`.
- Fused / converted: `data/train_run1/fused_iter1_thinking/` (14 GB, MM config),
  `data/train_run1/text_iter1_thinking/` (vLLM text-only).
- Eval: pod H100 `dzzuyo489zxzvz`; outputs `data/eval_iter1/{base,trained}/`;
  runbook `data/train_run1/GPU_EVAL_RUNBOOK.md`. **Tear the pod down once results pulled.**
- Local MLX work uses `.venv` (python3.12); eval env vars: `VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0`.

## 9. Caveats

- The +1.15 mean-floor gain is **modest and ~1 SE at N=60** — directionally consistent across metrics but not strongly powered; confirm with more seeds + the loader control.
- Loader confound (base via `gemma4_mm`, trained via `gemma4` text-only) not yet controlled (step 7.2).
- Trained on thinking-mode data with the marker-stripped target (§6) — a known defect to fix.
- Single seed-quarantine split (4 val seeds); small N for the binary boss-clear metric → prefer mean floor.
