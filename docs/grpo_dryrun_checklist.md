# GRPO dry-run checklist (validation discipline)

**Why this exists.** A prior session lost a multi-hour training launch because a dry-run
omitted `--wandb-project`/`--manifest` and a latent crash slipped through. The rule
(deliverable 6 of [`grpo_readiness_handoff.md`](grpo_readiness_handoff.md)): **every new GPU
code path gets a dry-run that mirrors the EXACT production flag set** — same flags, smaller
numbers — *before* the real run. Cheap dry-runs >> failed multi-hour GPU cycles.

Run these on the pod (CUDA-13 H100, `.[vllm]` + `.[train-cuda]` installed, simulator built)
in order; each is seconds-to-minutes. Set `VLLM_USE_FLASHINFER_SAMPLER=0`,
`export PYTHONPATH=src`, `PY=.venv/bin/python`. Use a small/fast model for the dry-run
(`DRY_MODEL`, e.g. a small Gemma/Qwen) where the flag set — not the model — is what's under test.

The golden rule: **copy the production command, shrink the numbers, change NOTHING else.**
If production passes `--wandb-project X --manifest Y --thinking`, the dry-run passes them too.

---

## 1. Lower-variance eval — `run_until.py` (K-per-seed + split)

Production: K rollouts/seed on the frozen eval split, base then trained.
```
$PY scripts/run_until.py --model "$MODEL" --backend vllm \
  --seeds-config configs/frozen_seeds.json --split eval --rollouts-per-seed 4 \
  --concurrency 96 --thinking --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 8192 \
  --combat-control llm --max-act 3 --battle-simulations 50 \
  [--adapter-path "$OUT/adapter" --max-lora-rank 16] --output-dir "$OUT/eval/{base,trained}"
```
Dry-run (same flags; smoke split, K=1, tiny model, shallow game):
```
$PY scripts/run_until.py --model "$DRY_MODEL" --backend vllm \
  --seeds-config configs/frozen_seeds.json --split smoke --rollouts-per-seed 1 \
  --concurrency 4 --thinking --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 1024 \
  --combat-control llm --max-act 1 --max-decisions 60 --battle-simulations 20 \
  --output-dir /tmp/dry/eval/base
```
And immediately re-run with `--adapter-path <a tiny adapter> --max-lora-rank 16 --output-dir /tmp/dry/eval/trained` so the LoRA-load path is exercised too.
Preflight: the script prints the launch line + an empty-specs guard (returns before building the agent if everything is already on disk). Confirm the outcome histogram prints and `seed_*_r*.meta.json` sidecars are written for each `(seed, idx)`.

## 2. Paired comparison — `compare_paired.py` (CPU, no GPU)
```
$PY scripts/compare_paired.py --base "$OUT/eval/base" --trained "$OUT/eval/trained" \
  --metric final_floor --min-act 1 --out "$OUT/eval/paired.json"
```
Dry-run: point it at the §1 `/tmp/dry/eval/{base,trained}` dirs. Confirm it prints a NON-ZERO
`paired_seeds` count plus the paired delta + sign-test p + bootstrap CI and the per-arm
`agent_invalid_rate`/`budget_truncated_rate`. `load_metas` recurses into the agent-label subdir
that `run_until` writes (`output_dir/<label>/seed_*_r*.meta.json`), and the CLI **errors (exit 2)**
if an arm has zero rollouts — so a `paired_seeds: 0` can only mean genuinely-empty inputs, not a
dir-level mismatch. This is CPU and unit-tested.

## 3. Offline PG dataset — `build_pg_dataset.py`
Production (signed-advantage stepping stone, from raw rollouts + final_floor):
```
$PY scripts/build_pg_dataset.py --rollout-dir "$ROLLOUT_DIR" --tokenizer "$MODEL" \
  --mode offline --baseline median --allow-thinking --out "$OUT/pg/offline.jsonl"
```
Dry-run: same flags, point `--rollout-dir` at a handful of existing rollouts (or the §1 dry
output). Confirm it writes the JSONL + `.manifest.json` and the report prints
`n_trajectories_with_advantage` + advantage min/max/mean. **Keep `--allow-thinking` if the
production data is native-thinking** (omitting it makes the builder refuse — catch that here).

## 4. PG training — `train_pg.py`
Production (offline signed-advantage):
```
$PY scripts/train_pg.py --dataset "$OUT/pg/offline.jsonl" --base-model "$MODEL" \
  --manifest "$OUT/pg/offline.jsonl.manifest.json" --out "$OUT/pg/adapter" \
  --lora-r 16 --lora-alpha 32 --epochs 1 --grad-accum 16 --max-seq-len 4096 \
  --learning-rate 1e-5 --clip-eps 0.2 --kl-beta 0.02 \
  --wandb-project sts-e4b-offline-pg --run-name offline-signed-adv
```
Dry-run: **identical flags incl. `--manifest` and `--wandb-project`** (this is exactly the pair
the lost-launch dry-run omitted), tiny model, a 5-10 row dataset, `--max-seq-len 512`. Confirm:
the manifest chat_template_hash skew-check passes (or fails loudly — that is the guard working),
wandb initializes, and an adapter is written to `--out`. A crash here costs seconds; in
production it costs hours.

## 5. GRPO loop — `run_grpo.py` (the in-process capstone)
Production:
```
$PY scripts/run_grpo.py --base-model "$MODEL" --tokenizer "$MODEL" \
  --train-seeds-config configs/frozen_seeds.json --train-split train \
  --out-dir "$OUT/grpo" --num-iterations 20 --group-size 8 --seeds-per-iter 8 \
  --concurrency 96 --temperature 1.0 --top-p 0.95 --top-k 64 \
  --combat-control llm --max-act 3 --max-decisions 1500 --battle-simulations 50 \
  --clip-eps 0.2 --kl-beta 0.02 --learning-rate 1e-5
```
Dry-run (1 iteration, 2 seeds × G=2, tiny model, shallow game):
```
$PY scripts/run_grpo.py --base-model "$DRY_MODEL" --tokenizer "$DRY_MODEL" \
  --train-seeds-config configs/frozen_seeds.json --train-split smoke \
  --out-dir /tmp/dry/grpo --num-iterations 1 --group-size 2 --seeds-per-iter 2 \
  --concurrency 4 --temperature 1.0 --top-p 0.95 --top-k 64 \
  --combat-control llm --max-act 1 --max-decisions 60 --battle-simulations 20 \
  --clip-eps 0.2 --kl-beta 0.02 --learning-rate 1e-5
```
This is the **#1 dry-run target** because it exercises the in-process co-residency that nothing
else does. Confirm in one iteration: vLLM constructs with `enable_lora=True, enable_sleep_mode=True`;
generation runs; **`agent.sleep()` actually frees enough GPU for the trainer to load** (the whole
point of sleep mode — if it OOMs here, lower vLLM `gpu_memory_utilization`); `train_pg` produces an
adapter; `agent.set_adapter()` hot-swaps it (the next iteration's generation uses it); the returned
summary lists the iteration + `final_adapter`. Watch `nvidia-smi` across the wake→sleep→train→wake
transitions to confirm the memory hand-off.

## Known guards already in the code (the dry-run confirms they fire)
- `train_pg` / `train_policy`: manifest `chat_template_hash` skew-check raises on a tokenizer/template
  mismatch between dataset and base model.
- `build_pg_dataset` / `build_sft_dataset`: refuse native-thinking data without `--allow-thinking`;
  the SFT builder also refuses a sparsity-fallback dataset without `--allow-fallback`.
- `run_until`: empty-specs guard returns before building the (expensive) vLLM agent.
- `run_grpo`: validates `num_iterations`/`group_size`/`concurrency`/`max_decisions >= 1` and `temperature > 0`.

## Regression-test rule
Every bug found by a dry-run gets a regression test (unit if the logic is pure; otherwise a
documented gated/integration test) before the real run. No silent fixes.
