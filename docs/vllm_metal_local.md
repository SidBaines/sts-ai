# Local vLLM on Apple Silicon (`vllm-metal`)

**Status:** adopted 2026-07-06 for local **generation** (eval + data collection). Not a training path.
**One-liner:** run our existing `VllmJsonAgent` + `streaming_rollout` locally on the Mac, so the code path we
develop/test on Apple Silicon is the *same* one we deploy on the CUDA pod — plus a ~1.3–2× speedup on thinking
rollouts from continuous batching that `mlx-lm` can't do.

## When to use which local backend

| | `--backend mlx` (existing) | `--backend vllm` (via vllm-metal) |
|---|---|---|
| engine | `mlx-lm` lockstep (`parallel_rollout`) | vLLM continuous batching (`streaming_rollout`) |
| raw batched throughput | ~parity | ~parity (vllm-metal runs MLX kernels underneath) |
| variable-length (thinking) rollouts | slower (waits for slowest-of-K each round) | **~1.3–2× faster** (refills finished slots) |
| LoRA / adapter eval | ✅ (`--adapter-path`) | Direct LoRA ❌; fused MLX model ✅ |
| code path parity with CUDA pod | different orchestrator | **same** `VllmJsonAgent`/`streaming_rollout` as the pod |

**Rule of thumb:** for local *thinking* rollout generation at K≥8, prefer `--backend vllm`. For a live LoRA
adapter (`--adapter-path`) or quick single rollouts, stay on `--backend mlx`. If MLX lockstep eval is too slow,
fuse the adapter into a standalone MLX model and run that model through `--backend vllm`. **Training
(SFT/RWR/GRPO) is unaffected** — it stays on `mlx_lm lora` (MPS) or TRL (CUDA); vllm-metal has no LoRA/sleep and
does not replace either.

## Benchmark (M5 Pro, 48 GB, macOS 26.5, gemma-4-e4b-it-bf16)

Measured 2026-07-06. vllm-metal **0.3.0.dev** (vLLM **0.24.0** core), `mlx-lm` **0.31.3**.

**1. Raw batched decode — parity with `mlx-lm`** (fixed-length decode, temp 0, batch 32):

| path | batch 1 | batch 32 | scaling |
|---|--:|--:|--:|
| vllm-metal (paged/continuous) | 24.7 tok/s | 321 tok/s | 13.0× |
| `mlx-lm` `batch_generate` | 27.0 tok/s | 337 tok/s | 12.5× |

No kernel-level speedup — both are MLX underneath. The ~13× batching lever is available on *either* backend.

**2. Continuous batching — the vLLM-exclusive win** (same vLLM engine, thinking on, 12 seeds, K=12, 144
decisions, `--combat-control llm`, only the orchestrator differs):

| orchestrator | wall | decisions/s | gen tok/s | valid |
|---|--:|--:|--:|--:|
| lockstep (`parallel_rollout`) | 21.9 min | 0.110 | 81.6 | 100% |
| streaming (`streaming_rollout`) | 16.4 min | 0.146 | 105.4 | 100% |
| **ratio** | **1.33×** | **1.33×** | 1.29× | — |

Work was matched (mean gen 744 vs 720 tok; think-token spread ~120–1560, the variance continuous batching
exploits). **1.33× is a lower bound** — this run used `specs == K` with a uniform decision cap, so it captures
*only* the per-round straggler effect. A full-game run (games terminating at wildly different decision counts,
with a refill queue `specs >> K`) adds a second source of streaming advantage; expect ~1.3–2× in practice.

## Setup (one-time)

Requires native **arm64 Python 3.12** + Xcode Command Line Tools (`xcode-select --install`). The installer builds
a **dedicated venv** at `~/.venv-vllm-metal` (it compiles vLLM 0.24 core from source via clang + installs a
prebuilt Metal plugin wheel — a few minutes). It pulls a CPU torch, so **keep it separate from the project
`.venv`; do not merge** (that torch would conflict with the CUDA `[vllm]==0.23.0` extra).

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
# verify:
~/.venv-vllm-metal/bin/python -c "import vllm; print(vllm.__version__)"   # -> 0.24.0
```

**No simulator rebuild needed.** The project `.venv` and `~/.venv-vllm-metal` use the *same* uv-managed CPython
3.12.13, so the already-built `external/sts_lightspeed/build/slaythespire.cpython-312-darwin.so` is ABI-compatible
and imports directly under the vllm-metal venv (verify with `PYTHONPATH=src ~/.venv-vllm-metal/bin/python -c "from
sts_ai.lightspeed import LightspeedHybridEnv; LightspeedHybridEnv(world_seed=1).advance_to_decision()"`). If the two
venvs ever diverge in Python patch version, run `scripts/build_lightspeed.sh` under the vllm-metal interpreter.

## Running local rollouts

Invoke the normal scripts with the vllm-metal interpreter (`PYTHONPATH=src` as always). Gemma-4 wants
`--top-p 0.95 --top-k 64`:

```bash
PYTHONPATH=src ~/.venv-vllm-metal/bin/python scripts/run_sweep.py \
  --backend vllm --models mlx-community/gemma-4-e4b-it-bf16 \
  --thinking on --top-p 0.95 --top-k 64 \
  --seeds 70,71,72,73,74,75,76,77,78,79,80,81 --concurrency 12 \
  --combat-control llm --max-act 1 --output-dir data/rollouts/local_vllm

# single-model full-game / eval-split runner works the same way:
PYTHONPATH=src ~/.venv-vllm-metal/bin/python scripts/run_until.py \
  --backend vllm --model mlx-community/gemma-4-e4b-it-bf16 --thinking \
  --top-p 0.95 --top-k 64 --concurrency 12 --split eval \
  --seeds-config configs/frozen_seeds.json --output-dir data/eval/local_vllm
```

Traces/metas are shape-identical to the MLX and CUDA paths (same `DecisionRecord` schema), so
`compare_models.py` / `compare_paired.py` / `visualize_rollout.py` all consume them unchanged.

### Fused adapter eval

`vllm-metal` cannot serve a LoRA adapter separately on Metal, but it can load a full model directory produced by
`mlx_lm fuse`. This is the local escape hatch for adapter evals that need continuous batching:

```bash
PYTHONPATH=src .venv/bin/python -m mlx_lm fuse \
  --model mlx-community/gemma-4-e4b-it-bf16 \
  --adapter-path data/local_curricula/gremlin_nob/adapters/rwr_sft_won \
  --save-path data/local_curricula/gremlin_nob/models/rwr_sft_won_fused

PYTHONPATH=src ~/.venv-vllm-metal/bin/python scripts/local_task_eval.py \
  --task gremlin_nob \
  --manifest data/local_curricula/gremlin_nob/manifests/source.json \
  --model data/local_curricula/gremlin_nob/models/rwr_sft_won_fused \
  --backend vllm --split holdout --thinking \
  --temperature 0 --top-p 0.95 --top-k 64 --max-tokens 8192 \
  --concurrency 11 \
  --output-dir data/local_curricula/gremlin_nob/eval/rwr_sft_won_fused_vllm
```

Observed caveat (2026-07-07, Gemma-4 E4B Nob RWR/SFT): fused-model eval itself works, but it faithfully reflects
whatever the adapter learned. The first Nob adapter loaded and generated cleanly, yet no longer used Gemma's native
`thought` channel because the MLX SFT data path had fed assistant `content` through the stock `ChatDataset`, whose
Gemma template stripped `<|channel>thought...<channel|>` before training. The corrected path bypasses that dataset:
`train_mlx.prepare_native_mlx_data` pre-tokenizes `prompt + completion + "<turn|>\n"`, preserves the thought channel,
and drops/counts samples whose thought would be truncated. A fused corrected adapter (`rwr_sft_won_native8k_stop`)
emitted Gemma thought tokens under vLLM and stopped cleanly. Treat fused-model eval as a valid local adapter arm, but
check `meta.extra.agent_config`, `agent.raw_response`, and token counts before comparing speed or reasoning behaviour
against a native-thinking base arm.

## Gotchas (Metal-specific)

- **macOS `spawn` needs an `if __name__ == "__main__":` guard.** vLLM's V1 `EngineCore` runs in a spawned
  subprocess that re-imports the entry module; without the guard it recursively rebuilds the `LLM` and dies with a
  `freeze_support()` error. The repo entrypoints (`run_sweep.py`, `run_until.py`, …) are already guarded — but any
  **ad-hoc script** that constructs a vLLM `LLM`/`VllmJsonAgent` at import time must wrap its body in `main()` under
  the guard.
- **Cosmetic `mps` allocator assert on engine teardown.** On process exit the EngineCore subprocess prints a
  `RuntimeError: ... Allocator for mps is not a DeviceAllocator` from torch's MPS caching allocator. It fires
  **after** all results are produced (exit code stays 0) — ignore it. Don't gate success on a clean stderr.
- **gemma-4 is "experimental"** in vllm-metal's support matrix (Qwen3 is fully supported). It worked cleanly here
  (288/288 decisions valid across both benchmark arms), but treat it as not-yet-guaranteed across upgrades.
- **No live LoRA, no sleep mode on Metal.** `VllmJsonAgent` defaults (`enable_lora=False`,
  `enable_sleep_mode=False`) keep plain generation compatible; do **not** pass `--adapter-path` on this path (that
  would request live LoRA). Use `mlx_lm fuse` first if you need local continuous-batching eval of an adapter. GRPO's
  co-resident sleep/wake hot-swap is CUDA-only.
- **`gpu_memory_utilization` (default 0.90) is honored on Metal** (unified memory); it loaded and KV-cached fine.
  Note vLLM reports "available" GB depressed by the OS page cache from the model read — not an OOM signal.

## What this is not
- Not a raw-speed upgrade (parity with the existing MLX batched path).
- Not a training path (generation only; no live LoRA/sleep).
- Not a dependency of the core package — it lives in its own installer venv, not a pip extra.
</content>
