# Rollout throughput benchmarks (local MLX + vLLM/H100)

Measured numbers to help **plan experiments** (how long will N rollouts of model X
take, which model to pick for a given budget). Generated from the 2026-06-15
multi-model sweep (`scripts/run_sweep.py` → `scripts/compare_models.py`).

## Setup

- **Hardware:** Apple M5 Pro, 48 GB (the dev machine). Per-token rates are
  hardware-specific — rescale for other machines.
- **Inference:** `mlx_lm` 0.29.1, 4-bit models, `temperature=0.2`, batched via
  `mlx_lm.batch_generate` (the `parallel_rollout` orchestrator).
- **Sweep config:** `--combat-control llm` (every in-combat micro-decision goes to
  the model), `--seeds 3,4,5,7` (4), `--max-decisions 140`, `--batch-size 8`
  (so effective concurrency K=4, capped by the 4 seeds), `--battle-simulations 50`,
  `--max-tokens 256`.

## vLLM / CUDA path

The vLLM backend now uses streaming continuous batching (`run_streaming_rollouts`)
instead of lockstep rounds. The throughput knob is `--concurrency`; effective
concurrency is `min(concurrency, number of rollout specs)`, so a pod run needs
enough seeds/rollout indices to keep the GPU busy. Prefix caching is enabled by
default for vLLM.

The MLX numbers below are unchanged because MLX still uses `parallel_rollout`
with `--batch-size`. vLLM thinking-mode numbers are measured in the next section
(Gemma 3 & 4 on H100); do not infer CUDA numbers from the local MLX table.

## Results — vLLM on H100 (Gemma 3 & 4)

Measured 2026-06-16 on a RunPod **H100 SXM 80GB** (vLLM 0.23.0, torch cu130 /
CUDA 13.0). Config: `--combat-control llm`, **16 seeds**, `--concurrency 48`
(effective **16** = #seeds, so the GPU is *under-saturated*), `--max-decisions 200`,
`--max-tokens 4096`, `--temperature 0`, `--battle-simulations 50`. Each model ran a
thinking-off and a thinking-on arm (Gemma 3 = *prompted* `<think>` reasoning; Gemma 4
= *native* thinking).

| Model · arm | rollouts | decisions | dec/roll | wall (s) | **dec/s** | s/dec | compl_tok | think_tok | invalid |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma-4-E4B-it · no-think | 16 | 2496 | 156 | 224 | **11.2** | 1.00 | 153 | 0 | 0% |
| gemma-4-E4B-it · think    | 16 | 2425 | 152 | 1103 | **2.2** | 5.16 | 828 | 0\* | 0% |
| gemma-3-12b-it · no-think | 16 | 2617 | 164 | 237 | **11.0** | 1.15 | 87 | 0 | 0% |
| gemma-3-12b-it · think    | 16 | 2828 | 177 | 568 | **5.0** | 2.68 | 212 | 168 | 0% |

- `wall` = max over the 16 concurrent rollouts of that rollout's summed per-decision
  `latency_s` (≈ arm wall-clock at effective concurrency 16; excludes one-time model
  load). `dec/s` = decisions ÷ wall; `s/dec` = avg per-decision turnaround under load.
- **0% invalid everywhere** — `--max-tokens 4096` is ample for these models (no
  truncated-before-JSON failures, even when thinking).
- **Thinking cost:** Gemma-3-12B ≈ 2.2× slower (5.0 vs 11.0 dec/s); Gemma-4-E4B ≈ 5×
  slower (2.2 vs 11.2) because it reasons far longer (~828 vs 153 completion tokens).
  Both no-think arms ≈ 11 dec/s.
- **Under-saturation:** only 16 rollouts were in flight (concurrency cap 48), so these
  dec/s are a **lower bound** — more seeds/rollout-indices raise them.
- **\* Gemma-4 `think_tok=0` is a capture gap, not "no reasoning":** Gemma-4 emits its
  native reasoning under a `thought` channel (the completion starts with `thought\n…`),
  not `<think>…</think>`, so the harness's `<think>` parser records 0 thinking tokens
  and the reasoning stays mixed into `raw_response`/`completion`. The final JSON still
  parses (hence 0% invalid). **This run predates the parser fix** — `parse_json_action`
  now captures Gemma-4's `thought` channel (tagging `metadata.reasoning_format =
  "gemma_thought"`), so re-runs record `think_tok` correctly; see `src/sts_ai/CLAUDE.md`
  (the `agents.py` reasoning-mode note).

Reproduce on a pod (needs a CUDA-13.0-driver H100 for the cu130 `vllm==0.23.0`):

```bash
# one model per process (vLLM doesn't reliably free GPU mem between in-process loads)
printf 'google/gemma-4-E4B-it\ngoogle/gemma-3-12b-it\n' > scripts/runpod/models_gemma.txt
HF_TOKEN=<token> bash scripts/runpod/run_sweep_on_pod.sh scripts/runpod/models_gemma.txt data/rollouts/gemma_bench
PYTHONPATH=src .venv/bin/python scripts/compare_models.py data/rollouts/gemma_bench
```

## Results — valid (no-thinking) arms

Each arm = 4 rollouts (Ironclad act 1, seeds 3/4/5/7).

| Model (no-think, 4-bit) | rollouts | decisions | dec/rollout | gen wall (s) | **dec/s** | s/dec | avg completion tok | invalid-JSON |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen3-1.7B | 4 | 443 | 111 | 210 | **2.11** | 0.47 | 84 | 3% |
| Llama-3.2-3B-Instruct | 4 | 367 | 92 | 285 | **1.29** | 0.78 | 62 | 5% |
| Qwen3-4B | 4 | 554 | 139 | 774 | **0.72** | 1.40 | 86 | 8% |

- **dec/s** is the headline planning number (decisions per second, batched K=4).
- "gen wall" = Σ per-decision `latency_s` (this equals the actual batched
  generation wall-time; see caveats). Excludes one-time model load (~1 s on this
  box) and env/sim stepping (small in llm mode).
- Bigger model ⇒ stronger play but slower: Qwen3-4B reaches ~floor 12 and ~138
  decisions/rollout (survives longer, so *more* decisions to generate) at ~0.72
  dec/s; Qwen3-1.7B is ~3× faster per decision but dies earlier (~floor 7).

### Estimating a run

`wall_seconds ≈ (rollouts × dec_per_rollout) / dec_per_s   (+ ~1 s model load)`

e.g. 100 Qwen3-4B no-think rollouts of full act 1 ≈ `100 × 139 / 0.72 ≈ 5.4 h`
at K=4. **More concurrent rollouts (more seeds, larger `--batch-size`) raise
dec/s** — K=4 here under-uses the batch; K=16 was ~2× the K=1 rate in the
throughput investigation, and grows with longer generations.

## Omitted: thinking / reasoning arms (degenerate at max_tokens=256)

The Qwen3-1.7B-think, Qwen3-4B-think, and DeepSeek-R1-Distill-1.5B arms ran but
are **excluded** because at `--max-tokens 256` they were **100% invalid**: the
model spends the whole budget reasoning, never emits the closing JSON, and the
old rollout loop fell back to action 0 (degenerate "always first action"). Their timings
(e.g. Qwen3-4B-think ~2.5 s/dec, ~256 completion tokens every step = the cap) only
measure truncated reasoning, not real decisions.

**To benchmark reasoning properly, rerun with `--max-tokens 4096`** (now the
default — see `scripts/CLAUDE.md`). Expect reasoning to be **roughly 8× slower per
decision** than no-think (the token counts above show ~255 vs ~85 completion
tokens), and the per-decision wall scales with tokens generated.

## Reproduce / refresh

```bash
PYTHONPATH=src .venv/bin/python scripts/run_sweep.py \
  --models mlx-community/Qwen3-1.7B-4bit,mlx-community/Qwen3-4B-4bit \
  --thinking off --seeds 3,4,5,7 --max-decisions 140 --batch-size 8 \
  --output-dir data/rollouts/sweep
PYTHONPATH=src .venv/bin/python scripts/compare_models.py data/rollouts/sweep
```
Per-arm timing is `Σ agent.latency_s` over the arm's decision records; per-rollout
counts/outcomes are in each `seed_*.meta.json` sidecar.
