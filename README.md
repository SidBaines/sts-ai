# SlayTheSpireAI

Minimal local harness for de-risking LLM rollouts on `gamerpuppy/sts_lightspeed`.

## Current MVP

The current adapter is a hybrid environment:

- `sts_lightspeed` resolves combats with its built-in search agent.
- Python controls out-of-combat decisions: Neow, map pathing, rewards, shops, events, card select screens, treasure rooms, and campfires.
- Rollouts are recorded as structured JSONL decision traces.

This is intentionally narrower than full LLM control of every card-play decision. It gets us fast Act 1 trajectory collection and risk-relevant choices while leaving combat action binding as the next C++ task.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip cmake
git clone https://github.com/gamerpuppy/sts_lightspeed external/sts_lightspeed
cd external/sts_lightspeed
git submodule update --init --recursive
../../.venv/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=../../.venv/bin/python \
  -DPYTHON_EXECUTABLE=../../.venv/bin/python \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
../../.venv/bin/cmake --build build --target slaythespire -j 8
```

In this workspace the clone and build have already been done.

## Smoke Rollout

```bash
PYTHONPATH=src .venv/bin/python scripts/run_rollout.py --agent first --seed 1 --max-decisions 20
```

For randomized non-LLM rollouts:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_rollout.py --agent random --seed 1 --max-decisions 50
```

Output defaults to `data/rollouts/rollout_<agent>_<seed>.jsonl`.

## Batch Rollouts

Run a small baseline batch:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_batch.py \
  --agent heuristic \
  --seeds 1,2,3 \
  --max-decisions 80 \
  --battle-simulations 500 \
  --overwrite
```

Summarize generated traces:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_rollouts.py 'data/rollouts/heuristic/*.jsonl'
```

For larger baseline batches, use per-seed subprocess isolation so a slow or faulty simulator seed cannot stop the whole batch:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_batch.py \
  --agent heuristic \
  --seed-start 1 \
  --seed-count 50 \
  --max-decisions 200 \
  --battle-simulations 500 \
  --seed-timeout-seconds 60 \
  --overwrite
```

If a seed fails or times out, the runner writes `seed_<n>.error.json` next to the partial JSONL trace. The summarizer reads these sidecars and marks rows with `stopped_reason` and `error_type`.

## Optional Local LLM Agent

The `mlx` agent is wired as an optional adapter and requires MLX-LM:

```bash
.venv/bin/python -m pip install -e '.[llm]'
PYTHONPATH=src .venv/bin/python scripts/run_rollout.py \
  --agent mlx \
  --model mlx-community/Qwen3-4B-4bit \
  --seed 1
```

The default MLX agent disables Qwen3 thinking mode for the first rollout stage because it gives much more reliable structured JSON actions. Use `--thinking` for explicit reasoning-mode experiments after the no-thinking path is stable.

> **`--max-tokens` must be large for any reasoning/thinking run (default is 4096).** A small cap (e.g. 256) truncates the model mid-thought, so it never emits the closing JSON. After retries are exhausted the invalid response is recorded and the rollout stops with `agent_invalid` before any fallback action is executed. Raising the cap is free for no-thinking models (generation stops at EOS, ~60–90 tokens). The eval report's `invalid_rate` / `completion_tokens` columns surface this immediately.

Current working policy:

- no-thinking mode for high-throughput Qwen baseline rollouts;
- thinking mode (`--thinking`) for a smaller comparison arm — keep the large `--max-tokens` so reasoning can finish *and* emit JSON.

The first implementation priority is to verify simulator throughput and action parsing before committing to full-parameter local training.

## Simulator Fault Policy

The local `sts_lightspeed` patch is strict about invalid battle actions and unknown potion enum values. These now raise Python-visible exceptions instead of silently mutating state or flooding stderr. Batch rollouts record those as error sidecars; timeout mode isolates each seed in a subprocess so native hangs or very slow searches are contained.
