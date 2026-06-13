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

## Optional Local LLM Agent

The `mlx` agent is wired as an optional adapter and requires MLX-LM:

```bash
.venv/bin/python -m pip install -e '.[llm]'
PYTHONPATH=src .venv/bin/python scripts/run_rollout.py \
  --agent mlx \
  --model Qwen/Qwen3-4B \
  --seed 1
```

The first implementation priority is to verify simulator throughput and action parsing before committing to full-parameter local training.
