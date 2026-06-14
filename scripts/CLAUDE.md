# scripts

CLI entry points and build tooling. See the top-level [`CLAUDE.md`](../CLAUDE.md) for project goals and repo-wide rules.

## Contents

- `build_lightspeed.sh` — clones `sts_lightspeed`, applies `patches/sts_lightspeed_python_api.patch` (only if not already applied), and builds the `slaythespire` pybind module. Run this before any rollout.
- `run_rollout.py` — runs a single hybrid rollout from the CLI. Always invoke with `PYTHONPATH=src`.
- `run_batch.py` — batch rollouts over a seed range/list. Seeds come from `--seeds`, `--seed-start/--seed-count`, or `--seeds-config configs/frozen_seeds.json --split {smoke,dev,eval}`. For `--agent mlx` add `--thinking` (captures `<think>` chain-of-thought into `agent.thinking`), `--max-tokens`, `--temperature`. Use `--seed-timeout-seconds` for LLM/thinking runs (per-seed subprocess + kill) — thinking mode can hang on the seed-2-class UB and is slow.
- `summarize_rollouts.py` — JSONL rollouts → summary CSV.
- `compute_risk_proxies.py` — JSONL rollouts → risk-event CSV + aggregate JSON (deterministic, computed from stored traces).
- `visualize_rollout.py` — Streamlit viewer for rollout traces (click-through / 1-decision-per-second autoplay; shows state, chosen action, reasoning, and the `<think>` trace). Needs the `viz` extra. Run: `PYTHONPATH=src .venv/bin/streamlit run scripts/visualize_rollout.py`. Rendering logic lives in the unit-tested `sts_ai.rollout_view` (no Streamlit import there).

## Area gotchas

- **`run_rollout.py` deletes the output file if it already exists** before writing. Pass a fresh `--output` path if you need to keep a prior trace.
- Records are appended per-decision during the run; a crash mid-rollout leaves a partial JSONL file.

## Planned (research_plan.md Stage 1)

Batch rollout (seed ranges) and a JSONL→metrics summarizer belong here. Add them as new scripts rather than overloading `run_rollout.py`.
