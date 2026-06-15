# scripts

CLI entry points and build tooling. See the top-level [`CLAUDE.md`](../CLAUDE.md) for project goals and repo-wide rules.

## Contents

- `build_lightspeed.sh` — clones `sts_lightspeed`, applies `patches/sts_lightspeed_python_api.patch` (only if not already applied), and builds the `slaythespire` pybind module. Run this before any rollout.
- `run_rollout.py` — runs a single hybrid rollout from the CLI. Always invoke with `PYTHONPATH=src`.
- `run_batch.py` — batch rollouts over a seed range/list. Seeds come from `--seeds`, `--seed-start/--seed-count`, or `--seeds-config configs/frozen_seeds.json --split {smoke,dev,eval}`. For `--agent mlx` add `--thinking` (captures `<think>` chain-of-thought into `agent.thinking`), `--max-tokens`, `--temperature`. Use `--seed-timeout-seconds` for LLM/thinking runs (per-seed subprocess + kill) — thinking mode can hang on the seed-2-class UB and is slow.
- `summarize_rollouts.py` — JSONL rollouts → summary CSV.
- `compute_risk_proxies.py` — JSONL rollouts → risk-event CSV + aggregate JSON (deterministic, computed from stored traces).
- `visualize_rollout.py` — Streamlit viewer for rollout traces (click-through / 1-decision-per-second autoplay). Out-of-combat decisions show deck/relics/potions + chosen action + reasoning + `<think>`; **combat decisions render a board** (enemy panels with HP/block/intent, a player panel with HP/block/energy/powers, and the hand as neutral card tiles with the chosen card + targeted enemy highlighted). Tiles are colour-by-state (chosen / unaffordable), **not by card type** — type isn't in the records (the "records only" data scope). Needs the `viz` extra. Run: `PYTHONPATH=src .venv/bin/streamlit run scripts/visualize_rollout.py`. All parsing lives in the unit-tested `sts_ai.rollout_view` (no Streamlit import there); the script is a thin render shell.

## Area gotchas

- **`run_rollout.py` deletes the output file if it already exists** before writing. Pass a fresh `--output` path if you need to keep a prior trace.
- Records are appended per-decision during the run; a crash mid-rollout leaves a partial JSONL file.
- **Streamlit HTML rendering in `visualize_rollout.py` has two sharp edges.** (1) Markdown treats any line indented ≥4 spaces as a code block, so each HTML tile/panel must be emitted as a **single-line** string (no pretty-printed newlines) via `st.markdown(..., unsafe_allow_html=True)`. (2) A rerun (e.g. autoplay/`st.rerun`) re-renders the page from scratch, so the `<style>` block is **re-emitted every run** rather than once — don't guard it behind a session flag or the tiles lose their CSS.

## Planned (research_plan.md Stage 1)

Batch rollout (seed ranges) and a JSONL→metrics summarizer belong here. Add them as new scripts rather than overloading `run_rollout.py`.
