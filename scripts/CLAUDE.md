# scripts

CLI entry points and build tooling. See the top-level [`CLAUDE.md`](../CLAUDE.md) for project goals and repo-wide rules.

## Contents

- `build_lightspeed.sh` — clones `sts_lightspeed`, applies `patches/sts_lightspeed_python_api.patch` (only if not already applied), and builds the `slaythespire` pybind module. Run this before any rollout.
- `run_rollout.py` — runs a single hybrid rollout from the CLI. Always invoke with `PYTHONPATH=src`.

## Area gotchas

- **`run_rollout.py` deletes the output file if it already exists** before writing. Pass a fresh `--output` path if you need to keep a prior trace.
- Records are appended per-decision during the run; a crash mid-rollout leaves a partial JSONL file.

## Planned (research_plan.md Stage 1)

Batch rollout (seed ranges) and a JSONL→metrics summarizer belong here. Add them as new scripts rather than overloading `run_rollout.py`.
