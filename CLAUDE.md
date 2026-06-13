# CLAUDE.md

## What this project is

A research harness for testing **how training-time framing changes what an LLM learns from the same reward signal**. The motivating question: when a model is reinforced along a graded axis like risk-taking, does the *framing* of the training context (e.g. "risk-reward tradeoffs" vs. "adventurous") decide which broader latent trait absorbs the update?

The first environment is Slay the Spire via `gamerpuppy/sts_lightspeed`, targeting Qwen3-4B. The current slice is a **hybrid harness**: Python controls out-of-combat decisions (Neow, pathing, rewards, shops, events, card select, campfires) while the built-in Lightspeed search agent resolves combats. Full LLM combat control is a later task.

**Source of truth for goals, design commitments, and what to work on next: [`docs/research_plan.md`](docs/research_plan.md).** Read it before starting substantive work — it carries the staged roadmap and the current near-term ordering.

## Project status: early development

We are **not yet in the research / data-collection phase**. The harness is still being hardened (serializer fixes, LLM JSON path, regression tests — see the research plan's near-term steps). This matters for the guidelines below: some are active now, others activate once we start keeping data.

## Coding guidelines

### Active now

- **Test before claiming done.** Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .` and at least one smoke rollout (`scripts/run_rollout.py --agent random`) before reporting a change as working. Add a regression test for every bug you fix. Tests are tiered: `tests/unit` (pure Python, no build) and `tests/integration` (needs the built simulator, gated by `@requires_simulator`); see [`tests/CLAUDE.md`](tests/CLAUDE.md). Note the `-t .` flag — it's required now that the tiers are packages.
- **Match the existing style.** Type hints, `@dataclass`, `from __future__ import annotations`, small pure functions. Read the neighbouring code and mirror it rather than introducing new patterns.

### Activate once we start keeping data (≈ research_plan.md Stage 1 onward)

These are **not yet binding** — they switch on the moment we freeze the first seed dataset or collect rollouts we intend to train on. When that milestone lands, update this section to move them to "Active now."

- **Reproducibility is a contract.** Frozen-seed trajectories must not silently change. Anything touching `lightspeed.py` / `rollout.py` / battle resolution must be checked for determinism (same seed + deterministic agent → identical trace across processes), and any intentional change to seed behaviour must be called out.
- **Schema stability.** `schemas.py` dataclasses define the on-disk JSONL format that training/eval scripts consume. Treat changes as breaking: flag them and note migration impact before editing.
- **Minimal dependencies.** Core stays dependency-free. New deps go in optional extras (like `[llm]`), pinned, and are raised explicitly — never added silently.

## Keeping docs current

When you surface something **non-obvious** during a session (a gotcha, a hidden invariant, a workaround, a sharp edge), record it before ending the turn. Place it at the right level:

- **Repo-wide** (affects agents working anywhere in the repo) → the Gotchas list below.
- **Area-specific** (only relevant to one part) → the nearest subfolder `CLAUDE.md` (e.g. [`src/sts_ai/CLAUDE.md`](src/sts_ai/CLAUDE.md), [`scripts/CLAUDE.md`](scripts/CLAUDE.md)) or a README if more appropriate.

When you complete substantive work that changes project status or the next-step ordering, update [`docs/research_plan.md`](docs/research_plan.md) (its status section and near-term steps).

## Pointers

- [`docs/research_plan.md`](docs/research_plan.md) — goals, design commitments, staged roadmap, current priorities.
- [`README.md`](README.md) — setup, build, and run commands.
- [`src/sts_ai/CLAUDE.md`](src/sts_ai/CLAUDE.md) — harness internals and area-specific gotchas.
- [`scripts/CLAUDE.md`](scripts/CLAUDE.md) — CLI entry points and build script.

## Gotchas (repo-wide)

- **Run with `PYTHONPATH=src` and `.venv/bin/python`.** The package isn't installed; imports fail otherwise.
- **The simulator must be built before any rollout.** Run `scripts/build_lightspeed.sh`. The built module lives in `external/sts_lightspeed/build/` and is located by `lightspeed_import.py`. `external/` and build outputs are gitignored — they are **not present on a fresh clone**.
- **The Python↔C++ binding is maintained as a patch**, `patches/sts_lightspeed_python_api.patch`, applied to the upstream clone by the build script. To change the binding, **edit the patch and rebuild** — do not edit `external/sts_lightspeed/` directly, as that clone is untracked and your changes won't be versioned. After editing `external/` for a real change, **regenerate the patch** (`cd external/sts_lightspeed && git diff > ../../patches/sts_lightspeed_python_api.patch`) and verify it applies to a fresh clone — the build script's apply step is skipped when the binding is already present, so a stale patch won't be caught locally.
- **The simulator is built in Release, so `assert()` is a no-op.** Any upstream guard written as `assert(false)` does nothing in our builds — an overflow/`while(true)` guard like that will hang instead of bailing. Defensive guards that must fire in production have to **throw**, not assert (see `BattleContext::executeActions` in the patch).
- **The upstream simulator has uninitialized-memory UB** (default-constructed `GameContext`/`BattleContext` had uninitialized `potions`). Symptoms are build-/layout-dependent (a given seed may crash on one build and hang on another), which breaks naive cross-build reproducibility. If you see a seed behave differently after an unrelated rebuild, suspect UB, not your change. Background: `docs/simulator_issue_handoff.md`.
