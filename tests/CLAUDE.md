# tests

Two tiers. See the top-level [`CLAUDE.md`](../CLAUDE.md) for project rules.

## Layout

- `unit/` — **pure Python, no native simulator.** Fast; runs on any checkout
  (uses fakes or `object.__new__` to avoid loading the built module). Use this
  in the edit loop.
- `integration/` — **drives the real built `sts_lightspeed` module.** Each
  `TestCase`/method that constructs `LightspeedHybridEnv` (or otherwise needs the
  binary) is decorated with `@requires_simulator` from [`support.py`](support.py).
- `support.py` — shared helpers (`simulator_available`/`requires_simulator`,
  `vllm_available`/`requires_vllm`).

## Running

Always `PYTHONPATH=src` and run from the repo root with `-t .` (so `tests.*`
imports resolve):

```bash
# everything
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .
# fast unit tier only
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/unit -t .
# integration only
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/integration -t .
```

## Conventions / gotchas

- **Build-gated, not build-required.** Without the simulator build, integration
  tests **skip with a reason** so the unit tier stays green. Set
  `STS_REQUIRE_SIMULATOR=1` (e.g. in CI) to make them **run and fail** instead, so
  a missing build can't silently pass the gate — fail-closed, matching the
  simulator-fault policy in `docs/research_plan.md`.
- **New test needs the binary? Put it in `integration/` and decorate it with
  `@requires_simulator`.** Don't construct `LightspeedHybridEnv` in `unit/`.
- **Be wary of asserting exact simulator output for a fixed seed.** The simulator
  has build-/layout-dependent UB (see `docs/simulator_issue_handoff.md`), so prefer
  structural assertions (e.g. "an EVENT_SCREEN option exists") over brittle
  exact-state checks where practical.
