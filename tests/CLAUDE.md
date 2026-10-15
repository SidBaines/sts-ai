# Environment tests

- `unit/` uses pure Python/fake engines; it runs without a native build.
- `integration/` exercises `sts_env` against the patched simulator. Decorate
  binary-dependent cases with `@requires_simulator` from `tests.support`.
- Run from the source root with `PYTHONPATH=src .venv/bin/python -m unittest
  discover -s tests -t .`. The `-t .` flag keeps package imports consistent.
- Set `STS_REQUIRE_SIMULATOR=1` in release verification and CI. A missing binary
  must fail the release gate instead of quietly skipping integration tests.
- Use bounded seeds/settings and the default isolated worker for native checks.
  Test timeout containment separately from ordinary game defeat.
- Reproducibility checks compare two fresh processes with the same seeds and
  native build. Cross-platform identical traces are not a release promise.
- Every fixed bug needs a regression test. Reward/observation/trace contract
  changes need explicit versioning and migration notes.
