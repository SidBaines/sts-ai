# Environment 0.1.4 verification record

Release 0.1.4 canonicalizes hidden pile order before seeded native search
samples a draw order, and maps sampled discard/exhaust selections back to live
card instances; see [the search record](canonical_search_014.md). Ordinary
execution is unchanged and the observation label stays `sts_env_public_v2`.
The build script gains the opt-in `STS_OFFLINE_DEPS=1` dependency reuse. The
change was developed on the non-LLM workstream (`ws/nonllm` commit `ab02849`).
This release carries the identical patch to `main`, plus these changes:

- `ENVIRONMENT_VERSION` is now `0.1.4`. The workstream candidate reported a
  package version of 0.1.4 but trace `environment_version` 0.1.3; release
  traces now agree with package metadata.
- `scripts/build_test_fixtures.sh` builds the test-only hidden-pile fixture, and
  CI now builds it and requires the canonical sampling regressions to run
  instead of skipping.

## Verification

Built on 15 October 2026 from a fresh clone of the unchanged upstream pin
`7476a81954020087da31d41d16fddf475746ec2d`, pybind11 v2.13.6, CMake 4.1.3,
Apple clang 21.0.0 and Python 3.12.13 on macOS arm64, in an isolated `.venv`.

- Patch SHA256 `335fcc360e0d986416af4c282dfe636aed94123bc085af844d316c2b13c561c4`,
  identical to the workstream candidate. Fresh and reverse application pass; no
  pybind11 submodule hunk and no one-space blank context lines.
- Native binary SHA256
  `630ff37067ee7caf0d1f84955416665ae242449fc75341050bd48d3537394902`, identical
  to the workstream's own core 0.1.4 build.
- With the fixture built and required (`STS_REQUIRE_SIMULATOR=1
  STS_REQUIRE_SEARCH_TEST_FIXTURE=1`), the full suite passes all 210 tests with
  no skips, including the three canonical sampling regressions.
- The bounded random smoke (seed 3, Act 1, 50 battle simulations) ends in an
  ordinary defeat after 21 decisions.
- Fresh-process determinism: the 24-world `sts-env evaluate --agent native
  --mode ooc --max-act 3 --battle-simulations 50 --seeds 0:24` cohort, run twice
  in separate processes, gives identical traces once wall-clock
  `elapsed_seconds` fields are excluded.
- Unchanged ordinary execution: excluding version and provenance fields, the
  0.1.4 traces are identical to 0.1.3 traces for the same 24 out-of-combat
  worlds and for 8 full-control worlds whose combats use unsampled native
  search (`--mode full --max-act 1 --seeds 0:8`). Each version ran from its
  own source tree.

The immutable release reference is `env-v0.1.4`. Earlier tags and binaries are
unchanged.
