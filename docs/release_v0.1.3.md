# Environment 0.1.3 verification record

Release 0.1.3 repairs upstream shop pricing and purchase validation; the
[migration record](shop_maintenance_013.md) describes the defects, the native
changes and the observation migration. The public observation label advances
to `sts_env_public_v2` because shop menus no longer offer ineffective potion
purchases, so menu indices can differ from 0.1.2. The trace container format is
unchanged. Native and interface identities change; replay older traces with
their original build. The repair was developed and used on the non-LLM
workstream (`ws/nonllm`, commits `f7b722f` and `e55259b`). This release carries
the identical patch to `main`.

## Verification

Built on 15 October 2026 from a fresh clone of the unchanged upstream pin
`7476a81954020087da31d41d16fddf475746ec2d`, pybind11 v2.13.6, CMake 4.1.3,
Apple clang 21.0.0 and Python 3.12.13 on macOS arm64, in an isolated `.venv`.

- Patch SHA256 `91e53b39638f5d0018a8f50a244a6f88ee6c89135e1e85e1f361836773e1e84d`,
  identical to the workstream's final shop patch. Fresh and reverse application
  pass; no pybind11 submodule hunk and no one-space blank context lines.
- Native binary SHA256
  `97731fa4f9fb07447f0121eba04e865bcc5fcabe61b32e5741782ad219e5bbd0`. The
  workstream's own 0.1.3 binary differs because its research build also applies
  a separate research-only overlay patch; the core patch is the same.
- `STS_REQUIRE_SIMULATOR=1 PYTHONPATH=src .venv/bin/python -m unittest discover
  -s tests -t .` passes all 207 tests with no skips, including the five shop
  mechanics regressions and the fresh-process CLI trace/replay check.
- The bounded random smoke (seed 3, Act 1, 50 battle simulations) ends in an
  ordinary defeat after 21 decisions.
- Fresh-process determinism: `sts-env evaluate --agent native --mode ooc
  --max-act 3 --battle-simulations 50 --seeds 0:24`, run twice in separate
  processes, produced identical traces for all 24 worlds once wall-clock
  `elapsed_seconds` fields are excluded. The runs executed 34 shop purchases,
  and `sts-env replay` verified a saved trace step for step.

The immutable release reference is `env-v0.1.3`. Earlier tags and binaries are
unchanged.
