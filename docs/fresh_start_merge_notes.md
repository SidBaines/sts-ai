# Fresh-start environment adoption: A1

Status: file adoption passed independent Claude specification and quality
reviews. Native build/verification is pending; this is not a completed Step A.

## Merge evidence

- Branch `nonllm/fresh-start`, unchanged HEAD
  `4e0d4abfad0627e76708e4784865e5351aa60e37`.
- The orchestrator started the merge before this builder resumed. `MERGE_HEAD`
  contains annotated tag object `a2a88b15191ce8baf8aa0ccff37ff67d8f5681e6`
  (`env-v0.1.5`), which peels to commit
  `2c83f969d062873a318b30b1914510f7cf96f6bf`. The index retains 13 unmerged
  paths intentionally; selected file contents have no conflict markers.
- Every tracked `src/sts_env` file and every core test supplied by the tag
  matches the tag byte for byte, including `tests/native/hidden_pile_fixture.cpp`.
  The core patch and `docs/canonical_search_014.md`,
  `docs/shop_maintenance_013.md`, `docs/environment_base.md` also match the tag.
- Research code, scripts and native overlay are unchanged from HEAD.
  `docs/research_plan.md` and the fresh-start plan are unchanged. The orchestrator
  separately corrected the progress timestamp and added live design/health notes;
  those unstaged log changes will be committed separately after the merge. Auto-merged README release/build guidance, CI,
  packaging and fixture support were inspected. CLAUDE retains workstream
  instructions and now identifies the adopted source and pending verification.

## Changed pins and compatibility

- `pyproject.toml` and `sts_env.__version__`: `0.1.4` to `0.1.5`.
  Policy extras and the `sts_policy*` package entry are preserved; version is
  the only `pyproject.toml` change from HEAD.
- `sts_env.env.ENVIRONMENT_VERSION`: `0.1.3` to `0.1.5` (the parent constant
  lagged its package version); observation `sts_env_public_v2` to
  `sts_env_public_v3`.
- Native manifest patch SHA256: `335fcc360e0d986416af4c282dfe636aed94123bc085af844d316c2b13c561c4`
  to `d1428d0125497e12a0182b067a9b07870ecf419cb134e2fc0381466572ef37fd`.
  The actual adopted patch has the new digest. Upstream commit, pybind11 tag
  and backend API pin are unchanged.
- No research test needed a current-version/menu assertion change in the pure
  suite. Legacy fixture version labels remain unchanged. The imported core
  tests cover keyed Act 3 victory, including its reward/outcome and Gymnasium
  decision-limit semantics; those tests pass with the new core contract intact.

## Actual post-merge checks

- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/unit -t .`:
  488 tests in 2.879 seconds; 469 passed, 19 errors, no assertion failures or
  skips; exit 1. All errors are sandbox `PermissionError`: three HTTP socket
  binds (`test_long_rl_dashboard`, `test_long_rl_dashboard_v2`,
  `test_long_rl_episode_api`) and sixteen `ps` executions
  (`test_long_run_control`, `test_long_run_processes`). Full output:
  `scratch/a1_merge_review/pure_unit_post_merge.log` (gitignored).
  These checks still need an unrestricted
  rerun; production code/tests were not changed to suppress sandbox errors.
- `git diff --check HEAD` passed. Tag/HEAD byte comparisons confirmed the
  selections and preservation described above.
- No native build or simulator-dependent validation was run; core/research
  native compatibility, smoke, fresh-process determinism and replay remain
  unverified for this adoption. The frozen runtime was not installed into or
  modified. No Git metadata writes, merge restart/abort, experiment launch,
  historical dataset/checkpoint access, other-worktree edits, commit or push
  were performed by this builder.

## Specification review follow-up

Independent Claude specification review passed the file adoption, with explicit
resolution of the orchestrator's progress-log edits above. The workstream README
now warns that its frozen `.venv` must not run the generic install/build recipe
with default `PYTHON`; the existing separate `.venv-native` supplies build tools.
CLAUDE retains the released Discovery/unsupported-behavior limitation and link.
The review log is `/private/tmp/sts-fresh-20261015/a1_spec_review/stdout.log`.

## Host verification and quality review

The unrestricted pure suite passes **488/488 tests**, no skips, in 5.133 seconds
(`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/unit -t .`).
Full output and both independent Claude reviews are preserved locally under
`data/rl_runs/fresh_start_20261015/agent_records/`, rather than relying on /tmp.
The quality reviewer read the earlier sandbox result; this later host result
resolves the nineteen permission errors without changing tests.

The review identified an inherited research-helper risk: build_native_rl.sh
could label stale source as the current patch. The fresh-clone pipeline verifies
patch application and avoids it. A fail-closed repair is planned for the research
helper before accepting builds for experiments. Native verification must exercise
both core and research binaries, record skips and build-source identity, and
include fresh-process smoke/replay; it remains waiting on AC power.

The 0.1.5 menu/game changes require new baselines: the study adapter hides OOC
potion drink/discard entries for all players, preserves ordinary indices, and
leaves combat potions intact. Existing legacy combat features do not add new
free-once/retain flags; the new entity representation will. Existing HP-based
research overlays now receive live combat HP. No old-bank migration is attempted.
