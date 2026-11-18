# Workstreams

`main` holds only the reusable environment. Each learning strategy lives on its
own long-lived workstream branch, so different agents can pursue completely
different approaches against the same tested environment. Start from the
workstream's front page, not from this file: `research/README.md` on its
branch, laid out as described under "Writing up research" in `CLAUDE.md`. A
workstream that predates that convention names its entry document below.

| Workstream | Branch and research front page | Environment used in the latest study |
|---|---|---|
| Neural policies for combat and out-of-combat decisions | [ws/nonllm](https://github.com/SidBaines/sts-ai/blob/ws/nonllm/research/README.md) | Environment 0.1.5.1 with a research overlay for battle copying and reseeding |
| Reasoning language models for Ironclad | [ws/llm](https://github.com/SidBaines/sts-ai/blob/ws/llm/research/README.md) | Public observations collected with the released environment and scored by a frozen non-LLM policy |

## Working conventions

- Do experiment work on a topic branch cut from `ws/<name>` and merge it back
  there when done. A new agent resumes from the `ws/<name>` tip.
- Environment bugs found in a workstream go to `main` as their own pull request,
  with a regression test and a new `env-v*` release tag. The workstream then
  merges the tag rather than carrying a private fix.
- Models, datasets, checkpoints and native builds live locally. Each
  workstream's `research/status.md` names its active checkout. On Sid's Mac,
  the main checkout's `data/worktree-archive/2026-11-18/inventory.json`
  maps retired checkouts to archived assets and source commits. Original
  paths used by saved runs resolve through compatibility links.
- Update this table when a workstream starts, stops or moves, not at every
  experiment.

## Environment releases and open faults

Releases 0.1.2–0.1.4 carried the non-LLM workstream's environment changes to
`main` with identical native patches. Release 0.1.5 then fixed the remaining
known faults (see `docs/release_v0.1.5.md`). Release 0.1.5.1 introduced a faster step path.
Each workstream records the environment and any overlays used for its results.

Remaining known limits are listed in
[the environment contract](docs/environment_base.md).

## Archived and superseded references

- Tag `archive/nonllm/critique`: the stopped 13 Oct critique branch, with typed
  residual policies and a public action-identity repair. It was never merged
  into `ws/nonllm`. Entry: `docs/nonllm_critique_handoff_2026-10-13.md`. Its
  `sts_env_public_v2` label names a different change from release 0.1.3's.
- Tag `archive/nonllm/long-curriculum-v1`: the paused v1 longer-curriculum
  study, superseded by v2, which is in `ws/nonllm` history.
- Older branch names remain available on the remote for historical studies.
  `astra-5sep` and `astra-6sep` are in `ws/llm`. `codex/nonllm-curriculum`,
  `codex/long-curriculum-v2`, `claude/slay-spire-rl-debug-da6d22`,
  `codex/reproducibility-fault-records`, `codex/public-search-canonical` and
  `codex/joint-sparse-win-rl` are all in `ws/nonllm` history.
  `environment-base` is replaced by `main`.
