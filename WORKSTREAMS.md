# Workstreams

`main` holds only the reusable environment. Each learning strategy lives on its
own long-lived workstream branch, so different agents can pursue completely
different approaches against the same tested environment. Start from the
workstream's front page, not from this file: `research/README.md` on its
branch, laid out as described under "Writing up research" in `CLAUDE.md`. A
workstream that predates that convention names its entry document below.

| Workstream | Branch | Start here, on that branch | Environment it runs on | Status (15 Oct 2026) |
|---|---|---|---|---|
| Non-LLM RL: candidate-scoring policy networks, PPO, distillation and native-search experiments | [`ws/nonllm`](https://github.com/SidBaines/sts-ai/tree/ws/nonllm) | `research/README.md` | The 0.1.4 patch, released as `env-v0.1.4`, plus a research-only native overlay for battle copying and reseeding | Joint sparse-win study complete; next study under discussion |
| LLM agents: Gemma students, search-teacher distillation, and the training-framing research question | [`ws/llm`](https://github.com/SidBaines/sts-ai/tree/ws/llm) | `CLAUDE.md`, then `docs/research_plan.md` | The pre-extraction `sts_ai` harness, not the `sts_env` package | Paused since 13 Oct; the tip is a WIP snapshot |

## Working conventions

- Do experiment work on a topic branch cut from `ws/<name>` and merge it back
  there when done. A new agent resumes from the `ws/<name>` tip.
- Environment bugs found in a workstream go to `main` as their own pull request,
  with a regression test and a new `env-v*` release tag. The workstream then
  merges the tag rather than carrying a private fix.
- Models, datasets, checkpoints and native builds are gitignored. They live in
  local worktrees, so a branch alone does not reproduce a result. Each
  workstream's `research/status.md` or experiment pages list the local
  dependencies they need.
- Update this table when a workstream starts, stops or moves, not at every
  experiment.

## Environment releases and open faults

Releases 0.1.2–0.1.4 carried the non-LLM workstream's environment changes to
`main` with identical native patches. Release 0.1.5 then fixed the remaining
known faults (see `docs/release_v0.1.5.md`), and 0.1.6 corrected the
out-of-combat potion line and tagged agent errors (`docs/release_v0.1.6.md`).
Neither workstream has adopted 0.1.5 or later yet; `ws/nonllm` runs on the
0.1.4 patch plus a research-only overlay, and `ws/llm` still uses its
pre-extraction harness.

Remaining known limits are listed in
[the environment contract](docs/environment_base.md).

## Archived and superseded references

- Tag `archive/nonllm/critique`: the stopped 13 Oct critique branch, with typed
  residual policies and a public action-identity repair. It was never merged
  into `ws/nonllm`. Entry: `docs/nonllm_critique_handoff_2026-10-13.md`. Its
  `sts_env_public_v2` label names a different change from release 0.1.3's.
- Tag `archive/nonllm/long-curriculum-v1`: the paused v1 longer-curriculum
  study, superseded by v2, which is in `ws/nonllm` history.
- Older branch names are superseded and kept only for existing local worktrees.
  `astra-5sep` and `astra-6sep` are in `ws/llm`. `codex/nonllm-curriculum`,
  `codex/long-curriculum-v2`, `claude/slay-spire-rl-debug-da6d22`,
  `codex/reproducibility-fault-records`, `codex/public-search-canonical` and
  `codex/joint-sparse-win-rl` are all in `ws/nonllm` history.
  `environment-base` is replaced by `main`.
