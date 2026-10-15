# Workstreams

`main` holds only the reusable environment. Each learning strategy lives on its
own long-lived workstream branch, so different agents can pursue completely
different approaches against the same tested environment. Start from the
workstream's entry document, not from this file: each workstream keeps the
opening paragraph of its own `README.md` (or the file named below) pointing at
its current handoff.

| Workstream | Branch | Start here, on that branch | Environment it runs on | Status (15 Oct 2026) |
|---|---|---|---|---|
| Non-LLM RL: candidate-scoring policy networks, PPO, distillation and native-search experiments | [`ws/nonllm`](https://github.com/SidBaines/sts-ai/tree/ws/nonllm) | `README.md` opening paragraph, which links the current handoff | Unreleased 0.1.3–0.1.4 environment candidates developed on the branch (see below) | Joint sparse-win study complete; next study under discussion |
| LLM agents: Gemma students, search-teacher distillation, and the training-framing research question | [`ws/llm`](https://github.com/SidBaines/sts-ai/tree/ws/llm) | `CLAUDE.md`, then `docs/research_plan.md` | The pre-extraction `sts_ai` harness, not the `sts_env` package | Paused since 13 Oct; the tip is a WIP snapshot |

## Working conventions

- Do experiment work on a topic branch cut from `ws/<name>` and merge it back
  there when done. A new agent resumes from the `ws/<name>` tip.
- Environment bugs found in a workstream go to `main` as their own pull request,
  with a regression test and a new `env-v*` release tag. The workstream then
  merges the tag rather than carrying a private fix.
- Models, datasets, checkpoints and native builds are gitignored. They live in
  local worktrees, so a branch alone does not reproduce a result. Each
  workstream's handoff lists its local dependencies.
- Update this table when a workstream starts, stops or moves, not at every
  handoff.

## Environment fixes not yet released on main

The non-LLM workstream forked at `env-v0.1.0` and has since made environment
changes that still need upstreaming as tested releases:

- 0.1.2: validate supported Ironclad Discovery instead of rejecting it as a
  fault ([PR #1](https://github.com/SidBaines/sts-ai/pull/1)).
- 0.1.3: shop economics repair and public observation `sts_env_public_v2`.
- 0.1.4: canonical pile ordering for sampled search, and the
  `BattleContext.copy` / `reseed_rngs` bindings.
- Build: `STS_OFFLINE_DEPS=1` reuse of already pinned dependencies.

## Archived and superseded references

- Tag `archive/nonllm/critique`: the stopped 13 Oct critique branch, with typed
  residual policies and a public action-identity repair. It was never merged
  into `ws/nonllm`. Entry: `docs/nonllm_critique_handoff_2026-10-13.md`.
- Tag `archive/nonllm/long-curriculum-v1`: the paused v1 longer-curriculum
  study, superseded by v2, which is in `ws/nonllm` history.
- Older branch names are superseded and kept only for existing local worktrees.
  `astra-5sep` and `astra-6sep` are in `ws/llm`. `codex/nonllm-curriculum`,
  `codex/long-curriculum-v2`, `claude/slay-spire-rl-debug-da6d22`,
  `codex/reproducibility-fault-records`, `codex/public-search-canonical` and
  `codex/joint-sparse-win-rl` are all in `ws/nonllm` history.
  `environment-base` is replaced by `main` and retires once open pull requests
  targeting it move to `main`.
