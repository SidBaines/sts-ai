# Non-LLM RL workstream (`ws/nonllm`)

This branch is the non-LLM reinforcement-learning workstream: candidate-scoring
policy networks (`src/sts_policy`), PPO, distillation and native-search
experiments (`src/sts_rl`, `scripts/rl`). It is built on a copy of the Slay the
Spire environment (`src/sts_env` plus the native patch). The maintained
environment base lives on `main`; see WORKSTREAMS.md there. Start with the
README's opening paragraph, which links the current handoff and the next-study
options; docs/research_plan.md holds the longer history.

## Workstream conventions

- Do new work on a topic branch cut from `ws/nonllm`, and merge it back into
  `ws/nonllm` when done. Resume from the `ws/nonllm` tip, not from an earlier
  agent's branch.
- Don't fix the environment here. Environment bugs (`src/sts_env`,
  `patches/sts_lightspeed_python_api.patch`) go to `main` as their own pull
  request with a new `env-v*` tag; this workstream then merges that tag.
  Research-only native helpers stay in `patches/sts_lightspeed_rl_panel.patch`.
- This branch adopts environment 0.1.5 (observation `sts_env_public_v3`)
  plus the research overlay. `main`'s current release is `env-v0.1.5`; it changes
  menus, observations and some game logic, so fresh baselines are required.
  See `docs/fresh_start_merge_notes.md` for adoption verification and build status.
- Models, datasets, checkpoints, the frozen runtime and native builds are
  gitignored and live in other local worktrees. The current handoff lists their
  absolute paths; preserve them, and never delete a worktree without an
  approved preservation plan.
- Keep the README opening paragraph pointing at the current handoff whenever
  the workstream's state changes.

## Active experiment monitoring

Actively monitor experiments and training runs: emit and interpret metrics that
show whether the work is progressing as intended, and investigate problems early.
This is about useful feedback and judgment, not accumulating smoke tests or pilots.
Discuss consequential changes with Sid when he is available; when working
unattended in goal mode, proactively diagnose and make safe, in-scope adjustments,
preserving results and recording changes. A dashboard or live process alone does
not establish that an experiment is healthy or learning.

## Development contracts

- Keep src/sts_env dependency-free; Gymnasium is an optional extra. No model,
  training, framing, or experimental reward policy belongs in the core.
- Test before release: PYTHONPATH=src .venv/bin/python -m unittest discover
  -s tests -t . and the bounded smoke in scripts/run_rollout.py --agent random
  --seed 3 --max-act 1 --battle-simulations 50. Set STS_REQUIRE_SIMULATOR=1.
- Run fresh-process determinism/replay checks for changes to execution,
  observations, seeding, native policies, or the patch. Record intentional changes.
- Bug fixes need regression tests. Public observation/trace changes require
  explicit version bumps and migration notes; never modify historical artifacts.
- Build via scripts/build_lightspeed.sh (research builds: scripts/rl/build_native_rl.sh).
  Native sources are gitignored, so native edits must live in a patch: environment
  fixes go to main (see above), research-only helpers in
  patches/sts_lightspeed_rl_panel.patch. Verify fresh/reverse apply.
- Never treat simulator failure, undefined behavior, or timeout as a game loss.
  Default process isolation and finite operation timeouts must remain supported.
- Environment release status and known limits are maintained on main's
  docs/environment_base.md. Release tags are immutable.

## Gotchas

- Isolated scripts need an if __name__ == '__main__' guard (spawn semantics).
- Native combat search is privileged and uses hidden state. The native OOC
  policy is scripted/random, not tree search; label those modes accurately.
- Native RNG streams 0 and 1 alias on libc++; do not count both as independent.
- `undefinedBehaviorEvoked` can mark unsupported behavior as well as faults. Discovery is certified only for Ironclad's pool; see `docs/discovery_fix.md`.
- Legal action indices refer to the current menu, not a global stable action ID.
- Installed wheels need STS_LIGHTSPEED_BUILD pointing to a compatible native
  build; the pure Python wheel does not embed an architecture-specific binary.
- Source histories remain in Git; clean means experiment-free checkout, not a
  rewritten history. Never alter unrelated worktrees or the research checkout.
