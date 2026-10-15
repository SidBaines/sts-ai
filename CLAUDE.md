# Environment base

`main` is the reusable Slay the Spire environment, extracted from the
historical research harness. Read docs/research_plan.md and README.md first.
Learning strategies live on workstream branches; read WORKSTREAMS.md before
starting or resuming experiment work.

## Workstreams

- main holds only the environment: core, native patch, CLI, baseline policies
  and their tests. Experiments, models and training code never land here.
- Do experiment work on a topic branch cut from the relevant `ws/<name>` branch
  and merge it back there. A new agent resumes from the `ws/<name>` tip, not
  from a predecessor's personal branch.
- An environment bug found in a workstream gets its own pull request to main,
  with a regression test and a new `env-v*` tag; the workstream then merges
  that tag rather than carrying a private fix.
- Keep each workstream's README.md opening paragraph pointing at its current
  handoff, so WORKSTREAMS.md on main only changes when a workstream starts,
  stops or moves.

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
- Build via scripts/build_lightspeed.sh. Native sources are gitignored; edits
  must live in patches/sts_lightspeed_python_api.patch. Verify fresh/reverse apply.
- Never treat simulator failure, undefined behavior, or timeout as a game loss.
  Default process isolation and finite operation timeouts must remain supported.
- Update docs/environment_base.md for changes to release status and known limits.
  Release tags are immutable; subsequent fixes receive new tags.

## Gotchas

- Isolated scripts need an if __name__ == '__main__' guard (spawn semantics).
- Native combat search is privileged and uses hidden state. The native OOC
  policy is scripted/random, not tree search; label those modes accurately.
- Native RNG streams 0 and 1 alias on libc++; do not count both as independent.
- Legal action indices refer to the current menu, not a global stable action ID.
- Installed wheels need STS_LIGHTSPEED_BUILD pointing to a compatible native
  build; the pure Python wheel does not embed an architecture-specific binary.
- Source histories remain in Git; clean means experiment-free checkout, not a
  rewritten history. Never alter unrelated worktrees or the research checkout.
