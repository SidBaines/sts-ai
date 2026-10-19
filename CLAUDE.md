# Non-LLM RL workstream (`ws/nonllm`)

This branch is the non-LLM reinforcement-learning workstream: candidate-scoring
policy networks (`src/sts_policy`), PPO, distillation and native-search
experiments (`src/sts_rl`, `scripts/rl`). It is built on a copy of the Slay the
Spire environment (`src/sts_env` plus the native patch). The maintained
environment base lives on `main`; see WORKSTREAMS.md there. Start with
`research/README.md`, laid out as described under "Writing up research" below.

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
- Models, datasets, checkpoints, the frozen runtime and native builds are
  gitignored and live in other local worktrees. `research/status.md` and each
  experiment page name the runs and worktrees that hold them; preserve them,
  and never delete a worktree without an approved preservation plan.

## Active experiment monitoring

Actively monitor experiments and training runs: emit and interpret metrics that
show whether the work is progressing as intended, and investigate problems early.
This is about useful feedback and judgment, not accumulating smoke tests or pilots.
Discuss consequential changes with Sid when he is available; when working
unattended in goal mode, proactively diagnose and make safe, in-scope adjustments,
preserving results and recording changes. A dashboard or live process alone does
not establish that an experiment is healthy or learning. Record monitoring in run
records and chat, and keep `research/status.md` current; see "When to update"
below.

## Writing up research

Research notes exist so that Sid, or an agent new to the work, can quickly see
what has been tried, what happened, and what to do next. Every workstream
branch follows this convention. On `main`, which holds no research, only the
style rules apply.

### Where things go

All research writing lives in `research/`, in these files and no others:

| File | Holds | Limit | How to update |
|---|---|---|---|
| `README.md` | The goal, the current best agent and its headline numbers, what is running now, next steps, open questions | about 60 lines | Rewrite it |
| `strategies.md` | One section per approach: the idea, its status (working, promising, stalled or abandoned), the key evidence, and our best explanation | about 15 lines per approach | Edit in place |
| `results.md` | The results ledger: one table row per finished comparison, newest first | one row each | Add rows; if one proves wrong, mark it superseded and link the correction |
| `metrics.md` | Definitions of every metric, evaluation set, baseline and term used elsewhere | a line or two per entry | Add or correct entries |
| `experiments/YYYY-MM-DD-name.md` | One page per experiment, from plan to conclusion | about 150 lines | Fill in as it runs |
| `status.md` | Live state for handoff: what is running, where, how to check or stop it, what happens next | about 40 lines | Rewrite it; "Nothing running" when idle |
| `figures/<experiment>/` | PNG figures (each under about 300 KB) and the CSV data behind them | | |
| `archive.md` | Index of retired documents and the git tag that holds them | | Add entries |

- Do not create other document types: no separate plans, handoffs, progress
  logs, summaries, audits or "detail" files. A handoff is `README.md`,
  `status.md` and the active experiment page. A plan is the first half of an
  experiment page.
- How to run code is documented beside the code (that directory's README.md
  or CLAUDE.md), not in `research/`.
- Machine detail (hashes, PIDs, absolute paths, full commands, seeds,
  receipts, monitoring heartbeats) goes in machine-readable records in the
  run's output directory, referred to by run name. Exceptions: `status.md`
  may hold the few commands needed to check or stop a live job, and an
  experiment page ends with its code folder, commit and run names.
- If a page grows past its limit, condense it rather than splitting it into
  more files.

### Experiment pages

```markdown
# <Plain-words title>

**Status:** planned | running | finished <date>. **Verdict:** <one sentence>.

## Question
What we want to know and why it matters; link the approach in strategies.md.

## Design
What changes relative to the baseline and what stays fixed; the training data,
evaluation sets and metrics (names from metrics.md); the compute budget; the
decision rule, fixed before looking at results; Sid's approval and its date.

## Results
The main comparison first, as a small table, then one to three figures, then
other results in brief.

## Conclusion
What we learned, in plain words: what is now established, what is not, and
one list of caveats.

## Next
What this suggests doing next.

## Log
Dated one-line entries for decisions, surprises and changes of plan only.

Code: `experiments/YYYY-MM-DD-name/` at commit <short hash>. Data: run
`<run name>` (local).
```

### Results ledger

```markdown
| Date | Experiment | Tested | Result | Evidence | Verdict |
|---|---|---|---|---|---|
| 2026-10-16 | [Fresh start](experiments/2026-10-15-fresh-start.md) | Combat distilled from search vs champion combat | DEV wins 222 vs 143 of 4,096 (+1.9 points, 95% CI +1.2 to +2.7) | Exploratory | Better |
```

- **Evidence** is one of: *Exploratory* (chosen or analysed after seeing
  results), *Planned* (decision rule fixed in advance, on reused evaluation
  worlds) or *Confirmed* (fresh worlds, after the candidate was frozen).
- **Verdict** is one of: *Better*, *Worse*, *No clear difference* (enough data,
  no meaningful gap) or *Inconclusive* (too little data to tell).

### Style

- Write plain English for a smart reader who is new to the work, in normal
  sentences with normal spacing. Never squeeze words together or drop spaces
  and articles to save room.
- Lead with the answer. The first line of each page and section says the
  conclusion.
- Use words, not codes: "Act 2 boss fights", not BOSS2; "learning rate
  0.001", not lr1e3. A run or configuration ID may appear once, in code
  format, beside its plain description.
- Define every term and abbreviation in `metrics.md`, or spell it out. Reuse
  existing names rather than inventing new ones.
- Give each number its sample size and baseline, and give comparisons as the
  paired difference with a 95% interval and the evidence level. Say "no clear
  difference" rather than implying that two things are equal.
- Say each thing once, in its home file, and link to it from elsewhere.
- Keep caveats to one list per page rather than hedging every sentence.
- Leave out process (reviews, builders, receipts, retries) unless it changed
  a result; it belongs in commit messages and run records.
- Fix recurring problems at their source, not in the notes. A trap that is
  likely to bite again (a tooling pitfall, a misleading default, a gap or
  error in these instructions) does no good in an experiment page, where the
  next agent won't look. Point it out to Sid and propose wording for the
  right CLAUDE.md: this file, a directory's own CLAUDE.md (for example
  `scripts/rl/CLAUDE.md`), or a skill. With his agreement, add it there. This
  is the "Help improve these instructions" rule in Sid's global instructions.
- Outside the results ledger, delete stale text rather than appending
  corrections. Git keeps the history.

### When to update

- **Starting an experiment:** write its Question and Design, add it to the
  README's "Now" section, and fill in `status.md`.
- **While it runs:** keep `status.md` current by rewriting it. Add Log lines
  only for decisions and surprises. Interpret monitoring in chat and in run
  records, not as entries in the docs.
- **When it finishes:** fill in Results, Conclusion and Next; add rows to
  `results.md`; update the approach in `strategies.md`; update the README's
  current best and next steps; reset `status.md`.
- Commit research notes together with the code and configuration that
  produced the result.

## Code on workstream branches

Keep the reusable code small, tested and easy to find, and keep one-off
orchestration out of it. Workstream code is one of three kinds:

| Kind | Where | What belongs there | Tests |
|---|---|---|---|
| Library | `src/<package>/` | Pieces that more than one experiment uses: encoders, models, trainers, collectors, evaluators, fight panels, statistics, run records | Unit tests in `tests/`, required |
| Tools | `scripts/<area>/` | A small set of general command-line tools over the library: train, collect, evaluate, compare, dashboard, job queue | A cheap smoke test |
| Experiment code | `experiments/YYYY-MM-DD-name/`, named like its research page | Everything specific to one study: launch scripts, configs, analysis and plotting scripts | Only calculations that decide results, and those belong in the library |

- New code starts in its experiment's folder unless it is clearly general.
  Promote it to the library, with tests, when a second experiment needs it.
  Never import from another experiment's folder.
- Library code never imports experiment code, and contains no study names,
  dates, seed ranges or paths. Those go in the experiment's config files.
- Before writing a new script, look for an existing tool or function that does
  the job, and prefer adding an option to it over writing a near-duplicate.
- Keep orchestration as simple as the study allows: the shared job queue, a
  short launch script and a config file. Reproducibility comes from the run
  record written by the library (source commit, config, native build hash,
  seeds, parent checkpoints). Do not build per-study controllers, receipt
  systems or hash-binding layers unless Sid asks for them.
- Check checkpoint and data compatibility with explicit version numbers that
  are bumped when behaviour changes. Record source hashes for provenance, but
  never refuse to load because a source file's bytes changed: a comment edit
  must not make old checkpoints unusable.
- Don't write tests for one-off orchestration plumbing. Test library code and
  the calculations that decide results.
- When a study finishes, promote what is worth reusing and leave the rest of
  its folder unchanged. Its research page names the folder and the commit
  that produced each result. A later cleanup may delete old experiment
  folders that nothing uses; git keeps them.
- Models, datasets and run outputs stay in gitignored `data/` (for example
  `data/rl_runs/<run name>/`), never in an experiment folder.


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
