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
- Each workstream's front page is `research/README.md` on its branch (see
  below), so WORKSTREAMS.md on main only changes when a workstream starts,
  stops or moves.

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
| `brainstorms/YYYY-MM-DD-topic.md` | A dated record of a strategy discussion with Sid: what was found, the ideas considered, and the decisions | | Written when Sid asks for one; append later decisions rather than rewriting |
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

- The native `undefinedBehaviorEvoked` flag also marks unsupported simulator
  behaviour; it is not proof of C++ memory UB. Discovery support is certified
  only for Ironclad's existing pool. Preserve explicit unsupported-class faults
  and provenance boundaries when changing that path; see docs/discovery_fix.md.
