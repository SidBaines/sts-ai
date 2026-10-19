# CLAUDE.md

## What this project is

A research harness for testing **how training-time framing changes what an LLM learns from the same reward signal**. The motivating question: when a model is reinforced along a graded axis like risk-taking, does the *framing* of the training context (e.g. "risk-reward tradeoffs" vs. "adventurous") decide which broader latent trait absorbs the update?

The first environment is Slay the Spire via `gamerpuppy/sts_lightspeed`. The default **hybrid harness** gives Python control of out-of-combat decisions (Neow, pathing, rewards, shops, events, card select, campfires) while the built-in Lightspeed search agent resolves combats; full LLM combat control is also available.

**Source of truth for goals, design commitments, and current state: [`docs/research_plan.md`](docs/research_plan.md).** Read it before starting substantive work.

## Coding guidelines

### Active now

- **Test before claiming done.** Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .` and at least one smoke rollout — use a **bounded** one, `scripts/run_rollout.py --agent random --seed 3 --max-act 1 --battle-simulations 50` (seconds) — before reporting a change as working. Do **not** use bare `--agent random`: at its defaults (seed 1, 2,000 battle simulations, 3 acts, random pathing into many fights) it spins at 100% CPU with no output for over an hour, so it is useless as a smoke test and indistinguishable from the known simulator hang. Add a regression test for every bug you fix. Tests are tiered: `tests/unit` (pure Python, no build) and `tests/integration` (needs the built simulator, gated by `@requires_simulator`); see [`tests/CLAUDE.md`](tests/CLAUDE.md). Note the `-t .` flag — it's required now that the tiers are packages.
- **Match the existing style.** Type hints, `@dataclass`, `from __future__ import annotations`, small pure functions. Read the neighbouring code and mirror it rather than introducing new patterns.
- **Reproducibility is a contract.** Frozen-seed trajectories must not silently change. Anything touching `lightspeed.py` / `rollout.py` / battle resolution must be checked for determinism (same seed + deterministic agent → identical trace across processes), and any intentional change to seed behaviour must be called out.
- **Schema stability.** `schemas.py` dataclasses define the on-disk JSONL format that training/eval scripts consume. Treat changes as breaking: flag them and note migration impact before editing.
- **Minimal dependencies.** Core stays dependency-free. New deps go in optional extras (like `[llm]`), pinned, and are raised explicitly — never added silently.

## Keeping docs current

When you surface something **non-obvious** during a session (a gotcha, a hidden invariant, a workaround, a sharp edge), record it before ending the turn. Place it at the right level:

- **Repo-wide** (affects agents working anywhere in the repo) → the Gotchas list below.
- **Area-specific** (only relevant to one part) → the nearest subfolder `CLAUDE.md` (e.g. [`src/sts_ai/CLAUDE.md`](src/sts_ai/CLAUDE.md), [`scripts/CLAUDE.md`](scripts/CLAUDE.md)) or a README if more appropriate.

When you complete substantive work that changes project status or design commitments, update [`docs/research_plan.md`](docs/research_plan.md) (or `research/README.md` once this workstream's notes are converted; see "Writing up research" below).

## Writing up research

Research notes exist so that Sid, or an agent new to the work, can quickly see
what has been tried, what happened, and what to do next. Every workstream
branch follows this convention. On `main`, which holds no research, only the
style rules apply.

> **This workstream predates the convention.** Its existing notes are in
> `docs/` (`research_plan.md`, `experiment_history.md` and others listed under
> Pointers). When work restarts here, first convert them into `research/`, as
> `ws/nonllm` did, then follow the convention for all new writing.

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

## Pointers

- [`docs/research_plan.md`](docs/research_plan.md) — research question, design commitments, architecture, and current state.
- [`docs/experiment_history.md`](docs/experiment_history.md) — concise record of past experiments and open empirical items.
- [`docs/ooc_rl_v1_status.md`](docs/ooc_rl_v1_status.md) — status, findings and resume instructions for the paused `ooc_grpo_v1` out-of-combat RL run.
- [`README.md`](README.md) — setup, build, and run commands.
- [`src/sts_ai/CLAUDE.md`](src/sts_ai/CLAUDE.md) — harness internals and area-specific gotchas.
- [`scripts/CLAUDE.md`](scripts/CLAUDE.md) — CLI entry points, build script, and run-planning notes (throughput, backends, pod discipline).

## Gotchas (repo-wide)

- **Run with `PYTHONPATH=src` and `.venv/bin/python`.** The package isn't installed; imports fail otherwise.
- **The simulator must be built before any rollout.** Run `scripts/build_lightspeed.sh`. The built module lives in `external/sts_lightspeed/build/` and is located by `lightspeed_import.py`. `external/` and build outputs are gitignored — they are **not present on a fresh clone**.
- **The Python↔C++ binding is maintained as a patch**, `patches/sts_lightspeed_python_api.patch`, applied to the upstream clone by the build script. To change the binding, **edit the patch and rebuild** — do not edit `external/sts_lightspeed/` directly, as that clone is untracked and your changes won't be versioned. After editing `external/` for a real change, **regenerate the patch — scoped to the source dirs** (`cd external/sts_lightspeed && git diff -- bindings include src > ../../patches/sts_lightspeed_python_api.patch`) and verify it applies to a fresh clone — the build script's apply step is skipped when the binding is already present, so a stale patch won't be caught locally. **Scope it:** a bare `git diff` also captures the `pybind11` **submodule pointer bump** the build script applies (it checks out v2.13.6 separately), and that extra hunk makes `git apply` fail on a fresh clone. Sanity-check the regenerated patch with `git apply --check --reverse` (must report clean) and `grep` it for `pybind11 b/pybind11` (must be absent). A regenerated nested diff can also add literal one-space blank context lines that the *outer* repo flags as trailing whitespace; normalize lines matching exactly `^ $` to empty, then repeat both fresh-apply and reverse-apply checks (`git apply` accepts the normalized form).
- **The simulator is built in Release, so `assert()` is a no-op.** Any upstream guard written as `assert(false)` does nothing in our builds — an overflow/`while(true)` guard like that will hang instead of bailing. Defensive guards that must fire in production have to **throw**, not assert (see `BattleContext::executeActions` in the patch).
- **The upstream simulator has uninitialized-memory UB** (default-constructed `GameContext`/`BattleContext` had uninitialized `potions`). Symptoms are build-/layout-dependent (a given seed may crash on one build and hang on another), which breaks naive cross-build reproducibility. If you see a seed behave differently after an unrelated rebuild, suspect UB, not your change. The hang variant is also **state-dependent, so seed screening does not transfer across policies**: a cohort screened clean under one agent can still wedge under another whose decisions reach different combat states (bitten 2026-08-17: ooc_grpo_v1 iter-4 wedged on a screened seed; in-process rollout runners freeze whole-loop). Long unattended runs need a stall watchdog that kills and resumes with a bumped `run_grpo.py --policy-seed-salt` so the redone iteration re-rolls instead of deterministically replaying into the same state. Enemy powers once blamed on a second "phantom-power" bug in this family (Gremlin Nob with Metallicize 4 or Regen 3) are the burning elite's random buff (`MonsterGroup::applyEmeraldEliteBuff`), not memory corruption; the player-side cases came from Fossilized Helix and Liquid Bronze. The quarantine in `configs/competence/state_sanity_quarantine_v1.json` (detector: `scripts/audit_state_sanity.py`) is a false positive (corrected 2026-10-15). Detailed write-ups live in git history (`docs/simulator_issue_handoff.md`).
- **Native search seeds `0` and `1` are the same RNG stream on libc++.** `BattleScumSearcher2` uses `std::default_random_engine`, whose seed normalization makes 0 alias 1 on this build. Never count both as independent consensus votes; `collect_search_teacher.py` rejects that pair and defaults to empirically distinct seeds `1,2,3`.
