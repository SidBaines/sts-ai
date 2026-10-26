# Status

Nothing is running. Updated 26 October 2026 by the agent that ran stage 1.

## If you are picking this up

Read, in order: `research/README.md` (goal, current state, next steps), the
stage-1 page `research/experiments/2026-10-25-overfit-test.md`, and
`experiments/2026-10-25-overfit-test/README.md` (how to run and watch a
screen on the laptop). The plan behind the stages is in
`research/brainstorms/2026-10-25-getting-the-model-to-learn.md`.

**Your next task: the stage-1 stability check.** Stage 1 passed at learning
rate 0.00003, but from a single seed and after a two-pass drift towards
answering A. Before stage 2, check that the result repeats and try to remove
the drift. Do not launch anything until Sid has approved a written design:
1. Write a new experiment page,
   `research/experiments/<date>-stage1-stability.md`, using the template in
   `CLAUDE.md`. It needs a question, design and a decision rule fixed in
   advance. Suggested arms, all at 0.00003 on the same 32 states, with 6
   passes and the same pass rule:
   - a repeat of the passing setting with a new `--seed`;
   - `--kl-beta` > 0. Never run before at scale: smoke 3 iterations first and
     check the KL term's size in the logs before choosing a value;
   - 8 states per iteration (`--states-per-iteration 8`, 24 iterations).
     That halves the steps for the same samples; say so on the page.
   Report each arm against the passing run on the same states: the pass
   criteria, the share of A answers per pass, and agreement when the top
   choice is not shown first (`position_split.py`).
2. Ask Sid to approve it, record the approval on the page, then launch with
   a copy of `run_screen.sh` holding the new settings (never edit a running
   script).
3. Then write the stage-2 page (many more decisive states, held-out
   top-choice agreement as the gate) for approval.

## Practicalities

- Branch `ws/llm` (this work, fast-forwarded from `codex/llm-throughput`),
  worktree `~/.codex/worktrees/llm-throughput/SlayTheSpireAI` on Sid's
  MacBook. Data, models and virtual environments are gitignored and live only
  in that worktree.
- Laptop setup: `.venv-mlx/` (mlx 0.32.4, mlx-lm 0.31.3, matplotlib); model
  `data/models/k2-horizon-0.9b-mlx`. Install packages with
  `uv pip install --python .venv-mlx/bin/python`, because the venv has no pip.
- Runs: `data/llm_runs/stage1_screen_ps_20261026` (the valid screen; passing
  adapter `lr_3e-5/adapters/iter_0048.safetensors`) and
  `stage1_screen_20261026` (shared menu order, flawed). States:
  `stage1_states_20261025`.
- Long jobs must run detached (`launch_screen.py`), on mains power. Re-arm
  watchers with `watch_screen.py ... resume`. A dashboard server may still be
  running on port 8766 (PID in
  `data/llm_runs/stage1_screen_ps_20261026/dashboard/server.json`); it is
  read-only.

## Waiting on Sid

- Merge environment pull request #10 (release 0.1.6, CI green). Then tag
  `env-v0.1.6` on the merge commit, push the tag, and merge it into `ws/llm`.
  Build stage-2 states under 0.1.6.
- `WORKSTREAMS.md` on `main` still describes `ws/llm` as paused on the old
  harness. It needs a small pull request: the branch now runs on `sts_env`,
  and its front page is `research/README.md`.
- Proposed instruction changes, not yet approved:
  - project `CLAUDE.md`: no verdict from fewer than 10 worlds; report
    decision metrics beside trivial baselines; runs meant as independent
    samples need fresh seeds;
  - the Mac's `LOCAL.md`: timers stop while the Mac sleeps, so long runs need
    `caffeinate`.
