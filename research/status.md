# Status

Stage 1's learning-rate screen is running on the laptop. Updated 26 October 2026.

- Run `stage1_screen_20261026` in `data/llm_runs/` of the worktree
  `~/.codex/worktrees/llm-throughput/SlayTheSpireAI` (branch
  `codex/llm-throughput`). Learning rates 0.00001, 0.00003 and 0.0001, one
  after another, 48 iterations each; about 90 minutes per rate. Started
  26 October; page: `research/experiments/2026-10-25-overfit-test.md`.
- It runs detached in its own process group (ID in `pids.txt`). A babysitter
  holds `caffeinate` on mains power and pauses the job on battery
  (`babysitter.log`).
- Check: `tail data/llm_runs/stage1_screen_20261026/screen.log` (one line per
  rate) and `lr_<rate>.log`; per-iteration records are in
  `lr_<rate>/iterations.jsonl`, evaluations in `lr_<rate>/evaluations.jsonl`.
- Stop after the current rate: `touch data/llm_runs/stage1_screen_20261026/STOP`.
  Stop the current rate between iterations:
  `touch data/llm_runs/stage1_screen_20261026/lr_<rate>/STOP`.
- Next: apply the page's pass rule to each rate, fill in Results.
- Laptop setup: `.venv-mlx/` (mlx-lm 0.31.3); model
  `data/models/k2-horizon-0.9b-mlx`; inputs `data/llm_runs/stage1_states_20261025`.
- Not pushed: this branch and the local tag `archive/llm/pre-cli`.
- Environment release 0.1.6 is pull request #10 to `main` (CI green); merge
  its tag here once it lands.
