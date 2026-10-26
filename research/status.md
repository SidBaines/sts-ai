# Status

Nothing is running. Updated 26 October 2026.

- Branch `codex/llm-throughput`, local only and not pushed, in the worktree
  `~/.codex/worktrees/llm-throughput/SlayTheSpireAI`. The local tag
  `archive/llm/pre-cli` is not pushed either.
- Stage 1 finished. Runs are in `data/llm_runs/`: `stage1_screen_ps_20261026`
  (an order per sample; the passing adapter is
  `lr_3e-5/adapters/iter_0048.safetensors`) and `stage1_screen_20261026`
  (shared order, flawed). Inputs: `stage1_states_20261025`.
- To run a screen: keep the Mac plugged in and run
  `python3 experiments/2026-10-25-overfit-test/launch_screen.py <run name> <rates…>`.
  Stop between rates with `data/llm_runs/<run>/STOP`, or mid-rate with
  `<run>/lr_<rate>/STOP`.
- Laptop setup: `.venv-mlx/` (mlx-lm 0.31.3, matplotlib for figures); model
  `data/models/k2-horizon-0.9b-mlx`.
- Next: the stage-1 stability check (KL penalty, larger steps, new seed), then
  a stage-2 page for Sid's approval.
- Environment release 0.1.6 is pull request #10 to `main` (CI green); merge
  its tag here once it lands.
