# Status

Nothing is running. Updated 25 October 2026.

- Branch: `codex/llm-throughput`, local only and not pushed. Its worktree is
  `~/.codex/worktrees/llm-throughput/SlayTheSpireAI`. The core environment is
  env-v0.1.5.1.
- Local tag `archive/llm/pre-cli` marks the old `sts_ai` harness. It is not
  pushed.
- Laptop setup:
  - The environment is `.venv-mlx/` in the worktree (PyTorch 2.13, Transformers
    5.19, PEFT 0.21.2, mlx-lm 0.31.3).
  - The MLX model is `data/models/k2-horizon-0.9b-mlx`, symlinked to the
    Hugging Face cache.
  - Mac runs are in `data/llm_runs/mac_local_20261025`.
- RTX runs are in `data/llm_runs/k2_throughput_20261023` (local). The RTX pod
  from 23 October no longer exists: `runpodctl` finds no pods.
- Stage 1 (`research/experiments/2026-10-25-overfit-test.md`) awaits approval.
  To launch it, keep the Mac plugged in and run
  `python3 experiments/2026-10-25-overfit-test/launch_screen.py stage1_screen_<date> 1e-5 3e-5 1e-4`.
  Stop between runs with `data/llm_runs/<screen>/STOP`, or mid-run with
  `<screen>/lr_<rate>/STOP`.
- Inputs: `data/llm_runs/stage1_states_20261025`. Smoke runs:
  `stage1_smoke_functional_20261025` and `stage1_smoke_timing_20261025`.
- Environment release 0.1.6 (potion line, agent-error source) is pull request
  #10 to `main`; merge its tag here once it lands.
- Tests: the full suite passes 355 with 36 skipped (PyTorch, gym and fixture
  extras). With PyTorch installed, the four LLM test modules pass 76 of 76.
