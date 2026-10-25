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
- The pilot is on hold until Sid decides the questions in
  `research/brainstorms/2026-10-25-getting-the-model-to-learn.md`.
- Environment bug for `main`: the out-of-combat text lists potions by count
  rather than slot (`slaythespire.cpp` around line 849). Also for `main`:
  `run_episode` swallows agent errors, so a CUDA out-of-memory error in the
  clean CLI path becomes an "error" game.
- Tests: the full suite passes 355 with 36 skipped (PyTorch, gym and fixture
  extras). With PyTorch installed, the four LLM test modules pass 76 of 76.
