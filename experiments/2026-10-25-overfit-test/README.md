# Stage 1: can the training loop make the model learn decisive states?

Research page: `research/experiments/2026-10-25-overfit-test.md`.

1. `select_states.py` builds the state set from the non-LLM corpus of saved
   champion games (`joint2_20261022/loop/corpus_01` in the `nonllm-large-batch`
   worktree):
   - it samples up to 3 out-of-combat and 6 combat decisions per world;
   - it labels them with the frozen champion and keeps those whose top action
     leads the next by at least 0.3;
   - it picks 16 states per phase from training-split worlds (`train.jsonl`)
     and 16 per phase from validation-split worlds (`probe.jsonl`), one per
     world and phase, spread across decision types;
   - it skips test-split worlds entirely and excludes out-of-combat states
     whose potion line hides a menu potion (the env-v0.1.5.1 fault fixed in
     0.1.6).

   The output used: `data/llm_runs/stage1_states_20261025`, with its
   `manifest.json`.

   ```sh
   NB=/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-large-batch
   PYTHONPATH=src .venv-mlx/bin/python experiments/2026-10-25-overfit-test/select_states.py \
     --corpus $NB/data/rl_runs/joint2_20261022/loop/corpus_01 --teacher-root $NB \
     --checkpoint $NB/data/rl_runs/champions/champion_4058cbf0e271.pt \
     --checkpoint-sha256 4058cbf0e271e8cd0ada21576cf0583368c112c53fc99c0523cc26d9af80943e \
     --teacher-python .venv-mlx/bin/python --output data/llm_runs/stage1_states_20261025
   ```

2. Training uses `python -m sts_llm.mlx_training`; see `src/sts_llm/README.md`.
   `run_screen.sh` runs one learning rate per call, with the settings fixed on
   the research page.
