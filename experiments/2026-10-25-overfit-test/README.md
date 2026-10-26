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


3. The read-only live dashboard uses the Python standard library and does not
   load the model. From the worktree root:

   ```sh
   python3 -u experiments/2026-10-25-overfit-test/dashboard.py --port 8766
   ```

   Open http://127.0.0.1:8766. It refreshes every five seconds, follows all
   three rates in `stage1_screen_ps_20261026`, and offers the original
   `stage1_screen_20261026` as a clearly labelled shared-order reference.
   `--screen` and `--reference` can select other screens with the same layout.
   Charts and tables show every evaluation, full and partial training passes,
   combat or out-of-combat metrics, position-bias checks, and saved reasoning
   with its exact displayed menu. Process state, stalled output, parser
   failures and power state are visible. An unfinished JSONL line is ignored
   until its writer completes it.

   The detached server's PID and URL are in
   `data/llm_runs/stage1_screen_ps_20261026/dashboard/server.json`; its output
   is in the adjacent `server.log`. To stop only the dashboard, read that PID
   and use `kill <pid>`. This does not stop or change the training run.
