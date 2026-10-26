# Stage 1: can the training loop make the model learn decisive states?

Research page: `research/experiments/2026-10-25-overfit-test.md` (finished:
learning rate 0.00003 passed). Run everything from the worktree root.

## Files

| File | What it does |
|---|---|
| `select_states.py` | Builds the 32 + 32 state set from the non-LLM corpus (below) |
| `run_screen.sh` | Runs `sts_llm.mlx_training` once per learning rate given, one after another, with the stage-1 settings |
| `launch_screen.py` | Starts `run_screen.sh` detached, with a power babysitter |
| `watch_screen.py` | Prints one line per progress event, error, stall or power change; exits when the screen ends |
| `pass_summary.py` | Per-pass averages from one run's `iterations.jsonl` |
| `position_split.py` | Teacher agreement split by whether the teacher's top choice was shown as A |
| `plot_passes.py` | The research page's figure and CSV |
| `dashboard.py` | Read-only live web dashboard (below) |

## 1. States

`select_states.py` reads the saved champion games
(`joint2_20261022/loop/corpus_01` in the `nonllm-large-batch` worktree):
- it samples up to 3 out-of-combat and 6 combat decisions per world;
- it labels them with the frozen champion and keeps those whose top action
  leads the next by at least 0.3;
- it picks 16 states per phase from training-split worlds (`train.jsonl`) and
  16 per phase from validation-split worlds (`probe.jsonl`), one per world and
  phase, spread across decision types;
- it skips test-split worlds and excludes out-of-combat states whose potion
  line hides a menu potion (the env-v0.1.5.1 fault fixed in 0.1.6).

The output used is `data/llm_runs/stage1_states_20261025` (with
`manifest.json`):

```sh
NB=/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-large-batch
PYTHONPATH=src .venv-mlx/bin/python experiments/2026-10-25-overfit-test/select_states.py \
  --corpus $NB/data/rl_runs/joint2_20261022/loop/corpus_01 --teacher-root $NB \
  --checkpoint $NB/data/rl_runs/champions/champion_4058cbf0e271.pt \
  --checkpoint-sha256 4058cbf0e271e8cd0ada21576cf0583368c112c53fc99c0523cc26d9af80943e \
  --teacher-python .venv-mlx/bin/python --output data/llm_runs/stage1_states_20261025
```

## 2. Training a screen on the laptop

The trainer is `python -m sts_llm.mlx_training`; its options are in
`src/sts_llm/README.md`. Since format 2 (commit f131813), every sample gets
its own menu order by default (`--menu-order per_sample`). Never use
`per_group` for real runs: it teaches "answer A".

1. Plug the Mac in. On battery, the babysitter pauses the job with SIGSTOP and
   resumes it on mains power; the pause is lost time, not lost work.
2. Launch with a new run name (the directory must not exist):

   ```sh
   python3 experiments/2026-10-25-overfit-test/launch_screen.py <run name> 3e-5
   ```

   This prints the process-group ID and writes it to `data/llm_runs/<run>/pids.txt`.
3. Watch it, for example from a Monitor:

   ```sh
   python3 -u experiments/2026-10-25-overfit-test/watch_screen.py data/llm_runs/<run> <pgid> 8
   ```

   Add `resume` as a fifth argument when re-arming, so old events are not
   repeated.
4. Stop between learning rates with `touch data/llm_runs/<run>/STOP`. Stop
   the current rate between iterations with
   `touch data/llm_runs/<run>/lr_<rate>/STOP`. An iteration already started
   runs to completion.

Settings other than the learning rate are fixed inside `run_screen.sh`. For a
new design, write a new script or call the trainer directly. Never edit
`run_screen.sh` while a screen is running: bash reads it as it goes. A trainer
code change also reaches later rates of a running screen, because each rate
starts a fresh Python process.

Each rate's output is `data/llm_runs/<run>/lr_<rate>/`:
- `config.json`: every setting, the source commit and the format version;
- `iterations.jsonl`: one summary per iteration;
- `generations/iter_NNNN.jsonl`: every answer, with its reasoning, menu order
  and reward;
- `evaluations.jsonl` and `evaluations/`: greedy evaluations;
- `adapters/iter_NNNN.safetensors`: adapter checkpoints.

Timing: about 70–110 s per iteration and 2–3 minutes per evaluation. Rising
reasoning length slows iterations, up to about 300 s, and repetition loops
that hit the 4,096-token cap slow them further.

## 3. Analysis

```sh
python3 experiments/2026-10-25-overfit-test/pass_summary.py data/llm_runs/<run>/lr_<rate>/iterations.jsonl
python3 experiments/2026-10-25-overfit-test/position_split.py data/llm_runs/<run>/lr_<rate> data/llm_runs/stage1_states_20261025/train.jsonl
.venv-mlx/bin/python experiments/2026-10-25-overfit-test/plot_passes.py data/llm_runs/stage1_states_20261025/train.jsonl <out dir> "label=data/llm_runs/<run>/lr_<rate>" ...
```

Read the first-displayed-option rate against 0.27, the rate on these states
for a policy that chooses by content alone. Read agreement against the case
where the teacher's top choice is not shown first. A position shortcut shows
as A answers rising while that agreement falls.

## 4. Dashboard

`dashboard.py` uses only the Python standard library and does not load the
model:

```sh
python3 -u experiments/2026-10-25-overfit-test/dashboard.py --port 8766
```

Open http://127.0.0.1:8766. It refreshes every five seconds and follows all
three rates in `stage1_screen_ps_20261026`. It also offers the original
`stage1_screen_20261026` as a clearly labelled shared-order reference;
`--screen` and `--reference` select other screens with the same layout.
Charts and tables show:
- every evaluation, and full and partial training passes;
- combat and out-of-combat metrics, and the position-bias checks;
- saved reasoning, with the exact menu the model was shown;
- process state, stalled output, parser failures and power state.

An unfinished JSONL line is ignored until its writer completes it. The
server's PID and URL are in
`data/llm_runs/stage1_screen_ps_20261026/dashboard/server.json`, and its
output is in the adjacent `server.log`. To stop only the dashboard, `kill`
that PID; this never touches a training run.
