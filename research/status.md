# Status

Broader local training is running, updated 26 October 2026.

Sid authorized a longer monitored run instead of another stage-1 seed repeat.
Read [the design](experiments/2026-10-26-broader-training.md). The code and
launch instructions are in `experiments/2026-10-26-broader-training/README.md`.

- Branch: `codex/llm-broader-training`, based on the current `ws/llm` tip.
- Worktree: `/Users/sidbaines/Documents/SlayTheSpireAI/scratch/worktrees/llm-broader-training`.
- Runtime and model: read-only links to the original LLM worktree. New outputs
  live in this worktree's own `data/llm_runs/`.
- Input: `broader_states_selected_20261026`; 512 training states, 96 validation
  states and a 64-state training evaluation panel. World overlap is zero.
- Starting evaluation: train 31/64 and validation 56/96 teacher matches;
  all 160 answers valid. The first update completed in 98 seconds with 32/32
  valid samples and finite nonzero gradients; no post-update evaluation yet.
- Parent: `stage1_screen_ps_20261026/lr_3e-5`, iteration 48. AdamW resets.
- Run: `broader_training_20261026`, at most 256 steps or eight elapsed hours,
  plus the current iteration and final evaluation; mains power only.
- App monitor: `llm-broader-training-monitor`, every 20 minutes; quiet unless
  meaningful progress, deterioration, failure or completion.
- Check: `cat data/llm_runs/broader_training_20261026/watch.log` and
  `cat data/llm_runs/broader_training_20261026/lr_3e-5/evaluations.jsonl`.
- Stop: `touch data/llm_runs/broader_training_20261026/STOP`.
- The watcher checks validity, held-out deterioration, process death and stalls.
  Inspect `alerts.jsonl` and both training/screen logs if it stops.
- Do not change a running script or trainer; new settings go in a new run.

The environment PR #10 remains open. This run retains the existing filtered
0.1.5.1 observations; it does not require a native build. Main's WORKSTREAMS.md
entry is stale. The prior handoff's instruction proposals remain unapproved.
