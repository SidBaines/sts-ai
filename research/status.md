# Status

No training or evaluation is running. Updated 27 October 2026.

The broader continuation finished after an adaptive stop at 21 updates.
Held-out teacher matches fell from 56/96 to 6/96; 86/96 final answers hit the
cap. Both phases worsened. Read [the result](experiments/2026-10-26-broader-training.md).

- Workstream: `ws/llm`; topic branch: `codex/llm-broader-training`.
- Worktree: `/Users/sidbaines/Documents/SlayTheSpireAI/scratch/worktrees/llm-broader-training`.
- Run: `broader_training_20261026`; data: `broader_states_selected_20261026`.
- Final adapter: `lr_3e-5/adapters/iter_0021.safetensors` within the run.
  Steps 8 and 16 and the original stage-1 adapter are also preserved.
- Original reference: `stage1_screen_ps_20261026/lr_3e-5`, iteration 48.
- Full paired statistics, valid-pairs sensitivity and timing: run `analysis.json`.
- Read-only dashboard: http://127.0.0.1:8767/; it shows the completed study and
  refreshes every five seconds. Its PID and logs are in the run's `dashboard/`.
- Completion monitor `llm-broader-training-monitor` is paused.
- Next proposal: lower learning rate versus measured KL regularization; no
  additional experiment has been launched. Sid prefers useful interventions to
  seed repeats while the project is in signs-of-life mode.

Run instructions and the analysis command are in
`experiments/2026-10-26-broader-training/README.md`.
The runtime/model links point to the earlier LLM worktree; outputs are local
here. Environment PR #10 still awaits merge/release before a new simulator
corpus. Main's WORKSTREAMS.md entry is stale; prior instruction proposals are
unapproved.
