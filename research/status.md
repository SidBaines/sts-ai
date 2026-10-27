# Status

Nothing training or evaluating. The four-hour local study finished on 27 October 2026.
Read [the complete result](experiments/2026-10-27-learning-trials.md).

- Worktree: `/Users/sidbaines/Documents/SlayTheSpireAI/scratch/worktrees/llm-learning-trials`.
- Branch: `codex/llm-learning-trials`, to be handed back through `ws/llm`.
- Run: `learning_trials_20261027`; every arm, checkpoint, response and interval is retained.
- All four screens used medium effort. The smaller-update arm continued, then
  stopped after 26 further iterations because reasoning and capped answers grew.
- Both full-panel evaluations finished. The selected candidate has 61/96 held-out
  matches versus 56/96 initially, but 90/96 valid versus 96/96. Keep the original adapter.
- The final evaluation paused on battery, resumed successfully and finished at
  18:05. Training stopped before the 17:51:49 deadline; no new allocation was used.
- Runner and evaluator exited successfully. The dashboard remains at http://127.0.0.1:8767/.
- Analysis: run `trial_comparisons.json` and `research/figures/learning-trials/`.
- No additional experiment is authorized or queued. Monitor `llm-broader-training-monitor` is paused.
