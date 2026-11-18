# PPO learning audit

The [research page](../../research/experiments/2026-11-18-ppo-audit.md) describes the evidence and proposed next run. All diagnostic output lives in the gitignored run `ppo_learning_audit_20261118` on Sid's Mac.

Use the non-LLM research interpreter, `PYTHONPATH=src`, and the workstream's env-v0.1.5.2 native build. The actual launch records are preserved with the output.

- `history.py RUNS OUT_JSON` reads the saved baseline, reward ablations and evaluations.
- `audit.py --checkpoint CHAMPION --previous PREVIOUS --out OUT` collects the frozen diagnostic buffer and cross-fits the five disposable settings. The fixed split and sample seeds are in the script. It writes summaries and leaves all model files untouched.
- `protocols.py --checkpoint CHAMPION --out OUT` samples the four execution rules on the same 256 worlds and counts the top actions on saturated mixed reward menus.
- `export.py RUN FIGURES` writes the CSVs and figure using Matplotlib.
- `next-control.json` contains proposed arguments for `OOCTrainConfig`. Paths are relative to the active non-LLM checkout. A proposed paired challenger differs only in `lr` (0.0001) and its output directory. Both would use the same reserved training worlds.
