# Status

Code preparation on 23 October 2026; no training study is running.

The saved goal is to prepare single-decision teacher-feedback RL and evaluation
on the latest core environment, benchmark generation and optimizer updates on
an RTX 2000 Ada, then stop before the main experiment.

- Branch: codex/llm-throughput, based on ws/llm with latest main merged.
- Current core files exactly match env-v0.1.5.1; no environment behavior changes.
- RTX 2000 Ada allocated after two stock failures: ra7v5afx03jcji,
  llm-k2-rtx2000-20261023, $0.24/hour; three-hour limit at 18:30 BST.
- Generation sweep is running; checks all efforts at concurrency 1, 8 and 32.
  Control: create /workspace/llm_throughput/STOP to stop between cases.
  Check: ssh runpod-llm-k2-rtx2000-20261023 and inspect logs/completed_cases.txt.
- The prematurely started A40 qccxnwlrjo151f is stopped; setup logs are local.
- Fixed current-environment prompt cohorts are ready: 12, 64 and 128 states.
- Training, teacher, generation and evaluation adapters are written.
  The actual optimizer-update checks follow serving measurements.
- Current champion queried on 128 states; portable reward runtime matches exactly.
- Do not launch the main learning study after these checks.
