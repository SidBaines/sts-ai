# Status

Nothing running. Medium reasoning effort selected on 23 October 2026.

- Branch: codex/llm-throughput; core equals env-v0.1.5.1.
- Next study: research/experiments/2026-10-23-medium-teacher-pilot.md.
- Medium effort is approved. Other pilot settings and its allocation are proposed.
- Immediate work: matched rollout/scoring checks, adapter refresh/cache handling,
  then corpus preparation and baseline evaluation before teacher-feedback RL.
- Proposed pilot cap: ten GPU-hours including new preflight and evaluations;
  the previous speed-study allocation is already finished.
- No new GPU allocation or learning run has been started for this plan.
- Existing evidence: 52 LLM tests pass with Torch; environment 247 pass, 19 skip.
  Teacher mechanics produced 16 valid responses and four nonzero updates;
  the trained adapter's three real CLI decisions replay correctly.
- Full-game outcome RL is implemented and unit-tested, with no learning run.
- Completed run: k2_throughput_20261023; outputs and checkpoints are local.
- RTX pod ra7v5afx03jcji stopped at 16:28 UTC; disk retained, timer cancelled.
- Earlier A40 qccxnwlrjo151f returned not found at final lookup; setup logs
  are local and no comparable throughput timing completed.
