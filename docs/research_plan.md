# Non-LLM curriculum research goals and current state

This experiment branch starts from immutable `env-v0.1.0`. The active objective
is non-LLM learning for combat and out-of-combat play on the local M5 Pro/48 GB
Mac, within the first ten-hour allocation ending 2026-10-12 06:30:00 UTC.

[The detailed curriculum and experiment plan](nonllm_curriculum_plan.md) is the
source of truth for hypotheses, action representation, alternatives, budget,
selection/evaluation gates, and operational recovery. Current status: fresh Act 1 and Act 2/3 banks collected and audited;
228 tests and the bounded release smoke passed before the first training jobs.
The GPU architecture screen is running. First small models won 42/48 development
fights; the matched public heuristic won 42/48 and native search 44/48. These
are development pilots, not final results or integrated-game competence.
[Measured throughput and operational evidence](nonllm_throughput.md) records
the selected execution path and remaining limitations. Keep the environment release and old research work intact.

Architecture selection is complete and combat improvement controls are running.
See the maintained [reverse-chronological summary](nonllm_summary.md) for the
current state and evidence; the [detailed progress log](nonllm_progress.md) records
exact conditions. Expanded later-act cohorts, genuine corrective aggregation,
matched OOC teacher ablation and final cohort-completion gates are implemented.
The current full regression gate is 255 tests. Final data remains embargoed.
