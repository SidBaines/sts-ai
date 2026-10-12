# Non-LLM curriculum research goals and current state

This experiment branch starts from immutable `env-v0.1.0`. The active objective
is non-LLM learning for combat and out-of-combat play on the local M5 Pro/48 GB
Mac. Sid extended the initial ten-hour allocation to **2026-10-12 06:30 UTC**.

[The detailed curriculum plan](nonllm_curriculum_plan.md) records hypotheses,
action representation, alternatives, budgets and evaluation gates. The maintained
[latest-first human summary](nonllm_summary.md) records current state, decisions
and compact evidence; the [automatic progress log](nonllm_progress.md) records
exact commands and measurements, including failures.

As of 2026-10-12 00:07 UTC, architecture and combat hyperparameter selection are
complete, and combat training has finished. The selected combat checkpoint wins
44/48 Act 1 and 30/32 later-act development fights, matching native reference
win counts on these panels. Expanded banks include Act 3 development/final
fights from 15/12 worlds. Matched HP controls did not pass the integrated Act 1
selection gate. These are repeated selection results, not final-test estimates.

OOC specialist BC is running; matched native/lookahead warm starts and five PPO
controls follow. Subsequent OOC PPO and joint full-run training remain pending.
Final data is embargoed until
selection freezes. The last full regression gate passed **289 tests**, with native
gates enabled, plus a bounded core smoke and reproducible full-control model CLI
rollouts. [Throughput evidence](nonllm_throughput.md) explains CPU rollout/MPS
update routing, replay optimizations and limitations. Preserve the immutable
environment release and original research checkout.

Known limitation: auto-native combat does not expose its temporary UB flag.
Zero surfaced hybrid faults do not certify fault-free native playouts. Explicit
full-control faults are quarantined. Scope is Ironclad ascension 0, Acts 1–3.
