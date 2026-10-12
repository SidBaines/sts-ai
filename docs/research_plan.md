# Non-LLM curriculum research goals and current state

This experiment branch starts from immutable `env-v0.1.0`. The active objective
is non-LLM learning for combat and out-of-combat play on the local M5 Pro/48 GB
Mac. Sid extended the initial ten-hour allocation to **2026-10-12 06:30 UTC**.

[The detailed curriculum plan](nonllm_curriculum_plan.md) records hypotheses,
action representation, alternatives, budgets and evaluation gates. The maintained
[latest-first human summary](nonllm_summary.md) records current state, decisions
and compact evidence; the [automatic progress log](nonllm_progress.md) records
exact commands and measurements, including failures.

As of 2026-10-12 00:41 UTC, architecture and combat hyperparameter selection are
complete, and combat training has finished. The selected combat checkpoint wins
44/48 Act 1 and 30/32 later-act development fights, matching native reference
win counts on these panels. Expanded banks include Act 3 development/final
fights from 15/12 worlds. Matched HP controls did not pass the integrated Act 1
selection gate. These are repeated selection results, not final-test estimates.

A native automatic-combat character-class bug was diagnosed. Old OOC and
lookahead results are quarantined; the original native binary and artifacts
remain intact. A separately built corrected backend and fresh data will restart
OOC from supervised full-control weights. The retained combat evidence is clean
for this specific class-initialization defect, with remaining UB limits noted.
Joint full-run training is still pending.
Final data is embargoed until
selection freezes. The last full regression gate passed **289 tests**, with native
gates enabled, plus a bounded core smoke and reproducible full-control model CLI
rollouts. [Throughput evidence](nonllm_throughput.md) explains CPU rollout/MPS
update routing, replay optimizations and limitations. Preserve the immutable
environment release and original research checkout.

Known limitation: auto-native combat does not expose its temporary UB flag.
Zero surfaced hybrid faults do not certify fault-free native playouts. Explicit
full-control faults are quarantined. Scope is Ironclad ascension 0, Acts 1–3.
