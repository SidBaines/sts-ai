# Non-LLM curriculum research goals and current state

This experiment branch starts from immutable `env-v0.1.0`. The active objective
is non-LLM learning for combat and out-of-combat play on the local M5 Pro/48 GB
Mac. Sid extended the initial ten-hour allocation to **2026-10-12 06:30 UTC**.

[The detailed curriculum plan](nonllm_curriculum_plan.md) records hypotheses,
action representation, alternatives, budgets and evaluation gates. The maintained
[latest-first human summary](nonllm_summary.md) records current state, decisions
and compact evidence; the [automatic progress log](nonllm_progress.md) records
exact commands and measurements, including failures.

As of 2026-10-12 02:32 UTC, combat and corrected-native OOC training are
complete. The retained combat specialist wins44/48 Act1 and30/32 later-act
development fights. The selected OOC policy clears31/32 development worlds
with no faults after a matched opt-in PPO stopping correction; native and
public-heuristic references clear15/32 and22/32. These are reused selection
results, not final-test estimates.

Signedenv-v0.1.1 repairs native automatic-combat character initialization and
reports flagged UB. Old automatic OOC/lookahead results are quarantined; old
binary/tag remainintact. Fresh banks use module40ab and interface1fa.
Composed full-control policy training is running: 37/48 development Act1
clears, with zero faults/truncations, versus its 29/48 initial reference.
Act3 training and independent final evaluation remain pending. The original deadline remains06:30 UTC.
Final data is embargoed until
selection freezes. The last full regression gate passed **308 tests**, with native
gates enabled, plus a bounded core smoke and reproducible full-control model CLI
rollouts. [Throughput evidence](nonllm_throughput.md) explains CPU rollout/MPS
update routing, replay optimizations and limitations. Preserve the immutable
environment release and original research checkout.

Known limitation: auto-native combat does not expose its temporary UB flag.
Zero surfaced hybrid faults do not certify fault-free native playouts. Explicit
full-control faults are quarantined. Scope is Ironclad ascension 0, Acts 1–3.

## Corrected environment revision

Environment-base0.1.1 initializes native automatic combat's character class
and rejects reported UB before committing a battle. It preservesenv-v0.1.0
and the old native module. New module40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd
uses patch77fddbd67c82bebcc4ff31eea6bb0f6d25e8216e44825d6459f184f47756ce16.
Fresh banks and a separate OOC campaign prevent mixed-backend comparisons.
Release evidence and limitations are in [the0.1.1 notes](release_v0.1.1.md).
