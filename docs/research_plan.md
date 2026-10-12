# Non-LLM curriculum research goals and current state

This experiment branch starts from immutable `env-v0.1.0`. The active objective
is non-LLM learning for combat and out-of-combat play on the local M5 Pro/48 GB
Mac. Sid extended the initial ten-hour allocation to **2026-10-12 06:30 UTC**.

[The detailed curriculum plan](nonllm_curriculum_plan.md) records hypotheses,
action representation, alternatives, budgets and evaluation gates. The maintained
[latest-first human summary](nonllm_summary.md) records current state, decisions
and compact evidence; the [automatic progress log](nonllm_progress.md) records
exact commands and measurements, including failures.

As of 2026-10-12 03:39 UTC, isolated combat and corrected-native OOC training are complete. The retained combat specialist wins **44/48 Act 1** and **30/32 later-act** development fights. The selected OOC policy clears **31/32** development worlds with no faults after a matched opt-in PPO stopping correction; native and public-heuristic references clear 15/32 and 22/32. These are reused selection results, not final-test estimates.

Signed `env-v0.1.1` repairs native automatic-combat character initialization and rejects reported undefined behavior before committing a battle. Old automatic OOC/lookahead results are quarantined; the old binary and tag remain intact. Fresh banks use module `40ab5056…` and interface `1fa75a1d…`.

Composed full-control policy training is running: **42/48 development Act 1 clears**, with zero faults/truncations and zero native-controlled decisions, versus its 29/48 initial reference. Three-act training starts at approximately 04:08 UTC; selection freezes for final evaluation at approximately 05:08 UTC. Training and allocated evaluation stop by **06:30 UTC**. Neither the three-act stage nor the final benchmark is complete yet.

The designated final benchmark remains embargoed until selection freezes. An earlier learned-agent integration smoke used seed 3, which hashes to the broader final partition but is outside that benchmark; [the protocol audit](nonllm_protocol_audit.md) records the exception and evidence limits. The full hash-final partition cannot be described as untouched.

The latest regression gate passed **337 tests**, with native gates enabled. Bounded core smoke and earlier model CLI checks passed; final selected-model CLI and reproducibility verification remain pending. [Throughput evidence](nonllm_throughput.md) explains CPU rollout/MPS update routing, replay optimizations and limitations. [Public encoding reuse](nonllm_encoding_reuse.md) removed repeated feature extraction with exact update parity; the first completed optimized chunk reached 1,460 accepted transitions/s. [Agent usage](nonllm_agent_usage.md) documents the training adapter and public CLI. Preserve the immutable environment releases and original research checkout.

Simulator flags detect reported errors rather than certify complete simulator fidelity. Faults are quarantined and counted separately from ordinary game defeats. Scope remains Ironclad, ascension 0, ordinary Acts 1–3.

## Corrected environment revision

Environment base 0.1.1 initializes native automatic combat's character class
and rejects reported UB before committing a battle. It preserves `env-v0.1.0`
and the old native module. New module `40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`
uses patch `77fddbd67c82bebcc4ff31eea6bb0f6d25e8216e44825d6459f184f47756ce16`.
Fresh banks and a separate OOC campaign prevent mixed-backend comparisons.
Release evidence and limitations are in [the 0.1.1 notes](release_v0.1.1.md).
