# Native initialization revision: evidence and resumption plan

The original native artifact remains immutable:
`8dba3d574ea3d4cd4c91b7031ffb38821b3e18c03358263407df3c05329c1724`.
On 12 October 2026 an audit found `Player.cc` declared without an initializer.
`BattleContext::init` did not copy the character from `GameContext`. The hybrid
resolver constructed `BattleContext bc;` on the stack. Consequently class-specific
card/potion generation could index pools using an indeterminate character.
The corrected native artifact must have a new binary/patch identity and separate
output directories; never overwrite the original binary, datasets or checkpoints.

## Construction-path audit

| Evidence | Construction and use | Treatment for this defect |
| --- | --- | --- |
| `CurriculumWorker` battle/run tasks | `workers.py` selects full mode; `LightspeedHybridEnv._advance_to_decision_llm` calls Python `sts.BattleContext()`; pybind `init<>` value-initializes the defaulted C++ class | Provisional retention for Ironclad, whose enum value is zero |
| Full-mode combat search labels | `searchBestBattleAction` copy-constructs `teacherState` from the live initialized battle; `BattleScumSearcher2` copies root state and assigns `curState=*rootState` before use | Provisional retention; verify old/new paired traces and labels |
| Hybrid OOC tasks and evaluation | `kind=ooc` selects search mode; `resolve_current_battle` uses stack `BattleContext bc;` | Quarantine all old outcomes and RL transitions |
| OOC lookahead labels | Prefix reset starts full, then `lookahead.py` explicitly switches `combat_control='search'` for branch continuation | Quarantine branch outcomes, scores, labels, and policy comparisons |
| Scripted matched labels derived from lookahead rows | Some baseline labels preceded contaminated continuation, but share the suspect pipeline/artifact | Regenerate under the corrected binary for a clean revision |

Relevant source: `src/sts_policy/workers.py`, `src/sts_env/env.py`,
`src/sts_env/_backend/lightspeed.py`, `src/sts_policy/lookahead.py`,
`patches/sts_lightspeed_python_api.patch`, and upstream
`src/sim/search/BattleScumSearcher2.cpp`. Upstream `Player.cc` is uninitialized;
`CharacterClass::IRONCLAD=0`; `BattleContext` and its copy constructor are defaulted.
This audit addresses this specific character defect, **not all previously known
simulator UB**. Clean full-mode models are reusable initial weights, not proof
that every historical state is correct.

## Artifact boundaries

Keep old auto-combat OOC/lookahead artifacts and their timestamps/checksums, mark
all their results exploratory, and exclude them from corrected-backend selection.
Retain full-mode combat banks, BC weights, and selected combat weights under their
original provenance. Run bounded old/new full-mode paired prefixes, public hashes,
search labels, action traces, and outcomes before declaring compatibility. Old bank
`expected_provenance` must not be silently rewritten to the new module SHA.
Either collect a fresh bank or create an explicit verified migration artifact with
old/new identities and paired-check evidence. Encoder dimensions/version remain
unchanged only if the public observation contract remains unchanged.

## Minimal corrected-backend OOC restart

1. Root stops the old dispatcher only after a completed checkpoint/metrics chunk.
   Record the last old OOC training chunk as exploratory and preserve the old journal.
2. Use a new campaign directory (suggested `data/nonllm_native_revision/campaign`).
   Snapshot selected clean shared BC and combat checkpoints with their source hashes.
   Copy stable architecture/combat selection evidence and completed combat phase
   progress; discard old OOC phase progress, OOC effective settings, OOC pilot
   selection, lookahead teacher selection, and joint/final selection from the new
   mutable journal. Add an explicit `native_revision` record with old/new native
   identities and quarantine scope. Do not relabel old metrics as corrected evidence.
3. Regenerate fresh OOC scripted teacher/frontier data using the corrected binary.
   Start OOC from the clean full-mode BC foundation; do not warm-start from old
   auto-combat PPO or lookahead-derived weights. Root may use the new artifact's
   scripted teacher before a corrected lookahead comparison is affordable.
4. Evaluate the fresh scripted warm-start on the same 32 development worlds:
   start 10000, seed 777, native 200, Act 1, horizon 1024. Run the five matched
   90-second controls again in the new directory, all from the same fresh source:
   progress 0.05 at LR 1e-4/3e-4/1e-3; sparse at 1e-4; progress 0.02 at 1e-4.
   Keep gamma 1, clip 0.1, entropy 0.01, HP weight zero, train start 22000,
   64 worlds, horizon 512, and seed 11. Select only completed matched cohorts;
   keep the warm-start if controls regress. The old LR 3e-4 result is not a default.
5. Reevaluate clean combat/health alternatives with the newly selected OOC head
   before composing and freezing a joint initializer. Full-run training must use
   the corrected binary and record component model/native provenance.

Preserve the original stage boundaries: OOC ends **02:08:46 UTC**, joint Act 1
ends **04:08:46 UTC**, joint Act 3 ends **05:08:46 UTC**, hard overall stop
**06:30:00 UTC** (03:08/05:08/06:08/07:30 BST). Revision preparation consumes the
remaining window; do not extend these deadlines. Keep enough time for final
checks and report any reduced OOC online duration explicitly. If warm-start or
pilot preparation approaches the OOC deadline, retain the validated reference
rather than launch a truncated comparison and claim it complete. Final-world
files remain unopened until the corrected policy selection is frozen.

The command recipes are available in `sts_policy.ooc_controls`; pass the new
campaign directory and set `STS_LIGHTSPEED_BUILD` to the new artifact before any
fresh job. New output names/directories prevent old cached metrics from crossing
native revisions. The corrected binary's actual hashes and test record must be
added by the native owner after successful compilation and validation.

## Corrected artifact now available

The native owner built and tested the isolated artifact at
`external/sts_lightspeed_v011/build` in the experiment worktree. Reported module
SHA256 is `40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`;
patch SHA256 is `77fddbd67c82bebcc4ff31eea6bb0f6d25e8216e44825d6459f184f47756ce16`.
Four new native regression tests passed, including rejection of actual automatic
Discovery UB before `exitBattle`. The old automatic path could also lose its
latched UB flag when the local battle exited, so timeout-free old OOC outcomes
are not thereby valid. The full-mode wrapper already propagates its battle UB
flag before exit; this distinction does not certify other simulator behavior.

A prepared, unexecuted reference-plus-five-control argv plan is saved at
`/private/tmp/nonllm_native_revision_ooc_plan.json`. It uses the isolated build,
new directory `data/nonllm_native_revision/campaign`, fresh
`ooc_specialist_bc.pt` source, identical 32-world development evaluation,
and the fixed 02:08:46 UTC OOC boundary. Root owns execution and final native
provenance validation; no GPU job was launched by this report task.
