# Representation controls: current continuation handoff

The user authorized actual feature, capacity and learning-rate comparisons, followed by a discussion. Keep working until the experiments and analysis are complete. Power/status messages do not cancel the task. Read the [fixed design](representation_controls_plan_2026-10-16.md) and newest-first [progress](representation_controls_progress.md). The [preparation archive](representation_controls_handoff_detail_2026-10-16.md) preserves earlier operational details; its old pending statuses are historical.

## Current state at 18:11 UTC, 16 October

All model, evaluator, factorial, controller, cache and cache-integration SPEC and QUALITY reviews pass. Integrated host suite: **979 tests passed, two identified historical-artifact skips**. Actual native parent/child, same-head, nonzero feature-on, CPU/MPS and fresh-process checks pass.

The fresh 1,024-world bank completed without faults, passed strict merge, and produced the frozen 400/100/100-world panel. The cache was created and independently compared against fresh raw encoding: every array byte, dtype, shape, target, metadata field, row order, epoch minibatch order and first collated batch matched. Python/NumPy/Torch CPU/MPS RNG states were unchanged. Loading and checks took 15.2 seconds; raw encoding took 194.1 seconds.

Both explicitly excluded technical training repeats completed 4,700 updates, selected step 4,100 and reproduced the prior structured curve within 3e-8 validation loss. Uncached runtime467.4s; cached308.2s. These duplicates NEVER enter the main matrix, LR selection, label TEST, fight panels or games.

**Next: commit the verified source, run the ready preflight publisher, enable and detach the main controller, and verify actual learning progress. No main matrix fit, global selection freeze or new TEST/fight/game outcome exists yet.** Root will update this paragraph after launch. The canonical command plan has already been revised for the cache; normalization proved no other command/science change.

## Paths and immutable inputs

- Active root: `/Users/sidbaines/.codex/worktrees/representation-controls/SlayTheSpireAI`, branch `codex/representation-controls` from `ws/nonllm` ab2cf2c. Source is currently uncommitted. Locally merge verified topic work into ws/nonllm, no push. The owned integration checkout is `/private/tmp/sts-fresh-20261015/workstream-integration`; inspect its status/tip before any merge.
- BASE: active root `data/rl_runs/representation_controls_20261016`.
- OLD: `/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start/data/rl_runs/fresh_start_20261015`. Never modify/delete its artifacts.
- TMP: `/private/tmp/sts-repr-controls-20261016`.
- Python: active root `.venv/bin/python`, symlink to the frozen nonllm-curriculum runtime, Python3.12.13/Torch2.14/NumPy2.5. No installations. Matplotlib is available separately in `/Users/sidbaines/Documents/SlayTheSpireAI/.venv/bin/python` for plots.
- Native: OLD's repository `external/sts_lightspeed_rl/build015`, SHA `6351c58c3c5a17275e8036145ea53c41c71cc711afe0167159a3273ac5723428`.
- Original manifest SHA: `c929b68e6636e9b2dee23739ce5d2038ef19eda105a341700b22c62216c837f8`; data identity `b1e2f16ad98d92b7dec358fdbb5536c0bd2bf238ebc5968ab97082de50b7b074`.
- Cache: `BASE/controlled_cache_v1/{receipt.json,dataset.pkl}`. Receipt SHA `6e51f60dfd43e968accc68119728bac3b37948f2833c49d3052bc16b0454a84c`; dataset SHA `bd6ce457806098dc7d5ed8f164bf9a22f0e7c570626491621f6fe787eb6e6c11`,1.73GB.
- Native/environment/legacy/entity model/encoder/vocabulary/protected fresh_study bytes remain unchanged. **Do not edit controlled_entity.py or controlled_encoding.py:** checkpoint loaders bind their exact source hashes. Cache `_LOGIC` also binds its helper/trainer/encoder/CLI sources. They are now final.

Writes in this root need host/escalated exec because the chat's original writable root is6278. All relevant escalations have been approved; no automatic approval rejection occurred. Use `PYTHONPATH=src` for direct Python module calls. System python3 is3.9 and incompatible with the native ABI; always use the frozen interpreter.

## Queue, controller and preflight

Use this root's isolated queue. Daemon PID30106 was current; `runner_status.json` and actual process identity are authoritative. Every heavy command uses `/usr/bin/env STS_LIGHTSPEED_BUILD=<original build015> PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 <root>/.venv/bin/python ...`. The queue's default build013_clean is stale; explicit env is essential. Actual host power is AC; sandbox `pmset` is misleading, so use escalated host/queue checks. Keep6GiB disk free. Free space was about40GB; cached-fit peak RSS8.8GB versus raw5.4GB. One MPS fit; bounded native lanes. Do not kill unrelated work.

Queue bookkeeping was repaired for boolean `--bank`. Watchdog now marks stalls failed and never automatically appends `--resume`. Failed pre-launch bank attempts are preserved; successful ones have `_r3`. The original collector bank uses `eval --bank --no-teacher-labels`, not unlabelled collect mode. All1,024 worlds are complete and strict merged.

Controller entry: `scripts/rl/representation_controls.py --daemon --interval30` (arguments are `--interval 30`). It must run outside the queue to avoid a nested deadlock. Root must detach it with a supervisor/status/exit-code log, because some unexpected load exceptions can leave stale state. The code reads STOP/control between stages. Already active work has its own STOP/deadline; explicit user stop requires propagating to owned jobs.

`control.json` is still `preparation_only`; no controller state or preflight is published yet. `TMP/publish_preflight.py` is ready: it verifies all12 reviews,979-test suite, native controls, actual cache audit, cached full-curve parity, exact counts, original manifest/raw/probe/native/runtime identities, and writes the exclusive PASS gate with exact cache provenance. **Read actual review reports before invoking it; substring assertions alone are not a review.** Root already read them all. It has not been executed yet.

Canonical `controller_plan_v1.json` SHA is now `63a9cd2029f6061f1b9bb9a75148ea27ca5807c9e9d9a7d68dd8be25f1a87b30`. The pre-cache plan is preserved as `controller_plan_v1_pre_cache.json`; `command_plan_cache_revision_v1.json` proves normalization changes only cache flags/path. No model selection or state pins existed when revised.

After committing, publish preflight, atomically enable `complete_pipeline`, detach the controller under explicit frozen env, and verify its first real fit. Source gate inventory matches begin_run plus the fixed design. **After pinning, leave ROOT code and its timestamps untouched through the eight game-arm merges.** Source zip SHA includes mtimes. Do not git-checkout/stash/format/touch code. Avoid further commits until collection completes. Place ad hoc analysis scripts in TMP; README/progress/report docs outside archived directories can update. The design document is pinned too.

Controller warnings are nonblocking: queue-add timeout30s, unrelated active job can block game-arm submission instead of waiting, no supported automatic retry interface, and deadline also blocks late validation. It validates all completed artifacts, never trusts exit0 alone. Preserve state/queue/output before a diagnosed manual retry; at mostone unchanged-grid/checkpoint retry. No faults/cuts/timeouts become losses. Operational safety deadline is17Oct08:00UTC; it is not a retroactive extension of the old completed study's user deadline.

## Completed evidence and reviews

Review directories under TMP: model_spec_review/model_quality_review; eval_spec_review/eval_quality_recheck; factorial_spec_review/factorial_quality_review; controller_spec_review/controller_quality_review; cache_spec_review/cache_quality_review; cache_integration_spec_review/cache_integration_quality_review. All are done/PASS. Builders and isolated worktrees are preserved. No agents remain active.

Host logs: integrated suite `data/rl_runs/repr_integrated_full_suite_v3/stdout.log` (979PASS); targeted `TMP/cache_host_tests_v1.log` (87PASS), `cache_integration_host_tests_v1.log` (53PASS), `parser_repair_host_tests_v1.log` (39PASS). Earlier native and955-test logs remain. Two historical skips are identified in `TMP/historical_skip_reasons_v1.log`.

Actual-data proof: `BASE/actual_cache_audit_v1/report.json`; excluded raw/cached technical runs and comparison/intent files; `technical_native_v1`, `technical_device_parity_v1.json`, `technical_fresh_process_v1.json`, `technical_wide_mps_v1`, capacity/range receipts. Cached curve selected loss1.2958403533691423 versus original1.295840346071353; max gap2.12e-8. All595,621 completed and519,328 selected-step presentations match.

Controller manifest reader was changed from the scanner to strict standard JSON. Actual read_json timing0.72s versus17.43s, same28,676 row metadata hash, lower peak memory. No validation was weakened. All controller tests and final integration QUALITY pass.

A latent writer limitation for a single64KiB PickleBuffer is documented in controlled_cache.md. It was absent from this successfully serialized, fully audited corpus. Do not invalidate the current cache for an unreachable future input; fix/test it before a different corpus. Loading is unaffected.

## Fixed scientific schedule

The design has all details; do not adapt from outcomes. The18 new main fits comprise16 controlled cells (featuresOFF/ON × width128/216 × LR.0003/.001 × seeds1919/2020), plus two old LARGE/NONE seed2020 rates. Reuse both old1919 rate references. All fresh weights, original113,042FIT/17,989VAL labels, identical soft targets/splits/AdamW.01/clip1/batch128/bucketing/min2000/max12000/eval100/patience6/MPS. Controlled cache is lossless and shared; old controls stay raw. Width216 active counts OFF2,056,537/ON2,084,185 versus old2,035,713; smallOFF858,241/ON874,625.

Only if .0003 beats .001 in ALL FOUR wide feature×seed VAL cells, add all eight secondary .0001 fits. They never alter primary effects or the full-game LR choices. Freeze every checkpoint/condition/source/data identity and primary VAL-LR choices before ANY new TEST/fight/game outcome. Score/panel every20(or28)condition, including all four old references. New score/panel runs must use ROOT environment; don't reuse old outputs.

Fight bank: source worlds92,100,000,000..+1023. Frozen grid SHA `4bdae9f4cfea3cc2edde97c4399e1760981c504a4bb8f43d067acae180a6cc44`:400Act2 boss,100Act1 boss,100Act2 hallway worlds; unsalted plus8salts. One hash-chosen fight/world/group, no model-based selection. Check whether original old-minus-plain gap repeats on NEW fights at both seeds (old VAL-selected rate versus plain128/.0003). Three primary effects on Act2-boss resampled HP loss: feature, size and rate, averaged over other factors per seed, then mean of the two seeds. Bootstrap WHOLE worlds with allsalts,10,000 draws/seed1919. Two training seeds do NOT double worldn. Replicated direction requires BOTH seed CIs exclude0 in the same direction. Show seed conflicts/interactions/fixed-level effects as secondary. No familywise/equivalence claims or subtraction of champion-relative CIs.

Eight fixed full-game arms on NEW DEV91,001,000,000..+4095: champion; old1919; four controlled feature/size profiles1919;old2020;plainOFF1282020. Student LR chosen only from original.0003/.001 grid by VAL. Champion OOC fixed. PRIMARY full-game comparisons are direct paired old-versus-plain wins at EACH seed, with whole-world bootstrap and exact McNemar; floor/boss metrics and other pairs secondary. Conditional boss HP is descriptive. `full_game_DEV_cohort_v1.json` freezes seeds; `combat_TEST_cohort_v1.json` has28,676 exposed labelTEST IDs. Confirmation91.2e9 remains closed.

## Monitoring, analysis and finish

Check useful learning every20–30min: bounded current validation/gradient/probe/parameter movement, exposures, actual completed cells/cohorts, throughput, memory/disk and faults. A PID alone is not progress. Diagnose safely and preserve settings/results. Main first OFF128/.0003/1919 must reproduce the now-verified technical curve. Keep evaluations even if compute tight. Do not leave fixable failures for Sid to re-authorize.

Strict generic `controlled_compare.py` supports labels (agreement AND CE), panels and games. Factorial `controlled_factorial.py` is PANELONLY. Before using it, root must create a compact freeze mapping `s1919_f0_w128_lr0.0003` keys to actual checkpoints from controller `conditions` (`rc_off128_lr3e4_s1919` keys). Independently verify checkpoint/condition, selection.global_freeze equals original globalfreezeID, and secondary=false; bind and record that audit. The tool binds the compact file but does not inspect the original freeze itself. Use CLI10,000resamples. Actual strict completeness/native/runtime/filter/cohort checks must pass before any confidence claims.

Produce a self-contained report and scientific plots using existing matplotlib runtime; explain which changes helped/repeated, whether original gap returned, agreement versus fights versus games, unresolved causes and a recommendation. No unexplained G1/stepD jargon. Both original combat students were from random weights; the historical champion's mixed training was not inherited. Old engineered features, larger actor and unequal LR tuning were differences under test. Count matching is not architectural equivalence; two fixed training seeds and bounded rates do not establish global superiority/optimality. These are development diagnostics until future fresh confirmation.

Locally merge verified work into ws/nonllm, no push. Deliver findings/discussion, then pause heartbeat `representation-diagnosis-continuation` only when required work is complete. It wakes this thread every20min and stays quiet on unchanged/nonactionable state. Do not resume unrelated paused studies or touch their automations. No persistent Goal was created.
