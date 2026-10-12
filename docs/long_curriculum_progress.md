# Long curriculum detailed progress log

## 2026-10-12 09:36 UTC — preparation and anchor evaluations

Created a separate `codex/long-curriculum` managed worktree from the completed first study. Original research and the frozen first campaign remain untouched. Shared the existing Python runtime by a local ignored symlink; the corrected native 0.1.1 binary is reused read-only, with manifest and SHA validation.

Initial policy: `joint_act3_0030.pt`, SHA256 `57f1e9613a69e31230b591ef0ff2cca27e23ddc81c3a24366cee173bc9909e70`. Native module SHA256 `40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`; Python interface SHA256 `1fa75a1d3d7de01ceae2ba0461f8ce3cae4051e8e6ba94ff3ff1897a9af02205`.

Prespecified matched 48-world DEV cohort beginning at 200000, policy seed 777, 200 battle simulations, horizon 2048 for Act 1 and 4096 for Acts 2/3. Fresh TRAIN worlds start at 1000000. New FINAL cohort is prespecified in `configs/long_curriculum_protocol.json` from 300000 and embargoed until selection freezes; no final game states/outcomes have been evaluated. Old final results do not select this campaign.

CPU-only anchor evaluations used four simulator workers and one CPU thread. Each completed every requested task, with exact task IDs recorded. Commands, command/input signatures and full metric files live in `data/long_rl_v1/anchor_dev_act*.{command.json,metrics.json,log}`; protocol pins metric/checkpoint/task-file digests.

| Goal | Completed | Clears/wins | Defeats | Faults | Cutoffs | Mean terminal floor |
|---|---:|---:|---:|---:|---:|---:|
| Act 1 | 48 | 41 | 7 | 0 | 0 | 16.75 |
| Act 2 | 48 | 8 | 40 | 0 | 0 | 27.42 |
| Act 3 | 48 | 0 | 48 | 0 | 0 | 28.60 |

The new sampler has 634 distinct TRAIN Act 2 native-generated starts and 86 Act 3 starts from the existing corrected bank. Exact live prefix replay independently passed for TRAIN worlds 12007 (Act 2, floor 18, 249 actions) and 12082 (Act 3, floor 35, 337 actions). A terminal Act 2 goal-cap observation cannot be reused as an Act 3 frontier because its public menu/hash differs; capture uses only live nonterminal frontiers under a three-act task, and publishes them only after a complete fault-free source episode. Continuation results stay stratified by start act/source/goal; they do not estimate unconditional full-run wins.

Twelve focused sampler/capture tests passed. The first complete-suite run exposed inherited tests coupled to the expired earlier campaign deadline and ignored local data fixtures; repairs are underway in this new worktree. The bounded random seed-3 core compatibility smoke passed on corrected 0.1.1 in 21 decisions, ending in ordinary defeat without error.

Observed resource preflight: battery 80% discharging at 09:18 UTC, about 35 GiB free. Later power polling reported AC source with battery 77%. Controller under construction: finite budget, owned process/GPU locks, true progress watchdog, atomic recovery state, drift rollback, bounded checkpoint retention, disk/battery pause. Main training has not launched. User preferences for duration/GPU sharing remain unanswered; defaults will be stated before launch.

## 2026-10-12 09:39 UTC — bounded throughput and GPU preflight

Repeated the same fixed Act 3 DEV panel under four and eight workers, one CPU thread. Four workers: 16144 decisions / 10.927 seconds = 1477 decisions/s. Eight: 16144 / 9.129 seconds = 1768 decisions/s, about 19.7% faster in this single paired comparison. All 48 action-trace digests and outcomes match the initial panel in both repeats; no faults/cutoffs. Preserve `data/long_rl_v1/cpu_preflight_comparison.json` and repeat metric files. Four workers are the background default to leave CPU capacity while Sid works; this bounded evaluation is not an estimate of sustained PPO throughput or a learning gain.

A small host-side MPS tensor calculation verified GPU availability and correct numeric result. Root disclosed a finite 24-hour fallback budget, four workers/one thread and cooling between chunks while optional preferences remain unanswered. Battery/disk safeguards still apply; no learner has launched yet.

## 2026-10-12 09:54 UTC — rolling tasks, continuation capture and replay verification

Implemented `sts_policy.long_run_tasks` in this new worktree. Training batches advance through disjoint phase/round scratch seed blocks beginning at 1000000 and filter the fixed world-level TRAIN partition. Act 2 mixes include at least 20% fresh full three-act runs, separately labeled from Act 2 goals and Act 1 rehearsal. Native continuation sampling selects the earliest recorded public battle in each TRAIN world/act, using the actual act/floor metadata and strict corrected 40ab/1fa provenance. Capacity is 634 distinct Act 2 worlds and 86 Act 3 worlds; oversubscribed distinct-world requests fail explicitly. New DEV/FINAL definitions use separate ranges beginning at 200000/300000, without evaluating final game states.

The opt-in trainer flag `--preserve-task-prefixes` retains all mixed task frontiers and per-task goal/horizon/search bounds instead of collapsing them by world. Each immutable task JSONL has a `.jsonl.manifest.json` sidecar with exact TaskSpec IDs, source hashes and `start_source`, `start_act`, `goal_act`, `horizon` family labels. One internal PPO update per controller job keeps successive scratch cohorts fresh. Reusing finite native continuation worlds is explicit and does not claim fresh whole runs.

`--continuation-output` buffers live TRAIN Act 2/3 entry states reached during max-act-3 full control. It publishes them only after the entire source episode terminates without a simulator fault or cutoff. A later fault discards all buffered frontiers from that episode. Saved histories retain canonical public indices, OOC preview markers, public hashes, backend/interface provenance, actual collection-actor parameter SHA and initial-checkpoint lineage. Terminal-cap or empty-menu states are never reused across act limits. Learner/native replacement is recorded by world/act and previous/chosen task IDs; the deterministic chosen row is not labeled a chronologically latest actor.

Real bounded CPU checks passed on TRAIN worlds only. Native Act 2/3 starts (12007/12082) reproduced recorded public hashes at floors 18/35. An actual sampler task for world 13560 retained run/max-act-2/horizon-4096/native-200 settings through the CLI task parser and reproduced its saved public hash. Initial actor `57f1…` chose one boss-relic decision after a native TRAIN-12007 prefix; the resulting live Act 2 map state (floor 17, 248 actions, last preview false) captured and replayed to an exactly equal complete public observation. This partial-episode probe is verification evidence, not published learner-pool training data.

Twelve focused sampler/capture tests and the restored inherited test fixtures passed. The complete native suite passed **374 tests in 10.540 seconds**, with one pre-existing missing-artifact skip; no checks were weakened. Historical campaign clocks/fixture mtimes are explicit and health task fixtures are synthetic temporary definitions, so those tests no longer depend on an expired deadline or ignored machine-local banks. Production old-campaign code, environment core, native binary and old worktree were unchanged. No PPO/GPU learner was launched by these checks.

Proof JSON, replay scripts, source hashes and the full test log are preserved in `data/long_rl_v1/preflight/`. Reproduction uses corrected `STS_LIGHTSPEED_BUILD`, `PYTHONPATH=src` for `prove_long_prefixes.py`, and `PYTHONPATH=src:.` for `prove_long_sampler_capture.py` because it imports the actual trainer task parser.

## 2026-10-12 10:34 UTC — final preflight acceptance

The user confirmed 10 active hours for the first pass, dedicated GPU, AC power, pause/resume at any point and later extension if worthwhile. These replace the provisional 24-hour/half-duty/four-worker defaults. The prepared pilot/main argv now specify eight workers, one CPU thread, niceness 10, duty 1 and the unchanged source checkpoint. A 15-minute pilot consumes part of the same active allocation.

The immutable scientific configuration keeps reward coefficients at +0.05 per positive floor advance, +1 task success and −1 ordinary defeat; gamma 1, GAE lambda .95, terminal HP weight 0. The requested floor-reward taper was explicitly hypothetical and is not implemented. A future pure-game phase would need full three-act terminal goals in addition to zero floor reward.

Final host/native test suite: 404 tests in 11.445 seconds, successful with one pre-existing optional generated-bank artifact skip. Required bounded random seed-3 core smoke: 21 decisions, ordinary defeat, no error. Source hash map was identical before and after both checks. Evidence: `data/long_rl_v1/preflight/final_preflight_checks.json`, final suite log and smoke trace.

Thirty focused new controller/process/active-budget tests passed. Real tests proved cleanup of both recorded and previously unrecorded orphan simulator workers while unrelated sleepers survive; altered ownership proof cannot signal a group. Duplicate controller cannot mutate the active journal. Mock-clock orchestration advanced wall time by 86400 seconds during a resource pause, then launched only a fake child using the rebased active stage limit; that pause did not consume the 10-hour budget. Saved resume argv strips both inline and separate extension/session flags to prevent accidental repeated extension.

Budget extensions require an explicit operator command and append only to mixed_full3; earlier 3h/2.5h/2h stage intervals remain fixed. Checkpoints save every completed update/chunk. A bounded CPU-only 55.24 MB payload serialization/durable-write probe took 69/44/43 ms, excluding GPU synchronization/transfer; temporary owned copies were removed and source checkpoint SHA remained unchanged. Actual pilot checkpoint ages and learner progress still need acceptance.

Latest resource observation: AC power, 87% and charging, about 35 GiB free. Disk pause threshold 10 GiB; battery/discharge pause threshold 25%, including weak AC discharge. Model training has not launched at this entry; independent final review and source commit precede pilot.

## 2026-10-12 10:42 UTC — active-budget tests and first pilot data audit

Ten independent active-budget/control-CLI tests passed without launching a learner, native simulator or GPU job. They verify that only active monotonic intervals count; close and a day-long offline gap preserve the shared pilot/main allocation; only an explicit extension enlarges the final mixed stage; reward weights remain progress 0.05 and terminal HP 0. A full mocked orchestration case crossed the old UTC stage boundary during a simulated day-long battery pause, then launched its fake child successfully with a rebased stage deadline and only 300 seconds charged. Status is read-only; pause signals only a matching controller birth; saved resume commands cannot repeat either separate or inline extension flags. Logs: `data/long_rl_v1/preflight/active_budget_tests.log`.

The actual pilot started at 10:37 UTC. A separate read-only audit checked its first three completed training batches. Each contains 128 canonical task identities, matching the immutable sidecar and collected metrics, all in TRAIN. Each has 58 fresh scratch worlds at or above 1000000; the 174 worlds are disjoint across rounds. Each mixture is 70 Act 2 continuations, 19 fresh Act 2 goals, 26 fresh full-three-act goals and 13 Act 1 rehearsal tasks. Reported family strata keep those goals and native/learner start sources separate.

All 384 source episodes completed without a cutoff or reported fault, yielding 82,945 valid transitions. Their 63 published live learner starts comprise 59 Act 2 and four Act 3 frontiers. Every captured source task is a complete, nontruncated, nonfaulting max-act-3 episode; every frontier retains strict 40ab/1fa provenance, canonical history/preview lengths, public hash and frozen collection-model parameter identity. Each batch has one collection-model identity, different across batches. Round 1 and round 2 already sampled two and five learner-generated Act 2 starts respectively. No fault quarantine breach appeared; these three live batches contain no fault case, while the synthetic later-fault discard regression remains the explicit fault-path evidence.

`data/long_rl_v1/preflight/first_pilot_batches_audit.json` records exact task-file/capture hashes, source-model hashes, counts and family tables; `audit_first_pilot_batches.py` reproduces the read-only checks. This audit changed no runtime source, live journal, model weights or reward settings and ran no additional GPU or final evaluation. It verifies data integrity and genuine fresh collection, not a learning gain.

## 2026-10-12 10:57 UTC — pilot accepted; main background run verified

The pilot completed 27 chunks, 736,386 valid transitions and 1,785 optimizer steps in 900.0055 active seconds. Every recorded owned process identity had exited before main resume. Source hashes and recovery checkpoints passed acceptance (`data/long_rl_v1/preflight/pilot_acceptance.json`). Best pilot checkpoint 14, SHA256 `9019cdec9c4c2b92520249919d7c97d6f6e626907511b510a44d88258cede278`, achieved 9/48 matched Act 2 clears and 42/48 Act 1 clears, without faults or cutoffs. This small descriptive DEV gain does not establish full-game mastery.

Main resumed detached at 10:53:40 UTC, controller PID 79441. Six main chunks had completed by this snapshot, reaching global round 33. The latest verified main chunk had 24,970 valid transitions and 16 optimizer steps, with a changed actor fingerprint and verified checkpoint SHA. Total authorized allocation remains 36,000 active seconds, with zero extensions. Consumed time was 1,105.31 seconds including the pilot, leaving 9.693 hours. CPU inference, MPS updates, eight workers, one thread, batch size 1,024 and all reward coefficients match the frozen condition. Main-resume proof: `data/long_rl_v1/preflight/main_resume_verification.json`.

Hourly thread monitor `slay-the-spire-local-rl-progress` updates the human documents and checks real progress and operational recovery. It stays quiet unless meaningful change or action is needed, respects operator pauses, and never extends automatically. The new final cohort remains embargoed until the user decides to finish and selection freezes.

## 2026-10-12 11:20 UTC — dashboard deployed and visually verified

Implemented independent stdlib local service `scripts/long_rl_dashboard.py`, static HTML/JS/CSS in `web/long_rl_dashboard/`, six regression tests and `docs/long_rl_dashboard.md`. Source changes are confined to new dashboard files and documentation, outside the learner's pinned source set. The service binds only 127.0.0.1:8786, with curated read-only `/api/telemetry` and fixed asset routes; it never loads checkpoint tensors or embargoed final metrics. Process launch record: `data/long_rl_v1/dashboard/launch.json`, PID 88759; logs remain in that directory.

Coverage/cache regressions include journal-only running-to-complete changes, exact requested task-ID multiset comparison, explicit null evaluation/goal handling, selected checkpoint/paired retention binding, intentional deletion ledgers and unavailable metrics. Full host suite ran 410 tests successfully with one optional generated-artifact skip; six focused dashboard tests passed, as did JS syntax and the bounded core smoke. Independent reviewer confirmed scientific distinctions and curated routes.

Root opened the actual dashboard in the in-app browser, verified live values and refreshes, and inspected SVG curves. A visual issue in a narrow 320-pixel panel was repaired by sizing chart coordinates to the container and retaining readable 12-pixel tick labels. DEV captions now distinguish initial and selected floors. Refresh controls and rendered metric choices are visible. Live API response at acceptance had no telemetry errors and completed in about 0.03 seconds; snapshot history grows using cached reduced metrics.

The hourly monitor now checks dashboard health as well as training. Restarting or editing the dashboard must not change or interrupt the learner; adding unlogged learner telemetry would be a separately recorded training source change. Main training continues under the existing active-time ledger and fixed rewards. New final evaluation remains embargoed.
