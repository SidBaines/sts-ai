# Long curriculum — latest first

## 2026-10-12 16:49 UTC — v1 paused and remaining allocation transferred to v2

The user paused v1 at15:31 UTC. Its controller andownedlearner/nativegroups haveexited; the monitoringobserver wasstopped andresults preserved. V1used17,575.872seconds oftheoriginal36,000;the remaining18,424.128seconds were explicitly transferred to a new run fromthismorning's originalcheckpoint. **Do not resume v1 or spend its historical remainder without a new budget decision.**

The revised continuous full-game study lives in `/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI`, with separate [summary](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/docs/long_curriculum_v2_summary.md) and runtimeoutputs. Mainv2started16:44:51 UTC aftera verified budget-counted pilot. V1's finalpanel remains embargoed; no frozen source, checkpoint or release tag waschanged. The old scientificresults andcriticalauditbelowremainhistorical v1evidence.

## 2026-10-12 15:25 UTC — critical audit: weak broad learning and frequent discarded updates

The user's skepticism is supported by the broader curves. In50-batch windows within the fresh-start phase, fresh Act2 success is6.4%→6.1%, full3 wins0/1300→2/1300, and Act1 rehearsal74.0%→71.8%. Candidate full3 DEV wins are0,1,0,1,2,1,1; only two worlds ever win, with one repeatedly solved. This is limited positive evidence, not sustained broad competence.

The two-win CP405 was rejected specifically because its Act2 DEV had8/48 clears versus selected11/48; Act1 retention41/48 passed, with no faults/cuts. Whole-game observer results do not affect selection. The selected CP236's ancestry has only6new chunks/511updates versus526committed chunks/36,217updates overall: most short branches are restored after three chunks. All60audited fresh-start-stage decisions were rollback or hold. This is a plausible design bottleneck, not an established simulator or optimizer failure.

Recommended next condition: full3-primary selection with larger fresh DEV confirmation, and separate continuing learner/evaluation champion to avoid resetting every noisy check. Rewards and architecture would initially stay fixed; any change needs a distinct recorded condition and the user's decision. The active v1 study continues unchanged, with roughly5.3active hours remaining; final data stays embargoed. [Detailed critical audit and concrete proposal](long_curriculum_learning_audit.md).

## 2026-10-12 14:21 UTC — more fresh starts; latest candidate achieves two full victories

Training crossed the planned three-active-hour boundary into **act2_full**, which places more weight on fresh Act 1 starts. A verified128-task batch now has32 Act2 continuations,57 fresh Act2-goal runs,26 fresh three-act runs and13 Act1 rehearsals:96 fresh worlds per chunk. No scientific settings or source pins changed.

| Fixed 48-world full-three-act DEV | Wins / requested | Defeats | Faults / cutoffs | Mean reached floor |
| --- | ---: | ---: | --- | ---: |
| Initial reference | 0/48 | 48 | 0 / 0 | 28.60 |
| Selected best CP236, reused exact-SHA result | 0/48 | 48 | 0 / 0 | 30.90 |
| Candidate captured at controller round353, evaluated13:44 UTC | 1/48 | 47 | 0 / 0 | 28.58 |
| Candidate captured at controller round405, evaluated14:14 UTC | 2/48 | 46 | 0 / 0 | 29.73 |

Recent candidate victory counts are0,1,0,1,2; this is encouraging but still rare and variable on repeated DEV worlds. Selected-best remains11/48 fresh-start Act2 clears with42/48 paired Act1 retention; the observer does not promote candidates. The latest full3 checkpoint copy, raw metric SHA and exact initial-matched task multiset passed independent checks; both winning worlds were defeated by the initial model. Final data remains embargoed.

At14:20:59 UTC,417 committed chunks requested53,376 episodes from**27,632 distinct TRAIN worlds =26,998 fresh +634 native**. Sampled learner starts reuse5,440 already-fresh worlds and addzero seeds. This manifest inventory counts requested worlds, not healthy completions. The live API reported11.43million valid transitions,27,628 accepted optimizer steps andabout6.30 active hours remaining. Learner, observer and dashboard identities are healthy; no repair was needed. [Seed audit](long_curriculum_seed_audit.md), [dashboard](http://127.0.0.1:8786/) and[generated detail](long_curriculum_live.md).

## 2026-10-12 13:21 UTC — interactive dashboard deployed and verified

[The live explorer](http://127.0.0.1:8786/) now offers training-range zoom/pan/brush, trailing rolling averages, independent left/right axes, source/start/goal/floor/horizon filters, signed HP-cost/return/floor/step histograms, configurable scatter plots, sample tables and loaded-cohort CSV export. INITIAL, SELECTED BEST and LATEST CANDIDATE full-run results remain prominent; detailed status and interpretation text are collapsed. [Controls and metric guide](long_rl_dashboard.md).

| Verification | Evidence |
| --- | --- |
| Python/native checks | 437 tests passed; one optional artifact skip; bounded smoke passed |
| JavaScript checks | Persisted numeric, denominator, missing-value and saved-filter regressions passed |
| Live browser | Dual axes, smoothing, zoom, filters, distributions, scatter and baseline selection verified |
| Episode API cost, 19,328 records | 24.04 MB JSON → 1.71 MB compressed; measured 0.394 seconds |

Data loads lazily with bounded, explicit coverage and consistent pagination identities. No new dependency or learner telemetry source change was needed. HP cost means net reset-to-end change; terminal HP and full health trajectories remain unavailable. Training remains live on unchanged source/rewards, approximately 2.7 active hours used and 7.3 remaining.

The 13:14 latest-candidate full-run panel returned 0/48 wins, zero faults/cuts and mean floor27.54; selected-best remains0/48 and meanfloor30.90. The earlier1/48 candidate victory remains in history, but has not become reliable completion. Final data stays embargoed.

## 2026-10-12 12:47 UTC — first monitored full victory; exploratory dashboard upgrade underway

The 12:43 periodic check observed **1/48 full three-act victories** for the candidate captured at round 242, versus the initial 0/48. It also had 47 defeats, zero faults/cutoffs/unstarted tasks and mean floor 28.46. Selected-best CP236 still scored 0/48, mean floor 30.90; its primary Act 2 DEV had improved to 11/48 with paired Act 1 retention 42/48. The candidate's isolated victory is encouraging descriptive evidence, while neither role has established reliable full-run completion.

Three subagents are upgrading the dashboard into an interactive explorer with training-range zoom, rolling averages, independent left/right axes, start/goal/source filters, distributions and logged health-cost metrics. This changes dashboard assets/telemetry only; the learner remains running on its pinned source and constant rewards. At round 246 the ledger recorded 2.125 active hours used, about 7.875 remaining of the original 10 hours.

## 2026-10-12 12:17 UTC — periodic three-act evaluation and audited seed coverage

An independent CPU-only observer now checks selected-best and latest completed training-candidate snapshots every 30 minutes on the same 48 fresh-start DEV worlds. It does not alter learner selection, rewards or the final embargo. The dashboard now separates INITIAL, evaluated SELECTED BEST and evaluated LATEST CANDIDATE; exact checkpoint hashes, capture/evaluation times, reuse and newer-unevaluated-weight warnings prevent stale results being presented as current.

| Three-act DEV snapshot | Wins / requested | Defeats | Faults / cutoffs | Mean reached floor |
| --- | ---: | ---: | --- | ---: |
| Initial checkpoint | 0/48 | 48 | 0 / 0 | 28.60 |
| Selected best, CP14 | 0/48 | 48 | 0 / 0 | 30.92 |
| Candidate captured at controller round 184 | 0/48 | 48 | 0 / 0 | 28.71 |

These are complete matched cohorts. Selected best gets farther on this cohort; full-game completion improvement is not established. Identical checkpoint/condition results are reused with their original evaluation timestamp, rather than plotted as new independent measurements. Observer process 8097 runs separately; only the dashboard was restarted (PID8502). The pinned learner remains live and unchanged. [Evaluation observer](full_run_monitor.md).

The seed audit at 12:04:30 UTC covered 167 committed chunks: **21,376 attempts from 10,320 distinct TRAIN worlds**, comprising **9,686 fresh worlds + 634 borrowed native worlds**. Sampled learner frontiers reuse 2,338 of those fresh worlds and add no new seeds. Current 128-task batches contain 70 Act 2 continuations, 19 fresh Act 2-goal runs, 26 fresh three-act runs and 13 Act 1 rehearsals. Fresh worlds rotate each chunk; this is not a fixed 128-seed set. [Exact environment/start/seed audit](long_curriculum_seed_audit.md). A later manifest-only update at 2026-10-12T12:30:47Z records 13,220 distinct requested worlds across 27,776 attempts; fresh sampling continues to grow.

## 2026-10-12 11:20 UTC — live dashboard available

The main run continues in the background. The read-only dashboard is open at **http://127.0.0.1:8786**, refreshing every eight seconds. It shows the active allocation, checkpoint/heartbeat ages, valid transitions, accepted updates, training returns and percentiles, loss/KL/entropy/clip metrics, throughput/timings, matched DEV decisions and separate conditional training families. Missing metrics are explicitly unavailable; rehearsal clears and continuation success are not full-game win estimates.

Two subagents implemented and reviewed it. Validation: 410 host tests successful with one optional artifact skip; six focused dashboard tests and JS syntax passed; the bounded core smoke passed. Root verified the live API and rendered charts, including readable labels in the narrow in-app panel. Responsive assets can be edited without changing the learner. [Dashboard usage and extension guide](long_rl_dashboard.md).

The API returned RUNNING, an error-free snapshot, and approximately 9.3 active hours remaining. Hourly monitoring also checks dashboard availability and keeps this summary and the detailed log current. The frozen learner source, rewards, earlier experiment and final-cohort embargo remain unchanged.

## 2026-10-12 10:57 UTC — main run running after accepted pilot

The detached main controller started at **10:53:40 UTC** (PID 79441). The validated 10:57 snapshot recorded six completed main chunks, 33 total rounds and **9.693 active hours remaining**. The authorized budget stayed **36,000 seconds**; consumed pilot time was preserved and no extension was added.

The accepted 15-minute pilot completed **27 chunks, 736,386 valid transitions and 1,785 optimizer steps**. Its owned processes had stopped cleanly and source pins were unchanged. Selected best remains round 14: **9/48 Act 2 clears and 42/48 Act 1 clears**, with zero faults/cutoffs in those DEV checks. This is a small DEV selection gain, not a whole-game or significance claim.

Main chunk 32 demonstrated changed actor weights, 24,970 valid transitions and 16 accepted updates. Device, source hashes and reward coefficients remain unchanged: eight workers, dedicated GPU duty 1, floor reward .05, terminal ±1, gamma 1, GAE .95, HP weight 0. Final evaluation remains embargoed.

Evidence: `preflight/pilot_acceptance.json` and `preflight/main_resume_verification.json` under `data/long_rl_v1/`. The hourly monitor stays quiet unless progress is meaningful or attention is needed; it respects manual pause and never auto-extends.

## 2026-10-12 10:48 UTC — first valid pilot DEV promotion

Round 14 passed the exact matched 48-world development and fixed Act 1 retention gates:

| Full-start DEV target | Initial anchor | Round 14 | Defeats | Faults / cutoffs | Mean reached floor |
| --- | ---: | ---: | ---: | --- | ---: |
| Act 2 | 8/48 clears | 9/48 clears | 39 | 0 / 0 | 28.77 |
| Act 1 retention | 41/48 clears | 42/48 clears | 6 | 0 / 0 | 16.88 |

Selected checkpoint: `act2_continuations_00000014_pilot_a0_9b67285648d9a428.pt`, SHA256 `9019cdec9c4c2b92520249919d7c97d6f6e626907511b510a44d88258cede278`. Its current bytes match the published job artifact hash. Primary/retention raw metric SHA and requested task IDs were independently verified against the journal; both cohorts completed with native module `40ab…70bd` and no missing, faulted or censored cases.

Earlier rounds 2/5/8/11 rolled back; round 8 lost Act 1 retention (37/48). Round 17 subsequently rolled back from 7/48 Act 2 and 39/48 Act 1 to the selected round-14 model. The guards preserve the initial reference while allowing this small improvement. **One extra clear in 48 reused DEV worlds is a descriptive selection result, not statistical significance or a three-act victory claim.** Initial concerns based on the first two rollbacks were qualified by additional pilot data; no reward or scientific-method change/critic probe was applied.

Raw evidence: `data/long_rl_v1/campaign/metrics/act2_continuations_00000014_825ea8832fcb2305.metrics.json` and `act2_continuations_00000014_retention_40b628a90e6fcd3e.metrics.json`, referenced by the durable journal. At the latest check 21 rounds were booked and the pilot remained running; root controls pilot acceptance/main transition.

## 2026-10-12 10:38 UTC — required pilot running; source frozen for execution

Root launched the 900-second wall-clock pilot at **10:37:00 UTC**, using the same **10-hour active allocation** intended for the main invocation: dedicated GPU duty `1`, eight workers, one CPU thread, niceness 10 and unchanged rewards. Source commit: `f23cad1`. The final native-host preflight passed **404 tests in 11.445 seconds** (one optional artifact test skipped), and the bounded core smoke passed 21 decisions with an ordinary defeat and no error.

At the 10:38 check, three training jobs had completed, two rounds were fully booked in the journal and the first scheduled primary DEV evaluation was running. The selected best was still the initial saved joint checkpoint; **no DEV improvement is claimed yet**. The controller records actual actor-weight changes, accepted updates, valid transitions and source/task hashes separately from model quality.

The active-time/resource-gap regression now rebases stage/job deadlines after excluded pauses; explicit extensions append only the final mixed stage. Fixed Act 1 retention, coherent fault/censor guards, bounded new-goal transfer, immutable task/native pool/learner files and published checkpoint identities passed scientific review. Root owns pilot acceptance and the pilot-to-main transition. Machine status: `data/long_rl_v1/campaign/progress.json`; detailed/latest-first controller view: `docs/long_curriculum_live.md`.

## 2026-10-12 10:24 UTC — checkpoint write probe and active-budget review

A bounded CPU probe serialized and durably wrote the existing 55.24 MB checkpoint payload using atomic replacement:

| Probe | Durable write time |
| --- | ---: |
| 1 | 69.4 ms |
| 2 | 44.0 ms |
| 3 | 43.1 ms |

Evidence: `data/long_rl_v1/preflight/checkpoint_write_probe.json`. The source `57f1…9e70` remained unchanged; temporary owned copies were removed. These timings **exclude MPS synchronization and model/device transfer**. They support frequent per-chunk saves, but the pilot must measure actual end-to-end checkpoint timing/age and verify the at-most-30-minute recovery-loss requirement.

The active-budget redesign is under review. A stale wall-clock stage deadline after an excluded resource pause was identified and sent to the implementation agent: resumed work must recompute stage limits from active remaining time, not consume the preserved allocation in idle wait or fail retries against the old wall cutoff. Launch remains gated on those orchestration regressions and the final host test suite; the main run has not started.

## 2026-10-12 10:20 UTC — confirmed user allocation and unchanged rewards

The user chose **10 active hours for the first pass**, a **dedicated GPU**, AC power connected, pause/resume at any time and extension if progress is worthwhile, with at most 30 minutes of lost work. The selected setup uses eight simulator workers, one CPU thread and GPU job duty `1`. This supersedes the earlier 24-hour/50%-duty fallback and four-worker default.

Active time excludes manual/resource pauses and time while the controller is stopped. A 15-minute pilot is required before the main run. The controller's active-budget ledger, resume/extension behavior and source identities are still being finalized and tested; main training has not started.

Reward coefficients remain unchanged: **+0.05 per positive floor advance, +1 success / −1 defeat, gamma 1, GAE lambda .95, terminal HP weight 0**. A future taper of floor reward is hypothetical and separately scoped; it is not implemented here. PPO continues training model parameters.

Scientific repairs now present in the draft: fixed 41/48 Act 1 retention reference, fault/censor guards before promotion, goal-specific leaders and bounded goal-transfer evaluation, published checkpoint SHA validation, fixed native-pool/learner-file hashes and exact DEV task-ID binding. Sixteen controller CPU tests passed independently in 0.056 seconds, plus direct zero-win drift/censor/retention/goal-compatibility checks. Root/env-agent report real owned-orphan/ownership regressions passing; active-budget redesign and final full-suite acceptance remain pending.

## 2026-10-12 09:46 UTC — independent draft safeguards review

The new sampler/capture and inherited fixture repairs passed 36 targeted CPU tests independently in 0.082 seconds. Historical deadlines are patched only inside tests; synthetic task fixtures replace unavailable ignored experiment banks. Production behavior of the earlier frozen controller is unchanged.

The controller draft is under review. Required repairs identified: keep Act 1 retention tied to the fixed 41/48 initial reference rather than allowing a two-clear tolerance to ratchet after promotions; apply censoring/fault guards before promotion; preserve goal-specific references so an Act 2 improvement cannot silently replace the original three-act progress anchor; bind actual evaluated task IDs and fixed native pool hashes across resume. Controller acceptance remains pending those repairs and its own tests.

The encoder exposes current act but not task goal/horizon. Mixed goals produce heterogeneous critic terminal targets; transferred critic weights are explicitly uncalibrated. Keep the first pilot's recorded multitask objective. Goal-conditioned features would be a separate encoder-breaking condition, not an incidental fix. Fixed stage time shares are exploratory allocation, not mastery certification.

## 2026-10-12 09:38 UTC — bounded CPU worker comparison

A matched CPU-only repeat of the same 48 DEV three-act worlds used one Torch CPU thread and produced exactly 16,144 transitions in both conditions. All 48 action-trace hashes and outcomes matched the anchor/repeat; neither condition had a fault or truncation.

| Simulator workers | Elapsed | Transitions/s | Decision |
| --- | ---: | ---: | --- |
| 4 | 10.93 s | 1,477 | Background default: retain CPU headroom |
| 8 | 9.13 s | 1,768 | 19.7% faster in this single paired repeat; optional throughput setting |

Evidence: `data/long_rl_v1/cpu_preflight_comparison.json` and adjacent metrics. This is a bounded development sampling measurement, not a demonstrated training-speed improvement or guarantee under interactive load. Keep four workers unless the chosen resource policy favors maximum throughput.

## 2026-10-12 09:36 UTC — fresh development anchor and sampler review

The saved joint anchor was evaluated on the same 48 new DEV worlds (minimum seed 200043), policy seed 777. Every cohort completed: no faults, truncations or unstarted tasks. These are this study's reference gates; earlier final results will not select new models.

| Full-start target | Clears/victories | Defeats | Mean reached floor | Horizon |
| --- | ---: | ---: | ---: | ---: |
| Act 1 | 41/48 | 7 | 16.75 | 2048 |
| Act 2 | 8/48 | 40 | 27.42 | 4096 |
| Three acts | 0/48 | 48 | 28.60 | 4096 |

Metrics and exact command argv: `data/long_rl_v1/anchor_dev_act{1,2,3}.{metrics,command}.json`. Root is recording protocol/input hashes. The 11 pure sampler/capture tests passed independently in 0.025 seconds. They cover fixed disjoint new panels, immutable batches, per-task prefix bounds, whole-source-episode fault quarantine and capture opt-out compatibility. Controller/resource/retention review and the required 15-minute pilot remain pending; this reviewer launched no policies or GPU jobs.

## 2026-10-12 09:22 UTC — preparation; pilot and main run not launched

A new isolated study is being prepared in `/Users/sidbaines/.codex/worktrees/long-curriculum/SlayTheSpireAI`, starting from the saved joint composite `57f1e9613a69e31230b591ef0ff2cca27e23ddc81c3a24366cee173bc9909e70`. The [provisional plan](long_curriculum_plan.md) records new world partitions, continuation/full-run practice, Act 1 retention, goal-change optimizer reset, drift detection and recovery safeguards.

At this historical preparation point, duration/GPU/power preferences were unanswered and a 24-hour/50%-duty fallback was considered. It was superseded by the confirmed settings above; background execution was already authorized. A 15-minute pilot is required before the main run. Provisional background limits are four workers, one CPU thread and niceness 10; pause guards are below 10 GiB free disk or at/below 25% battery while discharging. No training-throughput or deployment claim has been measured yet.

| New-study evidence | Status |
| --- | --- |
| Controller/sampler implementation and independent review | In progress |
| New DEV baseline | Completed; same 48 new worlds for all three targets |
| 15-minute training pilot | Not run |
| Main training | Not launched |
| New final policy evaluation | Embargoed; not run |

The earlier completed campaign is historical transfer evidence only. Its frozen final cohorts and results will not be reused for this study's selection or retuning. Add new measured entries above this one, retaining actual checkpoint/task identities, faults/truncations and the distinction between continuation wins and full-run wins.
