# Long curriculum — latest first

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
