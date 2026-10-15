# Fresh-agent handoff — 15 October 2026

## Start here

**The joint RL run is complete, training is stopped, and its monitor is paused. No further training condition is authorized.** Sid requested this write-up and a code/docs push before handing off to a fresh agent and discussing next steps. Do not resume older paused studies, recycle their budgets or interpret proposals below as launch instructions. No new chat was created automatically.

Start new work on a topic branch from `ws/nonllm`, remote `git@github.com:SidBaines/sts-ai.git`; it contains this handoff plus later work (training speedups and these entry docs). Read `CLAUDE.md`; `AGENTS.md` resolves to it. Then read [next-study options](../next_study_options_2026-10-15.md), written after this handoff. Keep experiment logic outside `src/sts_env` and preserve unrelated checkouts.

This study ran in `/Users/sidbaines/.codex/worktrees/4a3c/SlayTheSpireAI` (branch `codex/joint-sparse-win-rl`, now behind `ws/nonllm`). That worktree still holds the run's gitignored data, so relative `data/...` paths below refer to it. Keep it, but don't commit new work there.

Read these in order:

1. [Completed run results](../joint_sparse_win_v2_results_2026-10-15.md).
2. [Critic trace diagnosis and calibration follow-up](../joint_critic_trace_diagnosis_2026-10-15.md).
3. [KL repair and original critic audit](../joint_kl_critic_diagnosis_2026-10-15.md).
4. [Restart protocol](../joint_sparse_win_v2_2026-10-15.md) and [verified machine snapshot](joint_sparse_state_20261015.json).
5. [Earlier training review](../training_review_2026-10-14.md) for the broader experimental history.

The [12 October handoff](../experiment_handoff.md) is historical: it names a different worktree and paused longer-curriculum study. Its allocation does not authorize work here. README now leads to this handoff.

## Work completed in this conversation

- Reviewed recent research, ongoing training and play-time search. Audited the multi-stage model lineage and whether it could be reconstructed. Preserved historical inputs, checkpoints, sources and native binaries in verified local archives, while documenting missing historical RNG, launch-source and online-trajectory records. Archiving cannot recover those gaps.
- Added reproducibility records for new launches, starts and resumes: effective settings, commands, code hashes/source archives, runtime/native identities, parent models, datasets and checkpoint RNG/optimizer state. Added compressed collection receipts with task specifications, complete action prefixes, outcomes and fault references. Exact joint optimizer buffers additionally support numerical replay.
- Hardened simulator-fault recording and bounded replay. Failures/timeouts remain diagnostic outcomes, never game losses. The captured Discovery case reproduced at the same operation on historical builds and passed that operation on a separately identified clean candidate. This does not certify the entire simulator. See [recording/fault replay](../rl_recording_and_fault_replay.md).
- Finished the outstanding search evaluations and requested Act 3 boss-head ablation. Wrote the [training review](../training_review_2026-10-14.md) and pushed code/docs. Sid explicitly chose to keep model/data archives local rather than publish public release assets.
- Created this branch and implemented guarded joint sparse-win PPO, independent public-feature critics, full-game binary reward, conservative actors and an interactive dashboard. The first pilot stopped on persistent numerical update rejection; it established no actor improvement.
- Repaired and tested masked KL arithmetic; replayed all four historical failed updates. Audited flat critics and isolated too few optimizer steps as a major cause, rather than broken targets, detached gradients or phase routing.
- Discussed moving to krill-mill using available evidence. Sid canceled that idea: no transfer, matched machine benchmark or remote training run was performed. Approved subsequent work stayed on this MacBook.
- Added a short active-monitoring instruction to project `CLAUDE.md`. Implemented 256-state critic optimizer minibatches at the same LR, critic-only warm-up with frozen actors, fresh validation, learning-health warnings and a fresh bounded joint run from the original champion.
- Monitored the new run every ten minutes, interpreted warnings and investigated them while it trained. Added the dashboard legend after Sid identified its omission; explained training versus greedy evaluation. Performed same-state checkpoint comparisons, paired whole-world uncertainty analysis and trace inspection without new simulator games or optimizer steps.
- Completed the allocation, wrote/pushed final results, paused the observer and prepared this handoff. No scientific setting changed mid-run; no new actor was promoted or further training started.

## Prior lessons and retained model

The successful later lineage was original composite → OOC PPO update 36 → smith distillation → lambda-1 PPO update 12 → smith-and-elite distillation → round-2 OOC PPO update 20. Combat tensors stayed original through those later stages. The macro-credit OOC recipe and strategic demonstrations helped; unchanged continuation plateaued. Do not infer that joint sparse learning works from that earlier OOC success.

The [training review](../training_review_2026-10-14.md) records exact settings and limitations of initial BC/DAgger/PPO, longer curricula, OOC macro credit, strategic distillation, lambda controls, combat imitation/value learning and search. Important retained findings:

- Initial training established strong Act 1 competence, but full three-act final evaluation was 0 wins/21 requested with one simulator fault. Isolated later-act fighting did not imply successful deck-building and progression.
- Longer v1/v2 exposed selection versus retained-learning problems and OOC-induced progression damage. Their studies, monitors and historical allocations remain separate and paused.
- Combat adaptation could damage endgame play despite better progression. The requested boss-head ablation recovered imitation damage (61/1,024 versus curated 46; parent 57), without establishing superiority to parent. Act/room gates correctly include Secret Portal bosses before floor 50.
- Corrected boss-plus-elite native search scored 333/4,096 on fresh frozen confirmation versus learned 213, paired p=9.14e-13, zero faults. This is native-assisted play-time search, not trained-policy improvement or certified public-only planning.
- Adding ordinary hallway search scored 98/512 on repeatedly used DEV versus learned 29. Its broader scope has no fresh confirmation. Do not present its 19.14% as independently confirmed generalization.

The retained original actor is:

```
/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/slay-spire-rl-debug-da6d22/data/rl_runs/ooc_r2_lam1/iter_0020.pt
SHA256 3d135d9bd53b92b5a5eb3dcc8bf26458140e282ee8ea63d9aaaf6ca3c2f222bb
```

## First joint pilot: failures and repair

`data/rl_runs/joint_sparse_win_20261014` completed 20 collections, about 1h54 active: 16 accepted/4 rejected rounds, 1,045 retained actor steps, 15,360 training games/708 wins, zero faults/cuts. Final joint greedy was 55/1,024 versus original 58; peak 65 was inconclusive. Final combat-only replacement scored 48, OOC-only 59. No head demonstrated benefit. Final retained actors were from accepted iteration 18; critics continued through 20. Historical artifacts remain unchanged.

The numerical failure was float32 underflow: reference probability 1.4013e-45 versus rounded-zero current probability made categorical KL infinite despite finite normalized log probabilities near −103/−104. `sts_policy.losses.masked_kl_divergence` now uses normalized log-space arithmetic, double on CPU/CUDA and float32 on MPS. **Only CPU was validated here.** Padding/nonfinite/empty-support guards remain; no trust coefficient was loosened. Rehearsal and original-policy drift use the helper.

`scripts/rl/joint_sparse_kl_check.py` successfully replays saved failures 14, 16, 19 and 20 after repair; the legacy iteration-20 control still fails and rolls back. Diagnostic repaired actors were never installed. The older pilot review's “only iteration 20 replayed” statement predates this follow-up.

`scripts/rl/joint_critic_diagnosis.py` audited all 245,736 selected states/15,360 worlds. Historical critic updates reproduce the saved weights bit-for-bit. Targets, phase routing and gradients are valid; zero-initialized output weights block trunk gradients only on the first step. A tiny memorization check verifies plumbing, not generalization.

The old schedule took only one full-batch AdamW step per collection: 20/head. One 256-state minibatch pass at unchanged LR 3e-4 gives 384 steps/head on the same exposures and improves held EV from near zero to 0.06266 combat/0.04133 OOC. Raising LR tenfold helped less. This justifies fixing optimizer coverage, not assuming unlimited extra fitting solves strategic prediction.

## Completed fresh joint condition

Frozen trainer source: `ed7b10279b4724fd267b374cc505b6b705cc0438`; configuration: `configs/joint_sparse_win_v2.json`. Later commits add presentation, independent analysis and docs. Recorded training-source/native hashes still match.

| Component | Actual setting |
|---|---|
| Parent | Original round-2 OOC update 20, never the previous joint actor |
| Reward/credit | Complete-game win 1/defeat 0, gamma/lambda 1; faults/cuts excluded |
| Actors | Combat LR 1e-5, OOC LR 3e-5; one epoch, clip .1, entropy .001; original-policy rehearsal beta .1, adaptive cap 1 |
| Exploration | Combat T .25/epsilon .001; OOC T .5/epsilon .002; evaluation uses argmax |
| Actor batching | Minibatch 4,096, microbatch 128, CPU with two Torch threads |
| Critics | Separate public-input 384→128→64→1 heads; AdamW LR 3e-4, weight decay .01, norm clip 1 |
| Critic inputs/fitting | 256 state features plus mean 128 candidate features; up to eight strided choice states/world/phase; total weight 1 per world/phase; true 256-state optimizer steps |
| Warm-up | Old feature cache, 12,265 fit/3,095 held worlds; 384 steps/head; actors frozen/hash unchanged; optimizer moments carried |
| Training | 768 games, expandable to 1,536 for sparse support; two collectors × six workers; chunk 16 |
| Evaluation | Same 1,024 DEV worlds every second round and before stopping; four collectors × one worker preserves inference batch shape |
| Guards | Likelihood error ≤1e-3; local KL .02, original mean KL .05/p95 .2/choice-change .2; sparse support; finite RPC/progress limits; paired retention/promotion; three supported urgent critic windows stop |
| Budget/resources | Two active hours including baseline/fresh validation/evaluation; RSS bound 12 GiB, free-space reserve 6 GiB; exact buffers use lossless gzip 6 |

`src/sts_rl/critic_training.py` implements fitting, world holdouts and diagnostics. Collected actor advantages remain R−pre-outcome V; fitting the critic does not recompute them on those outcomes. The historical `critic_minibatch=0` default exists for replay, not as the restart recipe.

Namespaces: TRAIN starts 60,010,000,000 with 10,000 stride/round; separate 512-game critic validation 60,020,000,000; repeated DEV 61,000,000,000; gated confirmation 61,200,000,000; rehearsal 63,000,000,000. No independent confirmation was triggered. Critic-held worlds are excluded from critic fitting; actors may train on them. Do not describe them as held out from all optimization.

Fresh validation before actor updates produced 20/512 wins, zero faults/cuts, no warnings and EV .11025/.05913. Original greedy 58/1,024 reproduced all historical action-trace hashes.

## Final outcome and remaining uncertainty

The run stopped **15 October 13:37:08 BST**, reason `budget_complete`. Final receipt used 7,154.08 seconds (1h59m14s) of 7,200; the terminal event reports 7,153.08 before clock cleanup. Another round requires more than 300 seconds remaining. No allocation extension occurred.

**19 accepted/0 rejected rounds; 1,248 retained actor steps; 372 critic steps/head after warm-up (756/head including warm-up).** Both actor heads changed every round. Training: 14,592 games/689 wins, zero faults/cuts/invalid games. All eleven baseline/evaluation cohorts also had zero faults/cuts. Maximum likelihood error 5.39e-5; peak learner RSS 9.69 GiB; last free disk 7.36 GiB. No KL rejection, rollback or promotion.

Greedy history: original 58; rounds 2/4/6/8/10/12/14/16/18/19 scored **52,54,61,56,55,57,54,50,59,62** of 1,024. Final 62 versus 58 is +0.39 percentage points, 28 newly won/24 newly lost worlds, paired exact p=.678. Nominal paired-world bootstrap interval is [−0.98,+1.76] percentage points. DEV was reused: these are descriptive comparisons, not independent confirmation. Final mean-floor gain +.3545 also has an interval spanning zero. Neither individual actor head's benefit is established; no new head-swap evaluation was launched.

On the unchanged 512-world original-policy panel, warm→final EV is .11025→.11461 combat/.05913→.06953 OOC. Predictions are non-flat and retain information, but means rise to 6.04%/5.80% versus 3.91% observed. Better centered-error point estimates do not imply uniformly improved calibration/Brier loss.

The final current-policy four-batch held window has 646 worlds/29 wins: aggregate EV .03352/.02652, Act 3 EV **−.03571/−.00412** with 225/210 worlds. Both have weak-value warnings, no urgent stop. Low late-act predictive value remains unresolved. Differences from the fixed panel cannot by themselves identify seed composition, policy-induced states, representational limits or stochastic sparse-return noise.

## Deeper trace investigation

`analysis/critic_trace_diagnosis_20261015.py` compares warmed/update-8/update-10 critics on identical held inputs: 24,746 states/1,547 worlds/77 winning worlds. It reproduces every update-10 score to within 1.2e-9, uses paired whole-world bootstrap intervals, and records room/action-semantic groups, reliability bins and replayable examples. It ran separately for about 82 seconds at reduced priority/two threads, without gradients or new simulator games. Reproduction requires a fresh output path; see its `--help` and the diagnosis doc.

The warmed critic already struggled on the warning cohort. Update 10 raised/spread OOC probabilities while largely preserving ranking (Act 3 prediction correlation .992 versus update 8); paired Brier-difference intervals spanned zero. Selected losing examples are illustrative, not representative or proof that an action caused defeat.

Update 13 gave stronger evidence of calibration sensitivity on unchanged states: OOC mean prediction 4.44%→7.01% at fixed 3.91% outcomes; Act 3 17.49%→26.53% at fixed 17.67% outcomes, while EV stayed steady. That batch had 39/615 fitting wins versus 1/153 held wins. Dependence on the latest fitting batch's success composition is plausible, not experimentally isolated. Warnings subsequently cleared and recurred in both heads. A cleared warning does not establish a repaired mechanism.

A metric caveat: beating a fixed 4.6% Brier reference is insufficient for late-act states with much higher conditional win rates. Read EV, calibration and world support together. New diagnostics did not silently alter the split, reward or optimizer recipe.

## Preserve these local dependencies

**A Git push does not back up models/data/native binaries. Sid explicitly requested local archives, not public release assets.** Latest joint artifacts are not inside the older pipeline ZIP; preserve both. No off-machine backup of the latest joint run was established. Do not archive/remove this worktree or donor worktrees without an approved preservation plan.

Relative paths below are under this worktree. The machine snapshot records full paths and critical hashes.

| Artifact | Location |
|---|---|
| Completed run | `data/rl_runs/joint_sparse_win_v2_20261015/` |
| Final state | `latest.pt`, SHA `090c4aee5c082279ba119fa0b1e081e8c3638ad2f5948165ae8a2efaa03c875b` |
| Final evaluated state | `evaluated_0019_10314e5b32af.pt`, SHA `4177999fd89cb072737494ceef26b2b28cf42b3bc8e1ffecee8aeb8d0dc31e67` |
| Warm critics | `data/rl_runs/joint_critic_warm_v2_20261015/warm.pt`, SHA `4714d4f621477d616b4a25910878eb8dc1687e4606903aa5c520a5c0a342874c` |
| Warm inputs | `data/rl_runs/joint_critic_diagnosis_repaired_20261015/public_critic_features.npz`; old pilot `before_0001_1827aab1b142.pt`; warm `report.json` binds the exact command |
| Exact training evidence | Run `records/.../launch.json`, source ZIP/native module, `before_*`, `buffer_*.pt.gz`, anchors and RNG/optimizer states |
| Collection/replay evidence | Nineteen `collection_*.json.gz`, baseline/evaluation/fresh receipts with task specs, complete action prefixes and hashes |
| Final/monitor records | `final_review.json`, `health.json`, `critic_fixed_cohort_final_0019.json`, `monitor_reviews/`, `monitor_review.json`, `monitor.json` |
| Diagnostics | Run `analysis_critic_0010/` and fixed-cohort reports for 10,13,16; older `joint_kl_repair_replay_20261014`, `joint_kl_repair_regressions_20261014`, `joint_critic_diagnosis_repaired_20261015` directories |
| Original pilot | `data/rl_runs/joint_sparse_win_20261014/`, unchanged |
| Verification | `data/rl_runs/joint_sparse_v2_validation_20261015/acceptance.json` and logs |
| Historical pipeline archive | `data/reproducibility/pipeline_20261014.zip` and receipt, 4,530,650,879 bytes; previously verified content-addressed objects, predating joint work |

`.venv` links to `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv`. It is the frozen Python 3.12.13/Torch 2.14.0/NumPy 2.5.4 runtime. Do not mutate it with pip or remove its owner worktree.

The native build is:

```
/Users/sidbaines/.codex/worktrees/public-search-canonical/SlayTheSpireAI/external/sts_lightspeed_rl_v2/build014_canonical
```

Set `STS_LIGHTSPEED_BUILD` explicitly for compatible native tests/diagnostics. Module SHA is `936734a2289e20971abdbe7ae2f703d2d82f4e357a4e2b8590817d9a96f81a04`. Package 0.1.4/episode diagnostic string 0.1.3 mismatch is known: hashes govern identity; do not rewrite historical metadata. No native/core contract changed for this restart.

New runs have strong source/input/RNG/optimizer bindings; exact optimizer replay is supported. This is not a guarantee of historical retraining or cross-platform bitwise equivalence. A fresh clone needs restored local dependencies and artifacts, not only Git.

## Operational state and verification

The queue reports the new job `done`, with no active jobs. Runner PID 76744 was left idle; trainer PID 14232 finished. The extra caffeinate assertion was tied to the trainer. Persisted PID files alone are not liveness evidence. Never kill unrelated processes or use self-matching `pkill -f` patterns.

The read-only dashboard is `http://127.0.0.1:8794/` (recorded PID 14214). If needed, restart only its server:

```bash
.venv/bin/python scripts/joint_sparse_dashboard.py \
  --out data/rl_runs/joint_sparse_win_v2_20261015 --port 8794
```

Old pilot port 8793 is separate. Teal is stochastic training, orange is greedy evaluation, purple is original greedy baseline. Chart series filtering is distinct from rollout table filters; episode filters do not change optimizer metrics. The champion card is the retained champion, not the latest candidate. Fresh critic-validation receipts are separate from the ordinary rollout table.

Both `joint-sparse-win-v2-learning-review` and older `slay-the-spire-local-rl-progress` automations are verified **PAUSED**. Run-owned next-review deadline is null. Do not reactivate them by default.

Future authorized jobs should use the queue/ownership/power/timeout mechanisms. The idle runner was loaded before the newest code and has an old default native build. V2 commands explicitly set the canonical native path; queue `--preparation` applied only to the wrapper's recording while the real trainer performed strict runtime checks. Do not use that flag or weaken resume identity checks to bypass a scientific mismatch.

Prelaunch verification passed 530 tests (three documented optional/generated-fixture skips) and the required isolated native random smoke. KL tests cover actual underflow, padding/nonfinite cases and gradients; critic tests cover actual step counts, held-world exclusion, retained advantages and warnings. The prior KL-only suite had 525 tests before the five new critic tests. Legend changes passed JS syntax/live visual/console checks. Analysis reproduced actual scores. This documentation handoff does not require more training or another full-suite run.

## Commits and next discussion

| Commit | Change |
|---|---|
| `50b24e1` | Guarded sparse joint PPO and dashboard |
| `87eea05`, `f4e8e0b` | Pilot/head-swap review and evidence qualification |
| `1b8844f` | Stable KL, saved-update replays and critic audit/tests |
| `96e305a` | Minibatch critics, warm-up, fresh validation, health checks, CLAUDE monitoring instruction |
| `ed7b102` | Final two-hour configuration; frozen training source |
| `b7e7b1a` | Launch acceptance and first-update review |
| `38d3eac` | Dashboard legend |
| `d02c83a`, `7f40576` | Trace analysis and calibration follow-up |
| `38a339c` | Completed results |

The new CLAUDE instruction emphasizes emitting **and interpreting** useful metrics, investigating early and using judgment, rather than accumulating smoke tests. The first pilot logged early critic weakness without timely interpretation; instrumentation alone did not meet Sid's monitoring intent. The restart has actual recorded reviews and investigations.

One instruction ambiguity is flagged without editing: CLAUDE still describes “this branch” as the reusable environment, although this is a research branch. Its core contracts still apply. Proposed wording for Sid's approval: “This repository contains a reusable environment base and branch-specific research; keep experiment logic outside `src/sts_env`.” No global/generated instruction change was made during handoff.

Before another condition, discuss calibration sensitivity to batch composition, act-conditioned controls/fixed current-policy panels and weak late-act/OOC discrimination. Replay mixing, smoothing or LR changes are proposals, not proven or authorized fixes. Feature coverage, exploration and sparse long-horizon credit remain possible constraints. The completed run provides no reason simply to extend the same condition or promote its actor.

Suggested fresh-agent prompt:

> Work in `/Users/sidbaines/.codex/worktrees/4a3c/SlayTheSpireAI` on `codex/joint-sparse-win-rl`. Read `docs/handoff/joint_sparse_win_20261015.md`, its machine snapshot, final results and linked diagnoses. Training is complete, both relevant monitors are paused, and no further training is authorized. Preserve local model/data/runtime/native artifacts and unrelated worktrees. Summarize the established results, uncertainties and most promising next decisions for discussion before changing settings or starting another run.
