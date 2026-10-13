# Non-LLM curriculum research goals and current state

The first local campaign is **complete**. Work is isolated on `codex/nonllm-curriculum`; the original `astra-6sep` checkout is preserved. Signed environment releases `env-v0.1.0` and corrected `env-v0.1.1` remain immutable. Core stays dependency-free; PyTorch/NumPy and Gymnasium are optional extras.

The [curriculum plan](nonllm_curriculum_plan.md) considers action representations, architecture/method alternatives, hyperparameters and bounded selection. Executed stages include supervised candidate policies, combat DAgger/PPO, corrected-native OOC PPO and joint full-control Acts 1–3 PPO. Separate 2.3M candidate heads are trained together in the final composite. Scope is Ironclad, ascension0, ordinary Acts1–3.

Selection froze at **2026-10-12 05:58:19 UTC**, after the last GPU learner ended at05:58:04, before the06:30 deadline. All22 declared final panels passed independent exact-cohort, command/status/hash, checkpoint, source, native-binary and Python-interface gates. Four full three-act CLI runs on two previously checked DEV worlds plus two replays passed; traces matched across processes and native-controlled decisions were zero. **362 tests** and the mandatory bounded core smoke passed. See [completion audit](nonllm_completion_audit.md).

Held-out results: **145/160 isolated Act1 fights**, **25/32 later-act fights**, **17/21 Act1 clears with native combat**, and **20/21 Act1 clears under full learned control**. Three-act full control has **0 victories,20 defeats,1 simulator fault**; no truncations. The specialist comparator has0victories/21defeats/0faults. Strong Act1 competence is supported; full-game competence is not established. [Final report and uncertainty](nonllm_final_evaluation.md).

[Latest-first summary](nonllm_summary.md), [detailed progress](nonllm_progress.md), [decision history](nonllm_progress_notes.md), [usage](nonllm_agent_usage.md) and [artifact hashes](nonllm_artifacts.json) preserve the experiment. Checkpoints, banks and binaries are gitignored local outputs; copy them before removing the worktree. Frozen shared role denotes the joint composite, not a single shared encoder.

Native0.1.1 repairs automatic-combat character initialization and rejects reported undefined behavior. Old auto-combat evidence is quarantined; legacy full-control combat transfer remains scoped, not a general simulator certification. The broad hash-final partition has an early seed-3 diagnostic exception outside the quantitative benchmark ([audit](nonllm_protocol_audit.md)). Final horizons and censoring differ from development and are reported explicitly.

Future research should investigate later-act OOC/deck strategy, continuation starts, credit assignment and learner-induced combat states with a new evaluation protocol. Short OOC bridges and GAE pilots were rejected before final selection. No post-final training or hyperparameter tuning occurred. Other algorithms/representations were considered rather than all implemented; this campaign does not claim game mastery or complete simulator fidelity.

## 12 October: longer local continuation study

A new authorized study is being prepared on `codex/long-curriculum`, transferring the frozen selected checkpoint `57f1e961...` without changing the completed first campaign. Its [plan](long_curriculum_plan.md), [latest-first summary](long_curriculum_summary.md), and [detailed log](long_curriculum_progress.md) are the live record. New TRAIN worlds begin at 1000000; fixed DEV worlds begin at 200000; a new 48-world final cohort begins at 300000 and remains embargoed until selection freezes.

Fresh matched 48-world DEV baselines are 41 Act 1 clears, 8 Act 2 clears and 0 three-act victories, with no faults or cutoffs. The curriculum mixes later-act native/learner continuations, new full starts and Act 1 rehearsal. Continuation outcomes remain conditional, separately labelled evidence. Reset optimizer state when the goal changes; protect Act 1 retention and use explicit later-floor/progress drift gates. The controller must provide finite duration, owned process locks, atomic recovery, bounded retries, true-progress watchdog, disk/battery pauses and run-owned checkpoint retention. The user confirmed a 10-hour active first allocation, dedicated GPU and pause/resume at any time; execution uses eight simulator workers, one CPU thread and no artificial cooling. Offline/resource pauses do not consume the allocation. Explicit extensions append to the final mixed stage without changing earlier stage boundaries or reward coefficients. A measured 15-minute pilot and independent safeguard review precede sustained training. No new final evaluation has occurred.

Preflight acceptance at 2026-10-12 10:34 UTC: the final host/native suite ran 404 tests successfully (one optional generated-artifact skip), and the mandatory bounded core rollout passed. Source hashes stayed unchanged during checks. Real process regressions cover parent crash, previously unobserved owned workers, unrelated-process preservation, altered ownership refusal and duplicate-controller journal integrity. Fake-clock orchestration proves a one-day resource pause preserves active time and correctly rebases the stage/child deadline. Initial rewards remain +0.05 per positive floor advance, +1 success and −1 ordinary defeat; future taper is hypothetical. Pilot acceptance and main launch remain next.

Pilot acceptance at 10:52 UTC: 27 chunks, 736,386 valid transitions and 1,785 optimizer steps; all owned process groups exited cleanly. Best DEV improved Act 2 clears from 8/48 to 9/48 with Act 1 retention 42/48. Main resumed in the background at 10:53:40 UTC with the same source, reward and remaining active budget; six further completed chunks were verified by 10:57. Hourly monitoring keeps the logs and latest-first summary current. New final evaluation remains embargoed until the user finishes the continuous study. No reward taper is enabled.

At the user's request, an independent monitoring-only CPU observer now evaluates selected and latest completed checkpoints on the fixed fresh-start 48-world Act 3 DEV panel every30 minutes. The dashboard explicitly distinguishes initial references from timestamped evaluated snapshots. First selected CP14 result is0/48 full wins withmeanfloor30.92 versusinitial28.60; a recent candidate also has0/48 withmeanfloor28.71. No full-game competence claim follows. Training remains the unchanged10h active curriculum; exact12:04 seed audit records10,320 distinct requested TRAIN worlds and rotatingfresh coverage. See[observer](full_run_monitor.md) and[seed audit](long_curriculum_seed_audit.md).

At13:21 UTC the user-requested interactive dashboard replaced static telemetry with zoom/pan/brush, rolling and dual-axis plots, source/start/goal/floor/horizon filters, episode distributions, scatter and CSV records. Lazy bounded read-only APIs retain exact task/checkpoint identity and show coverage limits; source/rewards remain frozen.437host tests plus JS regressions and bounded smoke passed. Latest periodic full3 candidate again scored0/48; the earlier1/48 snapshot is descriptive history. See[dashboard guide](long_rl_dashboard.md) and[live summary](long_curriculum_summary.md).

At14:21 UTC, theplannedact2_full phase isrunning with96freshworlds perchecked128-task chunk. Latestmonitoringcandidate scored2/48 full3victories(46defeats,0faults/cuts), followingcandidatewinhistory0,1,0,1,2;selected CP236remains0/48 full3and11/48Act2. Thesearevariable repeated-DEVevidence,withnoobserverselectionfeedback orfinalopening.417committedchunks cover27,632distinctrequested TRAINworlds;about6.3activehoursremain. See[latest-first summary](long_curriculum_summary.md).

The15:25UTC critical audit found weak broad learning, variable full3 candidate wins and frequent short-branch resets. Selected CP236 retains only6new training chunks despite526completed jobs; full3two-win CP405 was rejected by the Act2 objective. A proposed separately identified next condition would select on full3, confirm on fresh DEV and separate continuing learner from evaluation champion. It has not been applied; v1 remains unchanged. See[learning and selection audit](long_curriculum_learning_audit.md).

## 12 October: authorized continuous full-game v2

The user paused v1 and authorized a revised study from the morning original checkpoint, using only the remaining18,424.128active seconds. The new branch codex/long-curriculum-v2 separates a continuous warm learner from a full3-confirmed champion, unifies all training goals atAct3, and logs retained lineage, optimizer resets, paired outcomes and public endpoint health. New48screen/192confirmation/paired192retention DEV references are fixed; no final data opened.467host/native tests and real baseline/controller proofs passed before the pilot. See[v2protocol](long_curriculum_v2_protocol.md),[latest summary](long_curriculum_v2_summary.md) and[detailed log](long_curriculum_v2_progress.md). The independently tested Discovery fix PR is kept separate from the pinned study backend.

V2pilot was accepted after26continuous chunks/3,086retainedupdates/862,697validtransitions, withonecoldstart and25warmoptimizerrestores. Main resumed16:44:51UTC andgenuinecontinuedupdatespassed;the10hcombinedallocation remainsunchanged. The dashboardnowdefaults tov2withpausedv1selector. First48screen remained1/48 andthe192championisstilloriginal2/192,so usefulperformancegainisnotyetestablished. See[v2summary](long_curriculum_v2_summary.md). DiscoveryfixdraftPR#1 targetsenvironment-base independently; studybackend0.1.1 staysfrozen.

At 17:42 UTC the user paused v2 for a causal combat/OOC regression investigation and forgetting prevention. Two matched confirmations remain 0/192 full3 victories versus the original 2/192; Act1 retention was 131 then139 versus151. All v2 learner processes are stopped and 14,252.174 active seconds remain unspent. Matched TRAIN head swaps and code/gradient review are being prepared under separate diagnostic identities; no automatic continuation, final opening or science change is allowed. See the latest v2 summary and progress log.

The 18:01 UTC phase diagnosis used 64 matched TRAIN starts and the exact192 DEV confirmation (190 common-valid). OOC-only replacement reduced Act1 clears151→130 andAct2 clears13→5; combat-only replacement preserved151Act1clears andincreasedAct2to22 butlosttheoriginal2fullwins. Savedcombat32bankwins24→25. OOCdriftthereforeexplainsbroadprogressionloss, whilebothphasesneedfull-winretentionchecks. A protectedoriginalrestartpilot is proposed, not launched; v2remainsoperator_paused with3.959activehoursremaining. Fournewdiagnosticregressionsplusfull471testsandboundedcoresmokepassed. See the forgetting diagnosis and v2summary for evidence and uncertainties.

## Independent critique on 12 October

A separate [training critique](nonllm_training_critique_2026-10-12.md) reviews the first campaign and both longer runs. New bounded CPU probes confirm non-equivalent action deduplication, exact encoder information loss, selected critic menu limitations, delayed strategic credit, and missing learning/evidence telemetry. The review branch preserves executable probes and output hashes. These are proposals and diagnosis only: no optimizer update, campaign resumption, new final evaluation, or runtime source revision occurred. V2 remains operator-paused with its remaining allocation unchanged.

## Authorized repaired training study

The user subsequently authorized implementing the critique in the current worktree with primary CPU/GPU access until approximately 13 October 07:30 UTC. The old v1/v2 studies remain preserved. The [new plan](nonllm_v3_plan.md) and [latest summary](nonllm_v3_summary.md) track implementation and scientific acceptance separately.

The unreleased candidate environment 0.2.0/public-v2/native-API-2 repairs action-instance identity, exposes typed public inventory/current action mechanics, and incorporates supported Discovery generation. Its initial acceptance passed 483 host/native tests (one optional artifact skip), bounded smoke, reached-state distinct-resource regression, deterministic fresh-process replay, and fresh/reverse patch checks. Old banks require explicit regeneration. No old experiment has resumed and no new training condition is claimed successful.

# Typed strategic policy implementation (12 October 2026)

The authorized critique implementation now includes `public_entities_v3`, an explicit frozen-legacy behavior bridge and an initially zero OOC entity residual with a disjoint entity/menu critic. All eight native micro/macro equivalence tasks match trajectories/rewards. The new fresh DEV192 baseline is 156 Act 1 clears / 20 Act 2 clears / zero wins. A first TRAIN-only MC critic calibration failed its holdout gate (MSE 0.178→0.203) and is preserved but not accepted. No game-learning improvement is claimed yet. See `docs/public_entities_v3.md` and `docs/nonllm_v3_summary.md`; old campaigns and final cohorts remain untouched.

Model v2 preserves controlled public context in trainable residual inputs and independently reproduces every baseline DEV trace. Its first eight-chunk macro pilot ended 156/18/0 after temporary small gains; it is not promoted. Eight of ten training wins had no actor-bearing OOC experience, motivating a replay-verified 44-frontier OOC start bank. Matched micro/macro pilots and expanded TRAIN collection are active. The setup now reports actual sample/gradient exposures, finite CPU-float64 drift and actor-bearing wins; useful game learning remains an empirical gate.

The corrected-start micro/macro comparison ended 157/18/0 versus 154/16/0, with no accepted gain. The expanded bank supplies 114 exact strategic frontiers. A separate frozen-strategy combat residual trained on fresh native labels produces a first mixed DEV192 candidate, 164/15/1 versus baseline 156/20/0. It needs independent confirmation; a zero-residual control has one trace discrepancy under typed batching despite equal outcomes. Neither condition is promoted. Detailed evidence and remaining gates are in `docs/nonllm_v3_summary.md`.

The short combat fit confirms one win on a separate DEV192 cohort, but a second optimizer seed regresses and ten-times-longer cloning severely regresses. No condition is promoted. The zero-residual discrepancy is resolved by phase-separated inference: all 192 reference traces/rewards now match exactly, with zero faults/cuts; 517 host/native tests and bounded smoke pass. Learner-induced TRAIN supervision and teacher-budget diagnosis are next, alongside remaining interface and reporting work.

At 03:40 UTC, learner-induced 200/1,000-budget fits and a search/draw consensus fit all fail live retention despite valid, pure-query banks. Teacher labels are unstable across search randomness and hidden draw orders; disagreement alone does not prove incorrect choices. Privileged 1,000-search combat improves conditional TRAIN success and Act 2 reach, but has no fresh wins on the small control cohort. A dependency-free read-only report now exposes outcome/retention, actual learning diagnostics, source identities and teacher controls. 529 host/native tests, bounded smoke and default-controller equivalence pass. Broader-start λ=.95/1 macro credit comparisons are next; no final data opened or condition promoted.

The broader credit comparison and optimizer replication finish without full-win
gains. Reached semantic audits then motivate environment 0.3/public-v3/native-API-3
and entities-v4: precise deck/bottled joins, reward-group alternatives and live
combat player scalars. 536 host/native tests, bounded smoke and patch checks pass;
all 192 canonical original-policy controls/rewards/endpoints transfer unchanged.
Old 0.2 sources/binaries/data are preserved; fresh compatible artifacts are
`data/nonllm_v4`. A new 64-world TRAIN bank completes with no faults. Useful
learning remains unproven; see `docs/public_entities_v4.md` and the latest summary.

The overnight study is saved with no promoted learner. Current-profile strategic
controls end 159/18/0 (uniform), 158/20/0 (act-balanced seed 0), 156/17/0
(act-balanced seed 1), and 157/20/0 (sparse-reward act-balanced seed 0), versus
156/20/0 initially on the same DEV192 panel. The final learner finishes at
06:25:46 UTC. Current acceptance is 540 host/native tests (one optional skip),
bounded smoke, 192/192 canonical transfer and 192/192 fresh-process full-trace
determinism. These establish execution and interpretable experiments, not
full-game mastery. See the [morning report](nonllm_overnight_report_2026-10-13.md)
for the next proposed outcome-validated supervision study. The expanded OOC
potion menu remains a separately tested candidate, not the default interface.

A bounded post-allocation [complete-menu OOC diagnostic](strategic_counterfactual_v1.md)
replays all 49 compatible late frontiers and 140 legal alternatives with frozen
public continuation. All control traces match, no branches fault/cut, and no
alternative converts a reference defeat to a full victory. Three menus improve
floor reach; 33 are tied on the recorded paired outcomes. This supports earlier
and broader strategic starts plus validation of continuation competence before
scaling an outcome teacher. No generated intervention data enters PPO; no
teacher targets are accepted. Latest full suite runs 546 tests with one optional
skip; bounded smoke passes.

The explicit native-1,000 comparator on four matched sources wins one reference
continuation where learned combat loses, yet finds no alternative first-choice
win gain. These are privileged conditional diagnostics on three source worlds.
The next strategic teacher must validate continuation competence as well as
start coverage. Protocol v2 records controller roles, and reporting now displays
counterfactual runs separately with manifest hash checks plus actual actor
exposure/loss mass by act. Current suite: 549 tests, one optional skip; browser
verification and bounded smoke pass. No new policy or teacher target is adopted.

The earlier-source follow-up now provides **1,051 exact replay-verified starts**
from 60 new TRAIN worlds, with 17 independent Act 3 worlds, 77 card-only reward
decisions and 60 late Act 1 boss-relic choices. Source play uses the original
frozen OOC parameters and an explicitly privileged winning-plan controller;
its independent 64-world source panel scores 60/17/0. This broadens curriculum
coverage but does not establish a trained public agent or validated teacher.
Per-action native replanning differs from the built-in agent's plan retention,
so old comparator conclusions are scoped accordingly. Automatic source-byte
snapshots and hash-checked replay completion preserve reproducibility. See
[source bank, comparisons and limitations](early_strategic_frontiers.md).

A world-first sampler now prevents menu-rich worlds from dominating diagnostic
selection. The matched six-world card-only reward pilot completes 30 legal
branches under each frozen-public/privileged-plan continuation, all controls
matching, without faults/cuts. Neither controller produces a full-win gain.
One public floor gain from skipping a card disappears with the other
continuation. No strategic target or learner is accepted; broader sequential
strategy and uncertainty/continuation validation remain necessary.
