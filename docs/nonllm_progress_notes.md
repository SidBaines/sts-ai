# Experiment decision notes and recovery instructions

This is the maintained human log; `nonllm_progress.md` is automatically refreshed
every 30 seconds from the atomic command journal. Keep both: the generated page
records exact commands and measurements, while these notes explain decisions.
Raw datasets/metrics/checkpoints are gitignored local artifacts, so do not assume
they exist on a fresh clone. The code and these logs belong on the experiment
branch; never move the environment release tag or modify the original checkout.

## 2026-10-11 20:08–20:50 UTC — setup, profiling and first policies

- Created attached worktree `codex/nonllm-curriculum` from `env-v0.1.0`, preserving
  the original research work and the clean release checkout.
- Wrote the detailed curriculum plan before substantial training. Budget includes
  setup and final evaluation; original deadline is 2026-10-12 06:08:46 UTC.
- Delegated independent encoder/model, curriculum/data, training, and evaluation
  work. The parent owns exclusive GPU scheduling; agents use CPU tests/collection.
- Verified MPS and MLX using actual matrix operations. The sandbox hides Metal
  devices; GPU commands require the approved host execution path.
- Audited fresh 256-world Act 1 bank: 49,944 legal labels, 2,245 starts, 252 valid
  worlds, four quarantined native-UB faults. No old Gemma/native-build banks reused.
- Found and corrected replay RNG accounting: prefixes must preserve OOC preview
  commits. Twenty-five sampled starts reproduce public state and native labels.
- Found potion description grammar mismatch before training and versioned encoder
  `public_candidate_v2`; old CPU smoke v1 checkpoints are not campaign inputs.
- Measured 1/4/8/12 workers, exact observation caching, CPU/MPS/MLX training and
  inference. Selected eight workers, CPU rollout inference, MPS batch updates.
  Measurements and qualifications live in `nonllm_throughput.md`.
- Ran 228 tests plus required bounded release smoke before GPU training. Added
  further regression tests for masking, PPO old-policy ratios, GAE time limits,
  checkpoint recovery, cache/gradient parity, split/report aggregation and replay.
- First small policies trained roughly 6–7 million examples each in 90-second
  jobs. Development imitation is only diagnostic; forced choices are now reported
  separately so they cannot inflate meaningful action accuracy unnoticed.
- Initial evaluation hit a missing CLI cache flag. Preserved both completed BC
  checkpoints, paused only our dispatcher, added flags, and reran evaluations.
  The original deadline remained fixed. First models each won 42/48 matched dev
  fights; same-task random 28/48, public heuristic 42/48, native search 44/48.
  No final worlds evaluated. Small panels do not establish integrated competence.
- Collected separate Act 3 enrichment: 68,520 labels, 2,983 starts, 246 valid
  worlds/ten flagged faults. Act 3 coverage is thin (123 fights, no final boss).
  Do not claim boss generalization from this bank.
- Collected separate privileged train-only OOC lookahead labels: 256 frontiers,
  four candidates, ≤30-decision continuations, 89 strict proxy improvements.
  Act-clear healing complicates endpoint-HP proxy; compare real held-out outcomes.
- Identified prefix-reset overhead: replay repeatedly renders intermediate states
  that are immediately discarded. A read-only prototype suggests a large gain;
  an optional production path must preserve per-step UB and RNG/terminal checks.
- Identified a released auto-native-combat limitation: its temporary BattleContext
  UB flag is not latched. Zero surfaced OOC/lookahead faults cannot certify zero
  native UB. Audit existing APIs before any change; do not silently change the
  frozen simulator or reinterpret faulted games as ordinary defeats.

## Next gates and remaining work

1. Finish six size/context BC screens, learning-rate comparisons and seed repeats.
2. Verify CPU-inference/MPS-optimizer routing with genuine multi-round updates.
3. Compare combat PPO hyperparameters/reward controls; use easy-to-hard replay
   starts and corrective labels, retaining earlier fights to limit forgetting.
4. Compare scripted versus lookahead OOC warm starts and optimize OOC with a fixed
   native combat controller. Keep its search budget consistent across conditions.
5. Train full-control traces, expand Act 1 to Act 3, compare shared/specialized
   composition, and evaluate actual learner control of both phases.
6. Freeze selection before final embargo access. Report grouped-world uncertainty,
   faults, truncations, Act clears and full victories. Stop training by the deadline.

## Resume safely

Worktree: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI`.
Use `.venv/bin/python`, `PYTHONPATH=src`, and the immutable base build path in the
campaign config. GPU operations need host execution; do not let independent
agents launch GPU jobs. Check process ownership and `campaign.json` heartbeat;
a live PID alone is not evidence of training progress. Preserve final checkpoint
artifacts before interrupting our dispatcher. Its child jobs use their own process
groups so bounded termination can include spawned simulator workers.

The supervisor reuses completed commands only when metrics exist. For an interrupted
command, a metrics file with top-level `elapsed_seconds` means the CLI finished;
recover that journal entry as complete after checking the checkpoint. A partial
PPO metrics file is not a completed job. Resume from its atomic checkpoint with
explicit matching data/reward/config identity, recording any trajectory restart.
Never erase failed commands or silently replay beyond the original deadline.

## 2026-10-11 21:28 UTC — laptop interruption and authorized extension

- Sid closed the laptop and explicitly asked to continue on return. All six
  architecture checkpoints were preserved. The dispatcher had been paused for
  an isolated GPU routing verification; its completed child was recovered from
  the final metrics/checkpoint instead of repeating training.
- Genuine CPU inference/MPS PPO updates passed: two rounds, 258 transitions,
  eight optimizer steps, 2.49s total, no surfaced faults. PPO ratios changed
  from one, KL was finite and the optimizer survived both device transitions.
  Artifact: `data/nonllm_v1/verification/mps_route.{pt,metrics.json}`.
- Sid extended the hard deadline to **2026-10-12 06:30:00 UTC**. The original
  06:08:46 deadline remains recorded for audit. Current phase boundaries retain
  time for OOC, integrated traces and final evaluation.
- The Mac reported battery power at 62%; Sid confirmed he will connect power.
  Continue checking power/progress rather than assuming a live process is enough.
- The detailed generated progress page refreshes every 30 seconds; its writer
  PID is 12028. It has no GPU usage. Raw command records and failed invocations
  remain intact. Each new condition records adapter source hashes.

## 2026-10-11 21:32 UTC — recovery verified and replay optimization

- Power is now AC, charging. Resumed architecture learning-rate jobs show fresh
  supervisor heartbeats and actual checkpoint/metric progress.
- Full experiment suite passed **242 tests** in 20.27s, with simulator tests
  enabled and no skipped native gates.
- Fast prefix replay passed matched RPC tests: twenty later-act starts, 7,852
  prefix actions, 4.118s → 1.579s (**2.61×**). Public observations, teacher labels,
  following rewards, diagnostics and seeded fault rejection match exactly.
- The optional replay path keeps the immutable native/Python base identities.
  Frozen bank collection source hash is `64389785…`; optimized adapter source
  hash is `08e37bd5…`. Replaying these banks is compatible; collection resume
  under a changed adapter is deliberately rejected. New commands log source hashes.
- Independent review caught optimizer reset between PPO chunks, selected reward
  mode being discarded, restart curriculum clock reset, and stale metrics reuse
  after a failed command. Repairs preserve compatible AdamW moments, selected
  reward, persistent phase frontier, and require successful fresh metrics.
  Incompatible BC/reward/task changes reset optimizer with a logged reason.
- Final Act 3 evaluations now include matching Act 3 random/heuristic/native
  baselines on the same embargoed worlds; selection is frozen first.

## 2026-10-11 21:41 UTC — real inference and optimizer checks

- Leading large policy real ABBA inference benchmark: warm CPU collection
  0.962s versus MPS 1.395s for 687 decisions/48 fights; actor+bootstrap CPU
  0.375s versus MPS 0.787s. All four runs had 45 wins and 687 transitions.
  This establishes matching counts, not a separate assertion of bitwise traces.
- First optimizer-carry smoke correctly reset because older checkpoint reward
  metadata differed. A second smoke using current metadata restored real MPS
  AdamW moments: source step 4 → 7, `compatible_ppo_chunk`, 1.377s.
- Two new regression tests prevent a failed command from reusing stale successful
  metrics and prevent the log attributing a later rerun's results to a failed job.
- Reviewed supervisor resumed PID 17369; generated-log writer PID 17370. These
  are recovery handles, not sufficient evidence of progress; check fresh metrics.
- User requested a short reverse-chronological summary in addition to the full
  log. Maintain `docs/nonllm_summary.md` at major phase changes and meaningful
  decisions; put dated updates at the top, include compact outcome/speed tables
  or plots, preserve uncertainty, and link underlying metrics rather than
  copying commands there. Keep this note and the raw timeline detailed.

## 2026-10-11 21:42 UTC — full verification gate

Full suite: **245 tests passed**, simulator gates enabled, in 20.719s.
The campaign is running with fresh heartbeats and continuing learning-rate
comparisons; current source and logs are being committed as a recovery point.

## 2026-10-11 22:07 UTC — curriculum audit and controls activated

- Independent audits found Act1-only model selection, sparse Act3 sampling,
  missing promised short controls, current-label-only DAgger fine tuning,
  confounded OOC teacher histories, and insufficient final completion gates.
  These were repaired before extended combat training.
- Ten matched90s PPO recipes cover LR1e-4/3e-4/1e-3, sparse/potential, clip.1/.2,
  entropy.005/.01/.02, gamma.99/1 matched to worker shaping, terminalHPweight.25,
  and same-architecture scratch initialization. All settings and faults are logged.
- Combat selection blends Act1 and later-act results. Early controls retain the
  original immutable hard cohort; topthree are reevaluated on expanded panels.
  Later training cycles Act1/2/3 task pools instead of a72%Act1 mixedbank.
- Additional bank seeds12000..14047:2025valid/23faults,560620labels,24276starts.
  Expansion uses the same native/Python identities. Expanded Act3 dev/final
  panels have15/12worldclusters, addressing the prior single-world limitation.
- Native rehearsal20000traincombatrows samples381997eligible records with all
  sourcehashes/transitiveprovenance. Each corrective update aggregates rehearsal
  plus all earlier DAggerlabels into a deterministic bounded30000row reservoir.
- Scripted-versus-lookahead ablation uses identical256frontiers and90s from the
  same specializedcheckpoint; baseline replacesonly teacher_index withstored
  scripted_index. Composite dev evaluation precedes integratedshared training.
- Nonbattle training world cohorts advance128seeds perchunk instead of replaying
  the same128worlds throughout aphase. Evaluation cohorts remain fixed.
- UTCdeadline checked eachpoll includingafterlaptopsleep, withsleep/grace capped
  byremaining deadline. CLI/script hashes join adapterhashes. Cached metrics
  require matchingSHA or validatedlegacycompletion.
- Budget-censored episodes preserve latestinfo/provenance and markzero-step
  truncations. Finalfinished status requires eachnamedpanel and exactrequested
  task-IDcohort classified withoutunstarted/budget-censored entries. Faults
  remain separate fromgame losses. Nativeauto-combatUBlimit remainsqualified.
- 255tests passed20.131s;focused campaign/reward/aggregation21tests also passed.
  Supervisor23112 islive;checkfreshmetrics aswellasPID.

## 2026-10-11 23:25 UTC — health study, faster collection and deployment proof

- Expandedharddev clip.1 vsnative pairedHPcostdelta+11.02,95%CI6.52..16.40.
  This prompted two matched180s health-reward pilots fromexact same source/OOC
  weights and192trainstarts64distinctworlds/act. No final worlds were evaluated.
- Reference battle70/80,meanHP14.9125,fullAct1clears5/24;lambda.25 battle72/80,
  HP13.6625,full2/24;lambda1 battle72/80,HP14.0625,full3/24. Allcohorts complete,
  matched andzero surfacedfaults. Prespecified criterion rejectedbothbecause
  compositeclearsregressed;selected.combat_health pointsreference snapshot.
  Keepalternatives forlaterOO Ccoupling; donotclaims generalharm from24worlds.
- Study ownjournal/plan/selection liveshealth_ablation_4b0ab67c028d. Source/code
  frozenbetweenconditions; directguardlandedONLYafterthe study finished.
- DeferredbootstrapCPUparitydeterministic+stochasticactualnativecases passed:
  actions/observations/rewards identical,targetdifferences<=7.45e-9;1.35..1.37x
  collectiongain. Enabledonlyafterhyperparametercontrols. Faultedtrajectories
  remainexcluded andactualcutoffstates stillbootstrap explicitly.
- ScratchCLIcontrol wasinitiallyrejectedbycheckpoint-onlywarmoptimizerflag.
  Correctedflagcondition and addedregression. Actual90s scratchlargeMLP:54/64
  Act1+20/26originalhard. Originalfailedinvocation ispreserved.
- Directguardmatched20laterstarts7852prefixactions1.613s->.915s; randomfirstbank
  andseededUBfaultproofs passed. Newadapter1f2e599f..., frozenbase49c/native8dba
  unchanged; oldcollectionresume remainsstrictsourcehash-incompatible.
- CompositeActorCritic nowroutesphasegroups ratherthancomputingbothnetsallrows:
  batch8large+large1.99x(allcombat),2.20x(allOOC),1.57x(mixed), withallparameter
  gradient/outputparity1e-5. Noencoder/modelarchitecturechange.
- DeployedpretrainedModelAgent throughofficialCLI modefull seed3 maxact1:
  twoindependentprocesses producedidentical459-rowcanonical traces,228external
  actions andnative_steps0, endingordinarydefeat. This verifiesboth-phaseagent
  integration/determinism, notwinningcompetence. Initialstdinlauncher failed
  spawn__main__ resolution; usedguardedCLIentrypointinstead, retainingfailure.
- Fullregressiongate beforedirectguard:259tests passed25.264s. Newhealth/dispatch
  gates andthree directguardpairedtests passed; currentfullgate follows.
- MainresumedPID40629 fromfreshjournal withphasefrontierpreserved, fasterreplay
  active anddeadlineunchanged. Healthstudyusedsame sourcefiles forbothcases.

## 2026-10-11 23:29 UTC — combined regression gate

**268 tests passed** in24.716s withnativegates enabled. Maincombat frontier
isadvancingagain afterhealthstudy; currentnext_round10 andnewguard source
active. OOC andjointfull-run stages remainrequired; nofinallearnerpanel yet.

## 2026-10-11 23:53 UTC — maintained summary and OOC preparation

- Sid requested a short human summary with newest milestones first.
  `docs/nonllm_summary.md` is manually maintained at major decisions/phase
  changes; the automatic command log remains the detailed audit source.
- Combat best round 14 improves matched dev Act 1 wins 41→43/48 and later-act
  mean HP cost 20.56→16.25, holding later-act wins at 29/32. Selection panels
  are reused, so this is not an independent final estimate.
- OOC phase will compare five equal 90-second PPO pilots from one warm start:
  progress 0.05 at learning rates 1e-4/3e-4/1e-3, sparse at 1e-4, progress
  0.02 at 1e-4. Same 64 train worlds, seed, native combat budget and 32-world
  development cohort. The supervisor owns all GPU execution and selection.
- Commissioned CPU-only plotting and scientific split/reward/cohort audits;
  no agents may launch overlapping GPU jobs or access final learner outcomes.

## 2026-10-12 00:03 UTC — evaluation gates and reproducible chart

- Saved a development-only combat chart, source CSV and manifest in
  `docs/assets/nonllm/`, linked from the short summary. Connected points share
  exact cohorts; changing panels are not silently combined.
- Full suite passed 284 tests in 24.53s with native gates, plus bounded random
  seed-3 smoke. Supervisor was paused while its already-running combat round 19
  completed normally; preserve that checkpoint and recover the journal before
  restart. Subsequent cache tests will separately cover semantic input identity.
- Evaluation now uses its declared full budget; PPO retains bounded 120-second
  collection rounds. Selection rejects incomplete cohorts and counts requested
  tasks in its coverage score. Faults remain separately classified.
- Live OOC reset confirmed search control and no learner combat menus. Ordinary
  native simulation budget 200 has default boss multiplier 2, whereas explicit
  teacher queries use 200. Net HP change is not cumulative damage, especially at
  healed act transitions; interpret integrated HP results accordingly.
- CLI logs may contain a single very large JSON metrics line. Inspect parsed
  metrics fields rather than printing an unrestricted tail of the job log.

- Final reviewed source passed **286 tests** in 18.05s. Recovered completed
  combat round 19 from its final elapsed metric/checkpoint; restarted only our
  supervisor at 00:05 UTC (PID 51535), writer PID 51536. The automatic log now
  links the short summary. Absolute deadline remains 06:30 UTC.
- Cache reuse now pins semantic arguments and input SHA256s. Legacy records
  require matching semantic knobs and inputs predating their launch; only three
  separately validated performance flags are normalized during migration.
  Completed architecture selection resumes directly instead of rerunning old
  screens after their budget expired.
- Final audit still must confirm actual nonzero PPO updates in OOC, joint Act 1
  and joint Act 3, not merely completed evaluation panels. Before final access,
  freeze file hashes for checkpoints and panel definitions, alongside paths.

## 2026-10-12 00:07 UTC — phase transition and recovery correction

- Combat phase finished; round 17 selected (44/48 Act 1, 30/32 later-act).
  OOC specialist BC is now running. Isolated wins do not establish integrated
  competence, and selected-panel evidence is not an independent test.
- Restart at 00:05 recomputed the last chunk's budget from 180 to 92.04 seconds.
  Strict new signatures correctly rejected that changed condition, but the
  curriculum reused its filename and repeated the chunk from the same source.
  Original 180-second metrics are preserved in
  `data/nonllm_v1/campaign/archive/combat_0019_original180.metrics.json`
  (SHA256 32e6b3aba75d738b1ff1af0cd2d002cb1cbefee9923c0b0ec1e95f17871a85cc).
  The non-selected original round-19 checkpoint was overwritten; the selected
  round-17 checkpoint and all earlier checkpoints remain intact. Keep both
  budgets' metrics distinguishable. The repeated job completed in 92.54 seconds.
- A recovered frontier must reuse a validated completed command's recorded time
  budget, not recompute it and overwrite a completed condition. Regression and
  a completed-artifact overwrite guard are being added. Existing parent continues
  OOC work; apply these supervisor changes at the next safe boundary if needed.

- The final full gate passed **289 tests** in 20.42s. New resume regressions
  recover the original budget token exactly (`180`, not `180.0`), archive
  verifiable completed artifacts before unselected-name replacement, and reject
  replacement of selected/frozen/current/best checkpoint paths. These final
  supervisor protections are staged; the already-loaded live parent continues
  OOC work and should load them at its next safe restart boundary.

## 2026-10-12 00:41 UTC — explicit native revision, no silent binary replacement

- Found the cause of OOC non-repeatability: automatic resolver stack-constructed
  BattleContext leaves Player.cc indeterminate. Entropic Brew and attack/skill/
  power potion discovery index class-specific pools using that field.
- Ironclad full-control contexts are value-initialized through pybind11; enum
  zero is Ironclad. Full-run native-teacher collection uses this full-control
  path; lookahead switches to automatic search and is affected. This only
  establishes class initialization, not global freedom from simulator UB.
- Quarantine all old native-auto OOC/lookahead evidence and checkpoints from
  clean-model selection; preserve them for diagnosis. Old binary8dba remains
  untouched. Environment-base fix targets a new0.1.1 release after tests.
- Paused dispatcher51535 at00:40:57 while old OOC0002 finished normally.
  Fresh campaign will retain full-control combat17 and shared BC initialization,
  use fresh native-matched banks, and restart OOC from clean supervised weights.
- New campaign CLI accepts explicit bank/hard-bank/panel/lookahead paths.
  retained_combat bypasses earlier combat stages, so completed clean combat
  work need not be repeated. Deadline remains06:30; phaseends remain02:08:46,
  04:08:46,05:08:46 for OOC, jointAct1, jointAct3.

## 2026-10-12 02:12 UTC — selected follow-ups and full-control handoff

- Corrected release env-v0.1.1 is signed at environment-base b31a898; first
  env-v0.1.0 tag/969fc484 and binary8dba are unchanged. Research native40ab,
  patch77fddbd, interface1fa75a1d. Full research gate passed307 tests.
- Fresh campaign data/nonllm_native_revision/campaign started01:12; fresh
  bank counts exactlymatchold full-modebank counts. Old automatic evidence
  remainsarchived/quarantined; active automaticlog tracksnewcampaign.
- Before-KL guard and two20-update teacher corrections usedthe SAME frozen
  ooc0004 source. Pairedold/before sixty-second PPO produced29+1fault /
  31+0fault clears onthe32worldDEVpanel. Scripted/lookahead correction
  produced19+1fault /16+0fault. Retainbefore-guard and rejectBCcorrections.
  Correctedlookahead256/256TRAINfrontiers valid,92changedlabels,0faults;
  fourcandidates/horizon30/native200, andANYfailedbranch rejectswholefrontier.
- Source/candidatefiles and follow-upmetrics retainedundernewcampaign; no
  final learnerdata read. Eight GPUfollow-up/train/eval commandrecords merged
  intoatomicjournal beforeparentrestart, aftercurrent0010 finishednormally.
- Newparent80850, existingwriter67532. SelectedOO C
  kl_guard_paired/before_update.pt (31/32). OOCphasecompleted; jointsettings
  adopt before-KL flag. Newidentity-named compositions verifybothsourceSHA
  andimplementationSHA insteadofsilentlyreusinganearlierpair.
- Fullgateafterguard passed308tests in18.18s; boundedpublicCLI smoke passed.
  SuppliedoriginalAGENTS stale-patchnote describesastra-6sep, whileclean
  basebuildscript alreadyuses currentpatchapply/reverse checks+check-source.
  Do not removeprovenancechecks oraddunnecessaryinstallerworkarounds.

## 2026-10-12 02:32 UTC — actual full-control progress and Act3 scope repair

- Sharedwarmstart7/32 vscomposed21/32 clearsonsamefull-control32worlds.
  Two90-secondjointpilotsdidnotbeatreference, sofulltrainingstartedfrom
  composedwarmstart. Larger48worldbaseline29clears; jointAct10002
  improvedto37/48 with0fault/0trunc,364updates207730acceptedtransitions.
- These aredifferent32/48cohorts, withseparateinitialreferenceevaluations.
  Do notconnecttheminasingleimprovementcurve. Finalstilluntouched.
- Recheckedoldfull-modeHPheadswithnewOOC31/32policy:combat17composite21/32,
  HP0.25=11clears1fault,HP1=13clears1fault. FulltaskIDs matchandcommonvalid
  rankingagrees;retaincombat17. Traininghistoriesdiffer, notcausalHPstudy.
- FoundAct3evaluationbug:trainingmaxact3/h4096 butDEVdefaultedAct1. Fixed
  initial/everyevaltoforwardstageact/horizon; Act3uses48worlds240s, Act1
  remains48/h2048/90s, OO C32/h1024.14focusedtests passed. Currentparent
  80850loadedearliercode; activatefixatnextsafejointboundarybefore04:08.
- Shortsummarycondensedtolateststate/majorchoices/evidencetables; detailed
  chronologyremainsinhuman/automaticlogs.

## 2026-10-12 03:04 UTC — Act 1 progress and recovery audit

Selected `joint_act1_0007.pt` completes the fixed 48-world panel with 38 act clears, 10 ordinary defeats, zero faults/truncations. The initial composed reference completed 29 clears and 19 defeats on the identical requested task set. Round 9 has 35 clears and is not selected. Its training chunk accepted 207,748 transitions and executed 443 optimizer updates; training remains live.

The 02:51 supervisor reload activated the three-act development evaluator. A v1 cache compared raw option order, so JSON-sorted effective settings caused `joint_act1_0008` to repeat despite equivalent values. Prior artifacts were moved into `history/`, selected round 7 was preserved, and the new chunk finished normally. Cache v2 canonicalizes effective options and migrates pinned v1 only with matching canonical values and complete path/SHA input identities. No extra restart was made for this repair; it loads on the next required supervisor restart.

A separate audit found two earlier learned-agent seed-3 CLI smoke traces. Seed 3 hashes to the broad final partition, so that partition was not wholly untouched. It is outside the designated final cohort (worlds >=10000), was not used in recorded training/selection aggregates, and the exception is recorded in `nonllm_protocol_audit.md` with an evidence snapshot. Future learned-agent integration fixtures must explicitly verify train/dev membership.

## 2026-10-12 03:39 UTC — optimized full-control training and safe reload

Selected `joint_act1_0019.pt`:42/48act clears,6ordinary defeats,zero faults/truncations/native-controlled decisions. First completed feature-reuse chunk17 accepted263,687transitions and470optimizer updates over180.644s (1,459.71/s); PPO preparation .755s versus39–40s in earlier chunks. CPU equivalence covers features, stochastic/deterministic action sampling, losses, gradients, Adam state and resulting weights. Peak numeric buffer71.1MB. Live adjacent conditions differ in evolving policy/worlds; no causal competence claim. Implementation2a23625, measured evidence5d638d8.

Full337regressions passed20.766s and mandatory bounded randomseed3core smoke passed. After one automatic-review timeout, the authorized retry succeeded. `/private/tmp/reload_nonllm_between_jobs.py` observed completed `joint_act1_0020_dev`, stopped only supervisor88818, proved no training child remained, checked the completed metrics SHA, and relaunched supervisor99267 at03:34UTC. Evidence: `/private/tmp/nonllm-supervisor-reload-record.json`. Fresh chunk21 and its evaluation completed with cachev2; chunk22 is running. This activates final core-Python freeze and explicit composite seed/reward settings before selection freezes. No training chunk was killed or repeated by this reload.

## 2026-10-12 04:00 UTC — final scope and deployment preparation

Read-only release audit confirms immutable env-v0.1.0/env-v0.1.1 targets, clean environment-base checkout and separate dirty astra-6sep original. Both managed worktrees remain attached to this chat. The designated final protocol uses horizon2048 for OOC/full runs; OOCDEV uses1024 and Act3DEV4096. These different censoring limits must be visible in final tables; differences cannot be attributed solely to generalization. Keep the planned final horizon unchanged. Selected final model CLI verification should run verified DEV seeds10121/10011 with `--max-act3 --max-steps4096`, through both learned heads and zero native-controlled steps. Stage3/final results are still pending.

## 2026-10-12 04:19 UTC — genuine three-act stage and backend launch binding

Act1 selection checkpoint26 clears45/48 with3ordinary defeats,zero faults/truncations/native-controlled steps. Three-act initial DEV config isrun/maxact3/h4096/48worlds/seed777:0victories,48ordinary defeats (3Act1/43Act2/2Act3),zero faults/truncations/native steps. This confirms retained Act1 ability and later-act weakness. Initialmeanterminalfloor28.40. CPU-only matched original-composite/native controls are underway separately; no finalworlds are read.

Release audit found that verifying `state.native_build` alone did not bind `Popen`'s inherited `STS_LIGHTSPEED_BUILD`. Repair bb03017 sets the child environment explicitly from frozenbackend for final runs, recordedbackend otherwise; records override and verifiedbinary/manifest identities. Baselines now stamp native and Pythoninterface provenance beforeworkersstart. Full346tests pass20.934s. Quiescent-boundary supervisor reload watcher is active, preserving old record separately at `/private/tmp/nonllm-supervisor-reload-backend-pin-record.json`. No scientific settings, seed cohorts, rewards or checkpoint lineage were changed by this repair.

## 2026-10-12 04:50 UTC — rejected OOC bridges and actual supervisor resume

Bridge evidence: `data/nonllm_native_revision/campaign/act3_bridge_20261012T044416Z.json`; preserved orchestration: `data/nonllm_native_revision/campaign/act3_bridge_runner_source.py` (SHA256 d2d769ffa2144704abc1c37141284ac52b9833eb344908591e33d81207c7d331). Main supervisor9961 was stopped only at a completed three-act DEV boundary with no training child. Its pending selector was completed with the same score/current/best/next-round logic. Both pilots started independently from the immutable selected Act3best2, with native-combat OOC macro decisions, LR3e-4/gamma1/clip.1/entropy.01/progress.05/HP0/before-KL and30-second budgets; both were evaluated on the exact same full-control DEV48/max3/h4096/seed777 tasks. Act2 accepted22updates/34438transitions:0victories0faults; Act3 accepted37updates/37068transitions:0victories1fault. Neither improved the anchor score, and the latter also failed the no-extra-fault promotion guard. Both checkpoints/metrics/commands remain preserved. Decision: retainmainanchor; supervisor resumed asPID15332 afterallGPUchildrenfinished. No finalworlds were used, no existing checkpoint or seed cohort was changed.

Read-only retention controls: fresh32later-actDEVfights, retainedcombat17=27wins,Act1best26=28,Act3best2=28,zero faults/truncations. The older30/32 result used a different bank and is not a same-panel decline. Corelaunchbinding is now active with explicit40ab backend. Full353tests passed20.789s.
