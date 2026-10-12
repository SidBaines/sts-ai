# Learning and selection audit — 12 October 2026

Read-only assessment at approximately15:25 UTC. The active v1 learner, source hashes, rewards, task curriculum and final embargo were not changed. This audit distinguishes completed compute from updates that survive in the selected model.

## Assessment

Parameter learning is running, but sustained useful whole-game improvement is not established. Some policy changes improve particular DEV worlds, and the selected model improved the Act2 subgoal. The broader training/evaluation curves are largely flat or variable. The current controller repeatedly restores an earlier model after three-chunk branches, which is a credible contributor to the plateau; this is an inference from the design and lineage, not an experimentally isolated causal result.

| Fixed48-world DEV evidence | Initial | Selected CP236 | Two-win candidate CP405 | Latest evaluated candidate, captured511 |
| --- | ---: | ---: | ---: | ---: |
| Full three-act wins | 0/48 | 0/48 | 2/48 | 1/48 |
| Mean ending floor, three acts | 28.60 | 30.90 | 29.73 | 26.92 |
| Faults / cutoffs | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Latest-candidate win counts are0,1,0,1,2,1,1 across seven distinct evaluated snapshots on the same48worlds. Only two distinct worlds ever won:200204 wins in five candidate panels, while200458 wins only in the two-win panel. This supports a repeatable improvement on a particular world, not broad independent evidence from many new worlds. Using Act3 as floor≥35, the selected model reaches Act3 on11/48 versus8/48 initially, but has zero full victories. Its paired ending-floor comparison has22worlds better,17same and9worse. The latest evaluated candidate reaches Act3 on3/48 and dies in Act1 on10/48, versus8 and7 initially.

The two-win candidate versus selected CP236 has11worlds with better ending floor,19equal and18worse; mean ending floor is1.17lower. It improves rare wins while worsening some broader survival outcomes. It is worth preserving and confirming, not declaring the unequivocal champion from2/48 alone. All ten raw full3 panels, including the initial and selected snapshots, passed raw-metric SHA and exact48world checks, with no faults/cuts.

## Training trend within the fresh-start phase

Each window has50completed batches from act2_full. These are rotating TRAIN worlds, not a paired benchmark; numeric floor means exclude faulted episodes. Every reported requested denominator includes faults.

| Fresh Act1 family | First50 batches | Latest50 batches at audit | Interpretation |
| --- | ---: | ---: | --- |
| GoalAct2 clears | 183/2850 (6.42%) | 174/2850 (6.11%) | No clear improvement |
| GoalAct3 wins | 0/1300 (0.00%) | 2/1300 (0.15%) | Rare wins, weak overall signal |
| Act1 rehearsal clears | 481/650 (74.00%) | 467/650 (71.85%) | No retention gain |
| GoalAct3 mean ending floor | 25.86 | 25.80 | Approximately flat |

The windows are frozen within the first526completed TRAIN jobs: act2_full rounds343–392 and476–525. Exact member task/metric hashes are retained in the audit JSON; later jobs do not silently change their definition. The initial continuation-heavy phase likewise showed no broad upward trend within its fixed fresh-start families. Native/learner continuation comparisons remain confounded by changing saved-start distributions.

## Why the two-win model was rejected

The actual saved artifact SHA9319081e716ca61cf941bab61fe168c21ffc2672543ab9abc4c9926daf45b76c maps to training checkpoint act2_full round405. Its own primary Act2 DEV is8/48 clears, versus the selected anchor11/48; mean floor27.75 versus28.77. Its Act1 retention41/48 passes the fixed41minus2 guard, and there are no faults or cutoffs. The recorded decision is `rollback: primary_clear_or_floor_regression`.

The code ranks the current stage's target clear count first and floor second; any decline in the primary clear count triggers rollback. Full3 observer results are explicitly monitoring-only and never enter this decision. The first5.5active hours use Act2 selection; the planned Act3/mixed stages later use full3 victory selection automatically. Thus a policy with fewer Act2 clears but more eventual game wins can be rejected now. This is an objective mismatch with the ultimate task, not a missing dashboard update.

`selection_decision` is in src/sts_policy/long_run_control.py; both `rollback` and `hold` restore the stage incumbent. Evaluation runs every three completed chunks. At the bounded snapshot, act2_full had60decisions:46rollbacks,14holds,0promotions and0continues. Selected CP236 was last promoted at12:40:49 UTC.

The selected model's parameter ancestry contains only six new chunks (12–14 and234–236),511accepted optimizer steps and163,186valid transitions. In contrast, the audit saw526committed training chunks,36,217accepted optimizer steps and14,859,718valid transitions overall. About98.6% of counted updates are absent from the selected model's parameter ancestry. They are not all devoid of indirect effect: healthy learner-generated continuation starts survive and affect later collection. Nevertheless, headline compute totals do not represent accumulated optimization in the selected model.

In act2_full,60 of180 audited training jobs start fresh Adam state after returning to an anchor with the previous stage's mixture/goal signature;120 restore optimizer state within their short branches. The first phase predominantly restores the incumbent optimizer rather than creating fresh state. Resetting for a true task-mixture change is deliberate, but repeated returns to the old-stage anchor restart that adjustment over and over.

## Proposed next condition — requires a scientific decision

1. Make full three-act victory count the primary checkpoint objective from the beginning; use reached-floor/act distributions as secondary evidence and Act1/Act2 as regression diagnostics rather than an unconditional Act2 veto.
2. Confirm the initial model, selected CP236 and preserved two-win snapshot on a larger prespecified fresh DEV panel before deciding the new starting champion. For example, retain the48screening worlds and add144new DEV worlds with a separately recorded seed/task identity. Keep all final worlds embargoed. Repeated48-world winner selection needs confirmation because it is noisy and adapts to the DEV cohort.
3. Separate the ongoing learner from the evaluation champion. Do not automatically erase a short branch because a small DEV panel fails to improve. Preserve an incumbent and periodic snapshots; reserve rollback for invalid/nonfinite updates or confirmed substantial regression, using a prespecified repeated/expanded check.
4. Initially retain rewards, architecture and optimizer hyperparameters to isolate the effect of selection/rollback. Bound the continuation within the remaining original active allocation unless the user explicitly extends it. Preserve v1 outputs and give the changed method its own source/protocol/run identity.

A secondary, untested hypothesis: the public encoder has no goal-act or task-horizon input, while the mixed task families stop and pay terminal reward at different goals. The critic can therefore see identical public states with different future returns. Goal-conditioned value input or a consistent full-run objective deserves a separate experiment; static inspection does not prove it caused the current plateau.

Evidence: data/long_rl_v1/preflight/learning_selection_audit.json, campaign journal/jobs/events, immutable full_run_eval snapshots and SHA-bound metrics. No new rollout/evaluation or GPU job was launched for this audit.
