# Joint sparse-win v2 — completed results, 15 October 2026

## Conclusion

The fresh joint run finished at **13:37:08 BST**, with no demonstrated win-rate improvement over the original pre-joint champion. It validates repaired KL and genuine minibatch critic optimization operationally. Critics retain useful signal on an unchanged original-policy validation cohort, but show calibration swings and weak late-act discrimination on recent rollouts. These remain research issues before another training condition; a finite loss or an accepted update is not proof of successful learning.

No further training has started. The original champion remains retained; no promotion or independent confirmation was triggered. See [protocol](joint_sparse_win_v2_2026-10-15.md) and [trace diagnosis](joint_critic_trace_diagnosis_2026-10-15.md) for the scientific setup, warnings and investigations.

## Execution and provenance

- Run: `data/rl_runs/joint_sparse_win_v2_20261015`, fresh actors from the original round2 OOC champion, sparse full-game win reward, conservative unchanged actor settings.
- Frozen training source: `ed7b10279b4724fd267b374cc505b6b705cc0438`; archived source files still match. Native module SHA256 `936734a2289e20971abdbe7ae2f703d2d82f4e357a4e2b8590817d9a96f81a04` still matches. Later commits contain dashboard presentation, independent analysis and documentation.
- Stop reason: `budget_complete`. Final budget receipt7,154.08 active seconds (**1h59m14s**) against7,200 allocated; terminal event7,153.08s, before final clock cleanup. The remaining46s did not justify beginning another stage. Startup validation/evaluation was included, with no excluded heartbeat gaps or budget extension.
- **19 accepted rounds,0 rejected**,1,248 retained actor optimizer steps. Both actor heads changed in every round. Original-policy drift remained within existing guards.
- Critics: **372 optimizer steps/head during joint training**, plus384/head warm-up;756/head in total,1,512 actual optimizer calls across both heads including warm-up. Minibatches256states, LR3e-4 throughout; no mid-run scientific setting changes.
- **14,592 complete training games,689 wins (4.72%)**,0faults,0censored,0invalid. All eleven baseline/evaluation cohorts also completed without faults/cuts. Stochastic training wins are not a greedy evaluation comparison.
- No KL/numerical rejection; maximum behavior-likelihood reconstruction error5.39e-5 against1e-3 guard. Peak learner RSS9.69GiB below12GiB; last reported free disk7.36GiB above6GiB reserve.
- Final `latest.pt` SHA256 `090c4aee5c082279ba119fa0b1e081e8c3638ad2f5948165ae8a2efaa03c875b`. Final evaluated snapshot SHA256 `4177999fd89cb072737494ceef26b2b28cf42b3bc8e1ffecee8aeb8d0dc31e67` has the same policy, with evaluation-specific checkpoint metadata.
- Machine-readable final receipt: `final_review.json`; exact checkpoints, optimizer buffers, source/native artifacts, per-game outcomes/action prefixes, critic diagnostic features and monitor-review history remain local. Code/docs are pushed; model/data archives are not published.

## Greedy evaluation

The original58/1024 baseline reproduced all1,024 historical action-trace hashes. Every subsequent evaluation uses those same DEV worlds with greedy actions. The final round19 was evaluated before stopping, even though ordinary evaluations run every second round.

| Joint round | Wins /1,024 |
|---|---:|
|Original baseline|58|
|2|52|
|4|54|
|6|61|
|8|56|
|10|55|
|12|57|
|14|54|
|16|50|
|18|59|
|19, final|62|

Final62/1024 is6.05%, against5.66% original: **+0.39 percentage points**. There are28 newly won worlds and24 newly lost worlds; the paired exact two-sided p-value is0.678. A20,000-resample paired whole-world bootstrap (seed1919) gives a nominal95% win-rate difference interval **[−0.98,+1.76] percentage points**. Reused DEV screening makes these descriptive comparisons; the final/best checkpoint has not passed an independent confirmation. There is no convincing gain or regression.

Final mean floor improves0.3545 relative to original, but paired95% interval[−0.1250,+0.8448] spans zero. The intermediate small floor gains were not sufficient to establish progress in terminal wins. Joint evaluation also cannot identify an individual actor head's benefit without a separately designed composition comparison; none was started after the agreed run.

## Critics: optimization improved, reliability remains limited

Warm-up gave the original actors' independent critics useful signal before joint updates. On the **same512-world never-fitted original-policy cohort**, final versus warmed predictions are:

| Metric | Combat warmed →final | OOC warmed →final |
|---|---:|---:|
|Explained variance|0.11025→0.11461|0.05913→0.06953|
|Brier loss, lower is better|0.033430→0.033690|0.035322→0.035286|
|Mean predicted win|4.47%→6.04%|4.12%→5.80%|
|Observed weighted win rate|3.91% unchanged|3.91% unchanged|

These point estimates show retained predictive information, with some improvement in centered error, rather than broad forgetting. Calibration worsens: predicted means rise, so higher EV does not imply lower total error. No statistically established critic improvement is claimed from these point estimates. Fixed panel details/hashes: `critic_fixed_cohort_final_0019.json`.

On the final **four-batch held-out current-policy window**,646worlds/29winningworlds, aggregate EV is0.03352combat/0.02652OOC. Act3 EV is **−0.03571combat/−0.00412OOC**, with225/210worlds respectively. Predictions are non-flat, but both late-act critics have weak-value warnings. Their Act3 mean predictions20.44%/21.25% exceed observed weighted returns15.04%/15.57%. No urgent three-window stop occurred. The difference from the fixed panel cannot by itself distinguish changing seeds, changing reached states/policy or representational limits.

Warnings were investigated during the run. Same-state checkpoint comparisons showed that the warmed critic also struggled on the update10 warning cohort, and paired whole-world Brier-difference intervals spanned zero. Update13 then showed a clear calibration swing on unchanged states: batch fitting wins39/615 versus held1/153 made recent fitting prevalence a plausible driver. Later warnings recurred in both heads. The recorded analyses separate this evidence from unproven causes; see the trace diagnosis for details and selected replayable examples.

A metric limitation matters: beating a global4.6% constant Brier reference is insufficient for late-act states with much higher conditional win rates. EV and act-conditioned calibration must be considered together. The agent monitored and interpreted the warnings rather than treating a dashboard/live process as health evidence.

## Decisions to discuss before another run

The repaired execution path is suitable for further investigation, but simply extending this same training condition is not supported by the win results. The most concrete issue is critic calibration's sensitivity to recent sparse-success composition, alongside limited OOC and late-act discrimination. Consider act-conditioned calibration controls and a fixed panel from the current policy distribution; then assess data mixing or probability-scale stabilization with whole-world held-out comparisons. Critic feature coverage and sparse long-horizon credit assignment remain possible limitations, not diagnosed causes. Any new condition or settings change should follow discussion with Sid and receive a separate record.

The run and its observer are complete; the heartbeat monitor is paused after the final review. Older paused studies stay paused. The dashboard remains available for inspection.
