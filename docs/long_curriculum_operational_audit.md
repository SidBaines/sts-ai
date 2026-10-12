# Long curriculum pilot operational audit

Read-only snapshot, 12 October 2026 around 10:42 UTC. The pilot began at 10:37 UTC on committed source `f23cad1`, after the recorded 404-test/core-smoke gate. This audit launches no learner/evaluator, changes no source or scientific settings, and reads only the new experiment's training/development artifacts.

| Operational check | Observed evidence |
| --- | --- |
| Allocation | 36,000 authorized active seconds, no extensions; boundaries 3h/5.5h/7.5h/10h. Clock excludes manual/resource/offline pauses. |
| Devices/resources | CPU actor, MPS optimizer; eight persistent workers, one Torch CPU thread, nice 10, duty 1. |
| Fixed learning recipe | LR 1e-4, batch 1024, gamma 1, GAE .95, clip .1, entropy .01, progress-floor weight .05, terminal success/failure ±1, HP bonus 0. No reward taper. |
| Genuine learning | First ten completed PPO jobs: 276,402 retained transitions and 923 accepted optimizer steps. Actor-affecting tensor fingerprints change, rather than only metadata. |
| Throughput | Initial completed jobs took 19–28 seconds of reported learner time, about 1,024–1,448 valid transitions/s. Aggregate including actual child-process wall time is about 1,061 valid transitions/s. |
| Profiling | CPU actor 8.5–9.7s, native step waits 5.3–6.2s, reset waits 2.1–2.4s; update time 1.7–8.8s, encoding preparation .042–.048s in sampled jobs. More accepted updates naturally cost more update time. |
| Optimizer goals | First max-act-3→Act-2 transfer starts a fresh optimizer and logs critic recalibration. Compatible same-stage chunks carry Adam state. A development rollback returns to the anchor and correctly starts fresh again. |
| Development guard | First paired check: Act 2 3/48 vs fixed anchor 8/48, Act 1 retention 44/48 vs 41/48. Coherent primary regression triggers rollback. This is protection against drift, not evidence of learned improvement. |
| Artifact integrity | 44 remaining artifact hashes verified; eight intentionally deleted checkpoint hashes match the durable retention ledger. Zero unexplained missing/changed artifacts. Initial/best/current/last checkpoints remain present. Latest retained checkpoint age at this check: about 1.3 seconds. |
| Fault/censoring handling | First ten jobs: three surfaced training faults among 1,280 requested tasks (~.23%); no truncation or unstarted tasks. Faults are separate from game defeats and below the declared attention threshold. |
| Fresh and learner-generated starts | Task-family joining shows native and newly captured learner Act-2 continuations, plus fresh Act-2/full-three-act starts and Act-1 rehearsal. Family goals are reported separately. |

Changing-policy training totals at this snapshot:

| Actual task family | Requested | Clears | Defeats | Faults | Truncated/unstarted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Act-1 rehearsal, fresh reset | 130 | 88 | 41 | 1 | 0 / 0 |
| Act-2 goal, fresh reset | 190 | 8 | 182 | 0 | 0 / 0 |
| Act-2 goal, native continuation | 626 | 58 | 567 | 1 | 0 / 0 |
| Act-2 goal, learner continuation | 74 | 4 | 69 | 1 | 0 / 0 |
| Full-three-act goal, fresh reset | 260 | 0 | 260 | 0 | 0 / 0 |

Operational pilot acceptance is supported: actual parameter updates, frequent valid checkpoint publication, honest per-goal counts, stable reward/device settings, integrity checks and active regression guards are all functioning. The first selection check is worse than the anchor, so the audit does not support a competence-improvement claim or a relaxed guard. Continue the bounded pilot and assess further fixed development results before judging learning. End-of-pilot pause/resume, remaining-budget preservation and group shutdown still need actual execution evidence; this snapshot is not their proof.

Source/raw evidence lives under `data/long_rl_v1/campaign/journal.json`, its immutable task/manifest pairs, per-job metrics/log/checkpoints, deletion ledger, and machine progress. Authoritative current values supersede this fixed snapshot. Generated `docs/long_curriculum_live.md` is the ongoing latest-first status; human-authored summary and decision notes remain separate.

## Follow-up at 10:51 UTC

Twenty-six completed training chunks contain 709,703 retained transitions and 1,677 accepted optimizer steps. All 26 actor fingerprints changed. Aggregate throughput is 1,102 valid transitions/s including 644.18 seconds of child-process wall time. A bounded integrity check verifies 121 existing artifacts and 23 deliberate checkpoint deletions against their SHA ledger, with no unexplained missing or changed artifacts.

Round 14 improved fixed Act-2 development clears from 8/48 to 9/48 and Act-1 retention from 41/48 to 42/48, with no faults or censoring. The controller promoted it and later regressions restored that same best checkpoint. This is a small observed development improvement; repeated selection on 48 worlds does not establish statistical significance or held-out generalization. Round 24 produced a different surfaced fault set and was held rather than promoted, preserving the coherent-comparison rule.

Sampled logged value loss stays approximately .008–.013, accepted mean KL .002–.006, ratio .998–1.000, and clipping fraction .044–.126. Pre-update KL guards reject some high-KL minibatches while genuine updates continue. The metrics do not log explained variance, raw advantage distributions, or per-phase gradient norms, so these quantities cannot be inferred. Early rollbacks did restore the original transferred critic, but the round-14 promotion now preserves a checkpoint trained on the Act-2 mixture. Current evidence supports continuing the unchanged condition; no critic-only calibration or learning-rate change was executed by this audit.

Pilot-end shutdown and resumed active-budget accounting remain pending at this timestamp.

## Pilot closure at 10:52 UTC

The durable journal reaches `pilot_complete` after 27 completed training chunks: 736,386 retained transitions and 1,785 accepted optimizer steps. It records 900.0055 active seconds used out of the same authorized 36,000-second allocation, leaving about 9h45m. There is no active job or pending iteration; the final paired DEV jobs completed. Best/current remain the round-14 checkpoint. A host process inspection finds no controller or final job PIDs and no surviving members of any of the 49 journal-recorded child process groups. This supplies actual bounded-pilot shutdown evidence. Main-resume accounting continuity still awaits the parent's authorized launch; this audit itself launches no processes.
