# Joint sparse-win pilot launch — 14 October 2026

The three-active-hour pilot is running on `codex/joint-sparse-win-rl` from source commit `50b24e1`. This is an interim launch receipt, not a completed experiment or an improved deployment model. [Protocol and failure protections](joint_sparse_win_plan.md), [fixed configuration](../configs/joint_sparse_win_v1.json).

The original round-2 update20 actor is the parent. Combat and OOC both receive complete-game binary returns (1 victory, 0 defeat), gamma/lambda 1, separate actor optimizers and independent public win critics. No native search or progress/HP rewards enter learning. Conservative exploration, fixed original-policy rehearsal, likelihood parity, finite simulator operations, whole-world fault quarantine and trust/retention/resource guards remain active.

Local dashboard: http://127.0.0.1:8793 . The active-hour budget started about 14:59 London; nominal stop is around 17:59 London, allowing the last complete stage to finish. Battery/sleep gaps extend wall time. The trainer is PID 19975; dashboard PID 19972; awake assertion PID 20116 is tied to the trainer. Do not kill processes by command-pattern matching or modify live training source.

## Verified progress at approximately 15:36 London

| Measurement | Value |
|---|---:|
| Initial greedy DEV wins | 58 / 1,024 |
| Completed training batches | 6 × 768 games |
| Training simulator faults/cuts | 0 / 0 |
| Accepted updates / rejected updates | 6 / 0 |
| Retained actor optimizer steps | 392 |
| Greedy DEV wins after updates 2 / 4 / 6 | 57 / 53 / 58 (each / 1,024) |
| Act 1 clears, baseline / update 6 | 959 / 956 |
| Peak learner RSS | 9.44 GiB |
| Approximate run artifacts so far | 1.5 GiB |

Both actor hashes change at every retained update; old critic tensor hashes remain unchanged. The latest paired full-win comparison has 24 original-only and 24 learner-only wins, nominal p=1.0. There is no demonstrated greedy win improvement. Stochastic batch wins were 34, 25, 33, 36, 38 and 51; the upward training curve is not evidence of deployment gain. No champion promotion occurred. Collection takes roughly 150 seconds per 768-game batch, with actor/critic updates about 66–68 seconds; fixed greedy DEV1024 takes about three minutes.

## Recovery and inspection

Run artifacts remain local in `data/rl_runs/joint_sparse_win_20261014`: `status.json`, `budget.json`, `log.jsonl`, `config.json`, full `latest.pt`/`champion.pt`, immutable pre-update/evaluated snapshots, compressed exact optimizer buffers, complete rollout action-prefix receipts, request/task ledgers and source/native/fault records. Artifact filenames include the launch attempt identity so a recovery attempt cannot overwrite earlier receipts. Rolling progress receipts and latest/champion pointers are explicitly mutable.

The native binary SHA is `936734a2289e20971abdbe7ae2f703d2d82f4e357a4e2b8590817d9a96f81a04`; parent actor SHA is `3d135d9bd53b92b5a5eb3dcc8bf26458140e282ee8ea63d9aaaf6ca3c2f222bb`. The source archive SHA is `c0342e607a474abbb29cf85bfd9a41eecebb0591a153d90e3a59ba15616df75f`. Source/native/runtime/rehearsal/parent identities and adaptive guard/RNG/optimizer state are checked on resume. The episode diagnostic version-string inconsistency is documented in the protocol; actual hashes govern this condition.

Full regression and native smoke passed, alongside focused reward/quarantine/masking/gradient-separation/rollback/identity tests. Fresh-process recovery matched both heads, critics, phase optimizer tensors and local sampler on the real 13-win preflight. The accelerator screen aborted and CPU was selected; no production MPS run is active.

The dashboard's live metrics, phase/act critic diagnostics, zoom and episode filters were inspected in the real browser without console errors. The browser connection failed during the CSV-download verification, so that particular end-to-end check remains unconfirmed; filtered API pagination passed. The dashboard was reopened and left visible. This browser failure did not interrupt training.

Next review should use the fixed greedy win/Act1 retention curve, original drift by phase/act, sparse support, critic calibration, retained-step accounting and all simulator-fault receipts. Do not start another training condition or alter this one until the pilot has been reviewed.
