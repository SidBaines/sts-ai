# OOC pilots on the original native build — exploratory, quarantined

Status updated 12 October 2026. These results used native module `8dba3d574ea3d4cd4c91b7031ffb38821b3e18c03358263407df3c05329c1724`.
A subsequent audit found uninitialized `Player.cc` in the automatic-combat path.
Consequently **none of these OOC outcomes, lookahead labels, or PPO comparisons
supports a clean competence claim or a fresh-backend hyperparameter selection**.
Original metrics/checkpoints remain intact for diagnosis. The corrected-backend
transition is recorded in [nonllm_native_revision.md](nonllm_native_revision.md).

All completed evaluations classified the same 32 requested development task IDs:
world start 10000, split dev, seed 777, Act 1, horizon 1024, native search 200.
Requested task IDs matched the multiset of completed episodes plus faults; no
collector-budget censoring occurred. File digests matched completed journal
records. Faults are reported separately from game defeats.

| Condition | Act clears / 32 requested | Defeats | Faults | Mean reached floor of valid episodes | Mean HP cost of valid episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scripted matched BC / warm-start reference | 19 | 12 | 1 | 15.48 | 30.81 |
| Lookahead matched BC | 12 | 19 | 1 | 15.00 | 47.74 |
| PPO LR 1e-4, progress 0.05 | 19 | 13 | 0 | 16.16 | 32.53 |
| PPO LR 3e-4, progress 0.05 | 21 | 10 | 1 | 15.58 | 25.19 |
| PPO LR 1e-3, progress 0.05 | 20 | 12 | 0 | 16.09 | 29.84 |
| PPO LR 1e-4, sparse | 17 | 14 | 1 | 15.26 | 35.45 |
| PPO LR 1e-4, progress 0.02 | 15 | 17 | 0 | 15.34 | 41.69 |

The original campaign selected LR 3e-4/progress 0.05 on this development panel.
That selection is now exploratory and must be rerun under the corrected backend.
Scripted/reference and sparse faulted on world 10121; lookahead on 10297; LR 3e-4
on 10177. On the 30 worlds valid for both reference and LR 3e-4, clears were
18 versus 20 and paired mean HP cost changed by -5.73. LR 1e-3 had 19 versus 19
clears on the 31 mutually valid worlds: its additional requested-world clear was
coverage of the reference's fault world. These are small development panels,
not significance claims, and the newly found UB invalidates causal conclusions.

## Executed training and throughput

Every PPO pilot used the same checkpoint and 90-second learner budget, 64 fixed
train worlds from 22000, seed 11, horizon 512, gamma 1, clip 0.1, entropy 0.01,
HP terminal weight zero, native search 200, CPU inference and MPS updates.

| Condition | Elapsed seconds | Valid transitions | Optimizer updates | Native faults | Collector seconds | Update seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR 1e-4, progress 0.05 | 90.03 | 48,665 | 103 | 4 | 73.30 | 15.58 |
| LR 3e-4, progress 0.05 | 90.49 | 53,577 | 48 | 4 | 75.92 | 13.36 |
| LR 1e-3, progress 0.05 | 90.40 | 78,585 | 40 | 2 | 71.46 | 17.22 |
| LR 1e-4, sparse | 90.75 | 40,018 | 77 | 5 | 77.46 | 12.39 |
| LR 1e-4, progress 0.02 | 90.08 | 81,635 | 177 | 1 | 62.72 | 25.72 |

Different transition/update counts are consequences of equal wall-clock budgets,
policy-dependent simulator timeouts, and PPO KL early stopping. They are not
matched-sample comparisons. Training faults included repeated world 22049
(three times in the first pilot), plus 22081; other conditions faulted on 22073,
22042, 22019, and 22074. All affected episode transitions and corrective labels
were excluded from learning.

Before instrumentation, failed RPC waits were absent from `reset_wait_seconds`
and `step_wait_seconds`. In the first pilot, collector time was 73.30 seconds,
but actor/reset/successful-step timings summed to 32.81 seconds: the approximately
40.49-second gap matches four ten-second timeouts. Sparse's corresponding gap
was approximately 50.4 seconds for five timeouts. New collector metrics explicitly
report `fault_wait_seconds`, fault stage, and wait duration; successful-action
trace digests support matched timeout comparisons without changing policy features.

Actual model lineage leads to `bc_large_none_lr0.0001.pt` (large/no-context).
Warm-start CLI `config.size=small` and `context=mean` are unused constructor
defaults and must not be interpreted as the effective architecture. Checkpoint
`model_config` is authoritative. No smaller-architecture comparison was run here.
The evidence first supported investigating timeout/backend correctness, not
reducing model size: fault waits dominated the initial pilot's lost throughput.
After the initialization defect, correction and fresh matched evaluations take
priority over any timeout or architecture recommendation.
