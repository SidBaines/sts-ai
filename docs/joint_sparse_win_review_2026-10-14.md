# Joint sparse-win pilot review — 14 October 2026

Recorded at 2026-10-14T17:30:19.241519+01:00. This is a diagnostic review; no further training was started and original model/checkpoint/buffer receipts were not modified. The pilot stopped at 16:53 London after 1h54 active time, iteration 20, reason `persistent_update_rejection`. Sixteen updates were accepted, four rejected (14, 16, 19, 20); 1,045 actor optimizer steps were retained and 223 attempted steps discarded. Final actor weights are those accepted at iteration 18. Critic fitting continued on valid batches through iteration 20. No champion promotion occurred; the starting actor pair remains champion.

## Learning evidence

All values below are exact greedy full-game outcomes on the same 1,024-world DEV cohort. The original full-control phase masks and one observation per evaluation worker were preserved. Head swaps use the final retained actors, not newly trained heads. Simulator faults and censored worlds were zero in both head-swap conditions.

| Combat head | OOC head | Wins / 1,024 | Paired p versus original pair | Act 1 / Act 2 clears |
|---|---|---:|---:|---:|
| Original | Original | 58 | — | 959 / 349 |
| New | Original | 48 | 0.1433 | 956 / 344 |
| Original | New | 59 | 1.0 | 966 / 349 |
| New | New | 55 | 0.7838 | 963 / 346 |

Neither head has demonstrated improved full-game play. The combat-only swap moves in a concerning direction (-10 wins), with insufficient evidence here to establish regression statistically. The OOC-only swap is essentially flat (+1 win). The two new heads interact: separate swaps do not support adding their effects arithmetically. The best intermediate joint checkpoint (iteration 12) had 65 wins versus 58, nominal paired p=0.435, so it was not promoted. Joint greedy evaluations at iterations 2–18 were 57, 53, 58, 52, 52, 65, 53, 54 and 55. Repeated DEV screening does not provide independent confirmation.

Stochastic TRAIN totals were 708 wins / 15,360 games (4.61%); every batch had 25–51 wins, exceeding the minimum-success guard. First-five versus last-five batches had 166/3,840 and 185/3,840 wins. This weak, noisy training change does not establish deployment improvement. All 15,360 training games and all ten original 1,024-game greedy evaluations completed without simulator faults or censoring.

## Optimisation health and weaknesses

The retained policies have small original-policy drift: mean KL combat 0.000842 / OOC 0.000950, and changed greedy choices 8/384 combat and 2/384 OOC on the rehearsal corpus. Per-act means/p95 stay well below the declared hard budgets. Sampling entropy remains about 0.25 combat / 0.10 OOC; gradients and likelihood reconstruction remain finite in accepted updates. There is no evidence here of entropy collapse or wholesale forgetting. Phase/act anchor coverage is 128 choices per bucket. These fixed-state diagnostics are not a guarantee of behaviour on every reached state.

Both independent critics have learned little useful state-dependent value. Predictions remain around 3.1% in each act; world-held-out explained variance is approximately zero (about 0.0001–0.0004 overall). At iteration 18, Act 3 held-state binary targets averaged 22.9% combat and 25.4% OOC, while mean predictions were only 3.14% and 3.11%. They provide little useful variance reduction. The gamma/lambda=1 outcome targets remain actual full-game returns; poor critics do not bootstrap incorrect terminal labels into this run. One full critic optimizer step per batch was insufficient in this pilot; critic fitting/calibration deserves separate diagnosis before further training.

Peak learner RSS was 9.44 GiB, with about 11.3 GiB free disk at stop. No resource guard fired. The environment has no new fault to patch from this pilot.

## Reproduced numerical failure

The exact saved iteration-20 optimizer buffer, pre-update checkpoint, anchor corpus, optimizer states and RNG reproduced `Nonfinite actor loss` in OOC. An audit localized it to the original-policy rehearsal KL, not importance ratios or simulator data. On one legal action, the reference probability was `1.401298464324817e-45`, while the current probability rounded to zero in float32. Their normalized log probabilities were finite: -103.29425 and -104.02094. The categorical KL routine treated this rounded-zero support as infinite, rejecting the entire joint transaction. A direct log-space KL calculation was finite, with batch mean 0.00119261; the affected row had a negligible rounding residual. The source of the first such failure at iteration 14 was not separately replayed, so do not claim all four causes are individually proven.

The replay returned exactly to the starting actor hash after rejection (`actor_restored=true`). This validates fault containment and the usefulness of saving exact optimizer inputs. It does not validate the rejected numerical implementation. The immediate training-code repair is a stable masked KL computation with underflow/padding/gradient regression tests and an exact saved-buffer recovery check. This is an RL-adapter numerical issue, not a native simulator fix. No source patch or restarted pilot was made during this diagnostic-only review.

## Records and next decision

The existing [protocol](joint_sparse_win_plan.md) and [launch receipt](joint_sparse_win_launch_2026-10-14.md) preserve the condition. New local diagnostics are in `data/rl_runs/joint_sparse_numeric_review_20261014/results.json` and `data/rl_runs/joint_sparse_heads_review_20261014/results.json`, with native/source/input records, immutable copies of the diagnostic script, and full ablation action-prefix receipts. The dashboard still serves the stopped run at http://127.0.0.1:8793 .

Before continuing, repair and validate KL arithmetic, improve critic fitting/calibration without changing binary outcome targets, and assess the combat regression direction. Keep the original champion until improvement is independently established. Any code repair/continuation needs explicit recorded source migration and preservation of the original used-time budget; do not bypass resume identity checks or overwrite this pilot.
