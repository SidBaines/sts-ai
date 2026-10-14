# KL repair and critic diagnosis — 15 October 2026

The KL calculation is repaired and tested. The critic investigation supports underfitting from too few optimizer steps as the main explanation of the near-constant values. The original joint RL pilot remains stopped at iteration 20 with 6,844.49 active seconds used; no continuation, changed critic training recipe, promoted actor, altered reward or native environment patch was applied. Diagnostic fits are separate local artifacts, not deployment checkpoints.

## KL repair and acceptance

`sts_policy.losses.masked_kl_divergence` computes masked KL from normalized log probabilities, avoiding categorical support checks on rounded-to-zero probabilities. CPU/CUDA inputs use float64 arithmetic, with a float32 MPS path because float64 is unsupported there. Only CPU is validated in this work. Padded entries are cleared before subtraction; empty menus and nonfinite legal logits remain errors. Tiny negative roundoff is clamped to zero. Both rehearsal loss and original-policy drift measurements in `joint_sparse.py` use the helper. Future nonfinite loss reports identify the microbatch, loss components and rehearsal seeds.

Regression tests cover the actual -103.29425/-104.02094 log-probability underflow boundary, padded NaN/inf inputs with zero padding gradients, empty/nonfinite legal menus, identical and saturated policies, and agreement with ordinary double-precision KL values and gradients. No guard, coefficient or reward was weakened to make a replay pass.

Exact offline replays loaded each original pre-update checkpoint, encoded buffer, original rehearsal corpus, optimizer state and RNG. Historical categorical KL still reproduces the iteration-20 rejection, with actor rollback. Repaired iterations 14, 16, 19 and 20 all complete and satisfy the original trust guards, retaining 65, 65, 66 and 64 diagnostic optimizer steps respectively; both heads change and legacy critic tensors remain unchanged. No repaired actor is installed in the stopped run. The original `latest.pt` identity is unchanged.

The final whole repository suite passes **525 tests**, with three documented optional/generated-fixture skips, and the required isolated random native smoke passes. The simulator binary is the existing pinned canonical candidate, SHA `936734a2289e20971abdbe7ae2f703d2d82f4e357a4e2b8590817d9a96f81a04`; no rebuild or core observation/trace changes were made. Resume identity checks remain strict: this source repair is not permission to bypass them or silently resume old artifacts under new code.

## What the critic audit established

All 20 saved batches were audited: **245,736 selected states**, **15,360 distinct worlds**, **708 successful worlds**, and 369 varying public input features. There were no phase mismatches, nonfinite features, nonbinary labels or conflicting labels within a world. Encoded act values match public act/3 within 2e-8. Source-world weighting and whole-world holdouts have dedicated regression tests. Replaying the historical critic updates reproduces the saved iteration-20 critic weights **bit for bit**, maximum parameter difference zero.

Gradients reach the critic parameters. Zero initialization of the output weights deliberately blocks trunk gradients on the first step; the next-step regression and historical gradient audit show finite, nonzero trunk gradients thereafter. Neither a detached graph, frozen critic, bad phase routing nor broken label plumbing explains the observed values. A separate 64-state-per-phase balanced fit reaches 100% training accuracy after 500 steps. That memorization check is a plumbing/capacity test, not generalisation evidence.

There are 241 identical-input groups with different eventual labels (1,454 states, about 0.59% of the selected corpus). Future stochastic outcomes and omitted state can make equal public encodings have different binary returns. This limits perfect prediction; the held-world results below still show useful learnable signal.

## Controlled fitting comparisons

The initial critic weights, public features, binary labels, world weighting and world split are fixed. There are **12,265 fitting worlds and 3,095 held-out worlds**, with no world shared across the split. No actor model or new rollout is used. The historical recipe accumulates gradients over all selected states, then takes **one AdamW step per rollout batch at LR 3e-4**: only 20 steps per head across the whole pilot. This was too little fitting; values stayed close to the 3% initial prior. Combat had some ranking signal already (AUC 0.694), but the predictions had very little amplitude; OOC ranking was weak (AUC 0.563).

Each pair below is combat / OOC on held worlds, weighted equally per source world within phase. Lower Brier and higher explained variance/AUC are better. Held metrics use the selected critic-state distribution, not the entire actor transition distribution.

| Condition | Explained variance | Brier | AUC |
|---|---:|---:|---:|
| Historical 20 steps | 0.00044 / 0.00007 | 0.04427 / 0.04429 | 0.694 / 0.563 |
| Act-only baseline | 0.03239 / 0.02473 | 0.04264 / 0.04298 | 0.645 / 0.624 |
| fullbatch_200_steps_same_lr | 0.05858 / 0.04051 | 0.04149 / 0.04229 | 0.750 / 0.707 |
| fullbatch_20_steps_10x_lr | 0.03972 / 0.01994 | 0.04236 / 0.04319 | 0.711 / 0.668 |
| minibatch256_one_pass_same_lr | 0.06266 / 0.04133 | 0.04131 / 0.04227 | 0.756 / 0.707 |

Keeping LR 3e-4 and increasing the original full-batch schedule to 200 steps improves calibration and discrimination. A single pass with real 256-state optimizer minibatches is at least as effective here: 768 optimizer calls in total, about 384 per head, using the same number of example exposures as the historical one-pass fit. The 200-step full-batch condition reuses the data for ten passes. Merely increasing LR tenfold for 20 full-batch steps helps less, especially in OOC Act 3.

In the minibatch condition, Act 3 held-state mean predictions become **16.39% combat / 15.95% OOC**, versus observed **16.75% / 17.33%**. This is materially closer than the original values around 3.1%. Cached-feature fitting takes about **0.59 seconds for one minibatch pass**, or **2.54 seconds for 200 full-batch steps**, on the recorded two-thread CPU configuration. These timings exclude loading/decompressing and auditing the historical buffers; they are not complete rollout/training costs.

Residual weakness remains: explained variance is only about 0.063/0.041, even after improved fitting. The comparison uses one fixed diagnostic split of historical TRAIN worlds and a few predetermined fitting choices; it is not an untouched final test or proof of improved RL/game wins. Inputs are partially observable and full-game targets remain noisy. The broader combat regression direction from the pilot is not resolved by this critic diagnosis.

## Proposal for discussion, not applied

Use cached public critic features and make a true optimizer step per 256-state critic minibatch, preserving source-world weighting and whole-world validation, initially keeping LR 3e-4, the existing architecture, independent actor/critic parameters and binary outcome targets. Fit before relying on the baseline, then monitor fitting versus held loss, calibration and explained variance by phase/act, plus ranking, update count and prediction spread. Set a bounded fitting budget with validation-based stopping so weak fitting and overfitting are both visible. This is a proposed production change, not a claim that one pass is universally optimal.

If continuation is approved, record an explicit source/config migration, preserve the original actor/reference/rehearsal identities and used-time budget, and confirm the first real on-policy critic fit before sustained actor updates. Do not silently replace the original champion or start another allocation.

## Local records

- `data/rl_runs/joint_kl_repair_replay_20261014/results.json`: historical failure control and four repaired replays.
- `data/rl_runs/joint_kl_critic_final_checks_20261015/acceptance.json`: final native/source/runtime gate and whole-suite/smoke results.
- `data/rl_runs/joint_critic_diagnosis_repaired_20261015/results.json`: complete data audit, gradient checks, fitting curves, held metrics and memorization results.
- That diagnostic folder also retains the exact public feature cache and each offline critic condition checkpoint. They remain local.
- The initial critic diagnostic completed its audit and 200-step fit, then failed to serialize a NumPy scalar in its JSON report. Its records and completed checkpoint remain under `joint_critic_diagnosis_20261014`; the repaired attempt converts the scalar and persists audit results before further fits. A diagnostic metric test checks JSON serialization and known weighted scores.

The fixed KL code, diagnostic scripts, tests and this report are committed on the research branch. Original pilot receipts, source snapshots, models and buffers remain unchanged. No joint RL run was resumed.
