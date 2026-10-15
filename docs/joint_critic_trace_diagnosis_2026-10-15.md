# Joint critic trace diagnosis — 15 October 2026

## Finding

The update10 OOC warning reflects a weak recent validation cohort, plus a modest increase in predicted probabilities. It does **not** establish global forgetting, a gradient/target bug or a significant deterioration relative to the warmed critic. It also does **not** establish that the OOC critic is sufficiently informative: late-act predictions remain weak on this cohort even before joint training. Updates11 and12 cleared the warning, but their small positive Act3 explained variance should not be described as strong recovery.

Training continues under the existing two-active-hour allocation. This investigation uses saved inputs/checkpoints only; no optimizer updates, new simulator games, actor settings, training source, splits, rewards or budget changed.

## Evidence and reproduction

Read ten immutable update buffers and selected the same up-to-eight evenly strided choice states per world and phase used by the learning diagnostics. Kept only the deterministic whole-world critic-validation split, which never receives critic gradients: **24,746 states from1,547 worlds**, including77 winning worlds. Evaluated the warmed, update8 and update10 critics on exactly the same states. Retained their original behavior-time predictions too. Each world/phase has total weight1; act-specific scores retain that weighting.

The reconstructed update10 rolling window (batches7–10) matches every logged phase/act mean, Brier score and explained variance to within **1.2e-9**. This verifies that the analysis examines the actual dashboard warning. Buffers/checkpoints have SHA256 identities in the machine-readable report. The analysis took82 seconds, used two Torch threads at reduced scheduling priority, and wrote separate outputs; it did not modify trainer files.

Local archive: `data/rl_runs/joint_sparse_win_v2_20261015/analysis_critic_0010/` contains `report.json`, selected feature/metadata caches, predictions, original executed script/log, whole-world score comparisons, reliability bins, trace examples and `acceptance.json`. `brier_decomposition.json` records the score decomposition below. These artifacts and model/data archives remain local.

Reproduce into a **new** output directory with the committed analysis program:

```bash
PYTHONPATH=src .venv/bin/python analysis/critic_trace_diagnosis_20261015.py \
  --run data/rl_runs/joint_sparse_win_v2_20261015 \
  --out /tmp/joint-critic-trace-reproduction
```

The program is specific to the first ten batches and the warm/update8/update10 comparison. It refuses an existing destination. It requires the original local buffer/checkpoint paths and the frozen Python environment. Its logic matches the executed archived script; the committed wrapper adds explicit input/output arguments. Whole-world bootstrap uses seed4007 and2,000 paired resamples. Results are exploratory across several inspected groups.

## Same states, different critics

Explained variance (EV) on the warning cohort, batches7–10:

| Cohort | Worlds / winning worlds | Warmed | Update8 | Update10 |
|---|---:|---:|---:|---:|
| OOC, all acts |614 /25|0.0273|0.0288|0.0225|
| OOC, Act2 |540 /25|0.0040|0.0030|−0.0028|
| OOC, Act3 |175 /25|−0.0060|0.0014|−0.0168|
| Combat, all acts |614 /25|0.0647|0.0720|0.0684|

The warmed critic already has almost no useful EV on these Act2/3 states. Updating did not create most of the weakness. In contrast, on the early batches1–4, OOC aggregate EV improves from0.0514 warmed to0.0611 atupdate10. On all ten held batches it improves from0.0493 to0.0544. Because early/late batches differ in both seeds and actor policy, these contrasts cannot identify how much is seed composition versus policy-induced distribution shift.

The earlier, independent512-world original-policy validation cohort gives another control: OOC EV0.0591 warmed →0.0640 atupdate8 →0.0681 atupdate10. Act3 EV0.0517 →0.0675 →0.0784. Those exact states support no broad forgetting claim. This cohort was never fitted and is a different policy distribution from the later training rollouts.

## Size and uncertainty of the deterioration

Paired Brier-score differences on the warning cohort, update10 minus warmed; positive means worse:

| Cohort | Difference |95% whole-world bootstrap interval|
|---|---:|---:|
| OOC, all acts |+0.000201|[−0.000256,+0.000640]|
| OOC, Act2 |+0.000312|[−0.000096,+0.000680]|
| OOC, Act3 |+0.002285|[−0.003615,+0.007762]|
| Combat, all acts |−0.000085|[−0.000862,+0.000712]|

All intervals span zero. These are resamples of entire worlds, preserving the dependence among decisions from the same game; they are not thousands of independent state observations. They quantify this diagnostic comparison, not a corrected sequential model-selection test.

There is a modest prediction-scale shift. On the same Act3 OOC states, the average prediction rises from14.38% atupdate8 to19.24% atupdate10, versus16.38% observed weighted returns. Prediction correlation is0.992; weighted ranking AUC barely changes (0.5759→0.5745). Prediction spread rises from0.0800 to0.0978. Thus rankings are largely preserved while predicted probabilities spread and rise. This is consistent with calibration/scale jitter, rather than a sudden loss of all learned ordering; the training mechanism causing the shift is not proven here.

The exact Brier increase relative to update8 is0.002915:0.002492 from the change in centered residual variance and0.000423 from squared mean calibration bias. It would therefore be inaccurate to attribute the whole change to a global mean/intercept shift. Both the scale and residual errors matter.

A monitoring caveat: beating the fixed4.6% win-rate Brier reference is a weak standard for an Act3 bucket whose conditional win rate is16.4%. A critic can beat that reference while doing worse than a constant appropriate to that bucket. The EV warning exposed this; the two metrics must be interpreted together. Keep the fixed reference for historical comparability, and consider adding an act-conditioned calibration baseline after this run, rather than treating its Brier benefit as sufficient evidence of useful late-act discrimination.

## Where the traces look weak

The weakness is not universal across OOC decisions. On batches7–10, room-conditioned EV is already negative for both the warmed and update10 critics at events, elites, rests and boss-reward contexts. Shops and treasure contexts remain positive. Selected-action semantic groups show a similar pattern: buying states improve while resting/reward states are weaker. These groups overlap, span multiple acts and sometimes contain very few winning worlds; they do **not** establish that the selected action caused a loss. The critic predicts state value before choosing an action, not action quality.

Act3 OOC weighted AUC for the warmed critic is0.562 on the warning cohort, compared with0.574 forupdate10. The cohort is difficult for both; it is not simply that the new critic ranks every state less accurately. Individual batches are particularly noisy: Act3 has only3–9 winning worlds in each of batches7–10. Batch8 looks relatively easy for both critics, while batch10 is poor for both (AUC about0.51/0.52). This argues against interpreting a rolling-window threshold crossing alone as a diagnosed implementation fault.

Saved examples make the effect concrete:

| World seed | Stored OOC state | Warmed →update10 prediction | Recorded finish |
|---|---|---:|---|
|60010060925|Act3, floor42,80HP|49.2%→58.3%|Defeat on floor50; entered boss at56HP|
|60010080511|Act3, floor39,65HP|33.4%→46.0%|Defeat on floor47|
|60010060349|Act3, floor39,26HP|31.3%→42.7%|Defeat on floor48|

These were selected as large error increases and are deliberately not representative samples. A high predicted win probability followed by one loss is not itself a bug: probability forecasts concern many possible futures. The first example retains all573 actions and action-trace SHA256 `6fa22a7806a40c38eb9fdb8027332ba82d3de8a18f09986618a7c8f04f8dc800` in its collection7 receipt. Exact menu inputs, historical predictions, action indices and complete outcomes are available for further replay. No fresh native replay was necessary for the score comparison, and none was performed in this investigation.

## Judgment and next decisions

No immediate training-setting change is justified by these results. Numerical and simulator guards remain intact; critic optimizer progress and non-flat predictions are verified. The more substantive open issue is **limited OOC discrimination on some reached states**, alongside some probability-scale instability. Existing data cannot establish whether that is mostly stochastic sparse-return noise, limited public feature representation, insufficient coverage or a policy-distribution change.

While the approved run finishes, continue monitoring rolling diagnostics, the same-state controls and paired greedy wins. The warning cleared atupdates11/12, but Act3 OOC EV at12 was only0.0059, so continued low signal deserves review. Do not label the critic healthy simply because a warning cleared.

After the run, discuss whether to strengthen act-conditioned calibration diagnostics and add a fixed diagnostic panel drawn from the current policy distribution. If weak discrimination persists with adequate independent positive-world support, investigate feature coverage and calibration before assuming additional optimizer steps or a higher learning rate will solve it. Any replay-mixing, smoothing or learning-rate change should be a separately recorded condition after that discussion. No further training condition was started here.
