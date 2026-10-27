# Four short trials to find a stable learning signal

**Status:** running 27 October 2026. **Verdict:** try interventions that address
update size, policy drift, noisy reasoning credit and growing reasoning length.

## Question
Can teacher-feedback RL improve decisions on broader states without teaching
repetitive output? Sid authorized about four hours on this MacBook and asked
for signs-of-life experiments with minimal setup, not seed repeats or elaborate gates.

## Design
Start every screen from the original passing stage-1 adapter, with a fresh
optimizer, the same 512 training states, 4 states × 8 sampled responses per
update, shuffled menus, teacher-probability rewards, invalid reward −1, and
learning rate 0.00001. Use a common 1,024-token cap; every response in the prior
160-state starting evaluation finished below this cap. This changes the protocol
from the failed 4,096-token run, so that historical run is context, not a matched control.

| Trial | Reasoning | Reward-gradient tokens | KL penalty |
|---|---|---|---|
| Smaller updates | Medium | Whole completion | None |
| Anchor the starting policy | Medium | Whole completion | 0.05, frozen passing adapter |
| Credit the answer | Medium | Reasoning closer, answer and end token | Same anchor |
| Keep reasoning bounded | Medium; length cost after 256 tokens | Whole completion | Same anchor |

Sid explicitly rejected low-effort mode: all trials use medium effort. The
length-cost arm subtracts 0.1 reward per 256 reasoning tokens beyond the first
256, for valid answers only; invalid answers retain reward −1. Evaluation and
selection still use unpenalized teacher probabilities. At this token cap the
maximum valid-answer cost is below 0.3, preserving the selected states' minimum
teacher gap between the best and second-best action.

The answer-credit arm still generates its own reasoning. Capped responses with
no closer have no answer-token gradient; their KL penalty still acts, and their
−1 reward contributes to the group baseline. No teacher-forced reasoning or SFT.
The frozen adapter anchor avoids pulling back toward an untrained foundation.
Log policy loss, KL loss and sampled per-token KL to inspect their actual scale.

Allow roughly 25 minutes per screen plus its final evaluation. Use 16 fixed
training-panel states and 48 validation states (balanced by phase), evaluated
at the start, every 12 updates,
and at stopping. Both phase results, validity, cap rate, length and position
bias matter. Stop an arm promptly if the latest 64 sampled answers are below
80% valid. This is a practical screen, not a statistical acceptance test.

Use the remaining allocation to continue the usable arm with highest held-out
teacher probability; report this selection as exploratory. The continuation
resets optimizer moments, uses a new training order and retains the original
KL anchor. Reserve time to evaluate its starting adapter and selected candidate
on the full 64 + 96 panels under the same cap and effort. No significance gate.
Stop launching work at the session deadline; retain every outcome, including
flat or worse arms. Everything runs locally, detached and on mains power.

## Results
All four screens stayed usable. Three showed tentative held-out gains; the
length penalty was flat. The smaller-update candidate was selected by its
narrow score lead, but its continuation stopped as reasoning length
and capped answers increase. The full-panel comparison is pending. All results
are exploratory.

| Smaller-update evaluation | Training matches / 16 | Held-out matches / 48 | Held-out teacher probability | Held-out valid |
|---|---:|---:|---:|---:|
| Start | 10 | 22 | 0.424 | 48/48 |
| Update 12 | 9 | 26 | 0.471 | 48/48 |
| Update 15, final | 8 | 27 | 0.478 | 47/48 |

Paired agreement changes (percentage points, 95% world-cluster intervals):
update 12, held out +8.3 [−5.9, +21.7] and training −6.2 [−25.0, +12.5];
update 15, held out +10.4 [−2.2, +23.9] and training −12.5 [−37.5, +12.5].
The held-out panel has 44 worlds; the training panel has 16. Final teacher
probability changes are +0.054 [−0.051, +0.160] held out and −0.132
[−0.319, +0.018] on training. The small-panel evidence is inconclusive;
its held-out direction is promising enough to keep this candidate in the screen.
Complete interval records are in the run's `small_step_comparisons.json`.

The KL-anchor trial finished after 11 updates: training matches 10 → 7/16
(−18.8 points, 95% interval −43.8 to +6.2); held-out matches 22 → 26/48
(+8.3, −6.2 to +23.9), all 48 valid. Teacher probability changed by −0.105
[−0.316, +0.104] on training and +0.054 [−0.063, +0.173] held out.
Its held-out score of 0.4779 is essentially the smaller-update arm's 0.4784:
paired difference −0.0004 [−0.107, +0.114]; agreement differs by −2.1 points
[−15.2, +12.2]. Training-panel KL-minus-small-step differences are −6.2
[−25.0, +12.5] points and +0.026 [−0.081, +0.175] teacher probability.
These comparisons remain inconclusive. KL used 11 updates versus 15 in the same
time allocation because reference scoring and regularization add work.
All saved evaluations and contrasts are in run `trial_comparisons.json`.

The answer-credit trial finished after 12 updates: training matches 10 → 8/16
(−12.5 points, 95% world-cluster interval −31.2 to 0.0); held-out matches
22 → 25/48 (+6.2, −4.3 to +16.7), with all 64 answers valid. Teacher probability
changed by −0.087 [−0.236, 0.000] on training and +0.051 [−0.041, +0.140]
held out. Median held-out reasoning length was 183 tokens, versus 188 at the start.
Against the smaller-update trial, its held-out agreement difference was −4.2
[−16.7, +6.7] points and teacher probability difference −0.003
[−0.102, +0.093]. Training differences were 0.0 [−18.8, +18.8] points and
+0.045 [−0.056, +0.190] probability. Restricting reward credit to the answer
has not shown an advantage in this short screen; the comparison is inconclusive.

The length-penalty trial finished after 12 updates: training matches 10 → 8/16
(−12.5 points, 95% world-cluster interval −31.2 to 0.0); held-out matches
22 → 21/48 (−2.1, −16.3 to +11.1), with all 64 answers valid. Teacher probability
changed by −0.084 [−0.221, 0.000] on training and −0.001 [−0.108, +0.105]
held out. Median held-out reasoning length rose from 188 to 199.5 tokens, so
this short screen showed neither shorter reasoning nor a learning benefit.
Against smaller updates, held-out differences were −12.5 [−26.5, +1.9] agreement
points and −0.055 [−0.162, +0.054] teacher probability; training differences
were 0.0 [−18.8, +18.8] points and +0.047 [−0.053, +0.193] probability.
These differences remain inconclusive on the small panels.

At continuation update 12 (27 total updates), training matches recovered from
8 to 11/16 and held-out matches stayed at 27/48; all 64 answers were valid.
Teacher probabilities rose from 0.345 to 0.539 on training and 0.478 to 0.504
held out. Against the continuation start, agreement changes were +18.8
[−6.2, +43.8] points on training and 0.0 [−11.8, +11.8] held out; probability
changes were +0.195 [+0.008, +0.405] and +0.025 [−0.075, +0.130], respectively.
Against the original adapter, training agreement was 10 → 11/16 (+6.2
[−18.8, +31.2] points) and held-out agreement 22 → 27/48 (+10.4 [0.0, +22.4]);
probability changes were +0.063 [−0.144, +0.271] and +0.080 [−0.018, +0.182].
These are descriptive 95% world-cluster intervals on repeatedly inspected
panels, not confirmation. Held-out median reasoning rose from 172 to 242 tokens
during this first continuation interval.

At continuation update 24 (39 total updates), training matches were 9/16,
with 15/16 valid; held-out matches were 28/48, with 44/48 valid. Teacher
probabilities were 0.454 and 0.509. Against the original adapter, agreement
changes were −6.2 [−25.0, +12.5] points on training and +12.5 [0.0, +25.0]
held out; probability changes were −0.023 [−0.205, +0.162] and +0.085
[−0.014, +0.188]. Against the continuation start, agreement changes were +6.2
[−18.8, +31.2] and +2.1 [−10.9, +15.2] points; probability changes were +0.109
[−0.057, +0.301] and +0.031 [−0.089, +0.154]. Held-out median reasoning reached
360 tokens. The five capped traces include copied menus/map text and a repetitive
"Clash+" loop. This is a useful learning signal with worsening reliability;
the continuation's STOP file was requested after 25 completed updates. It
finished the in-flight update, stopped at 26 and began its final evaluation.

## Next
Finish saving and evaluating the stopped continuation, then let the existing
launcher compare the original and selected adapters on the full panels.
Further training should target reasoning growth while retaining medium effort;
the short length-cost screen has not established how to do that successfully.

Code: `experiments/2026-10-27-learning-trials/` and `src/sts_llm/mlx_training.py`.
Data: `learning_trials_20261027` and `learning_panels_20261027` (local).
