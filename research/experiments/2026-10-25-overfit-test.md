# Stage 1: can teacher feedback teach the model 32 decisive states?

**Status:** finished 26 October 2026. **Verdict:** stage 1 passes: at learning rate 0.00003 the loop taught the model the teacher's choice on 22 of 32 training states (from 10), mostly through content rather than menu position.

## Question

Can the MLX teacher-feedback loop make K2-Horizon-0.9B learn the teacher's
choice on a fixed set of 32 states where the teacher is decisive? Half are
combat states and half out-of-combat. This is a learnability test, not a
generalisation test: if the loop cannot fit states it trains on repeatedly,
something in the reward, loss, parsing or learning rate is wrong, and no
held-out result would mean anything. It is stage 1 of
[the proposed plan](../brainstorms/2026-10-25-getting-the-model-to-learn.md).
Approach: [teacher feedback](../strategies.md#teacher-feedback-rl-followed-by-full-game-outcomes).

## Design

**States** (`stage1_states_20261025`, from `select_states.py`): champion
games from the non-LLM corpus, excluding the env-v0.1.5.1 potion-line fault,
kept only where the teacher's top action leads the next by at least 0.3
(mean margin 0.66).
- Training set: 16 out-of-combat states (card rewards, events, map, rest sites,
  shops) and 16 combat states (6 boss, 5 elite, 5 hallway), from distinct
  training-split worlds.
- Probe: the same shape from validation-split worlds; it decides nothing.

**Teacher:** the 25 October champion (`champion_4058cbf0e271`, 29.4% wins).

**Model and prompt:** K2-Horizon-0.9B in MLX (BF16), medium effort, a fresh
LoRA adapter (rank 16, alpha 32, all projections). Prompt version 2: the state
text, then a letter-labelled menu in random order, answered with
`{"action": "<label>"}`. In the rerun, every sample gets its own order; the
first screen shared one per group.

**Learning:**
- Each iteration takes 4 states and 8 samples per state (temperature 0.6,
  top-p 1, 4,096-token cap).
- Reward: the teacher's probability of the chosen action, or −1 if invalid or
  capped.
- Group-mean advantages with no standard-deviation scaling. One AdamW step per
  iteration. Summed log-probabilities divided by 2,048, gradient clip 1, no KL
  penalty.
- 48 iterations, which is 6 passes over every state. Learning rates 0.00001,
  0.00003 and 0.0001, one run each.

**Measurements:** per pass, from 256 sampled answers: teacher probability,
top-choice agreement, validity, reasoning length and first-displayed-option
rate, per phase. A greedy evaluation of one fixed menu order per state, on the
training states and the probe, at iterations 0, 24 and 48.

**Decision rule, fixed before any run.** A learning rate passes if all of
these hold:
- Mean teacher probability in its sixth pass is at least 0.15 above its first
  pass.
- In the greedy evaluation, at least 8 more of the 32 training states match
  the teacher's top choice at iteration 48 than at iteration 0.
- Validity stays at or above 95%.

The stage passes if any learning rate passes; report the best one per phase.
If none passes, investigate before stage 2 rather than adding states.

**Also reported, deciding nothing:** reasoning length, the
first-displayed-option rate, group statistics and probe results.

**Budget:** about 90 minutes per learning rate on the laptop (timing smoke:
about 100 s per iteration, 2–3 minutes per evaluation), on mains power only.

**Approval:** Sid approved this design on 26 October 2026, and the rerun with an order per sample the same day.

## Results

**0.00003 passed; 0.00001 learned too slowly and 0.0001 broke the answer
format.** Rerun with an order per sample (`stage1_screen_ps_20261026`); 32
training states, 256 sampled answers per pass:

| Learning rate | Teacher probability, pass 1 → 6 | Training states matching the teacher's top choice (greedy), iteration 0 → 48 | Lowest validity in a pass | Verdict |
|---|---|---|---|---|
| 0.00001 | 0.336 → 0.433 (+0.10) | 10 → 14 (+4) | 98.8% | Fail |
| **0.00003** | **0.354 → 0.512 (+0.16)** | **10 → 22 (+12)** | **96.1%** | **Pass** |
| 0.0001 | 0.263 in pass 1; stopped after 9 iterations | not evaluated | 81.6% (pass 1) | Fail (validity) |
| Shared order, 0.00001 (first screen) | 0.346 → 0.468 (+0.12) | 10 → 17 (+7) | 98.8% | Fail; flawed design |

Training-state gains, paired by state (each from a different world), 95%
bootstrap interval: +12 at 0.00003 (13 gained, 1 lost; +6 to +18), and +4 at
0.00001 (7 gained, 3 lost; −2 to +10).

![Per-pass teacher probability, agreement when the teacher's top choice is
not shown first, and the share of answers picking A](../figures/overfit-test/passes.png)

Data: `figures/overfit-test/passes.csv`.

**Position against content.** A content-only policy matches the teacher
equally often wherever its top choice is shown, and answers A at 0.27 here.
- At 0.00003, agreement when the top choice was not shown first fell from 0.33
  to 0.25 over passes 1–3 while A answers rose to 0.65. It then recovered to
  0.61 by pass 6, with A at 0.23 (16% in the final greedy evaluation).
- At 0.00001 it rose from 0.33 to 0.42, with the A share at 0.32–0.44.
- With a shared order, agreement rose from 0.57 to 0.88 when the top choice
  was shown first, but only from 0.30 to 0.43 otherwise.

**By phase at 0.00003** (greedy evaluation, iteration 0 → 48; 16 states each):
- out of combat: 3 → 10 matching, teacher probability 0.21 → 0.59;
- combat: 7 → 12 matching, 0.40 → 0.50.

**Other results:**
- **Probe** (validation worlds; decides nothing): 15 of 32 matching at the
  start. It reached 19 at 0.00003 and 14 at 0.00001, both within the noise of
  one 32-state evaluation.
- **Format failures at 0.0001.** 47 of 288 answers were invalid in 9
  iterations: 27 had malformed endings (the reasoning closer and JSON written
  twice, or a bare letter), 16 hit the 4,096-token cap in repetition loops, and
  4 used labels not on the menu. 0.00003 showed the duplicated ending briefly
  (6 of 32 at iteration 6) and recovered by iteration 8.
- **Reasoning length** (median sampled tokens, passes 1 → 6): 271 → 201 at
  0.00001, and 180 → 196 at 0.00003, with a peak of 647 during the drift.
  The shared-order run shortened to about 100.
- **Time:** about 90 minutes per learning rate. Iterations took 37–300 s,
  depending on reasoning length.

## Conclusion

The loop can teach this model. At learning rate 0.00003, with a fresh menu
order for every sample, it more than doubled the training states on which the
greedy model picks the teacher's choice. By the end that came from reading
the options, not from position. Both phases improved, out of combat more than
combat.

Two failure modes bound the usable learning rate:
- At 0.0001, updates break the answer format faster than the −1 penalty
  repairs it.
- Without a KL anchor, the label distribution can drift, as 0.00003 did in
  passes 2–3. A content-blind lean towards one label earns the group average,
  so the reward neither pushes it nor pulls it back. This explanation is
  untested.

Sharing one menu order across a group made that drift much worse, and it was a
design flaw.

Caveats:
- One run per learning rate and one seed. 0.00003 passed after a dip, so
  whether a repeat passes, or how often runs collapse, is unknown.
- 32 states; this tests fitting, not generalisation.
- The rerun's design change (an order per sample) was made after seeing the
  first screen. The pass rule itself was fixed in advance.
- 0.0001 was stopped once its validity failure was certain. It might have
  recovered on teacher probability, but it could not pass.
- Teacher agreement measures imitation of a 29%-win policy, not game skill.

## Next

Sid chose a [longer broader continuation](2026-10-26-broader-training.md) on
26 October. Continue from the passing adapter on many more decisive states,
watch held-out choices and validity, and change course if the metrics worsen.
The suggested seed repeat, KL arm and larger-batch arm are deferred: the
current goal is signs of life, not reliability estimates. Keep the strict
answer format and inspect menu-position dependence separately from agreement.

## Log

- 25 October: Sid chose combat and out-of-combat decisions together, with the
  latest champion as teacher. The timing smoke seemed to drift fast at 0.0001,
  so the screen was lowered by a factor of 3. On 26 October that drift turned
  out to be mostly batch composition: every run draws the same states at each
  iteration.
- 26 October: Sid approved the design; first screen `stage1_screen_20261026`.
- 26 October: design flaw. All 8 samples of a state shared one menu order, so
  the reward for an action was also the reward for its label. 0.00001 failed
  (+0.12; +7), much of it positional. 0.00003 drifted to 88% A answers by
  iteration 8 and was stopped at 15. The trainer now gives each sample its
  own order.
- 26 October: Sid approved the rerun under the same rule
  (`stage1_screen_ps_20261026`). 0.0001 was stopped after 9 iterations, once
  its first-pass validity of 81.6% had already failed the rule.

Code: `experiments/2026-10-25-overfit-test/` and `src/sts_llm/mlx_training.py`
at commit f131813 (an order per sample; the first screen ran at 1052bd4).
Data: runs `stage1_screen_ps_20261026` and `stage1_screen_20261026`; states
`stage1_states_20261025` (local).
