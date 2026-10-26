# Stage 1: can teacher feedback teach the model 32 decisive states?

**Status:** running since 26 October 2026. **Verdict:** not yet known.

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

**States** (`stage1_states_20261025`, from `select_states.py`):
- Source: saved champion games from the non-LLM corpus, with the env-v0.1.5.1
  potion-line fault excluded.
- Selection: kept only if the teacher's top action leads the next by at least
  0.3 probability. Mean margin 0.66.
- Training set: 16 out-of-combat states (3 card rewards, 3 events, 3 map
  choices, 2 rest sites, 2 shops, 3 other) and 16 combat states (6 boss, 5
  elite, 5 hallway). All come from distinct training-split worlds.
- Probe: the same shape from validation-split worlds. It is reported but
  decides nothing.

**Teacher:** the 25 October champion (`champion_4058cbf0e271`, 29.4% wins).
Its combat network is the same as the previous champion's.

**Model and prompt:**
- K2-Horizon-0.9B in MLX (BF16), medium effort, a fresh LoRA adapter (rank 16,
  alpha 32, all projections).
- Prompt version 2: the unchanged state text, then a letter-labelled menu in a
  fresh random order at every presentation. The answer is
  `{"action": "<label>"}`.

**Learning:**
- 4 states per iteration, with 8 samples each at temperature 0.6, top-p 1 and
  a 4,096-token cap.
- Rewards: the teacher's probability of the chosen action; invalid or capped
  answers get -1.
- Group-mean advantages, with no standard-deviation scaling.
- One AdamW step per iteration (REINFORCE with a baseline).
- Each response's token log-probabilities are summed and divided by 2,048.
- Gradient-norm clip 1, no KL penalty.
- 48 iterations, which is 6 passes over every state (8 samples per state per
  pass).
- Learning rates 0.00001, 0.00003 and 0.0001, one run each, run one after
  another on the laptop. The screen goes no higher than 0.0001 because the
  timing smoke seemed to drift fast at that rate. That reading was mostly
  wrong (see Log): each iteration draws the same states in every run, and the
  first ones happened to differ.

**Measurements:**
- From the training samples of each pass (256 decisions): mean teacher
  probability, top-choice agreement, validity, reasoning length and
  first-displayed-option rate. All are reported per phase.
- A greedy evaluation of one fixed menu order per state, on the training
  states and the probe, at iterations 0, 24 and 48.

**Decision rule, fixed before any run.** A learning rate passes if all of
these hold:
- Mean teacher probability in its sixth pass is at least 0.15 above its first
  pass.
- In the greedy evaluation, at least 8 more of the 32 training states match
  the teacher's top choice at iteration 48 than at iteration 0.
- Validity stays at or above 95%.

The stage passes if any learning rate passes; report the best one per phase.
If none passes, investigate before stage 2 rather than adding states.

**Also reported, deciding nothing:**
- Reasoning length per pass: median, 95th percentile and share empty.
- First-displayed-option rate. Under random menu orders, "always A" cannot
  earn reward, so a rising rate signals drift.
- Per pass: zero-variance group share, group reward range and gradient norm.
- Probe results.

**Budget:** measured in the timing smoke at this shape:
- 81–118 s per iteration (about 100 s), so 48 iterations take about 80 minutes;
- 2–3 minutes per evaluation of all 64 states;
- about 90 minutes per learning rate, and about 4.5 hours for the screen.

The screen runs only on mains power: `launch_screen.py` holds `caffeinate`
while plugged in and pauses the job on battery. No GPU rental is needed.

**Approval:** Sid approved this design on 26 October 2026.

## Results

Not run. The smoke runs only check the mechanics: `stage1_smoke_functional_20261025`
and `stage1_smoke_timing_20261025`.

## Conclusion

Not run.

## Next

If stage 1 passes: stage 2, training on a larger set of decisive states from
training worlds, with held-out top-choice agreement on decisive states as the
gate.

## Log

- 25 October: Sid chose to train combat and out-of-combat decisions together
  and to use the latest champion as teacher.
- 25 October: the untrained model answered validly in the new label format
  (64/64 at the first evaluation). Over three steps at learning rate 0.0001,
  the timing smoke's median reasoning fell from 374 to 113 tokens and its
  first-displayed-option rate rose from 0.06 to 0.44, so the screen moved down
  by a factor of 3.
- 25 October: the Mac slept for about 20 minutes during the smoke. Iteration
  timers exclude sleep; wall time did not. Use the launcher for the screen.

- 26 October: Sid approved the design; the screen started as run
  `stage1_screen_20261026`, learning rates in the order 0.00001, 0.00003, 0.0001.
- 26 October: the smoke's drift was mostly batch composition. With the same
  seed, every run samples the same states at each iteration. At 0.00001, ten
  times lower, iterations 2 and 3 gave first-displayed-option rates of 0.37 and
  0.34 against the smoke's 0.38 and 0.44. Medians were 246 and 212 tokens
  against 210 and 113. Iteration 1's batch is simply unusual (rate 0.06 in
  both). Runs can therefore be compared iteration by iteration on the same
  states.

Code: `experiments/2026-10-25-overfit-test/` and `src/sts_llm/mlx_training.py`.
Data: `stage1_states_20261025` (local).
