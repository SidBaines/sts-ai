# Continue learning on a broader set of decisions

**Status:** running 26 October 2026. **Verdict:** the small-set result is enough
to try broader learning; reliability across training seeds is not the present goal.

## Question

Does continuing the passing stage-1 adapter on many more decisive states improve
held-out choices while preserving valid answers and useful reasoning?
Approach: [teacher feedback](../strategies.md#teacher-feedback-rl-followed-by-full-game-outcomes).

## Design

Sid authorized a longer monitored laptop run on 26 October and explicitly
preferred progress over another seed repeat. This is exploratory signs-of-life
work, not a confirmation study. The proposed repeat, KL arm and larger-batch arm
are deferred. No external compute is used.

Start from the stage-1 adapter at iteration 48, learning rate 0.00003. The saved
adapter has no optimizer state, so AdamW starts fresh. Keep medium effort,
rank-16 LoRA, 4 states per step, 8 samples per state, temperature 0.6, the
4,096-token cap, per-sample shuffled menus, group-mean advantages, fixed
2,048-token loss denominator, invalid reward −1 and zero KL penalty.

Use 512 training states (256 per phase, 436 worlds), versus 32 in stage 1.
One selected state overlaps the earlier 32. Select by teacher margin at least
0.3 and spread over decision types. Labels come from the same frozen champion
and the existing local corpus. Training remains on the filtered 0.1.5.1
observations; no simulator or prompt change is mixed into this run. The known
potion-text inconsistencies are excluded by the existing selector.

The fixed held-out panel has 96 states (48 per phase, 73 worlds), with zero
world overlap with either this run or the parent adapter's training states.
These are validation worlds and may have been inspected previously, not fresh
confirmation worlds. A fixed 64-state training subset provides a cheaper
comparison. Evaluate both before training and every 32 steps with one fixed
menu order per state. Report paired world-cluster 95% intervals for changes
from this run's starting adapter, separately by phase and overall.

Budget: up to 256 optimizer iterations, 8,192 responses and two passes over the
512 states, or eight elapsed hours before the final checkpoint/evaluation.
The earlier passing run spent about 108 minutes in 48 training iterations;
continuation may be faster or slower as reasoning lengths change. Save every
eight steps. Pause on battery and preserve completed outputs when stopping.

Monitor teacher agreement and probability, validity, cap rate, reasoning-token
distributions, group diversity, A-answer share against the content-only rate,
and agreement when the teacher's top action is not A. The probe is diagnostic,
not a reward. No reasoning trace is supplied and no length reward is added.

Rules chosen before training:
- Stop and inspect if validity is below 90% in each of two consecutive
  256-response windows.
- Stop and inspect if held-out agreement is at least 10 percentage points
  below the starting adapter and teacher probability is lower, at two
  consecutive evaluations.
- Flag sustained A-answer share above 70%; inspect content agreement before
  deciding whether it is a harmful shortcut.
- Stop and inspect a process error or 20 minutes without new progress.
- Otherwise continue to the budget. A rising held-out score motivates more
  training; a flat or worsening score motivates a targeted change rather than
  seed repeats. No automatic checkpoint selection is a confirmation result.

## Results

No post-update learning result yet. The starting adapter is valid on all 160
evaluation decisions. This is a new panel, so its scores cannot be compared
directly with the earlier 32-state panel.

| Starting evaluation | Decisions | Teacher top-choice matches | Mean teacher probability |
|---|---:|---:|---:|
| Fixed training subset | 64 | 31 (48.4%) | 0.396 |
| Held-out validation | 96 | 56 (58.3%) | 0.505 |

The evaluation took 289 seconds. The first optimizer step completed in 98
seconds with 32/32 valid samples, finite nonzero gradients and 21.2 GB peak
memory. Its four-state sampled score is not comparable to the fixed panels.
No post-update evaluation has completed.

The 112-test LLM suite passed after the seed
fixes; the final 113-test suite also passed, including the training-panel subset
regression. Shell syntax and Python compilation checks passed.

## Conclusion

Stage 1 establishes that the loop can improve selected training decisions;
whether learning transfers is the next useful question.

Caveats:
- Teacher preferences are not game values, and no full games have been tested.
- Decisive teacher states are a selected, easier subset of gameplay.
- Warm-starting resets optimizer moments; this is a continuation, not an
  exact optimizer resume or a controlled comparison with training from scratch.
- Frequent validation and adaptive stopping make results exploratory.
- One fixed evaluation permutation cannot establish menu invariance.

## Next

Use the observed failure mode or held-out improvement to choose the next
intervention. Adopt environment 0.1.6 before a newly simulated corpus or game
evaluation. Test game transfer once decision learning is promising.

## Log

- 26 October: Sid rejected a seed repeat as premature for signs-of-life work
  and authorized longer monitored training. Broader states replace repetition
  on the original 32; KL and batch-size changes are deferred.
- 26 October: only 50 distinct decisive combat validation worlds were available
  in the saved corpus; use 48 states per phase rather than the initially planned
  64. All labels were preserved and reused after this selection adjustment.
- 26 October: seed LoRA initialization explicitly and decouple evaluation-menu
  randomness. Earlier results remain valid, but their initial adapters cannot
  be reconstructed from the recorded seed alone.

Code: `experiments/2026-10-26-broader-training/` and `src/sts_llm/` at commit 05476ed.
Data: runs `broader_states_selected_20261026` and `broader_training_20261026` (local).
