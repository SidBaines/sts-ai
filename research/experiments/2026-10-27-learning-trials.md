# Four short trials to find a stable learning signal

**Status:** planned 27 October 2026. **Verdict:** try interventions that address
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
selection still use unpenalized teacher probabilities.

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
Pending. Existing tiny-model checks plus two checks for the new mask/reference
are the only setup tests. The initial adapter, data and previous runs are preserved.

## Next
Continue what learns, or use the observed failure to choose a different intervention.

Code: `experiments/2026-10-27-learning-trials/` and `src/sts_llm/mlx_training.py`.
Data: `learning_trials_20261027` and `learning_panels_20261027` (local).
