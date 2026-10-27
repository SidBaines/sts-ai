# Learning full-game play with a reasoning language model

Four local learning trials found tentative combat learning, but no reliable
overall improvement: the selected adapter matched 61/96 held-out teacher choices
versus 56/96 initially, while valid answers fell from 96/96 to 90/96.
The paired agreement change was +5.2 points [−4.1, +14.9], 95% world-cluster
interval over 73 worlds (Exploratory). [Complete results](experiments/2026-10-27-learning-trials.md).

The goal is an LLM making combat and out-of-combat decisions for Ironclad,
Ascension 0, Acts 1–3, while studying its reasoning during RL. No SFT or
teacher-forced reasoning traces are used.

## Current best

Retain the original stage-1 adapter as the working reference
(`stage1_screen_ps_20261026/lr_3e-5`, iteration 48). It learned 22/32 trained
teacher choices from 10/32 and answers all 96 broader held-out states validly.
The newer candidate has more teacher matches but worse reliability; it is saved
for analysis and has not replaced the working adapter. Neither has full-game results.

Held-out combat matches rose 35 → 41/48 in the new candidate; out-of-combat
matches were 21 → 20/48. The other 48 validation states outside the screening
set were flat at 31/48 before and after (0.0 points [−13.0, +13.0], 41 worlds).
Generalisation remains unestablished. Definitions are in [metrics](metrics.md).

## Now

The four-hour MacBook study is complete. All arms used medium effort. Short
screens compared smaller updates, a frozen-adapter KL penalty, answer-token
credit and a soft length cost. The selected unregularized continuation was
stopped when reasoning grew and traces began repeating. Final evaluation is
complete, outputs preserved, and no additional experiment was started.
The [dashboard](http://127.0.0.1:8767/) remains available; see [status](status.md).

## Next

Test action learning and length control separately: scale answer-token reward
gradients independently, apply a full-completion length cost, and retain KL to
the passing adapter with medium effort. This is a proposed intervention for a
new allocation, not an experiment already running. No seed-repeat prerequisite.

## Evidence and open questions

- Small-set learning works; broader learning still trades reliability for teacher agreement.
- The short length-cost screen showed no shortening or learning benefit.
- Answer-only credit reduced gradient scale as well as changing credit assignment.
- Teacher probabilities describe policy preferences, not correctness or game values.

See [strategies](strategies.md), [results](results.md) and [historical references](archive.md).
