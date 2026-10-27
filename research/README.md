# Learning full-game play with a reasoning language model

Stage 1 passed on 26 October: teacher-feedback RL on the laptop taught the
model the teacher's choice on 22 of 32 decisive training states (from 10), at
learning rate 0.00003 ([page](experiments/2026-10-25-overfit-test.md)).

The goal is an LLM that makes combat and out-of-combat decisions for Ironclad,
Ascension 0, Acts 1–3, while we watch how its reasoning changes during RL. No
SFT or teacher-forced reasoning traces are used.

## Current best

The last completed model was trained on the 32 stage-1 states (adapter
`stage1_screen_ps_20261026/lr_3e-5`, iteration 48), which shows no clear
transfer to held-out states. As expected for a model that has never played, the
unchanged K2-Horizon-0.9B scores no better than always choosing the first menu
option on the teacher measure. This is the "before" reference only:
- teacher probability 0.26–0.33 at medium effort, against 0.30 for
  always-first;
- 0.21 for a random legal action and 0.54 for the teacher's own top choice;
- 64 states, independent samples from the RTX and Mac runs.

It has won no games; none have been attempted. Definitions are in
[metrics](metrics.md).

## Now

The laptop continuation stopped after 21 updates because repetition and
answer validity deteriorated; the final held-out evaluation has resumed on mains
([design](experiments/2026-10-26-broader-training.md), [status](status.md)).
Sid chose signs-of-life progress over another seed repeat on 26 October.
The existing teacher, prompt and learning rate stay fixed. Optimizer moments
restart because the parent checkpoint holds only adapter weights.

## Next

1. Finish the stopped run's evaluation and compare it with its starting adapter.
   Repetition and token-cap failures motivate a targeted stability intervention.
   Do not launch another run automatically.
2. Assess transfer by phase and inspect position dependence and reasoning.
3. Adopt env-v0.1.6 after release before collecting newly simulated states or
   testing game transfer. The current run uses existing filtered observations.

## Evidence and open questions

- Teacher probabilities are the preferences of a 27–29%-win policy, not game
  values, and transfer to full games is untested.
- Open questions:
  - Does learning on decisive states transfer to held-out states?
  - Does its reasoning shrink under this reward?
  - Is the champion's diffuse combat policy a usable combat teacher, or is a
    search teacher needed?

See [strategies](strategies.md), [results](results.md), the [log](log.md) and
[historical references](archive.md).
