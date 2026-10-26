# Learning full-game play with a reasoning language model

Stage 1 passed on 26 October: teacher-feedback RL on the laptop taught the
model the teacher's choice on 22 of 32 decisive training states (from 10), at
learning rate 0.00003 ([page](experiments/2026-10-25-overfit-test.md)).

The goal is an LLM that makes combat and out-of-combat decisions for Ironclad,
Ascension 0, Acts 1–3, while we watch how its reasoning changes during RL. No
SFT or teacher-forced reasoning traces are used.

## Current best

There is no model trained beyond the 32 stage-1 states (adapter
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

Nothing is running ([status](status.md)). Stage 1 results, 26 October
([page](experiments/2026-10-25-overfit-test.md)):
- 0.00003 passed: greedy matches 10 → 22 of 32 (+6 to +18), and teacher
  probability rose 0.35 → 0.51, mostly through content rather than menu
  position.
- 0.00001 learned too slowly; 0.0001 broke the answer format.
- A first screen that shared one menu order per group taught "answer A" and
  was rerun.

## Next

1. Stability check on the same 32 states at 0.00003: a repeat with a new
   seed, a small KL penalty, and 8 states per step. Write its page first and
   get Sid's approval; [status](status.md) has the step-by-step handoff.
2. Stage 2: many more decisive states from training worlds at 0.00003, gated
   on held-out top-choice agreement on decisive states. Needs a page and Sid's
   approval.
3. Merge env-v0.1.6 into this branch once pull request #10 merges, and build
   stage-2 states under it.

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
