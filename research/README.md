# Learning full-game play with a reasoning language model

No learning run has happened yet. The pilot plan is on hold pending Sid's
decisions on a staged replacement. The laptop can run the training under MLX.

The goal is an LLM that makes combat and out-of-combat decisions for Ironclad,
Ascension 0, Acts 1–3, while we watch how its reasoning changes during RL. No
SFT or teacher-forced reasoning traces are used.

## Current best

There is no trained model. The unchanged K2-Horizon-0.9B scores no better than
always choosing the first menu option on the teacher measure:
- teacher probability 0.26–0.33 at medium effort, against 0.30 for
  always-first;
- 0.21 for a random legal action and 0.54 for the teacher's own top choice;
- 64 states, independent samples from the RTX and Mac runs.

It has won no games; none have been attempted. Definitions are in
[metrics](metrics.md).

## Now

Nothing is running ([status](status.md)). Completed on 25 October:
- A [review of the plan](brainstorms/2026-10-25-getting-the-model-to-learn.md):
  - the model is at trivial-baseline skill;
  - combat rewards are mostly near-ties, amplified by standard-deviation
    scaling;
  - the plan allowed only about 150 updates at a low LoRA learning rate.
- The [Mac speed study](experiments/2026-10-25-mac-local-speed.md): an exact
  MLX conversion, with an estimated 95 s per pilot iteration on the laptop.
- Code fixes: full-game RL scores model-invalid games, errors are no longer
  swallowed, held-out worlds are guarded, and few-world intervals are
  withheld.
- Corrected speed-check ledger rows, with trivial baselines added.

## Next

1. Sid decides the questions at the end of the review: changing the prompt,
   starting with out-of-combat decisions only, mean-centred advantages, and
   hardware.
2. Fix the potion line in the environment on `main`. Out-of-combat text lists
   potions by count rather than slot.
3. Write a small MLX teacher-feedback trainer: grouped generation and
   checkpointed updates.
4. Stage 0: a prompt A/B on the unchanged model on the laptop.
5. Stage 1: an overfit test and learning-rate screen on 32 decisive
   out-of-combat states.
6. Stage 2: an out-of-combat pilot with a held-out probe every 10–20 steps.

## Evidence and open questions

- Teacher probabilities are the preferences of a 27–29%-win policy, not game
  values, and transfer to full games is untested.
- Open questions:
  - Can the model learn decisive out-of-combat choices at all?
  - Does its reasoning shrink under this reward?
  - Is the champion's diffuse combat policy a usable combat teacher, or is a
    search teacher needed?

See [strategies](strategies.md), [results](results.md), the [log](log.md) and
[historical references](archive.md).
