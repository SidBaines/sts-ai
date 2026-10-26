# Learning full-game play with a reasoning language model

No learning result yet. Stage 1, an overfit test on the laptop, has been
running since 26 October.

The goal is an LLM that makes combat and out-of-combat decisions for Ironclad,
Ascension 0, Acts 1–3, while we watch how its reasoning changes during RL. No
SFT or teacher-forced reasoning traces are used.

## Current best

There is no trained model. As expected for a model that has never played, the
unchanged K2-Horizon-0.9B scores no better than always choosing the first menu
option on the teacher measure. This is the "before" reference only:
- teacher probability 0.26–0.33 at medium effort, against 0.30 for
  always-first;
- 0.21 for a random legal action and 0.54 for the teacher's own top choice;
- 64 states, independent samples from the RTX and Mac runs.

It has won no games; none have been attempted. Definitions are in
[metrics](metrics.md).

## Now

Stage 1's learning-rate screen is running on the laptop
([status](status.md)): an overfit test of the training loop on 32 decisive
states (16 combat, 16 out of combat), approved by Sid on 26 October
([its page](experiments/2026-10-25-overfit-test.md)). It uses:
- prompt version 2: letter-labelled menus in a fresh random order;
- the MLX teacher-feedback trainer in `src/sts_llm/mlx_training.py`;
- the state set, labelled by the 25 October champion.

Earlier on 25 October:
- a [review of the plan](brainstorms/2026-10-25-getting-the-model-to-learn.md);
- the [Mac speed study](experiments/2026-10-25-mac-local-speed.md);
- code fixes;
- the environment fixes in `main` pull request #10 (release 0.1.6).

## Next

1. Finish the stage-1 screen (about 4.5 hours, plugged in) and apply its
   pass rule.
2. Merge env-v0.1.6 into this branch once pull request #10 merges, and build
   later state sets under it.
3. If stage 1 passes: stage 2, a larger set of decisive states from training
   worlds, gated on held-out top-choice agreement on decisive states.

## Evidence and open questions

- Teacher probabilities are the preferences of a 27–29%-win policy, not game
  values, and transfer to full games is untested.
- Open questions:
  - Can the model learn decisive choices at all, in combat and out of it?
  - Does its reasoning shrink under this reward?
  - Is the champion's diffuse combat policy a usable combat teacher, or is a
    search teacher needed?

See [strategies](strategies.md), [results](results.md), the [log](log.md) and
[historical references](archive.md).
