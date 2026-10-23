# Learning full-game play with a reasoning language model

Medium reasoning effort is selected for the first pilot; the next step is a bounded rollout check and teacher-feedback RL.

The goal is an LLM controlling combat and out-of-combat decisions in Ironclad,
Ascension 0, Acts 1–3, while we observe its reasoning changing during RL.
No SFT or teacher-forced reasoning traces are used. There is no new LLM win-rate result.

## Now

Sid selected medium effort on 23 October. The
[medium-effort pilot plan](experiments/2026-10-23-medium-teacher-pilot.md)
records the proposed remaining settings, ten-GPU-hour budget and decision rules.
Nothing is running; see [status](status.md).

The [completed speed study](experiments/2026-10-23-llm-throughput.md) found
useful batching and repeated-state cache gains, but the current Hugging Face
training rollout is slower than dedicated vLLM serving. Sampling/scoring
consistency and adapter refresh need checking before adopting faster rollouts.

## Next

1. Check matched rollout/update correctness and end-to-end throughput; freeze
   one backend, precision, cache policy and pilot configuration before training.
2. Prepare a broader frozen teacher-labelled corpus with disjoint training,
   validation and test worlds; collect the unchanged model's baseline.
3. Run the medium-effort teacher-feedback pilot with one shared adapter and
   fixed checkpoint evaluations, retaining all generated reasoning.
4. Compare the frozen candidate with the base model on untouched states and
   a small paired full-game panel; report completion failures alongside wins.
5. Design terminal-outcome RL after the bootstrap and full-game validity checks.

## Evidence and open questions

Teacher-feedback updates, checkpoint continuation and trained-adapter evaluation
have bounded mechanics checks. Whole-game outcome RL is implemented and
unit-tested, with no full-game learning run. Teacher probabilities are action
preferences, not game values; transfer to full-game play is untested.

See [strategies](strategies.md), [results](results.md), [metrics](metrics.md),
and [historical references](archive.md). Backend/precision acceptance and the
remaining proposed pilot settings are the immediate open questions.
