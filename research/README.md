# Learning full-game play with a reasoning language model

Training and evaluation code is ready; the speed checks are complete, and no main learning study has started.

The goal is an LLM controlling combat and out-of-combat decisions in Ironclad,
Ascension 0, Acts 1–3. We want to observe reasoning changing during RL, without
SFT or teacher-forced reasoning traces. Initial teacher feedback and later
terminal-outcome training are implemented. There is no new LLM win-rate result.

## Now

[Speed checks](experiments/2026-10-23-llm-throughput.md) measured all three
K2-Horizon-0.9B efforts on RTX 2000 Ada, serving controls, actual HF generation,
real optimizer updates, prefix caches and numerical controls.

Four responses per state with serving prefix caching reached 3.27 versus 2.17
valid decisions/second, with 75% prompt-token hits. Distinct states showed no
gain. Medium HF generation at batch eight reached 0.326 valid decisions/second
on 16 responses; high effort had long tails. These are exploratory timings.

Teacher-feedback RL, checkpoint continuation and trained-adapter evaluation have
bounded mechanics checks. Whole-game outcome RL is unit-tested and untrained.
See [status](status.md), [results](results.md) and [metric definitions](metrics.md).

## Next

Choose effort, rollout backend/dtype, output cap and response-loss normalization
before the approximately ten-hour teacher-feedback pilot. HF remains slower
than dedicated serving. BF16 batch/cache changes can change generated actions,
and recomputed old scores are not bitwise identical to cached sampling scores.

The teacher is an action-preference bootstrap, not a game-value oracle. Later
full-game RL must report model/simulator failures and group eligibility alongside
wins. Whether the bootstrap transfers to full games remains untested.

## Evidence

[Strategies](strategies.md) records the current interpretation. Historical
saved-fight imitation improved execution, but stable full-game reasoning RL was
not established; see [historical references](archive.md).
