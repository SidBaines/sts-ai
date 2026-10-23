# Learning full-game play with a reasoning language model

The next step is a short serving benchmark before choosing settings for RL.

The goal is an LLM controlling combat and out-of-combat decisions in Ironclad,
Ascension 0, Acts 1–3. We want to observe how its reasoning changes during RL.
Sid does not want SFT or teacher forcing on reasoning. A non-LLM policy may
provide action feedback during an initial RL stage, followed by full-game
outcome RL. These training stages have not started.

## Now

[Serving benchmark](experiments/2026-10-23-llm-throughput.md): compare
K2-Horizon-0.9B at low, medium and high reasoning effort, with different
request concurrency and supported attention settings on one A40.
See [status](status.md) for the live job.

## Next

Choose serving settings using valid completed decisions, token cost, output
completeness and memory; do not choose solely by generated tokens per second.
Then design the first teacher-feedback RL pilot within about ten GPU-hours.

## Evidence

No new learning result has been obtained. Historical saved-fight imitation
improved execution and fight outcomes, but stable full-game reasoning RL was
not established. See [strategies](strategies.md), [results](results.md),
[metrics](metrics.md), and [historical references](archive.md).
