# Learning full-game play with a reasoning language model

The broader continuation failed: after 21 updates, held-out teacher matches
fell from 56/96 to 6/96 because most answers repeated text until the token cap
([results](experiments/2026-10-26-broader-training.md)).

The goal remains an LLM making both combat and out-of-combat decisions for
Ironclad, Ascension 0, Acts 1–3, while we study how its reasoning changes during
RL. No SFT or teacher-forced reasoning traces are used.

## Current best

The original stage-1 adapter remains the working reference
(`stage1_screen_ps_20261026/lr_3e-5`, iteration 48). It learned the teacher's
choice on 22/32 trained states, from 10/32, at learning rate 0.00003
([stage 1](experiments/2026-10-25-overfit-test.md)). Generalisation is not established.

On the broader study's starting panel it matched 56/96 held-out decisions and
answered all 96 validly. The failed continuation matched 6/96, a paired loss
of 52.1 percentage points (95% world-cluster interval −62.1 to −41.8;
73 worlds, Exploratory). Neither adapter has been tested in full games.
Definitions and the untrained-model references are in [metrics](metrics.md).

## Now

Starting an authorized four-hour MacBook session of short learning trials:
smaller updates, KL to the passing adapter, answer-token credit, and a soft
length penalty. All use medium effort; Sid explicitly excluded low-effort mode.
See [the design](experiments/2026-10-27-learning-trials.md) and [status](status.md).

## Next

Screen the four interventions briefly, then spend the remaining allocation on
the most promising usable result. Preserve every outcome and finish with a
paired full-panel comparison when time permits. Signs of life take priority;
no seed-repeat prerequisite and no full-game run in this allocation.

## Evidence and open questions

- Small-set learning works; the unregularized broader continuation is unstable.
- Whether decision learning can transfer while preserving coherent reasoning is open.
- Teacher probabilities describe a 27–29%-win policy's preferences, not game values.
- It remains open whether diffuse combat labels need a stronger search teacher.

See [strategies](strategies.md), [results](results.md), the [log](log.md) and
[historical references](archive.md).
