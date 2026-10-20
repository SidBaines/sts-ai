# Experiment log

A running, dated record of what was run, how it went and what came out,
newest first, kept at Sid's request from 20 October 2026. One or two lines
per entry; numbers and conclusions live in the linked pages.

## 20 October 2026

- **Oracle option test: the oracle cannot choose card rewards.** 160 Act 2
  card rewards, 4 options x 32 common-seed boss fights: best option clearly
  beats the champion's in 12% (rule 20%); gain from following it +0.002 boss
  damage [−0.005, +0.009]. One card moves the fight less than the noise.
  The oracle mostly agrees with the champion's picks. The oracle moved into
  the library (`src/sts_rl/deck_oracle.py`, with a simulator test).
  [Page](experiments/2026-10-20-oracle-options.md).
- **Environment step 3.1x faster, byte-identical** (Codex task 1, reviewed by
  Claude). 1,119 -> 356 µs per decision; old and new code agree on all 18,344
  steps of 40 complete champion games (22 reaching Act 3, 2 wins); 266 tests
  pass. End to end with the policy, 2.5x faster. Pull request
  [#8](https://github.com/SidBaines/sts-ai/pull/8) to `main`, awaiting Sid.
- **Deck-strength oracle spike: worth building on.** Simulated Act 2 boss
  fights (16 per state, full HP, champion's combat network) from the last
  decision before the boss predict the real outcome with AUC 0.78 against
  entry HP's 0.68; together 0.83 against 0.64, log loss −0.127 [−0.18,
  −0.09], reliability 0.91, independent of HP. 300 worlds, 9 minutes on 4
  cores. Decks differ a lot in strength after all; next is whether it
  separates options. [Page](experiments/2026-10-20-deck-oracle.md).
- **Codex tasks started** for throughput: (1) remove redundant deep copies and
  speed up text rendering in the environment step, output byte-identical
  (branch `env/fast-step` from `main`); (2) a batched in-process game runner
  for evaluation that must reproduce existing games exactly (branch
  `nonllm/throughput`).
- **Throughput profile.** Per decision through the worker process: 0.58 ms
  environment step (over half deep copies and text rendering), 0.32 ms
  process round trip, 0.19 ms encoding, 0.20 ms network. One idle evaluator
  process: 635 decisions/s; seven at once: about 285 each. Realistic gain 2
  to 3 times per core. Re-run shards replayed 1,022 of 1,022 worlds
  identically.
- **Representation controls written up.** Old encoder beats the original
  structured encoder in full games at both seeds (+2.4 and +3.1 points); rule
  features recover about a point, width none; width 216 at learning rate
  0.001 trains badly, which drives the panel main effects. Merged into
  `ws/nonllm`. [Page](experiments/2026-10-16-representation-controls.md).
- **Brainstorm written** with the deck analysis and ideas; Sid's decisions in
  its section 7. [Brainstorm](brainstorms/2026-10-19-brainstorming.md).

## 19 October 2026

- **Remaining representation-controls game arms finished** overnight on 7
  lanes after a four-hour battery pause; two timed-out shards re-run; one
  world cut at the decision limit, counted as a loss by Sid's decision.
- **Work handed from Codex to Claude.** Research notes moved to the
  `research/` convention.
