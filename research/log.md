# Experiment log

A temporary, running record of what was run and what came out, newest
first, kept at Sid's request from 20 October 2026, one or two lines per
entry. It will be deleted: everything that matters is also recorded in its
home file (experiment pages, the results ledger, strategies, README).

## 20 October 2026

- **Instruction updates agreed with Sid.** `log.md` is now a recognised but
  temporary document (`main` via pull request
  [#7](https://github.com/SidBaines/sts-ai/pull/7), `ws/nonllm`, `ws/llm`),
  together with `brainstorms/`. `scripts/rl/CLAUDE.md` now says the batched
  evaluator is reproducible only at fixed batch size and process count, and
  that battery-pause faults leave evaluation cohorts incomplete.
- **Faster environment released and merged.** Pull request
  [#8](https://github.com/SidBaines/sts-ai/pull/8) merged into `main`,
  checked from a fresh checkout (266 tests, smoke rollout), tagged
  `env-v0.1.5.1`, and merged into `ws/nonllm` (1,010 tests pass). On the
  merged tip, `fast_eval.py` on 7 processes makes 18.8 games/s, against 5.3
  for the old evaluator, with all 224 test games identical.
- **Batched game runner merged into `ws/nonllm`** (Codex tasks 2 and 2b,
  reviewed by Claude; 1,005 tests pass). `scripts/rl/fast_eval.py` plays many
  games per process with batched inference; its stall detection ignores
  battery pauses and it re-runs healthy peers of a failed process. On 7
  processes: 6.7 games/s against 5.3 for the existing evaluator; with the
  faster environment of PR #8, 18.2 games/s (3.4x), all 224 test games
  identical. The speed-up needs PR #8 merged and tagged.
- **What makes a deck strong (exploratory).** Across the 300 oracle decks,
  counts and boss explain 24% of deck strength; power cards matter most
  (r = +0.31; Demon Form +0.21, Juggernaut +0.15, Inflame +0.11 simulated win
  share), then potions held (+0.25). Correlational only.
  [Deck-oracle page](experiments/2026-10-20-deck-oracle.md).
- **Batched runner benchmark (host, 256 worlds).** 7 processes x 32 games:
  6.68 games/s against 5.31 for the existing evaluator on 7 processes
  (1.26x). One process: 1.19 -> 1.32 games/s with batching. The environment
  step dominates, which task 1 cuts 3x; the combination is to be measured.
  Batch-size divergences (7 of 256 worlds) were between equivalent actions:
  same lengths, floors and outcomes.
- **PPO hygiene merged into `ws/nonllm`** (Codex task 3, reviewed by Claude;
  978 tests pass). Weight decay now explicit, default 0 (was a silent 0.01);
  out-of-combat lambda default 1.0. New options, default off: value loss kept
  out of the actor's trunk, value targets scaled by a running return scale,
  entropy bonus on strategic menus only. Runs after this are not directly
  comparable with older ones on the first two points. No training yet.
- **Batched game runner reviewed (Codex task 2).** At one game per process it
  reproduces all 256 recorded champion games exactly; larger batches flip
  exact ties between equivalent actions in about 3% of games (same lengths,
  floors and outcomes). Three fixes requested and made: pause-aware stall
  detection, re-running healthy peers instead of faulting them, and recording
  the batch size and process count.
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
