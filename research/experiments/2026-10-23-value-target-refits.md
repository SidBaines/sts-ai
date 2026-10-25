# Value targets in combat refits

**Status:** finished 25 October. **Verdict:** on the same 1,752 worlds, value-target refits beat vote-target refits by about 6 points of wins (fixed rule met: both seeds, pooled p < 0.0001); they pass the Act 3 boss far more often than the champion but lose more early fights, so overall they roughly match today's recipe, which uses three times as much data.

## Question

[Value targets](2026-10-23-value-targets.md) made better students than vote
targets on a small, boss-heavy corpus, but those students were far weaker
than the champion's own combat head. The refits that carried joint run 2's
gain are fitted on the current player's own fights with the teacher's
luckiest-line votes, plus two older corpora. Do value targets also make
better refits there? Approach:
[expert iteration for combat](../strategies.md#expert-iteration-for-combat).

## Design

- **Labels** (`value_refits_20261023/corpus`): 2,000 worlds from
  90,502,000,000 played by the champion (`champion_c8553bc33e53.pt`),
  labelled exactly as the joint loop labels (24 searches of 1,000
  simulations, the joint loop's per-fight-type sampling rates, luckiest-line
  votes), with the full search statistics recorded.
- **Fits** (old encoder, the refit recipe), two seeds each:
  - *Votes, new corpus only:* luckiest-line votes on the new corpus.
  - *Values, new corpus only:* value targets at scale 100 on the new corpus.
  - *Today's recipe:* luckiest-line votes on the new corpus plus the round 2
    and round 3 corpora (value targets cannot use those, which lack search
    statistics).
- **Play:** each fit spliced into the champion (since 25 October
  `champion_4058cbf0e271.pt`, whose combat head is identical to the one that
  played the labelled games), with the champion itself,
  on 8,192 unused worlds from 111,000,000,000, paired by world.
- **Decision rule, fixed in advance:** values are *Better* than votes on the
  same corpus if both seed pairs favour values and the pooled paired
  comparison (16,384 world pairs) has exact McNemar p < 0.05. Reported
  alongside: each fit against the champion and against today's recipe, and
  validation boss regret for every fit.
- **Changes before any fit (25 October):** labelling stopped at 1,752 of
  2,000 worlds, which are used as they are; the play tests use the new
  champion. Sid asked for both.
- **Approval:** under Sid's standing instruction to keep doing useful
  research (22 October); follows the value-target result.

## Results

**Main comparison (rule fixed in advance): value targets beat votes on the
same corpus.** Each refit spliced into the champion; 8,192 unused worlds from
111,000,000,000 (environment 0.1.5.1), paired by world:

| Combat head | Wins | Mean floor | Act 1 boss pass | Act 2 boss pass | Act 3 boss pass | Validation boss regret |
|---|---:|---:|---:|---:|---:|---:|
| Champion's own | 2,480 (30.3%) | 42.11 | 94.5% | 73.9% | 51.1% | |
| Votes, new corpus, seed 1 | 1,830 | 39.50 | 91.8% | 70.1% | 45.5% | 81.6 |
| Votes, new corpus, seed 2 | 1,874 | 38.67 | 89.8% | 64.9% | 50.1% | 76.6 |
| Values, new corpus, seed 1 | 2,349 | 38.58 | 87.8% | 74.4% | 58.4% | 51.4 |
| Values, new corpus, seed 2 | 2,402 | 38.69 | 87.7% | 72.9% | 61.0% | 55.4 |
| Today's recipe, seed 1 | 2,484 | 41.21 | 93.9% | 71.5% | 53.8% | 76.8 |
| Today's recipe, seed 2 | 2,320 | 40.82 | 93.9% | 70.2% | 51.6% | 73.2 |

- Values against votes on the new corpus: seed 1 2,349 against 1,830
  (discordant 1,249/730), seed 2 2,402 against 1,874 (1,263/735); pooled over
  16,384 world pairs 2,512 against 1,465 (exact McNemar p < 0.0001): *Better*,
  by about 6.3 points.
- Values (1,752 worlds) against today's recipe (5,752 worlds, votes): −1.6
  points at seed 1 (p = 0.003), +1.0 at seed 2 (p = 0.08): no consistent
  difference overall.
- Against the champion: values −1.6 (p = 0.005) and −1.0 (p = 0.10); today's
  recipe +0.0 (p = 0.94) and −2.0 (p < 0.001). No refit beat the champion's
  own combat head, so none would have passed the joint loop's gate.
- The value refits trade early strength for late strength: they pass the Act
  3 boss 58 to 61% of the time against 51 to 54% for every other head, but
  lose more Act 1 boss fights and hallway fights (mean floor 3.5 lower than
  the champion). The new corpus labels only 4 to 15% of hallway fights.
- Validation boss regret again separated value from vote fits (51 to 55
  against 73 to 82) in line with play at the Act 3 boss.

## Conclusion

Value targets make much better combat students than vote targets on the
same labelled fights, here by about 6 points of wins, and the gain is
largest where the agent loses most, at the Act 3 boss. On a boss-heavy
corpus a third the size, they roughly match today's recipe overall, losing
early what they gain late. The obvious next step is to give value targets
the hallway coverage of the older corpora.

Caveats:
- Two seeds per arm; the arms share one labelled corpus.
- 1,752 of the planned 2,000 worlds were labelled.
- The champion's own head was trained differently (several corpora and two
  gated refits), so it is a reference, not a matched control.

## Next

- Fit one student with value targets on the new rows and vote targets on
  the round 2 and round 3 corpora (`--target-mode pooled_values_or_votes`,
  being added), two seeds, and play it in the champion.
- Label future refit corpora with `--record-search`, and with more hallway
  fights, so value targets cover every fight type.

## Log

- 25 Oct, 10:54: all play tests done: values beat votes by about 6 points;
  no refit beat the champion's own head.
- 25 Oct: six fits started on the 1,752 labelled worlds. The two fits of
  today's recipe were refused because the fitter would not mix corpora
  recorded with and without search statistics (which does not change the
  votes); after a fix (commit 42bc115) they were rerun.
- 23 Oct, 18:18: the queue's stall watchdog stopped all eight labelling
  jobs after two hours without log output (the collector writes no progress
  to its log), with 1,753 of 2,000 worlds labelled and saved. The plan can
  go ahead on those worlds.
- 23 Oct, 16:18: labelling started (eight shards of 250 worlds).

Code: worktree `nonllm-value-targets` (commit 9dead85; do not edit it).
Data: run `value_refits_20261023` (local, worktree `nonllm-large-batch`).
