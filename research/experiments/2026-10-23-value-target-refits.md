# Value targets in combat refits

**Status:** running (fits started 25 October). **Verdict:** none yet.

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

## Conclusion

## Next

## Log

- 25 Oct: six fits started on the 1,752 labelled worlds.
- 23 Oct, 18:18: the queue's stall watchdog stopped all eight labelling
  jobs after two hours without log output (the collector writes no progress
  to its log), with 1,753 of 2,000 worlds labelled and saved. The plan can
  go ahead on those worlds.
- 23 Oct, 16:18: labelling started (eight shards of 250 worlds).

Code: worktree `nonllm-value-targets` (commit 9dead85; do not edit it).
Data: run `value_refits_20261023` (local, worktree `nonllm-large-batch`).
