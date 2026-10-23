# Combat students trained on the teacher's move values instead of its votes

**Status:** running (stage 1 fits started 23 October, 13:13). **Verdict:** none yet.

## Question

Combat students copy the search teacher poorly: they agree with the pooled
teacher on about 57% of positions, against about 80% for a second run of the
teacher, and more data helps slowly ([better teacher](2026-10-22-better-teacher.md)).
Today's targets keep only which move the teacher chose. Each search also
records how good every move looked (its mean evaluation), which says which
alternatives are nearly as good and which are bad. Does training on those
values give students that play closer to the teacher, and better? This is
option C1 of the [23 October discussion](../brainstorms/2026-10-23-next-steps.md);
approach: [distilling the search teacher](../strategies.md#distilling-the-search-teacher).

Agreement is also a misleading measure at bosses: half the time the
teacher's two best moves differ by under 32 evaluation points (about a third
of an HP in a won fight), so many disagreements cost almost nothing. This
study also measures **value regret**: how much teacher value the student's
choice gives up.

## Design

- **Data:** the pooled-teacher corpus (`combat_pooled_20261023`, 2,000
  worlds from 90,501,000,000 played by the 22 October champion; 134,376
  combat rows with the full search statistics), with the same split and fit
  recipe as the earlier pooled students (old encoder, learning rate 0.001,
  patience 6, up to 12,000 steps).
- **Targets:**
  - *Votes:* the pooled teacher's chosen move (today's target).
  - *Values:* over the moves the pooled searches visited at least a tenth
    as often as the most-visited move, a target proportional to
    exp((value − best value) / scale), at scales of 30, 100 and 300
    evaluation points; other moves get zero.
- **Stage 1 (fits only):** one fit per target at seed 1. Measures on the
  held-out worlds, by fight group: agreement with the pooled teacher's move
  and value regret, with the Act 1 to 3 bosses primary.
- **Stage 2 (play), if a value target lowers boss regret:** the best scale
  and the vote target, two fit seeds each, spliced into the champion
  (`champion_c8553bc33e53.pt`), play 8,192 worlds from 110,000,000,000 with
  the champion itself. Decision rule, fixed in advance: value targets are
  *Better* than vote targets if both seed pairs favour them and the pooled
  paired comparison (16,384 world pairs) has exact McNemar p < 0.05. These
  students see far less data than the champion's own combat head, so beating
  the champion is not expected; the question is which target is better on
  the same data.
- **Caveat known in advance:** the joint loop's refit corpora did not record
  search statistics, so using value targets in refits needs labelling with
  `--record-search`.

## Results

## Conclusion

## Next

## Log

- 23 Oct, 13:13: four stage 1 fits started (votes; values at scales 30, 100
  and 300). A read-only scan of all 192,591 rows found the pooled value
  winner always equals the stored pooled vote; at scale 100 the median
  target puts 0.60 on its best move.

Code: fitter options `--target-mode` and `--value-scale` (commit 9dead85,
run from worktree `nonllm-value-targets`; do not edit it). Data: run `value_targets_20261023`
(local, worktree `nonllm-large-batch`).
