# Combat students trained on the teacher's move values instead of its votes

**Status:** running (stage 2 play test). **Verdict:** none yet.

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

**Stage 1: value targets cut the value students give up at bosses by about a
sixth.** Held-out test rows of the pooled corpus (worlds the fits never
saw), one fit per target at seed 1, paired row by row against the vote
target; intervals are 95%, clustered by world. Regret is in search
evaluation points (about 100 per HP in a won fight).

| Target | Boss regret (n = 18,869) | Act 1 boss | Act 2 boss | Act 3 boss | Agreement at bosses |
|---|---|---|---|---|---|
| Votes | 55.1 | 65.8 | 54.7 | 50.9 | 52.5% |
| Values, scale 30 | −3.0 [−5.6, −0.3] | −1.2 | −1.7 | −6.0 | −2.0 points |
| Values, scale 100 | **−8.6 [−11.8, −5.4]** | −12.2 [−18.6, −5.7] | −5.7 [−9.9, −1.6] | −12.0 [−18.3, −5.7] | −1.5 points |
| Values, scale 300 | −6.0 [−9.0, −2.9] | −9.0 | −2.8 | −10.1 | −2.7 points |

- Over all 38,040 test rows, scale 100 lowers regret from 72.0 by 6.3
  [3.8, 8.8], with agreement 0.5 points lower [−1.1, +0.0]: students
  trained on values pick the teacher's exact move slightly less often but
  give up less value when they differ.
- Validation rows rank the scales the same way (boss regret 56.1 for votes,
  53.1 at scale 100, 53.4 at scale 300), so scale 100 goes to stage 2.
- Between one fit and another of the same recipe, play strength varies a
  lot, which these intervals do not include; stage 2 uses two seeds.

## Conclusion

## Next

## Log

- 23 Oct, 13:42: stage 1 meets its condition (scale 100 lowers boss
  regret by 8.6 [5.4, 11.8]); stage 2 started: seed 2 fits and play on
  8,192 worlds from 110,000,000,000.
- 23 Oct, 13:13: four stage 1 fits started (votes; values at scales 30, 100
  and 300). A read-only scan of all 192,591 rows found the pooled value
  winner always equals the stored pooled vote; at scale 100 the median
  target puts 0.60 on its best move.

Code: fitter options `--target-mode` and `--value-scale` (commit 9dead85,
run from worktree `nonllm-value-targets`; do not edit it). Data: run `value_targets_20261023`
(local, worktree `nonllm-large-batch`).
