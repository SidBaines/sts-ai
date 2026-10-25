# Value targets with the older corpora's coverage

**Status:** finished 25 October. **Verdict:** worse: mixing value targets on new rows with votes on older rows lost both to today's recipe and to value targets alone (pooled p < 0.0001), and kept none of the Act 3 boss gain.

## Question

[Value-target refits](2026-10-23-value-target-refits.md) on 1,752 of the
champion's worlds passed the Act 3 boss far more often than any other combat
head (58 to 61% against 51 to 54%) but lost more early fights; their corpus
labels few hallway fights. Today's recipe (votes on that corpus plus the
round 2 and round 3 corpora, 5,752 worlds) is strong early and weaker late.
Does one student fitted with value targets on the rows that have search
statistics and votes on the older rows keep both strengths? Approach:
[expert iteration for combat](../strategies.md#expert-iteration-for-combat).

## Design

- **Data:** the same 5,752 worlds as today's recipe: 180,891 combat rows get
  value targets (scale 100) and 352,140 get votes
  (`--target-mode pooled_values_or_votes`).
- **Fits:** old encoder, the refit recipe, seeds 1 and 2
  (`value_refits_20261023/fit_hybrid_seed*`).
- **Play:** each fit spliced into the champion
  (`champion_4058cbf0e271.pt`), on the same 8,192 worlds from
  111,000,000,000 as the other arms, paired by world.
- **Decision rule, fixed in advance:** the hybrid is *Better* than today's
  recipe if both seed pairs favour it and the pooled paired comparison
  (16,384 world pairs) has exact McNemar p < 0.05. If it also beats the
  champion, it is confirmed on 8,192 unused worlds from 112,000,000,000
  before any promotion. Reported alongside: against value targets alone,
  against the champion, boss pass rates and validation boss regret.
- **Approval:** under Sid's standing instruction to keep doing useful
  research; follows the value-target refit result.

## Results

**Main comparison (rule fixed in advance): the hybrid is not better than
today's recipe; it is worse.** Each fit spliced into the champion; the same
8,192 unused worlds from 111,000,000,000 as the other arms, paired by world:

| Combat head | Wins | Mean floor | Act 1 boss pass | Act 2 boss pass | Act 3 boss pass | Validation boss regret |
|---|---:|---:|---:|---:|---:|---:|
| Champion's own | 2,480 | 42.11 | 94.5% | 73.9% | 51.1% | |
| Today's recipe, seeds 1 / 2 | 2,484 / 2,320 | 41.21 / 40.82 | 93.9% / 93.9% | 71.5% / 70.2% | 53.8% / 51.6% | 76.8 / 73.2 |
| Value targets alone (1,752 worlds), seeds 1 / 2 | 2,349 / 2,402 | 38.58 / 38.69 | 87.8% / 87.7% | 74.4% / 72.9% | 58.4% / 61.0% | 51.4 / 55.4 |
| Hybrid, seed 1 | 1,975 | 39.69 | 92.2% | 66.2% | 49.2% | 64.9 |
| Hybrid, seed 2 | 2,317 | 40.34 | 92.6% | 69.6% | 54.2% | 59.4 |

- Against today's recipe: seed 1 1,975 against 2,484 (p < 0.001), seed 2
  2,317 against 2,320 (p = 0.96); pooled 1,631 against 2,143 discordant of
  16,384 pairs (p < 0.0001). Not better; worse, though seed 1 looks like a
  bad fit of the kind seen before.
- The hybrid kept the older corpora's early strength only partly and lost the
  Act 3 boss gain of value targets alone. Its validation boss regret sits
  between the two (59 to 65).

## Conclusion

Value targets do not combine well with vote targets in one fit: two thirds
of the rows carrying votes pulled the student back toward vote-trained
behaviour, and the mixture was worse than either recipe on its own. The
Act 3 boss gain seen with value targets alone needs a corpus labelled
entirely with search statistics.

Caveats:
- Two seeds; seed 1 is an outlier, so the size of the loss is uncertain,
  though seed 2 also showed no gain.
- Why mixing hurts is not tested: inconsistent target types for similar
  positions, or simply the vote rows outnumbering the value rows, are both
  possible.

## Next

- Label more of the champion's fights with `--record-search`, with more
  hallway fights sampled, and fit value targets alone on the larger corpus.

## Log

- 25 Oct, 11:55: play tests done: the hybrid is worse than either recipe.
- 25 Oct, 10:56: fits started.

Code: commit a8b0ea4, run from worktree `nonllm-hybrid-targets` (do not edit
it). Data: run `value_refits_20261023` (local, worktree `nonllm-large-batch`).
