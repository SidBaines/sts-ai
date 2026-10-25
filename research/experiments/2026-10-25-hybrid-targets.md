# Value targets with the older corpora's coverage

**Status:** running. **Verdict:** none yet.

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

## Conclusion

## Next

## Log

- 25 Oct: fits started.

Code: commit a8b0ea4, run from worktree `nonllm-hybrid-targets` (do not edit
it). Data: run `value_refits_20261023` (local, worktree `nonllm-large-batch`).
