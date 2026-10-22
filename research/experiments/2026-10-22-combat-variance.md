# Combat student: fit-to-fit variance, all corpora together, and boss-only labels

**Status:** running. **Verdict:** pending.

## Question

[Round 3](2026-10-21-combat-student-round-3.md) of combat expert iteration
produced a much worse student than round 2, with the same recipe. Before
trusting any single refit we need to know: how much do students fitted to
the same corpus with different seeds differ in play? Does a student fitted
on all three corpora together (511,847 labels; Sid's option 3 of 22
October) beat the champion's round 2 student? And, as the next corpus, does
labelling only boss fights, across more of the champion's games, help?

## Design

- **Player for every comparison:** the champion's out-of-combat head
  (`champion_85ea9019ff95.pt`) with the combat head swapped, so only the
  combat student differs; `fast_eval.py` on the 4,096 second-DEV worlds,
  paired with the champion (701 wins; its combat head is the round 2
  student, seed 1919).
- **Students:** the round 2 corpus refitted with seed 2; the round 3
  student (seed 1919); the round 3 corpus refitted with seed 2; all three
  corpora fitted together with seed 1919, using the new opt-in
  `--mixed-sources` (under review; it accepts corpora from different
  players and from environment 0.1.5 and 0.1.5.1, which have byte-identical
  observations). All with the 15 October fitting recipe.
- **Reading, fixed in advance:** the two round 2 fits measure fit-to-fit
  variance on one corpus; if seed 2 of round 2 is as bad as round 3, the
  round 3 result is fit variance, not data. The all-corpora student is
  *Better* if it wins more than the champion with exact McNemar p < 0.05,
  then confirmed on 4,096 unused worlds before any promotion.
- **Boss-only corpus:** the champion plays 4,000 new training worlds from
  90,500,300,000 (eight shards of 500); the teacher labels every Act 1, 2
  and 3 boss decision and nothing else, so the same labelling time covers
  about twice as many distinct boss arrivals. Its use is decided after the
  variance check.
- **Approval:** Sid, 22 October ("do option 3", "the boss-only labelling
  round next").

## Results

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 22 Oct, 12:42: fits, splices and evaluations queued (detached
  `combat_variance.sh`); boss-only labelling started.

Code: `scripts/rl/fresh_train.py` (`--mixed-sources` from worktree
`nonllm-mixed-corpora` until merged), `scripts/rl/combine_heads.py`,
`scripts/rl/fast_eval.py`. Data: runs `combat_variance_20261022` and
`combat_bosses_20261022` (local, worktree `nonllm-large-batch`).
