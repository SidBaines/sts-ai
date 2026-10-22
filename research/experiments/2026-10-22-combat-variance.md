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

**Fit-to-fit variance is heavy-tailed: three of four fits land within half
a floor of each other, one collapsed.** Each student in the champion's
out-of-combat head, second DEV, paired with the champion (701 wins; its
combat head is the round 2 student, seed 1919):

| Combat student | Wins | Discordant, champion only / student only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Round 2 corpus, seed 1919 (the champion's own) | 701 | | | 40.16 | |
| Round 2 corpus, seed 2 | 681 | 337 / 317 | 0.46 | −0.47 [−0.80, −0.12] | −0.7 points [−2.3, +0.9] |
| Round 3 corpus, seed 2 | 689 | 379 / 367 | 0.69 | −0.09 [−0.45, +0.26] | +3.4 points [+1.7, +5.0] |
| Round 3 corpus, seed 1919 | 588 | 400 / 287 | < 0.001 | −1.96 [−2.33, −1.60] | −2.7 points [−4.4, −1.0] |
| All three corpora, seed 1919 | running | | | | |

- The two corpora look alike (2,000 worlds each; the collecting players won
  274 and 284; 174,859 and 177,281 labels with the same room mix), and
  the round 3 corpus with seed 2 is as good as the champion's student.
  The [round 3](2026-10-21-combat-student-round-3.md) failure was a bad fit,
  not bad data.
- Fits: round 2 seed 2 stopped at step 3,400 (validation loss 1.418);
  round 3 seed 1919 at 3,300 (1.469); round 3 seed 2 is in the run's
  status file. Validation loss did not flag the bad fit clearly.

## Conclusion

Pending.

## Next

Pending.

## Log

- 22 Oct, 13:33: round 3 seed 2 is fine (689 against 701): the round 3
  collapse was fit variance. The all-corpora fit is being evaluated.
- 22 Oct, 12:42: fits, splices and evaluations queued (detached
  `combat_variance.sh`); boss-only labelling started.

Code: `scripts/rl/fresh_train.py` (`--mixed-sources` from worktree
`nonllm-mixed-corpora` until merged), `scripts/rl/combine_heads.py`,
`scripts/rl/fast_eval.py`. Data: runs `combat_variance_20261022` and
`combat_bosses_20261022` (local, worktree `nonllm-large-batch`).
