# Combat student: fit-to-fit variance, all corpora together, and boss-only labels

**Status:** finished 22 October. **Verdict:** fit variance is heavy-tailed; round 3's collapse was a bad fit; neither pooling all corpora nor adding boss-only labels beat the champion's student under the rule fixed in advance.

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
  about twice as many distinct boss arrivals. Use, fixed before the corpus
  finished: two students (seeds 1 and 2) fitted on the round 2 and round 3
  corpora plus the boss-only corpus (new opt-in `--mixed-sampling`, which
  pools corpora whose teacher-probability profiles differ), each in the
  champion's out-of-combat head, on second DEV paired with the champion;
  *Better* if a student wins more with exact McNemar p < 0.05, then
  confirmed on unused worlds. Because two seeds are tried, a pass is
  exploratory until confirmed. Early stopping uses pooled validation loss,
  which the boss-only corpus tilts towards boss fights.
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
| All three corpora (511,847 labels), seed 1919 | 647 | 358 / 304 | 0.039 | −0.47 [−0.80, −0.14] | −3.5 points [−5.2, −1.9] |

- The two corpora look alike (2,000 worlds each; the collecting players won
  274 and 284; 174,859 and 177,281 labels with the same room mix), and
  the round 3 corpus with seed 2 is as good as the champion's student.
  The [round 3](2026-10-21-combat-student-round-3.md) failure was a bad fit,
  not bad data.
- Fits: round 2 seed 2 stopped at step 3,400 (validation loss 1.418);
  round 3 seed 1919 at 3,300 (1.469); round 3 seed 2 is in the run's
  status file. Validation loss did not flag the bad fit clearly.
- The all-corpora fit (357,792 fit rows, stopped at step 4,200, validation
  loss 1.38) is worse than the champion's student under the rule fixed in
  advance. One seed: it may be fit variance (it matches round 2 seed 2's
  size) or the 15 October corpus, collected by a player winning 3.7%,
  pulling the student towards states today's player rarely sees.

**Boss-only labels: no student passes the rule; one is worse, one shows no
clear difference but wins more boss fights.** Students fitted on the round 2,
round 3 and boss-only corpora (435,285 fit rows), each in the champion's
out-of-combat head, second DEV, paired with the champion's 701:

| Seed | Wins | Discordant | McNemar p | Mean floor | Act 1 / 2 / 3 boss pass, paired difference (points) |
|---|---:|---:|---:|---|---|
| 1 | 646 | 350 / 295 | 0.033 | −0.78 [−1.13, −0.42] | +0.4 [−0.4, +1.2] / −0.5 [−2.2, +1.2] / −1.0 [−3.8, +1.7] |
| 2 | 733 | 328 / 360 | 0.24 | +0.19 [−0.16, +0.53] | +1.0 [+0.3, +1.8] / +1.9 [+0.2, +3.6] / +1.4 [−1.4, +4.1] |

- Seed 1 loses before the bosses, not at them: it reaches the Act 1, 2 and
  3 bosses 1.7, 2.4 and 3.7 points less often, while its boss pass rates
  barely move. Seed 2 reaches the bosses as often as the champion and
  passes all three a little more often.
- Fits stopped at steps 3,600 (seed 1, validation loss 1.418) and 6,900
  (seed 2, 1.408). As before, validation loss did not separate the good fit
  from the bad one.
- Exploratory follow-up (chosen after seeing these results): seed 2's
  student in the new champion's out-of-combat head won 927 against the new
  champion's 899 on second DEV (discordant 387/415, p = 0.34; floor −0.07
  [−0.41, +0.28]; Act 2 boss pass +0.6 points [−1.0, +2.2]): no clear
  difference. It is not adopted.

## Conclusion

- A single refit cannot be trusted: three of four fits of one recipe lie
  within half a floor of each other and one fell 2 floors.
- The champion's combat student (round 2, seed 1919) is the best of the
  five fits made; part of round 2's confirmed gain may be a good draw.
- More labels from older, weaker players did not help in one fit.
- Boss-only labels did not clearly help either: one of two fits was worse
  (it lost hallway fights), the other was not clearly better overall,
  though it passed bosses slightly more often. With fits this variable,
  two seeds cannot separate a small benefit from luck; a better teacher
  is the more promising lever ([better teacher](2026-10-22-better-teacher.md)).

Caveats:
- One or two seeds per condition; every comparison on second DEV only.

## Next

- Check every refit before use, and fit at least two seeds per corpus.
- Find a cheap check that flags bad fits before a full evaluation
  (validation loss does not).
- Improve the teacher before collecting more labels
  ([better teacher](2026-10-22-better-teacher.md)).

## Log

- 22 Oct, 16:52: boss-only fits evaluated: seed 1 worse (646), seed 2 no
  clear difference (733 against 701); seed 2 queued in the new champion
  as an exploratory check.
- 22 Oct, 17:03: in the new champion seed 2 won 927 against 899 (p = 0.34):
  no clear difference; not adopted.
- 22 Oct, 13:54: `--mixed-sampling` merged after review; boss-corpus fits
  armed to start when the corpus completes.
- 22 Oct, 13:33: round 3 seed 2 is fine (689 against 701): the round 3
  collapse was fit variance. The all-corpora fit is being evaluated.
- 22 Oct, 12:42: fits, splices and evaluations queued (detached
  `combat_variance.sh`); boss-only labelling started.

Code: `scripts/rl/fresh_train.py` (`--mixed-sources` from worktree
`nonllm-mixed-corpora` until merged), `scripts/rl/combine_heads.py`,
`scripts/rl/fast_eval.py`. Data: runs `combat_variance_20261022` and
`combat_bosses_20261022` (local, worktree `nonllm-large-batch`).
