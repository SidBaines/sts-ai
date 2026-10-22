# Combat student, round 3: another expert-iteration step

**Status:** finished 22 October 2026. **Verdict:** worse, sharply: the round 3 student loses 2 floors and about 3.7 points of wins against the player whose games it was fitted to. The cause (fit variance or the data) is being tested.

## Question

[Round 2](2026-10-21-combat-student-round-2.md) raised wins from 12.5% to
13.9% on unused worlds by refitting the combat student to teacher labels
from the champion's own games. Does a third round, labelling the round 2
candidate's games, gain again, or does the loop flatten? (Approach:
[expert iteration for combat](../strategies.md).)

## Design

- **Player:** the round 2 candidate (`candidate_623b01850482.pt`: the
  afternoon champion's out-of-combat head with the round 2 student).
- **Labels, student, fitting:** exactly as round 2: 2,000 new training
  worlds from 90,500,200,000 in eight shards, the same teacher and room
  probabilities, the new corpus only, the 15 October fitting recipe, 8
  probe worlds from 93,000,000,200.
- **Evaluation:** the new student spliced into the candidate; `fast_eval.py`
  on second DEV, paired with the candidate's 576 wins; candidate and new
  checkpoint both on 4,096 unused worlds from 98,000,000,000.
- **Decision rule, fixed in advance:** *Better* if it wins more than the
  candidate on second DEV with exact McNemar p < 0.05, confirmed if it also
  wins more on the worlds from 98,000,000,000 with p < 0.05.
- **Approval:** Sid's standing instruction of 21 October to keep doing
  research until stopped. The round 2 candidate's promotion is still Sid's
  decision; this round uses it only as a starting point.

## Results

**The round 3 student is much worse than the round 2 student it was meant
to improve on.** Spliced into the round 2 candidate (which carries the
round 2 student), paired:

| Worlds | Wins | Discordant, candidate only / round 3 only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference |
|---|---:|---:|---:|---|---|
| Second DEV | 576 → 438 (14.1% → 10.7%) | 357 / 219 | < 0.001 | −2.08 [−2.45, −1.71] | −5.2 points [−6.9, −3.4] (3,037 worlds) |
| From 98,000,000,000 (unused before) | 547 → 388 (13.4% → 9.5%) | 339 / 180 | < 0.001 | −1.89 [−2.26, −1.53] | −4.1 points [−5.9, −2.2] (3,011 worlds) |

- Corpus: 2,000 worlds, no faults. Fit: 121,981 fit rows, early-stopped
  at step 3,300 (best validation loss 1.47; round 2's fit stopped at 4,900
  with 1.41).
- On the worlds from 98,000,000,000 one of 4,096 worlds was not completed
  by the round 3 checkpoint (4,095 paired).

## Conclusion

- The same recipe on a corpus from a stronger player produced a much
  weaker student. Expert iteration is not monotone here.
- Not yet known whether this is fit-to-fit variance (the 15 October study
  found students with equal teacher agreement playing very differently) or
  something about this corpus. If variance, round 2's gain may owe part of
  its size to a lucky fit; the round 2 checkpoint itself stays confirmed.

Caveats:
- One fit per corpus so far; no seed variation.

## Next

A fit-variance check (round 2 and round 3 corpora refitted with a second
seed, every student spliced into the current champion) and a fit on all
three corpora together, recorded on the
[2026-10-22 combat variance page](2026-10-22-combat-variance.md).

## Log

- 22 Oct, 12:42: round 3 student much worse on both world sets (438
  against 576; 388 against 547).
- 21 Oct, 19:48: labelling started (eight shards); a detached pipeline fits,
  splices and evaluates when the corpus is complete.

Code: as round 2. Data: run `combat_round3_20261021` (local, worktree
`nonllm-large-batch`).
