# Combat student, round 3: another expert-iteration step

**Status:** running. **Verdict:** pending.

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

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 21 Oct, 19:48: labelling started (eight shards); a detached pipeline fits,
  splices and evaluates when the corpus is complete.

Code: as round 2. Data: run `combat_round3_20261021` (local, worktree
`nonllm-large-batch`).
