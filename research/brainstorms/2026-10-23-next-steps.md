# Discussion, 23 October 2026: where we are and what to do next

Written by Claude on 23 October 2026, after joint run 2's final checkpoint
was promoted, at Sid's request: a step back over everything so far, the
current training setup, where the headroom is and the options for
continuing. Section 6 records what Sid decided.

## 1. Summary

- The champion wins 26.8% of fresh games (Ironclad, Ascension 0), up from
  about 5% three days earlier. Almost all of the gain came from two
  mechanisms: out-of-combat PPO on whole games, and refitting the combat
  network on search-teacher labels from the current player's own fights.
- 80% of the champion's losses are at bosses, 41% at the Act 3 boss alone.
- The largest measured headroom is combat at bosses: the pooled search
  teacher, playing only the boss fights, roughly doubles the win rate. But
  students copy it poorly, and more data helps slowly. Imitation, not the
  teacher, is the bottleneck.
- Out-of-combat PPO still works, but each round gains less.
- Recommendation: keep PPO with refit cycles running in the background as
  the steady source of gains, and make the boss imitation gap the main
  research thread.

## 2. Progress so far

Each step was confirmed on unused worlds:

| Step | Wins on unused worlds |
|---|---|
| Round 3 (old-encoder combat student, large-batch PPO) | 9.6% |
| First continued training | 11.6% |
| Second continued training | 12.4% → 16.9% |
| Third continued training | 17.3% → 21.1% |
| Fourth continued training | 21.7% → 23.7% |
| Joint run 2 (two gated combat refits) | 26.8% |

## 3. The training setup

The player is two networks spliced together. At play time it uses only
public game information; search is never used to play.

**Combat network.**
- The old hand-built feature encoder with a small network on top. Each new
  version is trained from scratch, by supervised learning, to copy the
  search teacher.
- The teacher: the native search runs 24 searches of 1,000 simulations each,
  on copies of the game whose draw order and random rolls are re-randomised.
  The training target is the spread of the 24 votes, using the old "luckiest
  line" vote (each search votes for the first move of its best playout).
- Expert iteration: the labels come from the current player's own games.
- Each refit is gated before adoption: it must win at least as many of 2,048
  reused worlds, with at least the same mean floor. The gate is needed
  because about one fit in four comes out badly.
- Cost: about 1.7 hours to label 1,000 worlds, and minutes per fit.

**Out-of-combat network.**
- PPO on whole games, with the combat network frozen and playing greedily.
  Credit for what happens between two out-of-combat decisions goes to the
  earlier decision.
- Reward: +1 for a win and −1 for a loss at the end, plus 0.05 per floor
  climbed. No discounting (gamma 1, lambda 1).
- Settings: learning rate 0.0003; entropy bonus 0.01 on strategic menus;
  potion filter; dead-action filter; normalised returns; detached critic.
- Built in: two strategy rules distilled earlier (upgrade at rest sites
  when HP is at least 85%; take elites when HP is at least 90%).
- One round is 24 iterations of 3,072 games, about 4.3 hours on this laptop.

**Joint loop.** The same PPO, plus a cycle that labels the current player's
fights, fits two students, gates them and swaps in a winner, about every 8
iterations.

**Evaluation.** 4,096 paired worlds with an exact McNemar test, which
resolves differences of about ±1.3 points.

**Compute.** Everything runs on one laptop, 8 to 12 processes at a time.

## 4. Where the remaining losses are

Of the champion's 3,025 losses on 4,096 worlds from 105,000,000,000:

| Where | Share of losses | Pass rate for games that get there |
|---|---|---|
| Act 3 boss | 41% | about 47%; the combat refits did not move it |
| Act 2 boss | 32% | 73% |
| Act 1 boss | 8% | 94% |
| Hallways and elites | 20% | |

## 5. Headroom by lever

1. **Combat at bosses: by far the largest measured headroom, but we cannot
   capture it yet.**
   - When the pooled search teacher plays only the boss fights, the previous
     champion goes from 20% to 42% of 1,024 worlds, and to 44% with twice
     the search budget.
   - Students agree with that teacher 66%, 54% and 46% of the time at the
     Act 1, 2 and 3 bosses. A second run of the teacher agrees with the
     first 87%, 78% and 77% of the time.
   - Agreement grows slowly with data: 47% at 5,000 rows, 51% at 20,000,
     57% at 134,000. Longer training and the structured transformer encoder
     made no difference.
   - The two refits in joint run 2 moved overall agreement with the pooled
     teacher by only 0.7 points, and by 0.5 at the Act 3 boss (not clearly
     different from zero).
   - Potions are the one concrete, separable piece. Letting the teacher
     decide only the potion drinks is worth about 4.3 points of wins. Where
     a drink is allowed, the teacher drinks 18% of the time and the new
     champion 11%, up from 6% before the refits.
2. **Out-of-combat PPO: still working, but returns are falling.** The last
   three rounds gained +4.5, +3.9 and +2.1 points. Entropy is low, and 78% of
   card-reward decisions have one option above 99% probability, so PPO
   cannot find rare choices by sampling. Expect a few more points, at a
   falling rate.
3. **Deck-building: unknown headroom, possibly large, and no cheap lever
   found.** Strong human players win most Ascension 0 games, so the gap is
   very likely mostly strategic. The deck oracle predicts boss outcomes well
   (AUC 0.78) but cannot choose between individual card rewards. Forcing
   Demon Form whenever offered was the only habit that measurably added wins
   (179 against 156 of 4,096).
4. **More refit cycles.** Each of the two adopted refits passed its gate.
   Whether later refits keep paying has not been measured; scoring each
   refit separately is cheap.

### Two things worth questioning in the setup

- **The reward mixes floors with wins.** 0.05 per floor adds up to about 2.5
  over a full game, against a ±1 terminal reward. That made sense when wins
  were rare. At 27% wins it may now push the policy toward surviving longer
  rather than winning: for example, resting rather than upgrading, which
  helps at the Act 1 boss but not at the Act 3 boss.
- **The combat targets throw away most of what the teacher computes.** We
  train on vote counts, but each search also records how good every move
  looked (its mean value). Training on those values would give the student
  a ranking of all moves at every position instead of a noisy single
  choice: the most direct attack on the data-efficiency problem. The
  134,000-row pooled corpus already holds the values, so this needs no new
  labels.

### Options

| Option | Cost | Expected gain | Risk |
|---|---|---|---|
| A. Keep going: another PPO round from the champion, refits every 4 iterations | about 5 hours, unattended | +2 to 3 points, falling each round | low |
| B. Potion distillation: dedicated potion labels at bosses, weighted up or with their own small head | a day | up to about 4 points | low to medium |
| C1. Train the combat student on the teacher's move values instead of votes | hours, existing data | unknown; tests the imitation bottleneck directly | low |
| C2. Missing information: add enemy move history and known top cards to the observation | days; needs a new encoder version, because the existing encoder files are hash-bound | unknown; tests why students cannot copy the teacher | medium |
| D. Reward ablation: floor weight 0.05 against lower | about 5 hours per arm | unknown | low |
| E. Deck-value network for shaping or menu choices | a week or more | potentially the largest long-term gain | high |
| F. Rent a many-core CPU machine for labelling | setup, a Linux build and determinism checks | 5 to 10 times the labels per day; helps B and C | medium; costs money |

### Recommendation

There is still plenty of headroom, but it is in combat imitation at
bosses, not in more PPO.

1. Keep option A running in the background all the time, with the reward
   ablation (D) as its second arm, since both are the same kind of job.
2. Make the boss imitation gap the main research thread: C1 first (the
   cheapest and most informative), then potion distillation (B), then the
   missing-information test (C2) if C1 does not close much of the gap.
3. Defer the deck-value network until combat imitation plateaus.
4. Rent compute only if C1 or B shows that more labels are what limits us.

Least certain: the Act 3 boss is now the biggest killer and the refits did
not touch it. That could be combat imitation (agreement is lowest there, at
46%) or decks that are not strong enough by Act 3. The teacher benchmark
points to combat: with the same decks, the teacher passes the Act 3 boss
much more often.

## 6. Decisions

- 23 October: Sid asked to start with the reward ablation (option D): two
  arms of 12 iterations each from the new champion, one with the current
  reward and one rewarded only for wins (floor weight 0), with the option
  of resuming both to 24 iterations if 12 does not separate them. See the
  [reward ablation page](../experiments/2026-10-23-reward-ablation.md).
