# Reward ablation: floors and wins against wins only

**Status:** finished 23 October. **Verdict:** rewarding only wins is better than wins plus floors (+1.0 points of wins after 12 iterations, p = 0.008, under the rule fixed in advance).

## Question

Out-of-combat PPO is rewarded +1 for a win and −1 for a loss, plus 0.05
per floor climbed (about 2.5 over a full game). The floor term was chosen
when wins were rare. At 27% wins it may now push the policy toward surviving
longer rather than winning. Does training on the win signal alone do better
from the current champion? Part of the
[23 October discussion](../brainstorms/2026-10-23-next-steps.md); approach:
[out-of-combat PPO](../strategies.md#out-of-combat-ppo-with-macro-credit).

## Design

- **Start:** the champion (`champion_c8553bc33e53.pt`, joint run 2's final
  checkpoint). The combat head stays fixed: no refits, so the arms differ
  only in reward.
- **Arms** (`reward_ablation_20261023`), run at the same time:
  - *Floors and wins:* the current reward (floor weight 0.05).
  - *Wins only:* floor weight 0.
- **Fixed:** the fourth continued training's settings (3,072 games per
  iteration, learning rate 0.0003, lambda 1, weight decay 0, potion filter,
  detached critic, normalised returns, entropy bonus 0.01 on strategic
  menus, evaluation every 4 iterations), training seed 6 in both arms. Both
  arms see the same training worlds, and the first iteration's games are
  identical.
- **Return scale:** both arms start their return normaliser empty rather
  than inheriting the champion's statistics, which were measured under the
  old reward (standard deviation 1.14; a win-only return has about 0.89).
  The critic is inherited in both arms, so the wins-only arm starts with a
  miscalibrated critic and must relearn it: this biases the comparison
  slightly against it.
- **Budget:** 12 iterations per arm (36,864 games each), about 4.5 hours
  with both arms sharing the laptop.
- **Evaluation:** each arm's iteration-12 checkpoint and the champion play
  16,384 unused worlds from 108,000,000,000 (study protocol, paired by
  world). Evaluation takes minutes, so the larger world set is cheap; it
  resolves differences of about ±0.6 points rather than ±1.3. Iterations 4
  and 8 are also played on the same worlds, as an exploratory trajectory.
- **Decision rule, fixed in advance:** wins only is *Better* (or *Worse*)
  than floors and wins if it wins more (or fewer) games at iteration 12 with
  exact McNemar p < 0.025. Otherwise both arms resume to iteration 24 and
  the same test, again at p < 0.025 (the threshold is halved for the two
  looks), is made on 16,384 unused worlds from 109,000,000,000. If neither
  look separates them: no clear difference, and keep the current reward.
- **Secondary:** each arm against the champion; mean floor; where games end
  (by act, boss or not); rest-site upgrades, elite choices and card skips.
- **Approval:** Sid, 23 October.

## Results

**Main comparison (rule fixed in advance): wins only beats wins plus
floors.** 16,384 unused worlds from 108,000,000,000, environment 0.1.5.1,
paired by world, iteration-12 checkpoints:

| Player | Wins | vs champion | vs wins plus floors |
|---|---:|---|---|
| Champion (start) | 4,372 (26.7%) | | |
| Wins plus floors, iteration 12 | 4,468 (27.3%) | +0.6 points (p = 0.11) | |
| Wins only, iteration 12 | 4,632 (28.3%) | +1.6 points (p < 0.001) | **+1.0 points [+0.3, +1.7]** (discordant 1,975/1,811, p = 0.008) |

Trajectory on the same worlds (exploratory): wins only gained +0.8, +1.2
and +1.6 points over the champion at iterations 4, 8 and 12; wins plus
floors +0.6, +0.7 and +0.6. Mean floor did not change in either arm.

**Where the difference comes from (exploratory).** How the games end:

| Games ending at | Champion | Wins plus floors | Wins only |
|---|---:|---:|---:|
| Act 1, hallways | 499 | 456 | 541 |
| Act 1 boss | 848 | 924 | 872 |
| Act 2, hallways | 829 | 772 | 826 |
| Act 2 boss | 3,808 | 3,818 | 3,694 |
| Act 3, hallways | 1,050 | 1,030 | 1,023 |
| Act 3 boss | 4,978 | 4,916 | 4,796 |
| Win | 4,372 | 4,468 | 4,632 |

- Wins only passes the Act 3 boss more often (49.1% against 47.6% of games
  reaching it) and the Act 2 boss slightly more often (73.9% against 73.2%),
  while losing slightly more games in Act 1 hallways: it trades early
  survival, which the floor reward pays for, for late wins.
- The guess behind the study was wrong in detail: wins only does *not*
  upgrade more at rest sites. It upgrades less (55% against 60% of rest
  sites with at least half HP; 10% against 15% below half HP) and enters
  the Act 1 boss with more HP (77% against 74% of maximum).

## Conclusion

At 27% wins, the floor reward no longer helps out-of-combat PPO: training on
wins alone improved the champion about 2.5 times as fast over 12 iterations
and won 1.0 points more than the current reward. The floor reward appears to
pay for surviving early at the expense of beating the late bosses.

Caveats:
- One training seed per arm; the arms shared training worlds and the first
  iteration, so the difference is the reward, but training noise after that
  is not replicated.
- Only 12 iterations; whether the gap keeps growing is not known.
- The behavioural explanation is exploratory.

## Next

- Use wins-only reward for out-of-combat training from now on.
- Resume the wins-only arm to 24 iterations (resuming is exact) to see
  whether it keeps improving; its iteration-12 checkpoint is a promotion
  candidate (Sid's decision).

## Log

- 23 Oct, 18:24: decision at iteration 12: wins only better (4,632 against
  4,468 of 16,384, p = 0.008); no 24-iteration look needed.
- 23 Oct, 12:13: both arms started. Before launch: the new trainer options
  were checked on real games (with floor weight 0 every return is exactly
  ±1), and 1 plus 1 resumed iterations gave bit-identical weights to 2
  straight, so resuming to 24 is the same as running 24 from the start.

Code: trainer options `--progress-floor-weight` and
`--reset-return-normalizer` (commit 8a78301, run from worktree
`nonllm-reward-ablation`; do not edit it). Data: run `reward_ablation_20261023` (local, worktree `nonllm-large-batch`).
