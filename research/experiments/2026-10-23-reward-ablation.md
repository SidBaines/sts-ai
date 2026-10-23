# Reward ablation: floors and wins against wins only

**Status:** running (started 23 October, 12:13). **Verdict:** none yet.

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

## Conclusion

## Next

## Log

- 23 Oct, 12:13: both arms started. Before launch: the new trainer options
  were checked on real games (with floor weight 0 every return is exactly
  ±1), and 1 plus 1 resumed iterations gave bit-identical weights to 2
  straight, so resuming to 24 is the same as running 24 from the start.
Code: trainer options `--progress-floor-weight` and
`--reset-return-normalizer` (commit 8a78301, run from worktree
`nonllm-reward-ablation`; do not edit it). Data: run `reward_ablation_20261023` (local, worktree `nonllm-large-batch`).
