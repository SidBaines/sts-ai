# Can the fighter and strategist learn together from the same games?

**Status:** running. The experiment compares enabling and freezing combat learning under the same measured exploration rule.

## Question

Does updating the card player on every batch improve whole-game wins over the same strategy training? The [previous direct trial](2026-10-14-joint-sparse-win-ppo.md) averaged 36 winning games per update. This champion already wins about 39% of full games.

## Design

Sid requested 24 matched updates and a pilot on 19 November, from the confirmed 39.42% player that started the equal-budget refit study. The choices below are the experimenter's, fixed before pilot results.

- Sample alternative legal moves according to the fighter's scores. Calibrate ordinary fights and final-boss fights separately on 192 training worlds. Aim for 5%, 12% and 22% probability of a strictly lower-ranked move on menus with a meaningful choice. Equal best scores count as a tie. Compare those three rules with the existing near-greedy fighter on 768 matched worlds each, including whole-game wins, Act 3 reach, faults and achieved exploration. Prefer 12%, then 5%, then 22%, among settings within five points of the baseline win rate and eight points of its Act 3 reach. If all fail, test a 2.5% target before selection. If none meets the thresholds, report the tradeoff and decide before launching production. These are screening thresholds, not population claims.
- Fit combat from a common frozen collection at the selected exploration rule, using 3,072 further training worlds. Learn from complete-game returns under the existing floor and terminal rewards. Estimate the expected return from other worlds in the same batch using public act, room and health. Sample at most 32 ordinary and 16 final-boss choices per game to bound memory and give the boss its own update. Train both combat policies from their current weights, with independent optimizers. Compare rates 0.00001, 0.00003 and 0.0001 with one pass, and 0.00003 with two passes. The frozen champion is the zero-rate reference. Hold strategy fixed for this screen.
- Evaluate all five on 2,048 common unused pilot worlds. Measure changes to preferred moves, action probabilities and update size on separate worlds. Select the best whole-game win count among fits with changes to at least 0.2% of meaningful preferred moves in both combat players and a mean change in action-distribution KL no greater than 0.05 in each. Prefer smaller rate, then fewer passes on ties. If the initial rates all fail the movement checks, tune a smaller or larger rate before production. A small pilot win difference cannot establish an optimal setting.
- Restart both arms at the original champion. Train strategy for 24 updates at 0.0001, three passes and 3,072 matched fresh worlds per update. Use identical combat exploration temperatures, sampling layout, world rosters, rewards, and strategy settings. Update both fighters every iteration in the joint arm. Freeze both in the control. The pilot weights do not enter the main runs. No search teacher is used.
- Freeze both update-24 endpoints and compare their exact preferred play on 16,384 matched unused worlds. Nominate joint only for a positive whole-cohort win gain, otherwise the control. On a second independent 16,384 worlds compare both endpoints and the starting champion. A schedule advantage requires matching directions and a strictly positive worst-case paired 95% lower bound for the winner. A successor must beat the starting champion on independent confirmation. Report simulator faults and decision limits separately. If another study has since promoted a different incumbent, a win over this parent alone cannot replace it.
- Periodic greedy checks on the familiar 1,024 worlds describe learning only. Compare wall time, actual exploration and Act 3 reach as well as greedy wins. There is one matched training seed. The frozen arm measures the value of enabling combat learning at the chosen exploration level.

### Expanded pilot

On 19 November Sid requested a broader search while the preceding study was still training. The first pilot results above informed this extension. The following rules were set before its new games or fits.

- Cross exploration targets of 5%, 12% and 22% with combat learning rates 0.00001, 0.00003 and 0.00006 and one, two or four passes. These 27 combinations test whether the best update size depends on the alternatives the fighter actually tries. Collect under each exploration rule on the same 3,072 assigned training worlds. Reuse identical first-pilot fits and data where available. All other settings are shared. The original reference is the 12%, 0.00003, two-pass fit.
- Compare all 27 fitted players on the same 768 unused worlds with exact preferred moves. Report each result and the original movement checks. A challenger must change at least 0.2% of meaningful favourite moves and have mean KL of at most 0.05 in each fighter. Pick the distinct eligible setting with the largest worst-case paired win gain over the original choice. Ties favour fewer passes, then a lower learning rate, then an exploration level closer to 12%, then the lower target. This panel is for selection only.
- Restart both finalists at the common starting champion. Give each three real joint updates on the same 1,536 fresh worlds per update, with a new matched training seed and the finalist's own exploration rule. Strategy learns alongside combat. Use the same collector layout and Mac accelerator in both arms. After the third update, freeze both players and compare them on 8,192 further unused full games. This measures a short trajectory at half the production batch size.
- Adopt the challenger for the 24-update run only if it gained wins in the screen, its worst-case paired 95% lower bound in the independent comparison is positive, and that result is available before the first production arm starts. Otherwise use the original setting for both arms. The 24-update comparison can start when the Mac becomes available. Late results inform future training. The primary and confirmation worlds reserved for the 24-update comparison stay separate.

## Results

The moderate rule produced measurable exploration in both roles. On 768 matched training-mode games per rule, all worlds completed. The intervals describe paired percentage-point changes in wins relative to near-greedy combat.

| Target | Lower-ranked move, ordinary / boss | Wins / 768 | Paired win change, 95% interval | Reached Act 3 / 768 |
|---|---|---:|---|---:|
| Near greedy | 0.01% / 0.10% | 285 | Reference | 514 |
| 5% | 4.80% / 5.02% | 276 | −1.17 [−4.85, +2.50] | 504 |
| 12% | 11.59% / 12.14% | 254 | −4.04 [−7.94, −0.13] | 492 |
| 22% | 20.88% / 21.47% | 220 | −8.46 [−12.33, −4.59] | 471 |

The 12% target met both screening thresholds. Its common learning-rate collection yielded 1,082 wins in 3,072 training games, with no unresolved outcomes. The 48-game wiring check found identical training trajectories and strategy updates, with both fighters changing only in the joint arm. The four fits and the unchanged player then chose their preferred moves on 2,048 separate worlds.


| Combat learning rate | Passes | Wins / 2,048 (starting player 800) | Paired gain in points, 95% interval | Changed preferred moves, ordinary / boss |
|---:|---:|---:|---|---|
| 0.00001 | 1 | 811 | +0.54 [-0.52, +1.59] | 0.62% / 0.73% |
| 0.00003 | 1 | 819 | +0.93 [-0.37, +2.22] | 0.98% / 0.90% |
| 0.00003 | 2 | 825 | +1.22 [-0.18, +2.62] | 1.33% / 1.24% |
| 0.00010 | 1 | 815 | +0.73 [-0.92, +2.38] | 2.36% / 2.42% |

The selection rule chose 0.00003 and two passes. All four comparisons were inconclusive individually. All 2,048 worlds completed for every player. The 0.0001 setting hit its update-size guard early for the boss fighter. The two-pass setting completed both passes for both fighters.

On the same 3,072 training worlds with the fighter still frozen, 5%, 12% and 22% exploration yielded 1,105, 1,082 and 905 wins. Relative to 12%, the paired differences were +0.75 points [−1.19, +2.68] for 5% and −5.76 [−7.65, −3.87] for 22%. These are the outcomes while gathering experience for the fits. All games finished.

## Log

- 19 November — after screening began, added descriptive averages across each factor. These compare all 27 settings within each world and do not affect selection.

- 19 November — one earlier trainer update took 735 seconds during overlapping pilot work, against 282 seconds before it. Reduced supplemental concurrency and added a 768-world trajectory check for the lower-memory evaluator.

- 19 November — chose the starting player, comparison and pilot rules before sampling.
- 19 November — Sid would prefer one general fighter long term and asked to keep the existing specialist for this comparison. A separate study could distil both fighters into one.

Code: `experiments/2026-11-19-online-combat/`. Data: local run `online_combat_from_9700`.
