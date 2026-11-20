# Can the fighter and strategist learn together from the same games?

**Status:** finished 20 November 2026. **Verdict:** Learning combat and strategy together gained 2.28 percentage points of whole-game wins over continued strategy training with frozen combat (95% paired interval +1.44 to +3.11).

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

On 19 November Sid requested a broader search while the preceding study was still training. The first pilot results above informed this extension. The screening and confirmation rules below were set before its new games or fits. The launch timing was amended during screening as recorded in the log.

- Cross exploration targets of 5%, 12% and 22% with combat learning rates 0.00001, 0.00003 and 0.00006 and one, two or four passes. These 27 combinations test whether the best update size depends on the alternatives the fighter actually tries. Collect under each exploration rule on the same 3,072 assigned training worlds. Reuse identical first-pilot fits and data where available. All other settings are shared. The original reference is the 12%, 0.00003, two-pass fit.
- Compare all 27 fitted players on the same 768 unused worlds with exact preferred moves. Report each result and the original movement checks. A challenger must change at least 0.2% of meaningful favourite moves and have mean KL of at most 0.05 in each fighter. Pick the distinct eligible setting with the largest worst-case paired win gain over the original choice. Ties favour fewer passes, then a lower learning rate, then an exploration level closer to 12%, then the lower target. This panel is for selection only.
- Restart both finalists at the common starting champion. Give each three real joint updates on the same 1,536 fresh worlds per update, with a new matched training seed and the finalist's own exploration rule. Strategy learns alongside combat. Use the same collector layout and Mac accelerator in both arms. After the third update, freeze both players and compare them on 8,192 further unused full games. This measures a short trajectory at half the production batch size.
- Both 24-update arms wait for the independent comparison on 8,192 worlds. Adopt the challenger if it gained wins in the screen and its worst-case paired 95% lower bound on the independent comparison is positive. Otherwise retain the original setting. Once the verdict is recorded and the earlier study releases the GPU, the coordinator starts the control and then the joint arm. Both use the same selected exploration rule. The production evaluations use separate worlds.

## Results

On independent confirmation, joint training won 6,857 games (41.85% of assigned worlds), frozen combat won 6,484 (39.58%), and the common starting player won 6,458 (39.42%). Both arms trained on the same 73,728 worlds. Their first 3,072 trajectories matched move for move. Only the joint arm changed the ordinary and final-boss fighting weights.

| Frozen full-game comparison | Paired wins and 95% interval |
|---|---|
| Joint against frozen, primary | 6,712 vs 6,502 on 16,384 assigned worlds, +1.28 points [+0.44, +2.12], 16,383 complete pairs |
| Joint against control, independent confirmation | 6,857 vs 6,484 on 16,384 assigned worlds, +2.28 points [+1.44, +3.11], 16,383 complete pairs |
| Control against parent, independent confirmation | 6,484 vs 6,458 on 16,384 assigned worlds, +0.16 points [-0.63, +0.95], 16,383 complete pairs |
| Joint against parent, independent confirmation | 6,857 vs 6,458 on 16,384 assigned worlds, +2.44 points [+1.60, +3.27], 16,384 complete pairs |

One control game reached its decision limit on each 16,384-world panel. The intervals in the table use complete pairs. Even assigning both unresolved outcomes in favour of the control, the lower 95% bound for joint training was +0.43 points in selection and +1.43 in confirmation. There were zero simulator faults. Training had one separately excluded decision-limit game in the joint arm.

**Where the added wins appeared.** The following exploratory counts use the same 16,383 completed confirmation pairs. Each change is a percentage of all paired worlds.

| Boss | Reached, joint vs frozen | Paired reach change in points [95% interval] | Cleared, joint vs frozen | Paired clear change in points [95% interval] |
|---|---|---|---|---|
| Act 1 | 16,000 vs 15,952 | +0.29 [+0.05, +0.54] | 15,338 vs 15,273 | +0.40 [−0.01, +0.80] |
| Act 2 | 14,649 vs 14,621 | +0.17 [−0.33, +0.67] | 11,675 vs 11,204 | +2.87 [+2.13, +3.62] |
| Act 3 | 10,838 vs 10,402 | +2.66 [+1.88, +3.44] | 6,857 vs 6,484 | +2.28 [+1.44, +3.11] |

The two players arrived at the Act 2 boss in similar numbers. The joint player cleared it in 471 more worlds. Changing combat can also change the decks, health and potions the strategist brings to those fights.

**Repeated development checks.** These descriptive measurements use the same 1,024 familiar worlds at every update.

| Update | Joint wins | Frozen wins | Paired win-rate change in points [95% interval] |
|---:|---:|---:|---|
| 0 | 381 | 381 | +0.00 [+0.00, +0.00] |
| 4 | 377 | 363 | +1.37 [-1.70, +4.43] |
| 8 | 372 | 362 | +0.98 [-2.13, +4.09] |
| 12 | 373 | 357 | +1.56 [-1.65, +4.78] |
| 16 | 399 | 360 | +3.81 [+0.61, +7.01] |
| 20 | 389 | 361 | +2.73 [-0.62, +6.09] |
| 24 | 419 | 390 | +2.83 [-0.58, +6.25] |

### Exploration pilot

| Target | Wins / 768 | Reached Act 3 / 768 | Lower-ranked move, ordinary / boss | Unknown |
|---|---:|---:|---|---:|
| 22.0% | 220 | 471 | 20.9% / 21.5% | 0 |
| 5.0% | 276 | 504 | 4.8% / 5.0% | 0 |
| 12.0% | 254 | 492 | 11.6% / 12.1% | 0 |
| 0.0% | 285 | 514 | 0.0% / 0.1% | 0 |

### Combat fitting pilot

| Rate | Passes | Greedy games against frozen player | Changed choices, ordinary / boss | Passed movement screen |
|---:|---:|---|---|---|
| 0.0001 | 1 | 815 vs 800 on 2,048 assigned worlds, +0.73 points [-0.92, +2.38], 2,048 complete pairs | 2.4% / 2.4% | yes |
| 1e-05 | 1 | 811 vs 800 on 2,048 assigned worlds, +0.54 points [-0.52, +1.59], 2,048 complete pairs | 0.6% / 0.7% | yes |
| 3e-05 | 1 | 819 vs 800 on 2,048 assigned worlds, +0.93 points [-0.37, +2.22], 2,048 complete pairs | 1.0% / 0.9% | yes |
| 3e-05 | 2 | 825 vs 800 on 2,048 assigned worlds, +1.22 points [-0.18, +2.62], 2,048 complete pairs | 1.3% / 1.2% | yes |

### Expanded pilot results

All 27 settings played the same 768 screening worlds. The original recipe won 319. Each cell gives wins and the paired difference in percentage points with a 95% interval against that recipe.

| Combat learning rate | Passes | 5% exploration | 12% exploration | 22% exploration |
|---:|---:|---|---|---|
| 0.00001 | 1 | 301, -2.34 [-4.62, -0.07] | 305, -1.82 [-3.80, +0.15] | 305, -1.82 [-4.19, +0.54] |
| 0.00001 | 2 | 302, -2.21 [-4.59, +0.16] | 302, -2.21 [-4.17, -0.26] | 297, -2.86 [-5.22, -0.51] |
| 0.00001 | 4 | 309, -1.30 [-3.67, +1.06] | 314, -0.65 [-2.54, +1.24] | 308, -1.43 [-3.81, +0.95] |
| 0.00003 | 1 | 294, -3.26 [-5.84, -0.67] | 310, -1.17 [-3.32, +0.98] | 308, -1.43 [-3.89, +1.03] |
| 0.00003 | 2 | 304, -1.95 [-4.46, +0.56] | 319, reference | 315, -0.52 [-3.07, +2.03] |
| 0.00003 | 4 | 309, -1.30 [-3.75, +1.15] | 322, +0.39 [-1.73, +2.51] | 309, -1.30 [-4.03, +1.42] |
| 0.00006 | 1 | 306, -1.69 [-4.38, +1.00] | 319, +0.00 [-2.58, +2.58] | 306, -1.69 [-4.40, +1.02] |
| 0.00006 | 2 | 312, -0.91 [-3.40, +1.58] | 318, -0.13 [-2.34, +2.08] | 305, -1.82 [-4.64, +0.99] |
| 0.00006 | 4 | 312, -0.91 [-3.40, +1.58] | 321, +0.26 [-2.34, +2.86] | 306, -1.69 [-4.59, +1.21] |

0 fits failed the movement screen. The screen selected among these fits after one combat update with strategy held fixed. [Full screening data](../figures/online-combat-pilot/screen.csv).

The 27 settings produced 26 distinct players. At 5% exploration and learning rate 0.00006, the per-update limit stopped both fighters early, giving the two- and four-pass settings identical weights.

All 768 reference trajectories reproduced with three active evaluator processes and the same seven world partitions.

Averaging over the other two settings gave these exploratory differences. Each interval uses independent worlds as its unit. A single training collection per exploration level leaves training-run variability unmeasured.

| Factor | Compared with the original level | Mean win-rate change in points, 95% interval | Complete worlds / 768 |
|---|---|---|---:|
| Exploration | 5% vs 12% | -1.17 [-2.56, +0.22] | 768 |
| Exploration | 22% vs 12% | -1.03 [-2.43, +0.37] | 768 |
| Learning rate | 0.00001 vs 0.00003 | -0.68 [-1.55, +0.19] | 768 |
| Learning rate | 0.00006 vs 0.00003 | +0.22 [-0.64, +1.07] | 768 |
| Passes | 1 vs 2 | -0.29 [-1.08, +0.50] | 768 |
| Passes | 4 vs 2 | +0.52 [-0.15, +1.19] | 768 |

![Screening game wins by exploration, learning rate and training passes](../figures/online-combat-pilot/screen.png)

The challenger used 12% exploration, learning rate 0.00003 and 4 pass(es). On 8,192 separate games after three joint updates, challenger against original was 3,242 vs 3,272 on 8,192 assigned worlds, -0.37 points [-1.43, +0.70], 8,192 complete pairs. Unresolved games were 0 faults and 0 cuts for the original, 0 faults and 0 cuts for the challenger.

The original recipe retained its place under the replacement rule.

| Short joint-training update | Original wins / 1,536 training worlds | Challenger wins / 1,536 training worlds |
|---:|---:|---:|
| 1 | 525 | 525 |
| 2 | 550 | 525 |
| 3 | 522 | 530 |

Descriptive checks on the same 128 worlds at updates 0 and 3 gave 43 and 52 wins for the original, 43 and 54 for the challenger. These checks had no role in selection.

The actual 24-update comparison used 12% combat exploration, combat learning rate 0.00003, and 2 pass(es). The selection was fixed from the first-pilot at first launch.

## Conclusion

Joint training beat both the frozen-combat control and the common starting player on fresh worlds. Strategy training alone showed no clear gain over the starting player at this exploration level. This comparison used one matched training seed.

## Next

The [early-refit player](2026-11-19-refit-timing.md) became workstream champion after this study began. A retrospective check on the same 1,024 development worlds gave 419 joint wins vs 365 for early refits, a paired difference of +5.27 points [+1.86, +8.68]. The [championship comparison](2026-11-20-joint-vs-early-refit.md) uses 16,384 independent worlds.

## Log

- 19 November, 13:32 GMT — At Sid’s suggestion, made the independent verdict a prerequisite for either 24-update arm. The earlier plan would have used the original settings if the GPU became available first. Thirteen of the 27 settings had finished screening, and the independent comparison had yet to begin.

- 19 November — after screening began, added descriptive averages across each factor. These compare all 27 settings within each world and do not affect selection.

- 19 November — one earlier trainer update took 735 seconds during overlapping pilot work, against 282 seconds before it. Reduced supplemental concurrency and added a 768-world trajectory check for the lower-memory evaluator.

- 19 November — chose the starting player, comparison and pilot rules before sampling.
- 19 November — Sid would prefer one general fighter long term and asked to keep the existing specialist for this comparison. A separate study could distil both fighters into one.

Code: `experiments/2026-11-19-online-combat/` at `d70feee` for the control and `bb94cc3` for joint training (same trainer source). Data: local run `online_combat_from_9700`.
