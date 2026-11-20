# Longer imitation and learning from game outcomes with one or two fighters

**Status** — queued 20 November 2026. The original one-fighter student and early-refit champion are frozen references.

## Question

Can more practice improve the [single fighter](2026-11-20-single-fighter.md), and which player produces the stronger successor when both combat and strategy learn from full games?

## Design

Sid requested the full overnight sequence on 20 November. The budgets and rules below were fixed before the new results.

1. Reproduce the 25% boss fit from the ordinary fighter with the original demonstrations, seed, learning rate 0.0001 and batches of 256. Compare the archived 3,200-update student and fixed 6,400- and 12,800-update students on 4,096 new matched worlds. Verify the replayed 3,200-update weights. Rank by worst-case additional wins over the archived student, keeping the shorter fit on ties.
2. If imitation loss improves and the best such longer fit's game interval includes zero, with no improved longer fit showing a positive lower bound, run one corrective round. That longer student plays 2,048 new training games. The frozen champion labels the moves on the student's own positions. Add those examples to the original demonstrations, preserve the 25% boss mix and train for up to 3,200 further updates. Compare it with the first nominee on another 4,096 matched worlds. Use it only for a positive worst-case win gain. Freeze the selected single fighter.
3. Compare the frozen student and original champion on 49,152 unused worlds. Separately freeze 1,024 Act 3 boss entries from 2,048 new champion games. Score original starts and one matched reseed. Retaining strength requires paired 95% lower bounds above −1 percentage point in full games and −3 on original boss fights. Promotion also requires a positive full-game lower bound. Unresolved games are assigned against the candidate for decisions.
4. Give the selected single fighter and the original two-fighter champion 24 joint strategy and combat updates on the same 73,728 new training worlds, regardless of the retention verdict. Use 3,072 games per update, strategy rate 0.0001 and three passes, combat rate 0.00003 and two passes. Calibrate both architectures to 12% lower-ranked combat moves in each fight category on a common set of public menus. Keep distinct ordinary and boss samples in both arms. Only the two-fighter player has separate combat weights.
5. Evaluate both update-24 players and both starting parents on 16,384 new matched worlds. Report the direct comparison and each player's change from its own start. Select the one-fighter endpoint only for a positive worst-case gain over the two-fighter endpoint. Confirm the nominee against the incumbent after step 3 on another 16,384 new worlds. Promote only for a positive conservative paired 95% lower bound. This is one matched training seed.

## Next

The queue runs the sequence on AC power. The [dashboard](http://127.0.0.1:8795/) and [progress log](http://127.0.0.1:8795/progress) show the live stage. The local run is `overnight_single_fighter_20261120` and the study code is in `experiments/2026-11-20-overnight-single-fighter/`.
