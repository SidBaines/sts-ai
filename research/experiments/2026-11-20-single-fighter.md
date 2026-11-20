# One fighter for every encounter

**Status** — running on 20 November 2026. The early-refit champion is teaching one fighter to play ordinary combat and the Act 3 boss.

## Question

Can one combat network retain the play strength of the [two-fighter champion](2026-11-20-joint-vs-early-refit.md)? Keeping its strategy network fixed isolates the change to combat.

## Design

Sid requested the experiment on 20 November. The following budgets, selection procedure and decision thresholds are study choices made before the run.

- Let the frozen champion choose its preferred moves in 4,096 training worlds. Sample up to 48 ordinary combat choices and 96 Act 3 boss choices per completed world. Exclude all examples from unresolved games. Use 512 separate worlds for imitation validation.
- Start the student from the ordinary fighter. Teach it the active teacher's scores over legal moves, with a temperature for each teacher calibrated on training data. The student has one combat network with the original public features and one fixed strategy network. It can see act and room in the usual game state.
- Try minibatches with 25% and 50% final-boss examples. Select the training step within each fit by the equally weighted validation loss for the two types of combat. Select between those two saved players by full-game wins on the same 2,048 development worlds, assigning unresolved games against the challenger. Ties retain the 25% fit. Freeze the finalist before opening either performance test.
- Play the finalist and champion on the same 16,384 fresh Ironclad worlds, Ascension 0 through Act 3. Use exact preferred moves, the same seven evaluator processes and eight games per process. Compare wins within worlds with a two-sided paired 95% interval. Report incomplete games separately.
- From 2,048 other champion games freeze 1,024 distinct Act 3 boss entries, one per world. Both fighters get the same deck, health, relics, potions and starting randomness. Original entries are the boss outcome. Repeat each entry with one matched new draw as a diagnostic and report it separately.
- For this study, call strength retained if the lower 95% bounds clear **−1 percentage point on full games** and **−3 points on original boss entries**. These are practical tolerances chosen for the experiment. For champion promotion require a positive full-game lower bound, robust to incomplete games, and the boss retention criterion. Otherwise the incumbent keeps its title.

## Next

Compare both outcomes before deciding how to use the single fighter.

The settings and code are in `experiments/2026-11-20-single-fighter/`. Local run `single_fighter_from_f3c8`.
