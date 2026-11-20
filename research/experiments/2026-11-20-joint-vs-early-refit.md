# Joint training against the early-refit champion

**Status** — queued 20 November 2026. Two frozen players will play the same 16,384 unused worlds to decide which should be champion.

## Question

Does the [player that learned combat and strategy together](2026-11-19-online-combat.md) win more full games than the [current champion that learned a new final-boss fighter](2026-11-19-refit-timing.md)? Both descend from the same player. Their previous confirmation results came from different worlds. Joint won 419 games to early's 365 on the 1,024 familiar development worlds.

## Design

Sid requested this head-to-head on 20 November. Both players use their frozen update-24 weights. Each plays the same 16,384 previously unused Ironclad worlds on Ascension 0, Acts 1 to 3. Each chooses its highest-scoring legal move. Both use the same environment, menu rules, native build, seven evaluator processes and eight games per process.

The primary outcome is the difference in full-game wins, paired by world. Before opening the panel we chose the workstream's usual promotion rule. Promote joint only if its two-sided paired 95% interval has a positive lower bound even with unresolved outcomes assigned against it. Otherwise retain the incumbent. Report wins unique to each player, faults, decision limits and the number of complete pairs. This comparison can select between these two saved players. Each came from one training run.

## Next

Use the result to choose the starting player for the next study with Sid.

The code and settings are in `experiments/2026-11-20-joint-vs-early-refit/`. The local run is `joint_vs_early_refit_20261120`.
