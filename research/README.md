# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| **Champion from 20 October:** large-batch out-of-combat PPO from the previous champion, iteration 24 (`champion_ba2fa6a2987d.pt`) | 5.7% on 4,096 fresh worlds (232 against the previous champion's 152; +2.0 points, p < 0.001; mean floor +1.24 [+0.91, +1.58]); 6.7% on the second-DEV worlds (276) | Confirmed; current default |
| New champion with the old-encoder combat student and the Demon Form rule (an overlay, not yet distilled) | 8.4% on 4,096 fresh worlds (346 against the new champion's 232 and the previous champion's 152); 8.1% on the second-DEV worlds (333 against 276; student alone 316, rule alone 292) | Confirmed; the rule must be distilled before this is a pure policy |
| Previous champion: round-2 out-of-combat PPO update 20 | 3.8% (156 of 4,096 second-DEV worlds; 3.7% on the fresh worlds above) | Baseline of every comparison before 20 October |
| Old-encoder combat distilled from the search teacher, with the previous champion's out-of-combat play | 4.8% (197 of the second-DEV worlds; +1.0 points [+0.2, +1.8]); a second training seed 5.4% (221) | Planned; being re-tested on the new champion |
| Champion plus search at Act 2 and 3 bosses and elites | 8.1% (333 of 4,096 fresh worlds, environment 0.1.4) | Confirmed, but search is not a pure policy |

Compare numbers only within one environment version and world set.

The agent loses mainly at bosses, above all the Act 2 boss (43% of the
champion's losses). The [19 October brainstorm](brainstorms/2026-10-19-brainstorming.md)
found weak deck-building (starter cards never removed, few powers, little
variety), and the [deck oracle](experiments/2026-10-20-deck-oracle.md) showed
that deck strength, measured by simulating the boss fight, varies widely and
predicts the real boss outcome far better than HP. Distilled combat helps by
winning hallway fights, not boss fights. Details in [strategies](strategies.md); every finished comparison is
in the [results ledger](results.md).

## Now

Running: [large-batch PPO](experiments/2026-10-20-large-batch-ppo.md)
(the champion's recipe at 3,072 games per iteration). Finished on 20 October:
[boss rares](experiments/2026-10-20-boss-rares.md): forcing Demon Form
whenever offered wins 179 against 156 of 4,096 games (p = 0.010), and 230
against 198 with the old-encoder combat head, the first single deck-building
habit shown to change wins;
[power picks](experiments/2026-10-20-power-picks.md): powers are not
under-picked as a class, the champion's power count is set by whether three
favourite powers are offered, and it skips about two thirds of boss card
rewards. Also landed: in-process PPO collection (`--in-process-games`; 27.5
games/s on 12 collectors against 8.6 for the old recipe), and the finding
that the trainer and `eval_policy.py` let the policy waste potions out of
combat, under which the champion wins 1.7%, not 3.8% (`--hide-ooc-potions`
restores the study protocol). See [status](status.md) and the temporary
[experiment log](log.md).

## Next

1. **Distil the Demon Form rule** into the out-of-combat head, confirm it on
   the confirmation worlds, and continue PPO from it with the potion filter
   on; extend the rule grid to other declined powers and removals.
2. **Exploration for a saturated policy** (discussion with Sid, 20 October):
   track per-menu entropy and rare-action sampling in the trainer; then try
   paired counterfactual exploration or an ε-mixture behaviour policy on
   strategic menus, with a fresh out-of-combat lineage held in reserve.
3. **Learn deck strength instead of simulating each choice.** Train a
   deck-value network on oracle scores from many states, then use it to
   score options. To discuss with Sid.
4. **Large-batch PPO with the hygiene options and the potion filter,**
   after the running recipe arm.
5. Confirm one old-encoder combat student, chosen in advance, on the
   confirmation worlds before promoting it.

The agent must stay a pure policy; search may be used to make better
teachers, not to play.

## Open questions

- Can a deck-strength oracle teach deck-building where full-game reward
  could not?
- Why does the old encoder still beat the structured encoder by about 1.4
  points once rule features and size are matched?
- Would a stronger teacher (visit counts or values instead of the single best
  rollout, a learned rollout policy) close the boss-fight gap?
- Does a recurrent policy (memory across turns) matter? Left for later.

## Pages

- [strategies.md](strategies.md): each approach, its status and evidence.
- [results.md](results.md): every finished comparison, newest first.
- [metrics.md](metrics.md): definitions of metrics, evaluation worlds and terms.
- [experiments/](experiments/): one page per study.
- [brainstorms/](brainstorms/): dated records of strategy discussions.
- [log.md](log.md): temporary running experiment log, newest first; it will be
  deleted.
- [status.md](status.md): what is running now.
- [archive.md](archive.md): where the retired documents went.
