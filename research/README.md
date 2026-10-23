# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| **Champion from 23 October, 09:58:** joint run 2's final checkpoint (`champion_c8553bc33e53.pt`, from `joint2_20261022/train/iter_0024.pt`; out-of-combat PPO from the previous champion with two gated combat refits) | 26.8% on 4,096 unused worlds from 107,000,000,000 (1,096 against the fourth continued training's 952, p < 0.001); 26.1% on unused worlds from 105,000,000,000 (1,071 against its same-seed control's 998, p = 0.022) | Confirmed; current default |
| The fourth continued training's out-of-combat head with joint run 2's combat head (`joint2_20261022/factorial/c4_ooc_joint_combat.pt`) | 26.4% on the same 107,000,000,000 worlds (1,083; no clear difference from the champion, p = 0.69) | Confirmed; not promoted |
| Fourth continued training's final checkpoint (`continued4_20261022/train/iter_0024.pt`; out-of-combat PPO from the 22 October champion) | 23.7% on 4,096 unused worlds from 104,000,000,000 (971 against that champion's 887, p = 0.005) | Confirmed; not promoted |
| Joint run's final checkpoint (`joint_20261022/train/iter_0024.pt`; out-of-combat PPO from the midday champion with one gated combat refit) | 23.1% on 4,096 unused worlds from 103,000,000,000 (945 against its control's 877, p = 0.029; the control is today's champion) | Planned comparison; not promoted |
| Champion from 22 October, 16:32: the third continued training's final checkpoint (`champion_af2fb9f1f260.pt`; out-of-combat PPO from the previous champion, its round 2 combat student unchanged) | 21.1% on 4,096 unused worlds from 100,000,000,000 (865 against the previous champion's 707, p < 0.001); 21.9% on second DEV (899 against 701) | Confirmed |
| Champion of 22 October, midday: the second continued training's out-of-combat head with the round 2 combat student (`champion_85ea9019ff95.pt`) | 18.4% on 4,096 unused worlds from 99,000,000,000 (755 against the trained checkpoint's 662, p = 0.001); 17.1% on second DEV (701) | Confirmed |

Older players and every comparison are in the [results ledger](results.md). Compare numbers only within one environment version and world set.

The agent loses mainly at bosses: of the new champion's 3,025 losses on
worlds from 105,000,000,000, 41% are at the Act 3 boss and 32% at the Act 2
boss. The [19 October brainstorm](brainstorms/2026-10-19-brainstorming.md)
found weak deck-building (starter cards never removed, few powers, little
variety), and the [deck oracle](experiments/2026-10-20-deck-oracle.md) showed
that deck strength, measured by simulating the boss fight, varies widely and
predicts the real boss outcome far better than HP. Distilled combat helps by
winning hallway fights, not boss fights. Details in [strategies](strategies.md); every finished comparison is
in the [results ledger](results.md).

## Now

Running (see [status](status.md)): an analysis of what joint run 2's
refitted combat head does differently, on matched game states
([joint run 2](experiments/2026-10-22-joint-run-2.md)). First look, from the
two-by-two games: the new combat head dies less at the Act 1 boss and in
Act 2 hallway fights, passes the Act 2 boss more often, drinks potions more,
and does no better at the Act 3 boss.

## Next

1. **Make refit-and-gate cycles standard** beside out-of-combat training, with
   more refits per run (shorter cycles); joint run 2 showed the refits, not
   joint training, carry the gain.
2. **Distil potion decisions at bosses:** the teacher deciding only potion
   drinks lifts the old champion from 205 to 249 of 1,024 boss-reaching games,
   and the champion matches the teacher's drinks only 12% of the time
   ([better teacher](experiments/2026-10-22-better-teacher.md)).
3. **Close the imitation gap:** students agree with the pooled teacher about
   57% of the time against about 80% for a second teacher run, whatever the
   network or training length; test whether missing enemy move history
   explains it.
4. **Learn deck strength instead of simulating each choice** (a deck-value
   network trained on oracle scores). To discuss with Sid.

The agent must stay a pure policy; search may be used to make better
teachers, not to play.

## Open questions

- Can a deck-strength oracle teach deck-building where full-game reward
  could not?
- Why does the old encoder still beat the structured encoder by about 1.4
  points once rule features and size are matched?
- Would a stronger teacher close the boss-fight gap, and is its weakness at
  bosses the fixed sampled future, the random playouts or the score?
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
