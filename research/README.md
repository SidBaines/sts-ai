# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| Fourth continued training's final checkpoint (`continued4_20261022/train/iter_0024.pt`; out-of-combat PPO from the champion) | 23.7% on 4,096 unused worlds from 104,000,000,000 (971 against the champion's 887, p = 0.005); 23.9% on second DEV (980 against 899) | Confirmed; pure policy; awaiting Sid's promotion decision |
| Joint run's final checkpoint (`joint_20261022/train/iter_0024.pt`; out-of-combat PPO from the midday champion with one gated combat refit) | 23.1% on 4,096 unused worlds from 103,000,000,000 (945 against its control's 877, p = 0.029; the control is today's champion) | Planned comparison; not promoted |
| **Champion from 22 October, 16:32:** the third continued training's final checkpoint (`champion_af2fb9f1f260.pt`; out-of-combat PPO from the previous champion, its round 2 combat student unchanged) | 21.1% on 4,096 unused worlds from 100,000,000,000 (865 against the previous champion's 707, p < 0.001); 21.9% on second DEV (899 against 701) | Confirmed; current default |
| Champion of 22 October, midday: the second continued training's out-of-combat head with the round 2 combat student (`champion_85ea9019ff95.pt`) | 18.4% on 4,096 unused worlds from 99,000,000,000 (755 against the trained checkpoint's 662, p = 0.001); 17.1% on second DEV (701) | Confirmed |

Older players and every comparison are in the [results ledger](results.md). Compare numbers only within one environment version and world set.

The agent loses mainly at bosses, above all the Act 2 boss (43% of the
champion's losses). The [19 October brainstorm](brainstorms/2026-10-19-brainstorming.md)
found weak deck-building (starter cards never removed, few powers, little
variety), and the [deck oracle](experiments/2026-10-20-deck-oracle.md) showed
that deck strength, measured by simulating the boss fight, varies widely and
predicts the real boss outcome far better than HP. Distilled combat helps by
winning hallway fights, not boss fights. Details in [strategies](strategies.md); every finished comparison is
in the [results ledger](results.md).

## Now

Running (see [status](status.md)): [joint run 2](experiments/2026-10-22-joint-run-2.md),
out-of-combat PPO with repeated gated combat refits, against the fourth
continued training as a same-seed control (training is deterministic, so the
two differ only through the refits); results expected late morning on 23
October.

## Next

1. **Decide on promotion:** the fourth continued training reached 23.7% on
   unused worlds (971 against the champion's 887, p = 0.005).
2. **Joint run 2's result** decides whether joint training replaces
   alternation.
3. **[A better search teacher](experiments/2026-10-22-better-teacher.md):**
   pooling the teacher's 24 searches makes it far stronger at bosses (with it
   playing the bosses, 42 to 45% of games are won), but students imitate it
   poorly (about 57% agreement, against about 80% for a second teacher run),
   whatever the network or training length; more data helps slowly. Next:
   test whether missing enemy move history explains the gap, or label far
   more boss states.
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
