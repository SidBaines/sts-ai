# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured. The agent must stay a pure policy; search may be
used to make better teachers, not to play.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| Reward ablation, wins-plus-floors arm, iteration 24 (`reward_ablation_20261023/floors/iter_0024.pt`; 24 more iterations of out-of-combat PPO from the champion, current reward) | 29.4% on 16,384 unused worlds from 109,000,000,000 (4,822 against the champion's 4,262, p < 0.001); the wins-only arm reached 28.9% (4,728; no clear difference, p = 0.15) | Confirmed; promotion candidate (Sid's decision) |
| **Champion from 23 October, 09:58:** joint run 2's final checkpoint (`champion_c8553bc33e53.pt`; out-of-combat PPO with two gated combat refits) | 26.8% on 4,096 unused worlds from 107,000,000,000 (1,096 against the fourth continued training's 952, p < 0.001) | Confirmed; current default |
| Champion from 22 October, 16:32 (`champion_af2fb9f1f260.pt`) | 21.1% on 4,096 unused worlds from 100,000,000,000 | Confirmed |

Earlier players and every comparison are in the [results ledger](results.md).
Compare numbers only within one environment version and world set.

The agent loses mainly at bosses: of the champion's losses, about 41% are at
the Act 3 boss and 32% at the Act 2 boss. Out-of-combat PPO still improves
it every round, and refitting the combat network on the search teacher's
labels from the current player's own fights adds more. The search teacher
playing only the boss fights roughly doubles the win rate, but students
copy it poorly: that imitation gap is the largest known headroom. The
[23 October discussion](brainstorms/2026-10-23-next-steps.md) sets out the
training setup, the headroom and the options.

## Now

Nothing running ([status](status.md)). Finished since 23 October:
- [Reward ablation](experiments/2026-10-23-reward-ablation.md): dropping the
  floor reward helped for 12 iterations but not over 24; keep it.
- [Value targets](experiments/2026-10-23-value-targets.md): combat students
  fitted to the teacher's move values beat vote-fitted students on the same
  data.
- [Value-target refits](experiments/2026-10-23-value-target-refits.md):
  labelling stopped at 1,753 of 2,000 worlds (killed by the queue's stall
  watchdog); fits not yet run.

## Next

1. **Value targets in refits:** fit and play-test refits on the 1,753
   labelled worlds (values against votes, two seeds each).
2. **Refit-and-gate cycles as standard** beside out-of-combat training, with
   shorter cycles, value targets if they win, and validation boss regret to
   screen fits before the gate.
3. **Distil potion decisions at bosses:** the teacher deciding only potion
   drinks is worth about 4 points; the champion drinks about half as often
   as the teacher ([better teacher](experiments/2026-10-22-better-teacher.md)).
4. **Close the imitation gap:** test whether missing enemy move history (or
   known top cards) explains why students copy the teacher poorly.
5. **Learn deck strength** (a deck-value network trained on oracle scores).
   To discuss with Sid.

## Open questions

- Can a deck-strength oracle teach deck-building where full-game reward
  could not?
- Why does the old encoder still beat the structured encoder by about 1.4
  points once rule features and size are matched?
- Does a recurrent policy (memory across turns) matter? Left for later.

## Pages

- [strategies.md](strategies.md): each approach, its status and evidence.
- [results.md](results.md): every finished comparison, newest first.
- [metrics.md](metrics.md): definitions of metrics, evaluation worlds and terms.
- [experiments/](experiments/): one page per study.
- [brainstorms/](brainstorms/): dated records of strategy discussions.
- [log.md](log.md): temporary running experiment log, newest first.
- [status.md](status.md): what is running now.
- [archive.md](archive.md): where the retired documents went.
