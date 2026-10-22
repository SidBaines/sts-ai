# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| **Third continued training's final checkpoint** (`continued3_20261022/train/iter_0024.pt`; out-of-combat PPO from the champion, its round 2 combat student unchanged) | 21.1% on 4,096 unused worlds from 100,000,000,000 (865 against the champion's 707, p < 0.001); 21.9% on second DEV (899 against 701) | Confirmed; pure policy; awaiting Sid's promotion decision |
| **Champion from 22 October:** the second continued training's out-of-combat head with the round 2 combat student (`champion_85ea9019ff95.pt`) | 18.4% on 4,096 unused worlds from 99,000,000,000 (755 against the trained checkpoint's 662, p = 0.001); 17.1% on second DEV (701) | Confirmed; current default |
| The second continued training's final checkpoint alone (`continued2_20261021/train/iter_0024.pt`) | 16.9% on 4,096 unused worlds from 96,000,000,000 (691 against the champion's 507, p < 0.001); 16.4% on second DEV (672 against 527); with the round 2 combat student 701 on DEV (exploratory) | Confirmed; pure policy |
| Round 2 combat candidate: the afternoon champion with its combat head refitted to teacher labels from its own games (`combat_round2_20261021/champion_with_student_r2_newonly.pt`) | 13.9% on 4,096 unused worlds from 97,000,000,000 (571 against the champion's 512, p = 0.017); 14.1% on second DEV (576 against 527) | Confirmed; pure policy |
| Champion of 21 October, afternoon: continued training with the hygiene options (run `round4`, `champion_2835f40fa439.pt`) | 11.6% on 4,096 new worlds from 95,000,000,000 (476 against the champion's 379, p < 0.001); 12.9% on the second-DEV worlds (527 against 413) | Confirmed |
| Champion of 21 October, morning: combat student plus retraining (run `round3_student`): the 20 October champion's out-of-combat head with the old-encoder combat student as its combat head, plus 24 more iterations of out-of-combat PPO with the potion filter (`champion_7d876c939a74.pt`) | 9.6% on 4,096 fresh worlds (394 against its start's 315, the harvest stack's 346 and the 20 October champion's 232); 10.1% on the second-DEV worlds (413) | Confirmed |
| Champion of 20 October: large-batch out-of-combat PPO from the previous champion, iteration 24 (`champion_ba2fa6a2987d.pt`) | 5.7% on 4,096 fresh worlds (232 against the previous champion's 152; +2.0 points, p < 0.001; mean floor +1.24 [+0.91, +1.58]); 6.7% on the second-DEV worlds (276) | Confirmed |
| Champion of 20 October with the old-encoder combat student and the Demon Form rule (an overlay, not yet distilled) | 8.4% on 4,096 fresh worlds (346 against the 20 October champion's 232 and the previous champion's 152); 8.1% on the second-DEV worlds (333 against 276; student alone 316, rule alone 292) | Confirmed; the rule must be distilled before this is a pure policy |
| Previous champion: round-2 out-of-combat PPO update 20 | 3.8% (156 of 4,096 second-DEV worlds; 3.7% on the fresh worlds above) | Baseline of every comparison before 20 October |
| Old-encoder combat distilled from the search teacher, with the previous champion's out-of-combat play | 4.8% (197 of the second-DEV worlds; +1.0 points [+0.2, +1.8]); a second training seed 5.4% (221) | Planned; on the new champion 316 against 276 (p = 0.050); round 3 is built on it |
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

Running (see [status](status.md)): the [third continued training](experiments/2026-10-22-continued-training-3.md)
of the out-of-combat head, which is also the control for the
[joint loop](experiments/2026-10-22-joint-loop.md), queued next, in which
out-of-combat PPO and teacher-labelled combat refits move together; and
two combat students fitted on round 2, round 3 and boss-only labels
([combat variance](experiments/2026-10-22-combat-variance.md)). Refits of
the combat student vary a lot: one fit in four collapses by about two
floors, so every refit is checked before use.

## Next

1. **Record the third continued training and the boss-label fits**, and
   follow the joint loop's gate decisions.
2. **[A better search teacher](experiments/2026-10-22-better-teacher.md)**
   (planned with Sid, 22 October): measure the teacher's noise and build a
   boss-fight benchmark for teachers; then fix its vote and score, pool its
   searches, and build information-set search; let it and the student see
   cards known to be on top of the draw pile. Network playouts follow.
3. **Learn deck strength instead of simulating each choice:** a deck-value
   network trained on oracle scores. To discuss with Sid.

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
