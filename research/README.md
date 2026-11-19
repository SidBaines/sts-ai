# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured. The agent must stay a pure policy; search may be
used to make better teachers, not to play.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| **Champion from 19 November:** Act 3 specialist refits at updates 2, 4 and 6 (`champions/champion_9700741dca15.pt`) | 6,459 of 16,384 independent worlds (39.42%) vs 5,575 (34.03%), +5.40 points [+4.56, +6.23] | Confirmed; current default |
| **Champion from 19 November:** 24 further strategy updates at 0.0001 with frozen combat (`champions/champion_5b06aba8a254.pt`) | 5,521 of 16,384 independent worlds (33.70%) vs 5,344 (32.62%), +1.08 points [+0.33, +1.83] | Confirmed; previous champion |
| **Champion from 18 November:** 24 iterations of strategy PPO at learning rate 0.0001 from the October champion, with frozen combat (`champions/champion_0f527a0fbf93.pt`) | 5,502 of 16,384 independent worlds (33.58%) vs 4,959 (30.27%), +3.31 points [+2.57, +4.06] | Confirmed; previous champion |
| Champion from 25 October: 24 more iterations of out-of-combat PPO from the previous champion, current reward (`champions/champion_4058cbf0e271.pt`, from `reward_ablation_20261023/floors/iter_0024.pt`) | 29.4% on 16,384 unused worlds from 109,000,000,000 (4,822 against the previous champion's 4,262, +3.4 points, p < 0.001) | Confirmed; previous champion |
| The same run rewarded only for wins (`reward_ablation_20261023/wins/iter_0024.pt`) | 28.9% on the same worlds (4,728; no clear difference from the champion's run, p = 0.15) | Confirmed; not promoted |
| Champion from 23 October: joint run 2's final checkpoint (`champions/champion_c8553bc33e53.pt`; two gated combat refits) | 26.8% on 4,096 unused worlds from 107,000,000,000 (1,096 against the fourth continued training's 952, p < 0.001) | Confirmed |

Champions are kept with their promotion records in `data/rl_runs/champions/`
(worktree `nonllm-large-batch`). Earlier players and every comparison are in the [results ledger](results.md).
Compare numbers only within one environment version and world set.

On the learning-rate confirmation panel about 43% of the 18 November champion's losses came at the Act 3 boss and 31% at the Act 2 boss. See the [completed paired comparison](experiments/2026-11-18-paired-ppo-rates.md) for the learning-rate and confirmation results.

## Now

The frozen-combat control started on 19 November at 21:58 GMT. The joint arm follows on the same GPU. Independent testing retained 12% combat exploration, a combat learning rate of 0.00003 and two passes for the joint arm. [Pilot evidence and 24-update comparison](experiments/2026-11-19-online-combat.md).

## Next

Test three early refits against three later refits on a second matched training seed. This would separate the effect of timing from the larger teacher budget used by the every-two arm.

## Open questions

- Can a deck-strength oracle teach deck-building where full-game reward
  could not?
- Why does the old encoder still beat the structured encoder by about 1.4
  points once rule features and size are matched?
- Why do value-target heads play Act 1 worse while copying the teacher
  better there?

## Pages

- [strategies.md](strategies.md): each approach, its status and evidence.
- [results.md](results.md): every finished comparison, newest first.
- [metrics.md](metrics.md): definitions of metrics, evaluation worlds and terms.
- [experiments/](experiments/): one page per study.
- [brainstorms/](brainstorms/): dated records of strategy discussions.
- [status.md](status.md): what is running now.
- [archive.md](archive.md): where the retired documents went.
