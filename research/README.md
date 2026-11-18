# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured. The agent must stay a pure policy; search may be
used to make better teachers, not to play.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| **Champion from 25 October, 09:42:** 24 more iterations of out-of-combat PPO from the previous champion, current reward (`champions/champion_4058cbf0e271.pt`, from `reward_ablation_20261023/floors/iter_0024.pt`) | 29.4% on 16,384 unused worlds from 109,000,000,000 (4,822 against the previous champion's 4,262, +3.4 points, p < 0.001) | Confirmed; current default |
| The same run rewarded only for wins (`reward_ablation_20261023/wins/iter_0024.pt`) | 28.9% on the same worlds (4,728; no clear difference from the champion's run, p = 0.15) | Confirmed; not promoted |
| Champion from 23 October: joint run 2's final checkpoint (`champions/champion_c8553bc33e53.pt`; two gated combat refits) | 26.8% on 4,096 unused worlds from 107,000,000,000 (1,096 against the fourth continued training's 952, p < 0.001) | Confirmed |

Champions are kept with their promotion records in `data/rl_runs/champions/`
(worktree `nonllm-large-batch`). Earlier players and every comparison are in the [results ledger](results.md).
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

Nothing running. The workstream has adopted `env-v0.1.5.2`. All 224
champion trajectories in the adoption check matched their historical records.
The [learning audit](experiments/2026-11-18-ppo-audit.md) covers 48 historical
iterations, 512 diagnostic games and ten disposable optimizer fits.

## Next

The recommended control is 24 iterations of out-of-combat PPO on fresh worlds,
starting from the champion at learning rate 0.0003 with frozen combat. If
running a second arm, change only the learning rate to 0.0001. The lower rate
was the strongest challenger on held-out decisions; its full-game effect remains
unmeasured. The [settings and decision rule](experiments/2026-11-18-ppo-audit.md#next)
are ready. The [boss combat alternatives](strategies.md#distilling-the-search-teacher)
remain separate from this comparison.

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
