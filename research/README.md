# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| Champion: round-2 out-of-combat PPO update 20 | 3.8% (156 of 4,096 second-DEV worlds, environment 0.1.5) | Current default |
| Old-encoder combat distilled from the search teacher, with champion out-of-combat play | 4.8% (197 of the same worlds; +1.0 points [+0.2, +1.8]); a second training seed 5.4% (221) | Planned comparisons on worlds new to both; not yet on the confirmation worlds |
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

Codex is building the throughput changes; the deck-oracle spike is done.
See [status](status.md) and the running [experiment log](log.md).

## Next

1. **Throughput.** Move network inference into the simulator worker
   processes and step several games per process, for three to ten times more
   games per hour. Everything else needs this.
2. **Deck-strength oracle as a teacher.** It measures decks well; next test
   whether it separates the options at a card reward, shop, rest site or
   removal, then distil its choices into the out-of-combat network.
3. **PPO hygiene:** separate critic, normalised returns, exploration on
   strategic menus, bigger batches, no silent weight decay.
4. Confirm one old-encoder combat student, chosen in advance, on the
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
- [log.md](log.md): running experiment log, newest first.
- [status.md](status.md): what is running now.
- [archive.md](archive.md): where the retired documents went.
