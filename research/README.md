# Non-LLM RL workstream

**Goal:** train a neural-network agent that wins full games of Slay the Spire
(Ironclad, Ascension 0, Acts 1 to 3) from public game information, measured
by win rate on many simulated worlds. See [metrics](metrics.md) for how
everything is measured.

## Where we are

| Player | Win rate | Evidence |
|---|---|---|
| Champion: round-2 out-of-combat PPO update 20 | 3.5% (143 of 4,096 DEV worlds, environment 0.1.5) | Current default |
| Old-encoder combat distilled from the search teacher, with champion out-of-combat play | 5.4% (222 of 4,096, same worlds) | Exploratory; not yet confirmed on fresh worlds |
| Champion plus search at Act 2 and 3 bosses and elites | 8.1% (333 of 4,096 fresh worlds, environment 0.1.4) | Confirmed, but the search uses hidden simulator state |

Environment 0.1.5 fixed several game-rule bugs and made the game harder: the
champion won 5.2% of fresh worlds on 0.1.4. Compare numbers only within one
environment version.

What has worked so far: out-of-combat PPO with macro credit and lambda 1,
distilling measured strategy rules, search at play time, and, most recently,
distilling the search teacher into the combat network with a real training
budget. What has not: direct combat PPO, joint training of both networks, and
the structured entity encoder. Details in [strategies](strategies.md); every
finished comparison is in the [results ledger](results.md).

## Now

Nothing is running. [Representation controls](experiments/2026-10-16-representation-controls.md),
a supervised diagnosis of why the structured encoder played worse than the old
one, is paused: its fight panels are complete, and seven of eight full-game
arms remain. See [status](status.md).

## Next

1. Confirm the distilled old-encoder combat model, with champion out-of-combat
   play, on 8,192 fresh confirmation worlds. If it holds, it becomes the
   champion.
2. Combat expert iteration from that model: play the student, label the states
   it visits with the search teacher, retrain, and check on the fight panel
   before full games. The aim is to close the gap to play-time search.
3. Move single-study code out of the library packages into experiment folders,
   following the code convention in `CLAUDE.md`.

## Open questions

- Why do two combat students that copy the teacher equally well play so
  differently? Average agreement with the teacher does not predict playing
  strength.
- How much of the distilled model's gain is adaptation to 0.1.5's rule changes,
  given that the champion was trained on earlier versions?
- How much of play-time search's advantage can a network learn, given that the
  search sees hidden state?
- Can critics for a sparse win reward be made well calibrated in Acts 2 and 3?

## Pages

- [strategies.md](strategies.md): each approach, its status and evidence.
- [results.md](results.md): every finished comparison, newest first.
- [metrics.md](metrics.md): definitions of metrics, evaluation worlds and terms.
- [experiments/](experiments/): one page per study.
- [status.md](status.md): what is running now.
- [archive.md](archive.md): where the retired documents went.
