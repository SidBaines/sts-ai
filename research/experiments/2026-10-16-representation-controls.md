# Representation controls: why did the structured encoder lose?

**Status:** paused 19 October, partly collected, handed from Codex to Claude.
**Verdict:** none yet; no outcome analysis has run.

## Question

In the [fresh-start study](2026-10-15-fresh-start.md), the structured entity
encoder copied the search teacher as well as the old encoder but played worse.
It also differed from the old encoder in three ways at once: it lacked the old
encoder's hand-made card and potion rule features, it was smaller (0.86 against
2.0 million parameters), and it had less learning-rate tuning. Which of these
explains the gap, and does the gap repeat with a different training seed?
(Approach: [structured entity encoder](../strategies.md#structured-entity-encoder).)

## Design

- **Factorial comparison,** run at two training seeds (1919 and 2020):
  - legacy rule features on or off (when off, the structured model is the
    original one);
  - width 128 (original) or 216 (chosen in advance to match the old encoder's
    parameter count within 5%);
  - learning rate 0.0003 or 0.001.
- **Old-encoder controls** at both learning rates and both seeds.
- **Extra rule, fixed in advance:** if the lower learning rate won on
  validation for every wide model, also fit learning rate 0.0001 for every
  structured model. It did, so 26 new models were fitted.
- **Training data:** the fresh-start study's combat labels, unchanged.
  Checkpoints and learning rates are chosen on validation loss only.
- **Evaluation:**
  - teacher agreement on the fresh-start test labels (already used, so
    these are development diagnostics);
  - a new fight panel from 1,024 unused worlds: 400 Act 2 boss fights
    (primary), plus 100 Act 1 boss and 100 Act 2 hallway fights, each with 8
    reseeds;
  - full games for eight arms, fixed in advance, on 4,096 new DEV worlds.
- **Primary measures:**
  - the effect of features, size and learning rate on HP lost in Act 2 boss
    fights, averaged over the other two factors and reported per seed;
  - in full games, the paired win difference between the old encoder and
    the original structured model, at each seed.
  - An effect counts as replicated only if both seeds agree in direction
    and both intervals exclude zero.
- **Approval:** Sid approved this supervised diagnosis on 16 October. No
  further RL, no promotion, and no use of the confirmation worlds.

## Results

Not yet analysed. Collected so far:

- all 26 fits, and label scoring for all 28 models;
- all 28 fight panels (10,800 results each, validated);
- the champion's full-game baseline on the 4,096 new DEV worlds.

Not yet collected: the other seven full-game arms. The old-encoder seed-1919
arm was stopped after 1,051 of 4,096 worlds; that partial run is kept but
excluded, and the arm needs a fresh full run. The other six have not started.

Validation loss barely separates the models: every fit landed between 1.28
and 1.35. For the wide model, the lower learning rate was better at both seeds.

## Conclusion

Pending.

## Next

Analyse the factorial results once collection completes, then decide with Sid
whether to confirm the old-encoder combat candidate and start step D.

## Log

- 16 Oct: design fixed and approved; a new fight bank was collected before
  any fitting.
- 16 Oct: the main fits finished. The lower-learning-rate extension was
  triggered and run.
- 17 Oct: collection paused while the laptop was on battery. It then stopped
  at a safety deadline the study had set itself.
- 19 Oct: restarted with no deadline at Sid's request. A second battery pause
  caused six simulator timeouts in one panel, which was re-run successfully.
- 19 Oct: all fight panels finished. Sid stopped the study during the game
  arms so that Claude could take over.

Code: `src/sts_rl/controlled_*.py`, `src/sts_rl/representation_controls.py`
and `scripts/rl/controlled_*.py`, at commit `1636034`. Data: run
`representation_controls_20261016` (local).
