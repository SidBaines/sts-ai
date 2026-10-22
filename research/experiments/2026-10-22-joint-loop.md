# Joint loop: out-of-combat PPO and teacher-anchored combat, moving together

**Status:** planned (being built). **Verdict:** pending.

## Question

So far the two halves of the agent improve in turn: out-of-combat PPO with
the combat head frozen, then a combat student refitted to search-teacher
labels and spliced in. Alternation is coordinate ascent and can miss
improvements that need both halves to move at once (a deck that only pays
off once combat learns to play it). Does a loop in which both move every
few iterations, on the same games, beat alternation at equal wall-clock?
This is a first route to joint training, not the only one (Sid, 22
October). Plain PPO on the combat head is not the route: it failed in
October ([combat PPO](2026-10-13-combat-ppo.md)) and would erode the
teacher-distilled combat head.

## Design (draft)

Three processes on the laptop, sharing files:

1. **Out-of-combat PPO** (`train_ooc.py` with the new `--combat-follow
   PATH`): the existing recipe (3,072 games per iteration, potion filter,
   hygiene options, exploration statistics); at each iteration boundary it
   adopts the combat head in PATH if that file changed, and records which
   combat head every iteration used.
2. **Teacher labeller**: `fresh_collect.py` shards that repeatedly take the
   trainer's latest checkpoint as the player and label boss decisions with
   the search teacher (boss-only probabilities, as in the
   [boss-only corpus](2026-10-22-combat-variance.md)).
3. **Combat updater**: every few hundred labelled games, updates the combat
   student on a rolling window of recent labels plus older ones, then gates
   it: the candidate head, inside the current player, plays a paired
   evaluation against the current head and is written to PATH only if it
   is not worse. A gate is needed because a single refit can be much worse
   ([round 3](2026-10-21-combat-student-round-3.md)).

Open until the fit-variance check reports:
- refit from scratch versus fine-tune from the current student (warm start
  with a penalty tying it to the current head);
- gate size and threshold (a 1,024-world paired check costs about 3
  minutes; noise at that size is about ±1.5 points of wins).

- **Comparison:** the joint loop against alternation (out-of-combat rounds
  then combat refits) from the same start and for the same wall-clock
  time; outcome on 4,096 unused worlds.
- **Approval:** Sid, 22 October ("Building the joint loop as the main
  engineering item ... Let's do it").

## Results

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 22 Oct, 12:44: design drafted; Codex building `--combat-follow` for the
  trainer.

Code: `src/sts_rl/ooc_trainer.py` (`combat_follow`), `scripts/rl/fresh_collect.py`,
`scripts/rl/fresh_train.py`; orchestration in
`experiments/2026-10-22-joint-loop/` (to come).
