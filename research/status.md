# Status

The [direct joint-training study](experiments/2026-11-19-online-combat.md) calibrated combat exploration and is testing the combat learning rate. Both eventual arms start from the original 39.42% champion.

- The moderate setting tried a lower-ranked move on about 12% of meaningful choices. Its frozen player won 254 of 768 pilot games. The near-greedy player won 285 on the same worlds.
- A 48-game paired training check reproduced every trajectory and the strategy update. Both fighters moved only in the joint arm.
- The shared [training dashboard](http://127.0.0.1:8795/) shows queue jobs. The [new study log](log.md) records milestones.
- In `data/rl_runs/online_combat_from_9700`, `status.json` names the current stage and `progress.jsonl` records heartbeats. A `STOP` file prevents further submissions. Queued jobs remain controllable through the queue.
- The existing refit study can finish training first on the Mac. Its participants have the same starting champion.
