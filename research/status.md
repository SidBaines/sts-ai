# Status

The 24-update comparison is running. The frozen-combat control started on 19 November at 21:58 GMT. The joint arm follows on the same GPU. Both train from the 39.42% starting champion on the same 73,728 assigned worlds per arm.

- Both arms use the 12% combat exploration target. The joint arm updates both fighters at learning rate 0.00003 for two passes. Four passes won 3,242 vs 3,272 games for two passes on 8,192 independent worlds, a paired difference of −0.37 points [−1.43, +0.70]. [Pilot results](experiments/2026-11-19-online-combat.md#expanded-pilot-results).
- After update 24, both players face 16,384 paired fresh worlds. Their independent confirmation covers a further 16,384 worlds and includes the starting champion.
- The queue pauses on battery and resumes on AC. Follow the [training dashboard](http://127.0.0.1:8795/) or the [progress log](log.md).
- In `data/rl_runs/online_combat_from_9700`, `status.json` names the stage and `launch-settings.json` holds the shared choice. A `STOP` file there prevents further submissions. Already queued jobs remain controllable through the shared queue.
