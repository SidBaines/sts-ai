# Single-fighter distillation

`config.json` fixes the teacher, world allocations, two training mixtures and fitting budget. `collect.py` records compact public combat menus from the champion's games. `fit.py` exports one deployable player and the same bare fighter for the replay tool.

To launch on the Mac, run `advance.py` through `scripts/rl/queue_runner.py` in its own lane. Pass the research Python interpreter, the checkout's `src` on `PYTHONPATH`, and the workstream native build on `STS_LIGHTSPEED_BUILD`. The launcher uses the existing queue for collection, both fits and both kinds of evaluation. Its fallback mixture is the 25% boss fit. A `STOP` file in the run directory stops new submissions. Already submitted jobs are managed by the queue.

The run directory contains the exact imitation-world roster, training and validation shard counts, each fit's curve and checkpoints, the frozen nominee, the boss grid, full evaluation records and `verdict.json`. The replay grid gives each policy the original boss entry and one matched reseed.
