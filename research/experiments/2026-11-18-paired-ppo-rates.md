# Does a lower learning rate improve full-game play?

**Status** — running. The frozen 29.4% champion is the starting player.

## Question

Does the lower rate preferred by the [offline learning audit](2026-11-18-ppo-audit.md) win more full games after the same 24 iterations?

## Design

Sid authorized both arms and independent confirmation on 18 November 2026. The control uses learning rate 0.0003 and the challenger 0.0001. Each starts from identical champion weights with fresh AdamW and inherited return scale. Combat is frozen. Each arm uses the same 73,728 fresh training worlds, in the same order, in 24 iterations of 3,072 games. The remaining settings are in the experiment config. The study uses env-v0.1.5.2, MPS for updates and diagnostics, and the original training menu filters.

Freeze the iteration-24 checkpoint from each arm. The 1,024-world checks during training are development measurements. Evaluate both endpoints greedily on the same 16,384 worlds from 125 billion, with seven processes and eight slots per process. Select the endpoint with more wins; an exact tie selects the control. Compare that selected endpoint with the starting champion on a separate 16,384 worlds from 126 billion, at the same topology. The paired win difference and a two-sided 95% interval determine confirmation. Retain the incumbent unless the interval lies above zero. Document faults and censoring separately, keeping every assigned world in the record. Use a paired large-sample interval on per-world win differences as in the audit. One training seed limits conclusions across seeds.

## Results

Pending the two frozen endpoints.

## Log

- 18 November: selected one GPU training lane to preserve the benchmarked 12-collector, 384-slot configuration and fit the 48 GiB Mac. Independent policies in each evaluation panel can run concurrently.

Code: `experiments/2026-11-18-paired-ppo-rates/`. Local data: run `ppo_fresh_worlds_from_4058`.
