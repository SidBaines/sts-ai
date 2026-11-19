# When should we refit the current champion’s Act 3 combat?

**Status:** queued on 19 November 2026. The study compares three early refits against three spaced refits from the same confirmed champion.

## Question

Does concentrating the refits at strategy updates 2, 4 and 6 produce a stronger successor than waiting until 8, 16 and 22? In the [previous cadence comparison](2026-11-18-act3-boss-refits.md#results), the early specialists all passed. The frequent schedule also spent 5.5 times the teacher budget. The new parent already carries the specialist accepted at update 6. This tests further improvement from that stronger position.

## Design

Sid requested the two schedules on 19 November and suggested the new champion as their starting point. Both arms start from the 39.42% confirmed player. The following training and evaluation choices were fixed before the first update.

- Give each arm 24 strategy PPO updates at learning rate 0.0001 on the same 3,072 new worlds per update. Use one new matched training seed, the previous reward and optimizer recipe, and the same native environment. Both arms inherit the champion’s Act 3 specialist. They share exact strategy checkpoints until their deployed combat policies diverge.
- Give each arm three opportunities and 512 on-policy teacher worlds per opportunity, totalling 1,536 per arm. Match the teacher world seeds, fight-bank world seeds and whole-game gate worlds by opportunity number. Opportunity one uses the same rosters at early update 2 and spaced update 8. Subsequent opportunities use separate rosters. Each arm plays its own games.
- Use the same specialist recipe and admission rules as the previous study. Fit pooled action values from 24 reseeded searches of 1,000 simulations. First require more wins from 192 identical original Act 3 boss entries. An additional reseeded replay of each entry is descriptive. A candidate that advances must gain wins on 4,096 paired full games even under the least favourable assignment of unresolved outcomes. Adopt it for public Act 3 boss combat only. All other combat weights stay frozen.
- Greedy development checks every two updates use the usual 1,024 worlds and cannot select a checkpoint. Freeze both update-24 endpoints. Compare them on 16,384 unused, paired full games at seven evaluator processes and eight games per process. Nominate spaced if its least favourable paired gain over early is positive. Otherwise nominate early, including ties.
- On 16,384 more unused worlds compare the frozen schedules directly and each against the starting champion. An advantage for one schedule requires the same direction on selection and independent confirmation, with the confirmed paired 95% interval excluding zero even under unresolved outcomes. Promote only the primary nominee and only if its worst-case paired 95% lower bound against the starting champion exceeds zero. Identical deployed policies share one evaluation per panel. Record faults and cutoffs separately.
- Report outcomes and uncertainty for all six refit opportunities, both final panels and any new champion. This is one matched training seed from a stronger parent. It answers which schedule to use next from this checkpoint.

## Results

The [progress log](../log.md) follows the run. The primary comparison is the update-24 head-to-head.

## Log

- 19 November: fixed the new parent, matched world rosters, equal teacher budgets and decision rules before training. The undiverged trajectory can supply immutable checkpoints to both arms.

Code: `experiments/2026-11-19-refit-timing/`. Local run: `refit_timing_from_9700`.
