# Can better Act 3 boss combat improve the champion?

**Status:** running. The two policies share their training history until the first specialist passes its gates.

## Question

Does refitting combat for the Act 3 boss improve on another 24 iterations of strategy training from the [November champion](2026-11-18-paired-ppo-rates.md)? On its independent confirmation panel the champion lost 4,714 games at that boss, including 2,626 entered at full health.

## Design

Sid approved the champion, 24 updates at 0.0001, a frozen-combat control, and refits after updates 8 and 16 on 18 November. The following sample sizes and decision rules were chosen for this run before inspecting outcomes.

- Both arms use the same 3,072 fresh training worlds per update, optimizer settings and evaluation protocol. The control trains continuously. Until the first accepted refit the joint arm shares the exact control checkpoints, including optimizer, training random state and return normalization.
- At each milestone collect decisions from the active policy on 512 separate worlds. The student plays. At Act 3 boss choices the teacher runs 24 publicly reseeded searches of 1,000 simulations. Fit one fresh large legacy combat head per milestone on pooled action values at scale 100. Use only the current milestone's corpus. Select the checkpoint on validation loss and defer test scoring.
- Freeze 192 Act 3 boss entries from 512 independent worlds per milestone before fitting. Compare both players on identical replayed fights. Require more wins on the original unsalted entries. One additional salted replay per entry is diagnostic. A tied fight comparison keeps the active head.
- If the fight gate passes, compare the proposed and active policies on 4,096 separate matched full games. Adopt the specialist only if its paired win gain is positive even under the worst assignment of unresolved outcomes. It plays only combat in a public Act 3 boss room, including bosses reached through Secret Portal. Strategy PPO resumes at the next iteration. A rejected refit leaves the current combat policy in place.
- Evaluate both iteration-24 endpoints on 16,384 fresh matched worlds. Use greedy evaluation with seven processes and eight games per process for every full-game comparison. Choose the joint arm if its worst-case paired gain is positive, otherwise the control. Freeze the selection before running it and the starting champion on another 16,384 matched worlds. Promote only on a positive lower bound of the paired 95% interval on confirmation, robust to unresolved outcomes.
- Development evaluations on the usual 1,024 worlds every four updates are descriptive. Report faults, cuts and missing outcomes separately. Operational failures require recovery and verification before moving on.

### Every two updates

Sid requested a second schedule on 18 November after the first update-8 result and before the first study's final evaluation. It begins only after that study finishes confirmation. Both schedules start from the identical November champion and use the same 24 PPO updates and assigned worlds.

- At updates 2, 4, …, 22, repeat the same fit and the same two gates. Each fit gets 512 on-policy teacher worlds. The total is 5,632 planned teacher worlds versus 1,024 for the first schedule. Both schedules use identical label, boss-entry and full-game gate rosters at their common milestones of 8 and 16. Other milestones have disjoint rosters.
- Resume the strategy optimizer, RNG and return normalization across each two-update segment. The trainer also records its usual 1,024-world development check at each segment boundary. Those extra observations do not select a refit or endpoint. Verify the trajectories against frozen combat while the two arms still share a policy. A denied gate keeps the previous combat policy. Report every gate and each run's elapsed time.
- On independent paired worlds compare the post-gate update-8 policies (4,096 games). At update 24 compare every two with every eight and with frozen combat (16,384 fresh worlds). Use the same evaluation batching and fault accounting as the first study.
- If the every-two endpoint beats both comparators under the least favourable assignment of unresolved wins, freeze it for independent confirmation on 16,384 further worlds against every eight and the incumbent resulting from the first study. Promote only if both paired 95% lower bounds exceed zero under that assignment. The update-8 result cannot select an endpoint.
- Follow the run on the [live progress page](http://127.0.0.1:8795/progress). The local heartbeat verifies processes, progress and power every 15 minutes. The follow-up run is `joint_a3_every2_from_0f52`.

## Results

Training is underway. The [progress log](../log.md) follows training, gate decisions and final evaluation.

## Log

- 18 November, 22:25 GMT: the routed policy completed a real optimizer resume, full games and identical-start replay checks. A separate small fit completed 2,000 updates. The smoke test caught the replay tool’s required bank namespace before either study gate opened.
- 18 November: fixed the design and reused the exact control trajectory until the first accepted refit. Each gate and each final panel has its own world roster.

Code: `experiments/2026-11-18-act3-boss-refits/`. Local data: run `joint_a3_from_0f52`.
