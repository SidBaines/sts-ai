# Corrected-backend health coupling check

Retain `combat_0017.pt` for the combat head. With the selected corrected-backend OOC checkpoint `kl_guard_paired/before_update.pt`, it cleared 21 of 32 development worlds with zero reported faults. Both existing health-trained alternatives cleared fewer worlds and reported one fault each.

| Combat head | Act 1 clears / valid | Faults | Mean floor | Mean HP cost | Campaign score |
|---|---:|---:|---:|---:|---:|
| combat17_reference | 21/32 | 0 | 16.375 | 26.50 | 67.00 |
| hp0.25 | 11/31 | 1 | 15.871 | 51.23 | 35.45 |
| hp1.0 | 13/31 | 1 | 16.129 | 46.29 | 41.77 |
| shared_bc_reference | 7/32 | 0 | 14.688 | 62.72 | 22.72 |

All four panels contain the exact same 32 task identities: development worlds generated from world start 10000, policy seed 777, Act 1 target, horizon 2048, native simulation budget 200, and corrected native module `40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`. The composites expose every combat decision to the combat head; native automatic combat is not used. Reported native-state faults remain separately quarantined, not ordinary losses.

The completed matching combat17 composite and shared-BC references were reused (CPU eight workers, two threads). The two alternatives ran CPU four workers, one thread, without GPU use or new training. All task cohorts completed with no unstarted worlds; the worker/thread differences affect elapsed time and do not define a matched speed comparison.

The λ0.25 head faulted in world 10218; its 31 common valid worlds contain 21 reference clears versus 11 alternative clears. The λ1 head faulted in world 10152; its 31 common valid worlds contain 20 reference clears versus 13 alternative clears. The ranking persists on those common valid subsets.

These are usable-checkpoint coupling comparisons on development data. The health heads were trained from the earlier `combat_pilot_clip0.1.pt` checkpoint; combat17 has a later training history. The result does not isolate the causal effect of HP reward under equal training budgets and makes no significance or final-test claim. The former health study used an older OOC head and a 24-world panel; this check updates that coupling evidence with the selected corrected-backend OOC head.

The campaign score shown is `100 × clears / requested_tasks + 0.1 × mean_floor − 0.01 × mean_hp_cost`. Faults are counted in requested coverage while omitted from valid floor and HP averages.

Exact commands, input-checkpoint hashes, native manifest and source hashes are in `data/nonllm_native_revision/health_coupling/records.json`; detailed outcomes and the task-ID list are in `summary.json`. No campaign journal or source code was changed, and no final learner results were accessed.
