# Repaired training study progress

## 12 October overnight allocation started

Work is active in `codex/nonllm-critique` under the user's new CPU/GPU authorization until approximately **13 October 07:30 UTC**. The earlier v1/v2 campaigns and their unused budgets are preserved. The [implementation plan](nonllm_v3_plan.md) and [critique](nonllm_training_critique_2026-10-12.md) define the scope. This study needs a compatible retaining warm start and evidence of useful learning before a sustained run.

The first candidate environment repair is implemented and verified: version **0.2.0**, public observation **sts_env_public_v2**, native binding API **2**. Distinct combat instances and exact teacher routing are preserved. Typed inventory/action mechanics provide source cards, public upgrades/counters/costs/targets and numeric offers. Discovery is repaired for supported Ironclad generation; unsupported classes still fail closed.

| Validation | Result |
| --- | --- |
| Full host/native suite | 483 tests, one optional generated-bank skip |
| Mandatory bounded random smoke | 21 decisions, ordinary defeat, no error |
| Reached free/normal Strike regression | Both choices preserved, resulting energy 3 versus 2 |
| Fresh-process full-control traces/replay | Identical and verified |
| Scoped patch application | Fresh apply and reverse apply pass |

The new runtime/source/build are independent copies inside this worktree. No old release, binary, bank, checkpoint or campaign is overwritten. [Interface changes and migration](environment_public_v2.md); detailed acceptance is `data/nonllm_v3/preflight/environment_acceptance.json`.

Fresh collection is complete: 64/64 TRAIN worlds, no faults, 20,222 teacher labels and 859 fight starts. This is a new provenance-pinned bank; native combat labels are privileged and native OOC labels remain a weak scripted teacher.

The [typed encoder and residual policy](public_entities_v3.md) are implemented. Initial zero residuals preserve the original public policy, including duplicate-menu routing. Actor and critic parameters/optimizers are separate; the original prior is immutable. The full **20,222-row** fresh TRAIN audit found zero unknown identities/statuses/action kinds or unresolved source/target joins; 2,344 unsupported damage-target calculations are explicitly unknown. Latest host/native acceptance: **507 tests**, one optional generated-bank skip, mandatory bounded smoke without an error, and a passing real-MPS subnormal regression.

Native micro/macro equivalence passed on all eight sampled TRAIN games: identical action traces and reward totals. Collection took 4.24 seconds for micro and 2.69 for macro in this small check. This is an execution comparison, not learning evidence.

The new 192-world fresh DEV baseline completed without faults or cuts: **156 Act 1 clears, 20 Act 2 clears, zero full-game wins**, mean floor 28.198. The zero-win Wilson 95% upper bound is 1.96%. Its tasks/results and immutable checkpoint identity are saved in `data/nonllm_v3/evals/common_initial_dev192.json`. This new cohort is development data, not a final test.

Original-policy calibration collection completed 63 TRAIN tasks (32 fresh, 24 Act 2 continuations, seven available Act 3 source worlds) plus 24 separate TRAIN holdout worlds. It produced one fresh win and one conditional win. The first five-epoch MC critic fit improved training MSE 0.645→0.536 but worsened holdout MSE **0.178→0.203**; holdout EV moved 0.437→0.425. Actor weights stayed bitwise identical. The failed calibration and its checkpoint are preserved under `data/nonllm_v3/calibration/common_mc_v1`; it is not accepted as the common warm start. Later-start sampling and broader calibration data need investigation.

Protected PPO and MC fitting now log actual choice/critic/rehearsal exposures, pre/post clipping gradient norms, final behavior/original KL, TV/greedy drift, and completed-return calibration by phase/act. Controlled tests verify learning a numeric choice signal and value target while preserving the prior, excluding forced actor rows and enforcing final-epoch guards. These are software acceptance checks, not proof of game improvement.

The bounded controller is implemented with permanent checkpoints before evaluation, task/source identities, distinct sampling/optimizer seeds, actual learner/champion pointers and failed-partial quarantine. It stops for missing evaluation games, paired loss of eight Act 1 clears, substantial floor loss without a win gain, or cumulative original-policy drift. Those are pilot investigation gates, not automatic rollback or proof of statistical significance. Resource reserves are 25% battery when not charging and 8 GiB free disk, with the UTC allocation checkpoint. Source checks cover loaded model/encoder/environment code and the native binary before updates/checkpoints/evaluation.

The first four-chunk macro pilot is complete: all 220 TRAIN requests finished without faults/cuts, and both DEV192 evaluations retain **every original greedy trace** (156/20/0). It collected 45,516 primitive steps; 8,656 actual choice rows entered actor optimization once each. This establishes execution/retention, **not useful game learning**. The initially zero residual needs a plasticity calibration rather than blindly inheriting a learning rate proposed for the previous architecture.

Matched-buffer rates 3e−5/3e−4/1e−3/3e−3 produced original rehearsal mean KL about 2e−11/6e−9/1.9e−7/4.7e−6 after one epoch, with no changed greedy rehearsal choices. The measurements caught a new telemetry bug: MPS flushed tiny float32 reference probabilities to zero in `p*log(p)`, yielding NaN in final behavior KL. Trainer v2 computes diagnostics in CPU float64 and fails closed on non-finite inputs; a real-MPS subnormal regression passes. Original affected outputs remain preserved, and corrected metrics are saved separately in `recomputed_float64_drift.json`. No condition qualifies for scaling from those software measurements alone.

The stronger charger is now verified: 69%, **charging** at 23:17–23:21 UTC. Unattended power gates remain in place. Next: improve residual context/plasticity, then run matched protected learning comparisons and broader independent retention/usefulness checks.
