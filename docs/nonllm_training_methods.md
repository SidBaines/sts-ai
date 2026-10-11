# Non-LLM curriculum methods

The first ten-hour window prioritizes sample-efficient public-observation policies.
No hidden simulator state is passed to the actor or critic. Privileged search labels
are teacher targets only, recorded separately with provenance. World seed splits are
fixed before fitting; development selects settings and final test worlds are used
only for the final report. Teacher search is a noisy target, not optimal ground truth.

## Architecture and decision space

Use a shared state encoder and candidate encoder with a shared action scorer, plus
a scalar value head. Menus have variable length, so the output is a masked
categorical distribution over the current legal candidates, never a global action
ID. Padding is masked in policy loss and mean candidate pooling. Numeric public
combat/card features and signed token hashes describe cards, enemies, rewards,
events, and shops. Renumbering/permuting menu indices must not change candidate
scores. Native combat in OOC tasks is a single macro transition: discount is per
external decision and reported as such.

Small, medium, and large widths compare compute and data efficiency; optional
mean menu context compares independent scoring with cheap DeepSets interactions.
Phase specialization compares all-phase fitting with combat-only and OOC-only
fitting. A composed policy dispatches by public phase. A recurrent or attention
model can follow if public feature coverage or partial observability limits the
feed-forward policy; that comparison must match samples and wall-clock budget.

## Curriculum

1. Collect search-labeled combat states and scripted-policy OOC states from train
   worlds, with a development-world collection kept separate. BC pilots vary
   learning rate, minibatch, model width, and context using short bounded runs;
   loss/accuracy by phase and real rollout outcomes jointly assess progress.
   OOC scripted imitation is a warm start, not a claim that its labels are strong.
2. DAgger collects corrective search labels on learner-reached battle states.
   Aggregate these with initial labels; retain original and round identifiers so
   covariate shift and teacher faults can be audited.
3. PPO fine-tunes individual battles, then OOC decisions with native combat,
   then complete runs with progressively longer horizons and larger max acts.
   Phase policies/shared policies are compared on development seeds before
   choosing a final model. Combat reward is bounded terminal success/failure
   and HP potential changes. Reward formulas are frozen and recorded per stage.
4. Evaluate held-out complete worlds, plus independent battle and OOC outcomes,
   against random and native policies with identical game/search settings. Report
   faults separately from game defeats; native faults invalidate affected episode
   transitions and teacher labels. Keep per-seed evidence and optimizer seeds.

PPO collection freezes actor weights and records the sampled action, behavior
log probability, value, and legal menu mask before any optimizer update. GAE
uses zero bootstrap for true game terminals, the actual next-observation value
for time limits, and cuts its recurrence at both terminal and truncation boundaries.
PPO optimizes stored behavior ratios with clipping and KL early stopping. Never
recompute old probabilities using the newly updated actor. Collector time limits
are explicit truncations; simulator faults are excluded rather than converted to
loss or zero reward. Atomic checkpoints contain model configuration, optimizer,
training step, dataset identity, reward identity, and Python/Torch/MPS RNG states.

## Alternatives and why they are not all first-window jobs

| Method | Appropriate role | Main limitation in this first window |
| --- | --- | --- |
| BC / DAgger | Fast legal-action warm start; stronger search feedback in combat | Teacher quality and cost; cannot exceed a teacher solely by cloning |
| Masked PPO | Online correction with shared candidate actor/critic | On-policy simulator throughput and noisy long-horizon rewards |
| DQN / distributional Q | Candidate Q scorer, replay reuse for battle tasks | Bootstrapping across variable menus and long reward delays need careful target/replay implementation |
| Discrete SAC | Replay-efficient candidate policy with entropy control | Extra Q networks, support/mask normalization, and offline extrapolation tests |
| Offline RL (IQL/CQL) | Reuse teacher and learner trajectories without simulators | Teacher labels alone lack reward/next-state coverage; dataset support is uneven |
| MuZero / learned world model | Long-horizon planning with model rollouts | Model training, public state abstraction, and planning cost exceed a useful ten-hour screen |
| Recurrent PPO / transformers | History-dependent beliefs, card/enemy structured attention | Establish feed-forward throughput and feature bottlenecks first; sequence reset and truncated BPTT add risk |
| Evolution / random search | Very small baselines and reward-function robustness | Poor sample efficiency for a rich action space |

The initial bounded pilots determine rollout throughput before allocating the
remaining window. Larger batch sizes and cached CPU encodings are considered
when MPS launch/synchronization overhead dominates. Failed jobs preserve data
and checkpoints; resuming uses the recorded dataset/reward identity rather than
silently changing the scientific conditions. A model trained for ten hours is a
first empirical result, not an assurance of strong full-game competence.

## Standalone training CLI

Run with `PYTHONPATH=src .venv/bin/python scripts/train_nonllm.py`. Point
`STS_LIGHTSPEED_BUILD` at the compatible environment-base binary. All commands
are bounded by `--seconds` and write atomic metrics; training writes atomic
checkpoints, and PPO retains a checkpoint after every update round.

```sh
PYTHONPATH=src .venv/bin/python scripts/train_nonllm.py bc \
  --teacher runs/teachers/teacher.jsonl --output runs/models/shared.pt \
  --device mps --size medium --context mean --batch-size 512 --updates 500 --seconds 120
PYTHONPATH=src .venv/bin/python scripts/train_nonllm.py ppo \
  --checkpoint runs/models/shared.pt --tasks runs/teachers/battles.jsonl \
  --output runs/models/combat.pt --task-kind battle --horizon 128 \
  --workers 8 --device mps --seconds 600 --updates 20
PYTHONPATH=src .venv/bin/python scripts/train_nonllm.py evaluate \
  --checkpoint runs/models/combat.pt --output runs/eval/combat.pt \
  --split dev --task-kind battle --episodes 64 --seconds 120
```

`bc --phase combat` and `bc --phase ooc` train specialized models, optionally
warm-started from a shared `--checkpoint`. PPO selects `--task-kind battle`,
`ooc`, or `run`; `--max-act` and `--horizon` define the curriculum stage.
`--teacher-simulations` additionally writes learner-state corrective labels to
the output's `.dagger.jsonl` sidecar. Retrain BC on an explicitly aggregated
teacher/DAgger dataset to perform a DAgger round. Original source/world/split
records should remain in that aggregation.

Use `--resume --checkpoint` to restore optimizer and RNG with the same dataset
identity and reward/stage settings; omit resume for deliberate warm starts on
a new dataset or stage. Training rejects development/final splits. The final
split is available to evaluation and must remain untouched during selection.
Independent phase policies can be evaluated using `--combat-checkpoint` and
`--ooc-checkpoint` together. The base environment CLI accepts the plugin
`--agent sts_policy.model_agent:make_agent` with JSON configuration containing
either `checkpoint` or both specialized checkpoint paths. Public phase dispatch
shares one encoder specification; the independently fitted models have separate
weights. A composite is therefore a mixture of specialists, not a shared encoder
parameterization.

Policy inference can use CPU (`--inference-device cpu`) while updates use MPS
(`--device mps`); model transfer occurs once per collection/update round. The
collector reports reset/step/teacher wait time and actor/bootstrap computation
time. `--fast-observations` enables an exact verified native-process render cache.
Rewards support explicit `--reward-mode potential|sparse|progress` and the
`--goal-act-clear` curriculum success boundary; ordinary horizons remain
bootstrap truncations. Reward mode and discount are checkpointed.

BC uses CPU-validated targets, direct masked cross entropy, and amortized metric
transfers (`--metric-interval`, default 32), preserving the same updates and
reported statistics. PPO encodes each immutable rollout buffer once. Optional
`--bucket-actions --tensor-cache` groups power-of-two action counts and stages
cached CPU tensors; all complex menus remain represented. Bucketing changes
minibatch composition and is an explicit experimental setting. CPU evidence
favored cached NumPy collation for mixed menus (0.135 versus 0.154 seconds for
100 1024-row batches), while same-bucket tensor collation was 3.08 times faster
on a 308-row panel. Tensor caching and bucketing therefore remain opt-in.

Successive bounded PPO chunks can use `--warm-optimizer`: compatible task kinds
and reward settings retain Adam moments while a new bank/horizon remains an
explicit curriculum change. New optimizer seeds and RNG initialization are
retained for the new chunk. Changing the method, reward, or task kind creates a
fresh optimizer and records the reason. Checkpoints retain source checkpoint and
old/new dataset identities. `--resume` is stricter and restores RNG as well,
requiring the same dataset and stage settings. `--fast-replay` enables verified
prefix replay without repeated intermediate public rendering; frontier observations,
teacher labels, rewards, termination, and fault behavior remain identical.

`--terminal-hp-weight` (default zero) adds an explicit terminal survival HP
bonus for successful battles/runs; this is an objective ablation, not pure
potential shaping. `--progress-floor-weight` (default 0.05) sets the progress
reward in progress mode. Both weights are passed to native task workers and
recorded in metrics and checkpoints, and changing either resets optimizer
carry across chunks.
