# sts_policy: encoders, models and training adapters

`sts_policy` encodes public game observations, scores legal actions and
supplies training, curriculum and isolated-worker adapters without changing
`sts_env`. Ironclad and Acts 1–3 are the supported research scope. See
[`sts_rl`](../sts_rl/README.md) for trainers, queues and dashboards;
[metrics](../../research/metrics.md) defines evaluation terms.

## Public encoders and models

`encoding.py` implements `public_candidate_v2`: a 32-bit floating-point state
vector of length 256 and one vector of length 128 per currently legal
candidate. Allowlisted public fields and controlled public text describe
health points, resources, cards, enemies, powers, offers and reachable map
routes. Fixed Blake2b signed hash bags preserve multiplicities and unordered
card piles. Candidate features join public card/potion rules to the matching
enemy's public state; duplicate enemy names use the public enemy index.
Multi-selection actions encode selected cards rather than using a fixed global
action head.

Only the `observation` enters the encoder. Diagnostics, world seeds, native
bits, hidden draw order and teacher values are excluded. Menu indices route a
selected row back to the current menu; they are never neural inputs. Version 2
corrects potion-name parsing and card-removal coverage. Version 1 encoded
caches and checkpoints are incompatible even though dimensions match.

`model.CandidateActorCritic` applies shared multilayer perceptrons to state
and candidate features, then scores their joint features.
`size=small|medium|large` selects capacity; `context=mean` adds masked mean
candidate embeddings and `context=none` omits them. Candidate permutation
permutes logits and preserves critic value. Padding is ignored and receives
logit -1e9. Terminal batches keep one masked slot for critic evaluation; never
sample that action distribution.

`entity_encoding.py` implements `public_entities_fresh_v1`: explicit player,
enemy, card, relic, potion, map and offer entities, with a frozen vocabulary,
numeric availability flags, owner-linked powers, recent public actions and
candidate source/target links. `entity_model.EntityActorCritic` uses separate
actor and critic transformer towers and one phase per model. `actor_logits`
unpads each row before arithmetic for stable processor inference;
`training_logits` uses masked vectorized batches. Full layouts belong in
[entity_encoding.md](entity_encoding.md). Controlled variants add optional
legacy candidate features; they are study adapters, not a new encoder default.

## Train and deploy a candidate policy

Use the repository root and a runtime with the optional `policy` dependencies.
In a new development runtime, install with `pip install -e '.[policy]'`;
preserve frozen study runtimes. Set `PYTHONPATH=src` and
`STS_LIGHTSPEED_BUILD` to a verified native build compatible with the Python
interpreter and environment patch. The package does not embed a native binary.
Use separate source/build directories when rebuilding; never reuse another
source tree's CMake cache.

These examples use your compatible teacher bank and task files and create new
artifacts. Behavioral cloning fits teacher actions; proximal policy
optimization (PPO) fits collected returns. Route heavy jobs through the
[queue](../sts_rl/README.md#run-jobs-through-the-queue).

```sh
.venv/bin/python scripts/train_nonllm.py bc --teacher BANK/teacher.jsonl \
  --split train --phase all --size large --context none --device cpu \
  --batch-size 1024 --learning-rate 0.0001 --updates 100 --seconds 120 \
  --output data/custom_nonllm/bc.pt
.venv/bin/python scripts/train_nonllm.py ppo --checkpoint data/custom_nonllm/bc.pt \
  --tasks PANELS/train_act1.jsonl --split train --task-kind battle --max-act 1 \
  --horizon 256 --workers 8 --threads 2 --simulations 200 --gamma 1 \
  --fast-observations --fast-replay --deferred-bootstrap --kl-guard-before-update \
  --updates 100 --seconds 180 --output data/custom_nonllm/combat.pt
.venv/bin/python scripts/train_nonllm.py evaluate --checkpoint MODEL.pt \
  --tasks PANELS/dev_act1.jsonl --split dev --task-kind battle --episodes 32 \
  --max-act 1 --horizon 256 --simulations 200 --output data/custom_nonllm/dev.pt
```

Evaluation writes metrics only; `--output` supplies their naming stem. PPO
uses `--device` for optimization and `--inference-device` for collection; each
accepts `cpu` (processor) or `mps` (Apple's Metal backend). `--task-kind ooc`
exposes out-of-combat choices with native combat, while `run` exposes both
phases. `--teacher-simulations` collects teacher labels on learner states.
Training permits only training worlds. `--resume` requires identical data and
compatible task/reward settings and restores optimizer/random state.
`--warm-optimizer` carries moments across compatible PPO chunks and resets on
method, task-kind or reward changes. `--preserve-task-prefixes` retains saved
frontiers and per-task settings; without it, out-of-combat/run task files
supply deduplicated worlds restarted from the beginning.

The public agent factory accepts candidate or candidate-composite checkpoints:

```sh
.venv/bin/python -m sts_env run --agent sts_policy.model_agent:make_agent \
  --agent-config '{"checkpoint":"MODEL.pt","device":"cpu","deterministic":true}' \
  --mode full --seed 10121 --policy-seed 777 --max-act 1 --max-steps 2048 \
  --battle-simulations 200 --step-timeout 10 --output data/custom_nonllm/run.jsonl
.venv/bin/python -m sts_env inspect data/custom_nonllm/run.jsonl
.venv/bin/python -m sts_env replay data/custom_nonllm/run.jsonl
```

Choose a declared development world; existing trace paths reject. `full` gives
both phases to the model, `ooc` gives combat to native search, and `combat`
gives out-of-combat choices to the scripted native policy with seeded pathing.
Deterministic play uses exact argmax. The public command line gives reward 1
only for game victory and truncates at act limits; curriculum shaping and
Act-clear promotion below do not apply there.

## Checkpoints and composition

Load format 1 with `training.load_checkpoint`, then reconstruct candidates or
composites through `model_agent.policy_from_payload`. It validates encoder
version, supported kind, fixed dimensions and exact weight structure.
Candidate configs may omit `kind`; composite configs use `kind: composite`
with candidate `combat` and `ooc` configs. Nested composites are rejected.

```sh
.venv/bin/python scripts/compose_nonllm.py --combat-checkpoint COMBAT.pt \
  --ooc-checkpoint OOC.pt --output data/custom_nonllm/composite.pt
```

Composition copies weights without overwriting either source, checks stable
inputs and rejects disagreement in recorded provenance. Missing legacy native
provenance stays explicitly missing; campaign/data manifests remain
authoritative. It starts at step zero without optimizer state. Later PPO
updates both heads; normal resume restores their parameter order and optimizer
moments. Phase dispatch runs only the appropriate head, using public encoder
state feature 7. `make_agent` also accepts both `combat_checkpoint` and
`ooc_checkpoint` instead of `checkpoint`. The training command line's
two-source loading is evaluation-only; compose one file for training.
Composition accepts candidate heads only.

Entity and controlled payloads require `entity_model.policy_from_payload` or
`controlled_entity.policy_from_payload`. They are unsupported by `ModelAgent`
and candidate composition; fresh-start inference uses `StudyPolicy`, and
controlled replacements use its controlled policy factory. Entity inference
receives one dictionary batch; candidate inference receives an unpacked tuple.
Source-file compatibility traps belong in [the tooling
instructions](../../scripts/rl/CLAUDE.md).

Checkpoints retain configuration, weights, optional optimizer state, update
count and random-number-generator state. Training records parent identities,
composition ancestry and current environment provenance. With
`STS_LIGHTSPEED_BUILD` set, training validates the manifest against the actual
native module. Preserve model files, banks, manifests and build identities
separately: they are gitignored. Freeze manifests identify the selected
artifact; do not infer selection from its filename. Preserve the original
identity record when relocating artifacts.

## Collection and replay

`workers.CurriculumWorker` keeps the native simulator in a persistent
subprocess; inference, learning and model memory stay in the parent. Reset,
step and teacher queries have finite bounds and asynchronous submit methods. A
native failure destroys the worker; reset starts a clean process. Faulted
episodes and labels never become ordinary defeats or targets. See [fault
recording and replay](../sts_rl/README.md#records-replay-and-dashboards).

`curriculum.TaskSpec` retains seed, kind, exact action prefix, optional public
start hash, horizon, act limit, ascension and search settings. Native
out-of-combat previews are queries, but executing the chosen action commits
the policy's random stream. Preserve aligned `ooc_previews` markers when
replaying prefixes. `task_start_prefix`/`task_start_ooc_previews` are
diagnostics, never inputs. Replay validates native and Python
interface/runtime provenance before execution; a matching public start hash
alone cannot establish compatibility.

`scripts/collect_curriculum.py --output DIRECTORY --worlds 32 --workers 4
--max-act 1 --simulations 200 --fast-observations` calls
`data.collect_dataset`. Every fight/state from a world shares its stable
training/development/final split. Atomic per-world journals preserve completed
worlds; errors discard the entire world's labels. Resume checks
source/settings and probes live native/interface identity against saved valid
worlds. Changed collection code/settings require a new directory, even when
old banks remain replay-compatible. Summary aggregation streams worlds to
bound memory.

`fast_observations=True` installs process-local compiled patterns and bounded
immutable-text caches without changing observations. `fast_replay=True` skips
intermediate prefix text rendering while preserving semantic actions, preview
commits, counters, terminal/menu checks and sticky/live undefined-behavior
flags; it renders the final frontier. Command-line flags are
`--fast-observations` and `--fast-replay` where supported. Candidate text hash
caching and phase dispatch are automatic. Rollouts normally retain their exact
encodings for PPO reuse; reuse checks observation identity/legal indices, ends
with the buffer and discards faulted trajectories. Custom collators keep their
original path; `reuse_encodings=False` disables reuse. `--threads` controls
Torch processor threads; measure full collection throughput before changing
threads or worker layout.

## Tasks and rewards

Curriculum rewards and termination belong to the adapter, not the environment.
Battle tasks terminate at fight outcome; out-of-combat tasks resolve combat
with native search; run tasks expose both phases. Horizons truncate for value
bootstrap. Default `goal_act_clear=True` promotes the base act-limit
truncation to terminal `act_clear`; set it false to retain the base cutoff.
Battle victory, Act clear and whole-game victory remain distinct outcomes.

Reward version `sts_curriculum_reward_v2` offers `potential` (default),
`sparse` and `progress`. Potential uses public player/enemy health, with
`gamma * Phi(next) - Phi(previous)`, zero terminal potential and no block
reward. Cutoffs retain potential for bootstrap. Match and record the learner's
discount to worker `gamma` (default 1). Progress adds `progress_floor_weight`
(default 0.05) per new floor and changes the objective. `terminal_hp_weight`
defaults to zero. Components/settings are diagnostics; faults produce no
reward targets.

## Evaluation

Match exact task identities, splits, horizons, act limits, policy seeds and
native search settings across arms. Out-of-combat/run evaluation deduplicates
worlds from fight banks; battle evaluation retains distinct prefixes. Select
on development data and reserve final worlds until selection freezes. Require
exact requested-task coverage and retain wins/requested for conservative
selection. Report faults, defeats, cutoffs, unresolved outcomes and unstarted
tasks separately. `scripts/report_nonllm.py` supplies equal-world estimates
and 95% bootstrap intervals with at least two worlds; many fights from one
world remain one cluster. Its success ratios condition on decisive completion,
excluding faults/censoring; keep outcome counts beside them. World-level
cloning intervals need per-record `bc_records` with world seed, phase,
correctness and loss, not just aggregate accuracy. For comparisons report the
paired difference, baseline, world count, interval and evidence level.
Repeated development selection is descriptive evidence.

`build_curriculum_panels.py --banks A B --output DIRECTORY` selects genuine
per-act training cohorts and balanced development/final fights across worlds,
without oversampling evaluation starts. Metadata retains coverage/source
identities. Publish changed panels separately; never change an active
selection cohort. `aggregation.aggregate_teachers` builds a deterministic
training-only rehearsal reservoir, validates splits/legal labels and preserves
source-line provenance. Compression and mixing remain explicit settings;
source files stay unchanged.

## Limits and optional teachers

Privileged labels remain separate from public policy inputs.

- Native combat search uses hidden state; native out-of-combat choices are
  scripted. `lookahead.CounterfactualOOCWorker` tests a bounded partial menu
  with native combat and scripted continuations. Defaults are four candidates
  and 30 decisions; the baseline is included and wins score ties. Any branch
  fault rejects the source label. Its proxy scores success +1, defeat -1,
  final health fraction weighted by 0.25 and new floors by 0.05; horizons and
  terms are logged. `collect_ooc_lookahead.py` samples training-bank states by
  default and separately validates source/code/settings on resume. Only the
  original public observation and selected index become training inputs.
- Act-boundary healing can saturate the proxy's health term. `hp_cost` is
  reset health minus endpoint health, can be negative after healing, and is
  not cumulative damage or remaining health. Fresh panel pre-healing loss has
  a separate definition.
- The current native patch checks reported undefined behavior during automatic
  combat initialization and playout. Older banks can predate this check; zero
  surfaced faults never certifies the absence of unflagged corruption.
  Preserve original conditions rather than silently relabelling them.
- Hash collisions, partial text parsing and missing public fields limit
  candidate features. Unknown symbols stay hashed; missing damage stays zero.
  Supported block rules use public Dexterity/Frail/No Block, while complex
  fallbacks can be approximate. Track coverage and bump encoder
  versions/re-encode when semantics change.
- Spawn-based workers require a saved script with an `if __name__ ==
  "__main__"` guard or an official guarded CLI; a Python standard-input entry
  point cannot launch them.
