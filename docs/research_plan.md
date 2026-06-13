# SlayTheSpireAI Research Plan

## Overall Goal

This repo is a research harness for testing how training-time framing changes what an LLM learns from the same or similar reward signal.

The motivating scientific question is:

> When a model is reinforced for behavior along a graded axis such as risk-taking, does the framing of the training context determine which broader latent concept absorbs the update?

For example, if two models see equivalent successful trajectories but one is framed as making "risk-reward tradeoffs" and the other as being "adventurous", do their downstream behaviors generalize differently toward risk-seeking, adventure-seeking, fun-seeking, confidence, impulsivity, or other nearby traits?

The first concrete environment is Slay the Spire through `gamerpuppy/sts_lightspeed`, starting with Qwen3-4B as the initial model target. Slay the Spire is not the cleanest possible scientific environment, but it is a useful ecological testbed because risk is partly emergent: pathing, low-HP campfire choices, elite fights, card rewards, shops, potions, and boss preparation all create real tradeoffs.

The repo should eventually support two complementary experiment arms:

- **Fixed-rollout arm:** generate neutral trajectories once, then train framing variants on the same states/actions/rewards. This best isolates interpretation effects: same data, different frame.
- **On-policy arm:** generate trajectories under each framing. This captures the full effect of framing on both interpretation and visited data distribution.

The MVP prioritizes the fixed-rollout arm because it is cheaper, cleaner, and better suited to debugging the pipeline.

## Current Implementation Status

The first working slice is a hybrid Slay the Spire rollout harness.

Current behavior:

- `sts_lightspeed` is cloned and built locally.
- A pybind patch exposes out-of-combat `GameAction`s to Python.
- Python can list legal out-of-combat actions, describe the state, execute selected actions, and record decisions as JSONL.
- Combats are resolved by the built-in Lightspeed search agent.
- The Python policy controls Neow choices, pathing, rewards, shops, events, card select screens, treasure rooms, and campfires.

This hybrid approach is deliberate. The existing upstream Python binding does not expose combat micro-actions. Letting Lightspeed resolve battles gets us useful Act 1 trajectories and risk-relevant decisions quickly, while full combat control remains a later C++ binding task.

Tracked implementation pieces:

- `scripts/build_lightspeed.sh` builds the local simulator binding.
- `patches/sts_lightspeed_python_api.patch` adds the Python API needed by the harness.
- `src/sts_ai/lightspeed.py` wraps the simulator.
- `src/sts_ai/agents.py` defines baseline agents and the optional MLX/Qwen JSON agent.
- `src/sts_ai/rollout.py` records structured rollout traces.
- `scripts/run_rollout.py` runs one rollout from the CLI.

Generated/local artifacts are intentionally untracked: `.venv`, `external/sts_lightspeed`, build outputs, and rollout JSONL files.

## Design Commitments

### Markovian State Prompting

The default policy input should be a canonical current-state serialization, not the full episode transcript.

This is standard RL practice when the observation contains all decision-relevant state. For Slay the Spire, the simulator state should encode the consequences of history: HP, deck, relics, potions, map position, current screen, reward state, and other run variables.

The risk is not Markovian prompting itself; the risk is an incomplete serializer. Before training, run a state sufficiency audit:

- inspect prompts for several screen types;
- compare decisions under compact and verbose state serializers;
- confirm risk-relevant fields are present;
- add missing fields before collecting fixed rollout data.

### Fixed Rollouts First

The first scientific arm should use neutral-frame rollouts and then train framing variants from the same stored trajectories.

This controls the data and reward, but not the gradient. The gradient remains frame-conditioned because token probabilities differ under different prefixes. That is part of the mechanism being tested.

Neutral rollout data should avoid framing leakage in reasoning. If Qwen reasoning is used, it should be generated under a neutral frame and audited for frame-specific language before reuse in framed training conditions.

### Thinking Policy

For StS-first MVP, allow thinking during generation for capability, but keep the primary supervised/action loss on the final structured action tokens.

The forward context should include the reasoning that preceded the action, otherwise the action likelihood is evaluated under a different context than the one that generated it. Reasoning can be masked from the primary action loss initially. Later experiments can compare:

- no-thinking rollouts;
- thinking in context with action-only loss;
- thinking in context with full trajectory loss.

### Local Training Is a Benchmark, Not an Assumption

The target model is Qwen3-4B. The MacBook Pro with 48GB unified memory may be enough for some local fine-tuning paths, but the repo should measure before depending on full-parameter local training.

Fallback order:

1. local inference plus rollout collection;
2. local LoRA or tiny-slice training smoke tests;
3. smaller Qwen model for end-to-end validation;
4. cloud GPU for full-parameter RL or large SFT-style runs if local training is too slow.

## Roadmap

### Stage 0: Reproducible Simulator Harness

Goal: make the local simulator importable and controllable from Python.

Done:

- clone `sts_lightspeed`;
- initialize submodules;
- build local pybind extension;
- expose legal out-of-combat actions;
- expose battle-only resolution helper;
- record JSONL decisions.

Acceptance criteria:

- `scripts/build_lightspeed.sh` completes on the local machine;
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` passes;
- `scripts/run_rollout.py --agent first` writes a valid JSONL trace.

### Stage 1: Baseline Rollout Dataset

Goal: produce reliable fixed-seed baseline trajectories before involving an LLM.

Tasks:

- add a batch rollout CLI for seed ranges;
- run `first`, `random`, and `heuristic` agents on a frozen Act 1 seed set;
- summarize floor reached, outcome, HP, gold, number of decisions, and decision screen distribution;
- inspect sampled traces manually for bad action descriptions or missing state fields;
- decide which seeds become the frozen dev/eval sets.

Acceptance criteria:

- no harness crashes across at least 100 seeds with a non-LLM agent;
- rollout JSONL schema is stable enough to consume in training/eval scripts;
- all common screen types have readable state and action descriptions;
- battle resolution returns control to Python after combat rewards.

### Stage 2: State Serialization and Risk-Proxies

Goal: make the prompts and measurements good enough for the research question.

Tasks:

- improve state text for map, rewards, shops, campfires, and events;
- add structured risk tags where they are clear and non-invasive;
- compute initial risk proxies:
  - campfire rest vs smith under low/medium/high HP;
  - elite pathing when current HP is low;
  - potion use and purchase patterns;
  - skip/take decisions for high-variance or self-damage cards;
  - Neow choices with obvious downside/reward tradeoffs;
  - shop spending vs saving before known threats.
- add a rollout summarizer that turns JSONL into CSV or JSON metrics.

Acceptance criteria:

- fixed-state prompts include enough context for a human to judge the listed actions;
- risk proxy code is deterministic and documented;
- baseline metrics can be computed from existing traces without re-running rollouts.

### Stage 3: Qwen3-4B Local Inference Loop

Goal: replace baseline agents with a local Qwen3-4B JSON-action agent.

Tasks:

- install optional `mlx-lm` dependency;
- run Qwen3-4B on a handful of states;
- enforce strict JSON final action format;
- add retry-on-invalid behavior;
- log raw response, reasoning, parsed action, validity, retries, and token counts if available;
- tune prompt enough to reach high action validity.

Acceptance criteria:

- Qwen action parsing is valid on >95% of decisions after one retry;
- a short fixed-seed rollout completes without harness crashes;
- generated JSONL records contain enough prompt/response metadata for later training;
- throughput is measured in decisions/minute and tokens/decision.

### Stage 4: Qwen Baseline Evaluation

Goal: determine whether Qwen3-4B can play the hybrid task well enough to train on.

Tasks:

- run Qwen3-4B on the frozen dev seed set;
- compare against random and heuristic baselines;
- inspect failure modes:
  - invalid actions;
  - obviously poor path choices;
  - pathological reward choices;
  - repeated format drift;
  - state misunderstanding.
- run a compact-vs-verbose serializer comparison on matched states.

Go/no-go criteria for training:

- Qwen produces valid actions reliably;
- behavior is non-random on at least some metrics;
- rollout throughput is adequate for hundreds of decisions overnight;
- state serializer failures are not dominating choices.

If these fail, improve prompt/serializer first or add a small SFT warm start from heuristic/search trajectories.

### Stage 5: Fixed Neutral Rollout Collection

Goal: create the shared data for the first framing experiment.

Tasks:

- choose fixed train/dev/eval seed splits;
- generate neutral-frame Qwen trajectories;
- store complete decision records;
- audit reasoning for frame leakage;
- filter or flag malformed examples;
- compute rewards and risk proxy labels after the fact.

Recommended initial dataset:

- small smoke set: 10 seeds;
- dev set: 50 seeds;
- first train set: 200-500 seeds, depending on throughput;
- frozen eval set: 100 seeds.

Acceptance criteria:

- train/dev/eval splits are fixed and documented;
- trajectory records are deterministic enough to replay or inspect;
- all framing conditions can consume the same records.

### Stage 6: Training Feasibility Benchmark

Goal: establish what training is practical on the Mac before designing expensive runs around it.

Tasks:

- convert rollout JSONL into a simple action-training dataset;
- build loss masking:
  - no loss on prompt/state tokens;
  - no primary action loss on reasoning tokens;
  - loss on final structured action tokens;
- run a tiny local training smoke test;
- measure memory, tokens/sec, checkpoint size, and wall-clock time;
- compare full-parameter vs LoRA feasibility if tooling supports both.

Acceptance criteria:

- one tiny training job completes;
- trained checkpoint can be loaded for inference;
- action likelihood changes on held-out examples in the expected direction;
- local full-parameter training is either validated or ruled out pragmatically.

### Stage 7: Initial Three-Frame Experiment

Goal: test whether different framing blocks produce different generalization from the same neutral rollout data.

Initial frames:

- neutral;
- risk-reward;
- adventurous.

Training setup:

- same base checkpoint;
- same fixed rollout records;
- same action targets and reward-derived weights;
- only the framing instruction block changes;
- compare at matched update count and, if feasible, matched KL from the base model.

Primary readouts:

- action likelihood shifts on held-out fixed states;
- rollout performance on frozen eval seeds;
- changes in risk proxy behavior;
- KL from base model;
- invalid-action and format-drift rates;
- optional reasoning-style classifier scores.

Acceptance criteria:

- all three variants train and evaluate through the same pipeline;
- eval uses frozen seeds and fixed prompts;
- results distinguish performance changes from risk-proxy changes;
- any apparent framing effect survives a basic sanity check for KL/update-size mismatch.

### Stage 8: Stronger Measurement

Goal: separate capability changes from motivational or preference-like generalization.

Tasks:

- build matched-state probes for risk-relevant decisions;
- sample multiple responses per state to estimate action propensity;
- add non-StS preference probes about risk, adventure, fun, safety, and prudence;
- optionally add utility-elicitation or pairwise preference prompts;
- compare before/after training for each frame.

Acceptance criteria:

- measurements are fixed before major training runs;
- probes include both in-domain StS decisions and out-of-domain generalization prompts;
- results can answer whether framing redirects generalization, not merely whether it improves gameplay.

### Stage 9: On-Policy Framing Arm

Goal: measure the combined effect of framing on data distribution and interpretation.

Tasks:

- run separate rollouts under neutral, risk-reward, and adventurous frames;
- train each condition on its own on-policy data;
- compare against the fixed-rollout arm;
- measure whether each framing visits different states before training and after training.

Interpretation:

- fixed-rollout differences estimate interpretation effects;
- on-policy differences estimate interpretation plus data-distribution effects;
- the gap between them is evidence about mediation through visited trajectories.

### Stage 10: Full Combat Control

Goal: let the LLM control every meaningful StS decision, including card play.

Tasks:

- expose `BattleContext` and combat `Action`s through pybind;
- serialize combat state: hand, draw/discard/exhaust piles, enemies, intents, powers, block, energy, potions, turn counters, and legal card targets;
- add combat action parsing and execution;
- decide whether each card play is a separate decision or whether one turn is a macro-action;
- update rollout schema to distinguish combat and out-of-combat decisions.

Acceptance criteria:

- LLM can complete at least one full combat through Python control;
- legal combat actions are complete and valid;
- hybrid and full-control modes are both supported for ablations.

## Evaluation Philosophy

The repo should avoid treating "beats Act 1" as the first success criterion. The first success criterion is a reliable measurement and training harness.

Early de-risking metrics:

- simulator build and import success;
- rollout stability;
- action validity;
- decisions/minute;
- prompt length;
- state serializer sufficiency;
- frozen-seed reproducibility;
- clear risk proxy extraction.

Later scientific metrics:

- performance-adjusted risk behavior;
- matched-state action propensities;
- out-of-domain preference shifts;
- KL-matched framing comparisons;
- fixed-rollout vs on-policy differences.

## Known Limitations

- Hybrid control means the LLM is not yet learning combat tactics.
- Built-in combat search may mask some consequences of bad pathing/reward choices.
- StS risk is messy and partly subjective; risk proxies must be treated as imperfect.
- Fixed-rollout action training is not exactly on-policy RL. It controls data and reward while deliberately allowing frame-conditioned gradients.
- Qwen3-4B local full-parameter training may be slower or tighter than expected; the repo should benchmark rather than assume.

## Near-Term Next Steps

1. Add a batch rollout script for seed ranges.
2. Improve state/action descriptions for map, shop, campfire, and reward screens.
3. Add a rollout metrics summarizer.
4. Install and smoke-test MLX/Qwen3-4B inference.
5. Run Qwen on 10 fixed seeds and inspect every trace manually.
6. Freeze initial dev/eval seeds only after serializer issues are fixed.
