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
- MLX/Qwen inference has been smoke-tested with `mlx-community/Qwen3-4B-4bit`; no-thinking mode produces valid structured actions on short rollouts.
- Simulator error handling is now strict: invalid battle actions and unknown potion values raise errors, batch rollouts write `.error.json` sidecars, and optional per-seed subprocess timeouts keep slow or faulty seeds isolated.

### Latest Stage 1 Run Notes

Baseline pass:

- Dataset directory: `data/baseline_rollouts_100`.
- Agents: `first`, `random`, `heuristic`.
- Seeds: `2-51`.
- Settings: `max_decisions=200`, `battle_simulations=100`, `seed_timeout_seconds=30`.
- Summary CSV: `data/baseline_rollouts_100/summary.csv`.

Observed baseline reliability:

- `first`: 46/50 clean seeds; timeout seeds `11, 17, 22, 48`.
- `random`: 47/50 clean seeds; timeout seeds `11, 25, 48`.
- `heuristic`: 46/50 clean seeds; timeout seeds `11, 17, 22, 48`.
- Clean intersection across all three agents: 45 seeds.
- First 10 clean intersection seeds used for Qwen smoke: `2, 3, 4, 5, 6, 7, 8, 9, 10, 12`.

The earlier `500` battle-simulation setting is useful as a simulator stress test but too slow/flaky for fast Stage 1 iteration. Seed `1` should be kept as a diagnostic regression seed, not included in the initial frozen dev/eval set.

Serializer audit on the clean baseline traces found no raw screen numbers, unknown potion names, fallback action labels, or missing major screen coverage. Covered screen types include Neow/events, map, rewards, shops, campfires, card select, treasure rooms, and boss relic rewards.

Qwen no-thinking smoke:

- Dataset directory: `data/qwen_smoke_100/mlx_Qwen3_4B_4bit_nothinking_128`.
- Model: `mlx-community/Qwen3-4B-4bit`.
- Seeds: `2, 3, 4, 5, 6, 7, 8, 9, 10, 12`.
- Settings: `max_decisions=20`, `battle_simulations=100`, `max_tokens=128`, `temperature=0`, `max_retries=1`, no thinking mode.
- Result: 10/10 seed files completed, 200 total decisions, no simulator error sidecars.
- Valid action rate: 98.0%.
- Retry count: 29/200 decisions used one retry.
- Invalid action count: 4/200 decisions. All observed invalids were truncated JSON after verbose reasoning, falling back to action `0`.
- Mean final floor after 20 decisions: 5.1.
- One run reached low HP by the cutoff: seed `6` ended at 15/80 HP on floor 5.

Before a larger Qwen batch, decide between:

- increasing the no-thinking output budget from `128` to `256`; or
- tightening the prompt/schema so the model emits very short reasoning or only the final action JSON.

Follow-up token-budget comparison:

- Same first five seeds: `2, 3, 4, 5, 6`.
- `128` tokens: 100 decisions, 98 valid, 14 decisions needed one retry.
- `256` tokens: 100 decisions, 100 valid, 1 decision needed one retry.
- No simulator error sidecars in either comparison.

This suggests `256` tokens is a better near-term no-thinking rollout budget if we keep the current JSON schema with a `reasoning` field. A stricter short-reasoning prompt may recover some of the throughput while keeping parse reliability high.

Thinking-mode comparison (2026-06-14):

- Dataset: `data/qwen_thinking_2048_cmp/mlx_Qwen3_4B_4bit_thinking_2048`, seeds `3, 4, 5`, `max_decisions=12`, `battle_simulations=100`, `max_tokens=2048`, `temperature=0`, `max_retries=1`, thinking mode, `--seed-timeout-seconds=1200`.
- Result: 36 decisions, **32 valid (88.9%)**, 9 retries; all 4 invalids were `no json object` — the model exhausted the 2048-token budget mid-`<think>` and never emitted the final JSON (the single retry also truncated).
- Throughput: ~11.5 min wall for 36 decisions ≈ **~19 s/decision** (incl. battle resolution), vs the sub-2 s/decision no-thinking arm.
- Same seeds under no-thinking `256` were **100% valid (60/60), 0 retries**.
- Takeaway: at 2048 tokens thinking mode is both slower and *less* reliable than no-thinking `256` on these states, because verbose reasoning truncates before the JSON. For a viable thinking comparison arm, either raise the budget (≥4096) or use a "think briefly, then emit JSON" prompt. This confirms no-thinking `256` as the Stage-1 high-throughput primary arm and leaves thinking mode as a still-unsettled comparison arm.

Fresh seed-2 check (2026-06-14):

- A current rerun of `mlx_Qwen3_4B_4bit_nothinking_256`, seed `2`, with `max_decisions=80`, `battle_simulations=100`, `temperature=0`, and `max_retries=1` again reached the old boundary: 48 decisions, then the floor-12 battle after the Entropic Brew path.
- On this build the child process pinned inside the native `slaythespire` extension rather than returning a Python-visible simulator error. The run was manually terminated and recorded under `data/qwen_rerun_100_256_current/.../seed_2.error.json`.
- The old `tests/integration/test_battle_search.py` replay stopped on the map before entering that battle (its `max_decisions` equalled the replayed decision count, so the battle-resolving `advance_to_decision()` never ran — it passed in ~40ms without exercising the bug). It now appends the map action *and* gives the rollout headroom past it so the floor-12 battle is actually entered whenever the path reaches the map node, and runs the replay in a subprocess with a timeout, matching the operational containment strategy in `scripts/run_batch.py`.
- The seed-2 path is non-deterministic across runs on this build (observed: >90s native hang, clean resolve, and early divergence before floor 12), so the test asserts only build-portable containment invariants (no hard crash, no garbage-potion `invalid battle action` regression), not that the battle is reached.
- Treat seed-2-class trajectories as unresolved simulator-search failures. Do not run long Qwen batches in-process; use `--seed-timeout-seconds`, and do not freeze seed `2` into an initial dev/eval set until this native battle-search issue is root-caused or explicitly accepted as an excluded seed.

Do not silently change this policy because output budget and reasoning verbosity affect rollout cost, parse reliability, and the training data distribution.

### Frozen Seed Policy (documented 2026-06-14; not yet a hard freeze)

Derived from the `data/baseline_rollouts_100` batch (seeds 2-51, agents `first`/`random`/`heuristic`).

Exclusion policy:

- **Errored seeds** (any agent wrote an `.error.json` sidecar): `11, 17, 22, 25, 48`. Excluded.
- **Seed `2`**: clean for the non-LLM agents (they path around it) but a known seed-2-class native battle-search hang on the Qwen LLM path (floor-12 Entropic Brew). Excluded from any LLM dev/eval set.
- **Seed `1`**: kept as a diagnostic regression seed only; never in dev/eval.

Clean intersection across all three baseline agents: **45 seeds** —
`2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51`.
Removing seed `2` leaves **44 LLM-safe seeds**.

Proposed splits (provisional, from these 44):

- **Smoke (10):** `3, 4, 5, 6, 7, 8, 9, 10, 12, 13`.
- **Dev (next ~24):** `14, 15, 16, 18, 19, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40`.
- **Eval (100):** NOT yet satisfiable — only 44 LLM-safe seeds exist in `2-51`. Requires a larger baseline batch (e.g. seeds `2-300`) before an eval freeze.

Blockers before this becomes a hard freeze:

1. **Serializer rebuild pending.** The `bits=` action-description prefix and `room INVALID` header label are being removed (binding change, rebuild in progress). Frozen traces should be generated under the final serializer, so freeze only after the rebuilt binding and a re-validated baseline.
2. **UB reproducibility caveat.** Cross-machine identical traces are not guaranteed while the seed-2-class uninitialized-memory UB is only contained, not root-caused (`docs/simulator_issue_handoff.md`). A single-machine freeze is usable for now; cross-machine reproducibility is not.

This hybrid approach is deliberate. The existing upstream Python binding does not expose combat micro-actions. Letting Lightspeed resolve battles gets us useful Act 1 trajectories and risk-relevant decisions quickly, while full combat control remains a later C++ binding task.

Tracked implementation pieces:

- `scripts/build_lightspeed.sh` builds the local simulator binding.
- `patches/sts_lightspeed_python_api.patch` adds the Python API needed by the harness.
- `src/sts_ai/lightspeed.py` wraps the simulator.
- `src/sts_ai/agents.py` defines baseline agents and the optional MLX/Qwen JSON agent.
- `src/sts_ai/rollout.py` records structured rollout traces.
- `scripts/run_rollout.py` runs one rollout from the CLI.

Generated/local artifacts are intentionally untracked: `.venv`, `external/sts_lightspeed`, build outputs, and rollout JSONL files.

## Review-Driven Priority Update

A first external review of the harness found that the simulator/control architecture is sound, but also identified several fixes that should happen before collecting any frozen seed dataset.

Immediate changes to make before Stage 1 batch rollouts:

- Make state/action serialization human-judgeable:
  - use screen names instead of raw integer screen codes;
  - fix reward labels, especially Singing Bowl max-HP choices;
  - remove cosmetic Neow labels such as empty trailing drawback slashes;
  - improve map/shop/campfire/reward descriptions before freezing seeds.
- Harden the LLM JSON action path:
  - use the model chat template for Qwen/MLX inference;
  - handle Qwen3 `<think>...</think>` output when extracting final JSON;
  - support modern MLX-LM sampling APIs;
  - implement retry-on-invalid output.
- Add regression tests for:
  - JSON extraction with think blocks, braces, and multiple objects;
  - invalid action fallback behavior;
  - known risk-relevant action labels;
  - Act 1 boundary behavior.

Updated near-term ordering:

1. Fix serializer/action labels. Done.
2. Harden JSON extraction and retry behavior. Done.
3. Smoke-test one real Qwen3/MLX decision. Done with `mlx-community/Qwen3-4B-4bit` in no-thinking mode.
4. Add batch rollout and metrics tooling.
5. Harden battle-search robustness against uninitialized-memory UB (seed-2 Entropic
   Brew crash/hang). Done — contained, not fully root-caused; see
   `docs/simulator_issue_handoff.md`.
6. Only then freeze dev/eval seeds. (Note: the residual UB is build-/layout-dependent,
   so frozen-seed reproducibility across toolchains is not yet guaranteed — resolve
   the deeper UB before depending on cross-machine identical traces.)

## Design Commitments

### Model Adapter Modularity

The simulator and rollout recorder should never depend directly on a model provider, tokenizer, or chat template.

The stable boundary is:

```text
state_text + legal_actions -> ActionAgent.choose_action(...) -> AgentDecision
```

Provider-specific concerns live inside agent adapters:

- tokenizer and chat template handling;
- thinking-mode controls;
- sampling parameters;
- retry behavior;
- raw response parsing;
- model-specific metadata.

This lets the repo add future adapters for other MLX models, Transformers models, hosted APIs, vLLM/OpenAI-compatible servers, or non-LLM policies without changing the simulator wrapper or rollout schema.

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

For the next Qwen rollout stage, use a two-arm policy:

- **High-throughput arm:** no-thinking mode with a small output budget for baseline rollout collection.
- **Comparison arm:** thinking mode with `2048` output tokens on a smaller seed set.

The benchmark motivating this choice was:

- no-thinking, `128` tokens: 3/3 valid decisions, 0 retries, about 1.9 seconds/decision;
- thinking, `512` and `1024` tokens: failed to emit final JSON in the tested cases;
- thinking, `2048` tokens: 4/4 valid decisions across tested runs, 1 retry total, about 27 seconds/decision in a 3-decision run;
- thinking, `4096` tokens: 1/1 valid, similar one-decision latency to `2048`.

This is not a permanent decision. It is the working policy for Stage 1/3 de-risking so we can collect enough no-thinking data while preserving a smaller thinking-mode comparison.

For later thinking-enabled training, allow thinking during generation for capability, but keep the primary supervised/action loss on the final structured action tokens.

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

### Simulator Fault Handling

Research traces should fail closed. A simulator warning is not safe to ignore, because a bad battle-search action can change the state distribution that the LLM later trains on.

Current policy:

- reject invalid battle actions in release builds before executing them;
- reject unknown potion enum values instead of indexing name/effect tables out of range;
- initialize `potions` arrays in `GameContext`/`BattleContext` and guard the battle
  search against transiently-corrupt potion slots / non-terminating playouts where
  the native code returns (see `docs/simulator_issue_handoff.md` for the
  unresolved seed-2-class native hang);
- convert the simulator's internal `while(true)`/overflow guards from `assert(false)`
  to thrown exceptions, since asserts are compiled out in release builds and would
  otherwise hang instead of failing closed;
- record recoverable simulator failures as `stopped_reason=simulator_error`;
- record batch failures and timeouts as `seed_<n>.error.json` sidecars;
- use `--seed-timeout-seconds` for larger non-LLM baseline batches so each seed runs in a subprocess and can be killed independently.

Hard C++ asserts remain useful for local debugging, but they are not the default batch mechanism because an abort tears down the whole Python interpreter. **They are also no-ops in our release builds**, so any guard that must fire in production has to throw, not assert. Python-visible exceptions plus subprocess timeouts give cleaner failure accounting. Some seed-2-class paths still require the subprocess timeout; they are not yet cleanly recoverable inside a single Python process.

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
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .` passes (the
  `tests/unit` tier runs without a build; `tests/integration` needs the simulator —
  see `tests/CLAUDE.md`);
- `scripts/run_rollout.py --agent first` writes a valid JSONL trace.

### Stage 1: Baseline Rollout Dataset

Goal: produce reliable fixed-seed baseline trajectories before involving an LLM.

Tasks:

- add a batch rollout CLI for seed ranges; done;
- run `first`, `random`, and `heuristic` agents on a frozen Act 1 seed set;
- summarize floor reached, outcome, HP, gold, number of decisions, and decision screen distribution;
- track `stopped_reason` and error sidecars separately from successful rollouts;
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

Done in the 2026-06-14 session:

- Serializer audit + fixes: removed the `bits=` action-description prefix (now on `LegalAction.bits` only) and render the Neow/event `room INVALID` header as `room none`. Binding rebuilt, patch regenerated, tests updated (`test_action_descriptions_omit_bits_prefix`, `test_state_room_label_is_not_invalid`).
- Thinking-mode `2048` comparison run (seeds 3-5): 88.9% valid, truncation-limited; no-thinking `256` remains the high-throughput primary arm (see Thinking-mode comparison above).
- Fixed the `tests/integration/test_battle_search.py` off-by-one that let it pass without entering the floor-12 battle; documented seed-2 non-determinism and the containment-only invariants.
- Documented the frozen-seed exclusion policy and proposed smoke/dev splits (see Frozen Seed Policy).

Remaining:

1. Root-cause or explicitly accept-and-exclude the seed-2-class native battle-search hang. Use a debug/sanitizer build or systematic value-initialization audit before depending on cross-build/cross-machine reproducibility.
2. Run Qwen no-thinking `256` on the full clean baseline intersection (excluding `{2, 11, 17, 22, 25, 48}`) under the rebuilt serializer, always with `--seed-timeout-seconds`; manually inspect traces.
3. If a thinking comparison arm is needed, retest with a larger budget (≥4096) or a "think briefly, then emit JSON" prompt to beat truncation.
4. Run a larger baseline batch (e.g. seeds `2-300`) so an eval freeze of ~100 seeds is possible, then hard-freeze dev/eval under the final serializer.
