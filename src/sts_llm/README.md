# Public-observation LLM adapter

This package provides single-decision teacher-guided training inputs and saved-state/full-game evaluation over the released clean environment. It does not change `sts_env` or start a training study. Teacher probability measures the champion's preference, not expected game wins or reasoning quality.

## APIs

- `observations.prompt(observation)` returns the shared two-message prompt. It copies only complete public observation text and the raw current action menu. It never serializes `info`, world seeds, teacher labels, provenance or policy features into messages. `legal_indices(observation)` preserves native display indices.
- `observations.labelled_prompt(observation, order=None)` is prompt version 2 (`public_text_labelled_menu_v2`). It uses the same state text, then the menu in the given order, with letter labels (`A) ...`, `B) ...`; `AA` after `Z`). The labels cannot be confused with the `[0]`-style indices of hand cards, enemies and potions. It returns `LabelledPrompt(messages, labels, order)`, where `labels` maps each label to its native index and never enters the messages. A fresh random order per presentation removes menu position as a learnable shortcut. Parse answers with `parse_action(..., labels=prompt.labels)`: the final JSON must be exactly `{"action": "<label>"}`, and the returned `action_index` is native. An unknown label is `illegal_action_label`.
- `decision_metrics` holds the quantities that decide results:
  - `centered_advantages` gives a group-mean baseline with no standard-deviation scaling;
  - `teacher_table`, `teacher_top` and `decisive_margin` (top minus second teacher probability) describe the teacher's target;
  - `normalized_teacher_score` is 0 for a random legal action in expectation and 1 for the teacher's choice;
  - `reasoning_token_count` counts tokens before the first reasoning closer.
- `generation.parse_action(text, legal_indices, finish_reason=None, reasoning_content=None, final_channel=False, require_reasoning_boundary=True)` returns `ActionParse(action_index, valid, error, reasoning, final_text, truncated)`. Raw K2 completions require a reasoning closer; any of its three effort closers is accepted, regardless of requested effort. The opening tag may be absent because the chat template already supplies it. A native separate reasoning field establishes a final channel. Truncation is a failure even if the visible suffix contains JSON. Final output must be one JSON object with only an integer `action_index`. The actual K2 assistant-end and end-of-text tokens are stripped after the boundary.
- `teacher.NonLLMTeacher(teacher_root=..., checkpoint=..., python=..., timeout=60, batch_size=64)` launches one isolated inference subprocess. `distribution(observation)` or `distributions(observations)` returns display indices, probabilities, excluded menu indices, checkpoint identity and source revision. Requests carry at most `batch_size` observations, and `timeout` applies to startup and to each request separately. The parent never imports the non-LLM environment package. No native simulator is loaded or queried. The worker removes its own directory (`src/sts_llm`) from its import path, so `sts_llm/statistics.py` cannot shadow the standard library. Teacher metadata reads the encoder and checkpoint format versions from the checkpoint; `version_source` says whether they came from the checkpoint or from the historical constants (`public_candidate_v2`, format 1).
- `teacher.teacher_reward(row, action_index, mode='probability', invalid_reward=None, log_floor=1e-8)` uses the prepared row's teacher distribution. Probability is the default bounded reward. A legal action filtered out of the champion's deployment menu receives zero probability and remains distinct from malformed/illegal output. Log probability is optional; its invalid reward must be no larger than `log(log_floor)`, to avoid rewarding invalid output over valid low-probability choices. Default invalid reward is zero for probability or `log(log_floor)` for log probability.
- `dataset.prepare_rows(rows, teacher)` prepares messages and complete target distributions with a stable train/validation/test split over whole source worlds. Source file and public-record identities are checked before loading. Repeated decisions from one world remain in one split. World identities are canonicalized before hashing (split scheme version 2, recorded on every row and in the manifest): an integer seed, its decimal string and `world:<seed>` are one world, so inline and source-referenced rows for the same seed share a split. Rows whose identity was already `world:<seed>` keep their earlier split. `dataset.verified_split(row)` recomputes a row's split from its world and refuses a stored `split` that disagrees; `dataset.check_row_versions(row)` refuses rows whose `prompt_version` or `observation_version` differs from the current prompt and environment.
- `eval.evaluate_saved(rows, client, output=...)` reports legality, truncation, mean teacher probability, top-action agreement and filtered-menu actions. `eval.make_agent(...)` provides `choose_action` for the clean CLI. Invalid model output raises `InvalidModelAction`; it is not turned into a game loss. Forced one-option menus can be stepped without generation. Full response logs retain messages, parse results and raw API responses outside model inputs.

The HTTP client accepts explicit `extra_body` for the serving engine's model-specific reasoning/template options. It does not guess how an endpoint implements effort settings. Raw Hugging Face training completions should retain special tokens when decoding so reasoning boundaries remain visible.

## Prepare data and a portable teacher

Run with this package on `PYTHONPATH`; the teacher's Python needs PyTorch and NumPy. The clean parent runtime needs only the standard library for preparation/evaluation adapters. `--teacher-batch-size` (default 64) and `--teacher-timeout` (default 60 seconds per request) bound each teacher request, so thousands of states need no single long deadline.

```sh
PYTHONPATH=src python -m sts_llm.dataset \
  --prompts PATH_TO_PUBLIC_PROMPT_ROWS.jsonl \
  --teacher-root PATH_TO_WS_NONLLM_CHECKOUT \
  --checkpoint PATH_TO_CHAMPION.pt \
  --teacher-python PATH_TO_NONLLM_PYTHON \
  --output data/llm_runs/pilot/teacher.jsonl \
  --export-runtime data/llm_runs/pilot/teacher_runtime \
  --include-checkpoint
```

Input rows can contain `observation` plus `world_identity`, or the immutable `source_record_identity` emitted by the throughput prompt preparer. Output JSONL contains `id`, `phase`, `legal_indices`, `messages`, `teacher`, canonical `world_identity`, `split`, `split_scheme_version`, `prompt_version`, `observation_version` and provenance. Submit only `messages` to the model. `teacher` uses `{legal_indices: [...], probabilities: [...], excluded_indices: [...]}`; its menu is a subset of the raw prompt menu because the champion hides OOC potion and key/recall actions.

The portable teacher export contains only the source modules actually imported for inference, a checkpoint if requested, and `teacher_runtime_manifest.json`. It is a dependency artifact for `data/`, not core environment code. Point `teacher_root` at this exported directory and `checkpoint` at the included file on another machine. Install compatible PyTorch/NumPy; no native simulator binary is needed. Source hashes record provenance and do not impose byte-for-byte checkpoint compatibility gates.

The prepared scratch example `data/teacher128.jsonl` has 128 distinct states from 93 worlds, 80 combat and 48 OOC; 92 training, 17 validation and 19 test rows. This tiny set is for smoke and timing checks, not a scientific training dataset. Its source observation files match `env-v0.1.5.1`; that speed-only release intentionally reports runtime `0.1.5` and `sts_env_public_v3`.

## Evaluate

```sh
PYTHONPATH=src python -m sts_llm.eval \
  --dataset data/llm_runs/pilot/teacher.jsonl --split validation \
  --base-url http://127.0.0.1:8000/v1 --model MODEL_NAME \
  --max-tokens 4096 --temperature 0 \
  --extra-body '{}' \
  --output data/llm_runs/pilot/saved_eval.jsonl
```

The default split is `validation`. The held-out `test` split requires `--allow-test-split`, for a final, planned evaluation only. Each row's split is recomputed from its world before selection, and rows built for another prompt or observation version are refused. The summary records the split and split scheme version. The example template option must match the selected serving implementation. For full-game evaluation use the existing clean CLI's `--agent sts_llm.eval:make_agent`, with endpoint/model configuration and a fresh `response_log`. Its ordinary game trace remains the authority for game outcomes, simulator faults and truncations; saved-state agreement is not a win-rate estimate. Greedy and sampled decoding differ through the explicit temperature. Environment operation timeouts/process isolation stay in the clean CLI.

## Validation

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Unit tests cover reasoning/final separation, actual K2 terminal tokens, truncation, JSON/schema/illegal-action rejection, public-only prompt construction, probability-to-display-index mapping, excluded legal actions, malformed target rejection, whole-world splitting, evaluator failure accounting, and exported-source revision preservation inside another checkout. A real current-champion integration queried all 128 saved states without playing any games. Exporting the 22 imported source modules and reloading the portable teacher gave exactly identical combat and OOC distributions, with `native_simulator_loaded=false`.

## Training and speed checks

This package trains generated reasoning and final actions with single-decision
non-LLM policy-probability rewards. It has no reasoning SFT targets.

The initial loop uses synchronous Hugging Face generation from the current
PEFT model. It samples the full categorical distribution (`top_p=1`, `top_k=0`)
and scores logits at the same temperature. Old-policy log-probabilities are
frozen once before any updates. The reference is the frozen base model with
adapters disabled. All generated reasoning/final/EOS tokens receive the outcome
advantage; prompt and artificial padding receive no loss. The KL diagnostic is
a sampled proxy.

Actual training requires an explicit `--loss-normalization`: `response_mean`
divides each response's token sum by its realized length, while
`fixed_token_budget` divides every response by the same positive
`--loss-token-budget` (default 2048), for both policy and KL terms. The choice
changes how response length weights the objective and must be fixed across
effort comparisons. Fixed normalization alone does not establish an unbiased
reward gradient: group standardization and clipping also affect it. Throughput
benchmarks default to `response_mean` and record the setting.

Run the CLI as `PYTHONPATH=src python -m sts_llm.training --help`. Training input
is the teacher-prepared JSONL; preserve held-out worlds separately. A bounded
mechanics run is `--smoke --group-size 4 --max-new-tokens 512 --loss-normalization response_mean`. The cap can cause
invalid truncated thoughts; report this with the reward variance and validity.
The main study requires separately approved settings and is not launched here.

Training requires an explicit `split` on every row and selects only `train`.
Validation/test rows are never sampled for updates. Every row's split is
recomputed from its `world_identity`; a missing world or a stored split that
disagrees stops the run, as does a `prompt_version` or `observation_version`
other than the current one (benchmark modes check these versions only when a
row records them). The selected training worlds are written to `config.json`
as `training_world_identities`, so full-game evaluation can refuse them. Benchmark modes may consume
all explicitly provided states and record that selection as benchmark-only.
The selected split, row count, resolved selection hash and limit are recorded;
exact resume validates them as well as the whole dataset hash.

`--benchmark-generation` measures this actual HF rollout path, excludes
`--warmup-steps` initial groups, retains generated IDs/text and performs no
optimizer updates. Compare generation microbatches using fixed `--group-size`.
vLLM serving throughput is a different execution path.

`--reuse-prompt-cache` is off by default. When enabled, each state prompt
(except its last token) is prefetched once under the current adapter, then
independent copies of its KV cache are expanded for each generation microbatch.
The cache is discarded before another state or any optimizer update. Full
input IDs and attention masks still reach HF generation, which processes only
the uncached suffix. The flag is recorded and validated on exact resume.
Before timing the enabled path, run `--check-prompt-cache` on representative
states with the intended `--generation-microbatch`; it compares next-token
logits, greedy token IDs and any completed legal actions, and verifies that the
shared prefix length stays unchanged. `--cache-check-tokens` controls the greedy
probe cap (default 32); an unfinished prefix cannot establish final-action
equivalence. The default BF16 logit tolerance is absolute 0.125, relative 0.01,
and actual differences are preserved. HF5.19 uses a different stopping-check
path for supplied caches, so the benchmark measures that full HF implementation.

`--static-cache-compile` is a separate, benchmark-generation-only option using
a static cache and HF partial-graph `torch.compile` with unmerged LoRA. It is
incompatible with prompt-cache reuse. Every prompt plus completion budget must
fit `--static-cache-size` (default 8192); HF's reservation option is a floor,
so this explicit validation prevents silent growth. Use an isolated command
with a shell deadline, for example `timeout -s TERM 180 ...`. A separate two-token
cold call records compilation startup/time before ordinary warmup and timed
generation. Static warmup uses the actual microbatch size, and partial tail
batches are rejected to avoid another unmeasured shape compilation. No compile
speedup or equivalence is claimed until actual K2 CUDA validation succeeds.

`--benchmark-updates` performs real forwards/backwards, gradient clipping and
AdamW optimizer steps. It excludes warmup steps and measures CUDA peak memory,
gradient norms, ratios, clipping and parameter changes. Optional
`--cohort-jsonl` consumes saved vLLM records with actual output token IDs; this
is explicitly a throughput-only replay, not on-policy training. Benchmark
advantages are synthetic to ensure the backward path is exercised. No adapter
is saved and the source checkpoint is never edited.

To stop a training run, create its output directory's `STOP` file, or the file
specified by `--stop-file`. The loop checks between iterations and optimizer
steps. Completed checkpoints remain intact; incomplete cohorts and the number
of partial updates are recorded. Partial updates are not promoted.

Every completed `adapter_NNNN/` contains LoRA weights, tokenizer files and
`training_state.pt` with optimizer and Python/Torch/CUDA RNG state. Resume with
`--resume /path/to/adapter_NNNN --output /new/empty/output` and the original
training settings; `--iterations` is the desired total iteration count.
The incomplete next iteration is regenerated. A changed model, data, sampling
or optimizer setting fails exact-resume validation. Device/kernel determinism
still depends on the pinned runtime. `--warm-start-adapter` (alias `--adapter`)
loads weights and explicitly resets optimizer/RNG; it is a new condition.
Before loading, the adapter's saved training state is read: a recorded model,
revision, effort, trainer format version or prompt version that differs from
this run is refused unless `--allow-adapter-mismatch` is given. The check result
is recorded in `config.json` as `adapter_compatibility`. A directory with no saved
training state loads and is recorded as `unrecorded`. Source-file hashes are
never compared.

Config records model revision, input and source hashes, teacher checkpoint,
actual framework/device details, and warm-start/resume semantics. Checkpoints
are staged in `.pending` directories and promoted only after state saving.
Failures write `failure.json` with the last completed checkpoint and retained
cohort paths. A pending checkpoint is never silently overwritten.

Tests: `PYTHONPATH=src python -m unittest tests.unit.test_llm_training -v`.
Numerical, gradient, token-alignment and cache-isolation tests require PyTorch.
Validate the full test set and a real CUDA update before a study.

The initial BF16 cache probes were not bitwise equivalent: a completed greedy
combat action changed. FP32 probes on the same two states matched within
0.0001 logit tolerance, but only at low effort with about 10-token outputs;
there is no medium-effort FP32 control, although the BF16 check also failed at
medium effort. Batch-size changes also
shifted BF16 logits. Keep cache/backend/dtype settings fixed within a study;
treat a change as a new numerical condition. The optional HF reuse path stays
off by default. Frozen old-policy scores are recomputed full-sequence scores;
freezing them does not prove equality to cached-decoding sampling probabilities.
This numerical difference should be quantified before the main learning study.

## Teacher-feedback training on Apple silicon (MLX)

`python -m sts_llm.mlx_training` trains a LoRA adapter on the MLX conversion of
K2-Horizon. Build the conversion with
`experiments/2026-10-25-mac-local/mlx_model.py`; its FP32 log-probabilities
match the original model within 0.0013. The input files are JSONL states with
`observation` (phase, text, menu), `teacher`, `world_identity` and `split`.
`--train` must hold training-split worlds and `--probe` validation-split worlds;
`dataset.verified_split` checks both.

Each iteration:
- takes `--states-per-iteration` states, cycling through a shuffled order so
  every state is seen equally often;
- presents each state under prompt version 2 with a fresh random menu order
  for every sample (`--menu-order per_sample`, format 2). Format 1 shared one
  order across the group, which lets the group baseline reward a label as much
  as its action; `per_group` keeps that behaviour for comparison. The permuted
  prompts have identical token lengths, so batching speed is unchanged;
- samples `--group-size` answers in one batch from the live adapter
  (temperature `--temperature`, top-p 1, cap `--max-new-tokens`);
- rewards a valid answer with the teacher's probability of the chosen action,
  and gives invalid or capped output `--invalid-reward` (default -1);
- subtracts the group mean and takes one AdamW step.

This is REINFORCE with a group-mean baseline. With one update per sampled
batch, a PPO ratio would be exactly one, so there is no clipping and no
old-policy scoring pass. Each response's summed generated-token
log-probability is divided by `--loss-token-budget` (default 2048), or by its
own length with `--loss-normalization response_mean`. Responses with zero
advantage are skipped in the backward pass. An optional sampled KL penalty to
the base model (`--kl-beta`, default 0) scores references with the adapters
switched off. Gradient checkpointing is on by default; on this machine it is
both faster and smaller.

Every completion is written to `generations/iter_NNNN.jsonl`, including its
raw text and reasoning, menu order, label, reward, advantage, teacher
probability, top-choice agreement, normalised score and reasoning-token count.
`iterations.jsonl` holds per-iteration metrics, overall and per phase:
- reward and validity;
- zero-variance groups, group reward range and distinct actions;
- first-displayed-option rate and reasoning length;
- gradient norm and timings.

Every `--eval-every` iterations, and before training, a greedy evaluation runs
fixed menu orders (`--eval-orders` per state) on the training states and the
probe. Results go to `evaluations/` and `evaluations.jsonl`. Adapters and their
metadata (format version, model revision, LoRA shape, effort, prompt version)
are saved to `adapters/`. `config.json` records the source commit, source
hashes, MLX versions, device, input hashes and teacher checkpoint. Create
`OUTPUT/STOP` to stop between iterations. `--adapter` warm-starts weights; the
optimizer and random state start fresh.

## Full-game outcome learning and evaluation

`sts_llm.game_training` provides the later full-game path. It reuses the clean
environment's `run_episode`, with native process isolation and finite operation
timeouts. One current HF/PEFT model in the parent chooses combat and out-of-combat
actions. Its own generated reasoning, final action and EOS tokens are trained;
there are no teacher targets or teacher reasoning in this module.

Games are synchronous and currently collected sequentially. This is a complete
minimal execution path, not a throughput-optimized collector or evidence of
successful full-game learning. The module has fake-environment/unit tests. No
actual full-game GPU learning run has been performed.

Use a JSON seed file containing only the explicitly selected training worlds.
Full-game checkpoints record those worlds (and, for a warm start, the worlds the
starting adapter was trained on) in `game_training_state.pt`.
World seeds must be distinct integers in `[0, 2**63 - 1]`, matching the clean
environment. Both training and evaluation require `--max-act 3`.
Training repeats each world `group-size` times with separate policy seeds, freezes
the behavior and base-reference log probabilities before any optimizer update,
then assigns each game's terminal advantage to all of its generated decisions.
Rewards must be explicit; the intended values are win +1 and defeat -1.

Every game ends in exactly one status, recorded in its `result.json` and
counted separately in each `collection_NNNN.json` (`status_counts`) and
`summary.json`:

| Status | Meaning | Scored |
|---|---|---|
| `win`, `loss` | The game reached a genuine ending | `--win-reward`, `--loss-reward` |
| `model_invalid` | The model's output was unparseable, illegal or cut off at `--max-new-tokens`; the game ends there | `--invalid-game-reward` (default and maximum: the loss reward) |
| `simulator_fault` | A simulator error, timeout or undefined behaviour | No |
| `budget_truncated` | The decision or act limit, or a prompt that cannot fit `--max-new-tokens` within `--max-sequence-length` | No |
| `unknown_terminal` | Terminated without a recognized outcome | No |
| `stopped` | A STOP file or interrupt during the game | No |

A model-invalid game stays in its comparison group, and all of its decisions,
including the invalid completion, receive the game's advantage; invalid output
is therefore pushed down like any losing move. One unscored game excludes its
entire group, and unscored games are never counted as losses. Any other
exception inside the policy (for example CUDA out of memory or a chat-template
error) is re-raised after the game's `result.json` is written (status
`policy_error`), so the run writes `failure.json` and exits nonzero. Every game
retains a clean replayable `game.jsonl`, a model `responses.jsonl` and `result.json`.

Each training iteration records `eligible_groups`, `groups_with_outcome_signal`,
`updated` and, when no update was possible, `update_skipped`. After
`--max-no-update-iterations` (default 3) consecutive iterations without an
optimizer update the run stops with an error naming the latest status counts.
Checkpoints up to that point remain valid; resuming with a larger limit is allowed.

These choices must be reviewed before a study:

- The invalid-game reward makes invalid output at least as bad as a loss. A game
  that ends early on invalid output and a game lost after hundreds of decisions
  receive the same reward by default.
- The baseline/normalization is the shared trainer's group mean and population
  standard deviation. Constant-reward groups have no outcome-policy gradient.
- Decisions are summed within each game, then games are averaged. The helper
  rescales both the policy advantage and KL coefficient to undo the shared
  optimizer's completion mean. Forced one-option menus need no model call and
  have no policy gradient.
- `--loss-normalization` is required. Response means still divide by each
  response's realized length; fixed normalization uses an explicit study-wide
  constant. The KL reference is the frozen foundation model with adapters
  disabled, including when warm-starting from a teacher-trained adapter.
- The prompt is the current public state/menu, without conversation history.
  This is not a claim that the public observation contains all useful memory.

Mechanics smoke, intentionally too bounded to establish learning:

```sh
PYTHONPATH=src python -m sts_llm.game_training \
  --seeds-file data/llm_runs/TRAINING_WORLDS.json --policy-seed 1729 \
  --output data/llm_runs/full_game_mechanics \
  --effort medium --temperature 1 --max-new-tokens 512 \
  --max-sequence-length 8192 --max-decisions 20 --max-act 3 --ascension 0 \
  --step-timeout 10 --win-reward 1 --loss-reward -1 \
  --group-size 2 --worlds-per-iteration 1 --iterations 1 \
  --loss-normalization fixed_token_budget --loss-token-budget 2048
```

The decision limit above normally produces budget-truncated groups, which
verifies fault handling without pretending to provide positive learning signal;
a longer run of such iterations stops at the no-update limit. Shorter
act limits are rejected: they could retain early defeats while truncating
stronger games and silently change the learning target. For an
approved learning study choose adequate decision/output budgets, the
world/group/update counts and normalization in advance. Run a bounded shell
timeout around diagnostic commands if a hard wall-clock limit is needed.

For full-game evaluation of a trained checkpoint, add `--evaluate-only
--warm-start-adapter PATH_TO_ADAPTER` and use a separate explicit seed file,
fresh output, `--group-size 1`, Acts 1–3 and the chosen evaluation budgets. It
loads the actual PEFT adapter into the same HF policy path; no serving endpoint
or merge/unmerge is required. Evaluation performs no optimizer update and saves
no adapter. Seeds that the adapter (or the adapter it was warm-started from) was
trained on are refused unless `--allow-training-worlds` is given; teacher-trained
adapters are checked against `training_world_identities` in their run's
`config.json`, and adapters from runs that did not record their worlds are
reported as unrecorded. The check is saved as `evaluation_world_check` in
`config.json` and `summary.json`. The adapter compatibility check above also
applies. `summary.json`'s `evaluation` block counts every status separately and
reports `win_rate_over_attempted` (wins over all games started, so faults and
truncations count as non-wins) and `win_rate_over_scored` (wins over win, loss
and model-invalid games). Model-invalid games count against the model; faults
and truncations are never losses. Temperature zero selects greedy evaluation;
positive temperatures sample. It does not by itself compute a paired win-rate
interval or reasoning faithfulness.

Create the output's `STOP` file to stop between decisions/games/whole optimizer
updates. Every completed checkpoint includes optimizer and Python/Torch/CUDA RNG
state, starting with `adapter_initial`. Resume using `--resume PATH --output NEW`
and the same scientific settings/worlds; `--iterations` is the desired total.
Partial iteration outputs remain intact and that iteration is regenerated from
the last completed checkpoint. Warm starts reset optimizer/RNG. Source hashes
record provenance; they are not byte-compatibility gates. Reproducing exact
trajectories additionally requires the same runtime and native build recorded
in the traces. Full-game format version 2 (scored model-invalid games and the
status names above) cannot resume a version-1 checkpoint; warm-starting from one
is allowed. Tests cover status classification, credit, failure propagation and,
with PyTorch, the HF policy's stop-token trimming, the iteration loop with a
fake environment, evaluation accounting and the command-line refusals.
