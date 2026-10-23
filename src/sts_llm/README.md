# Public-observation LLM adapter

This package provides single-decision teacher-guided training inputs and saved-state/full-game evaluation over the released clean environment. It does not change `sts_env` or start a training study. Teacher probability measures the champion's preference, not expected game wins or reasoning quality.

## APIs

- `observations.prompt(observation)` returns the shared two-message prompt. It copies only complete public observation text and the raw current action menu. It never serializes `info`, world seeds, teacher labels, provenance or policy features into messages. `legal_indices(observation)` preserves native display indices.
- `generation.parse_action(text, legal_indices, finish_reason=None, reasoning_content=None, final_channel=False, require_reasoning_boundary=True)` returns `ActionParse(action_index, valid, error, reasoning, final_text, truncated)`. Raw K2 completions require a reasoning closer; any of its three effort closers is accepted, regardless of requested effort. The opening tag may be absent because the chat template already supplies it. A native separate reasoning field establishes a final channel. Truncation is a failure even if the visible suffix contains JSON. Final output must be one JSON object with only an integer `action_index`. The actual K2 assistant-end and end-of-text tokens are stripped after the boundary.
- `teacher.NonLLMTeacher(teacher_root=..., checkpoint=..., python=..., timeout=60)` launches one isolated inference subprocess. `distribution(observation)` or `distributions(observations)` returns display indices, probabilities, excluded menu indices, checkpoint identity and source revision. The parent never imports the non-LLM environment package. No native simulator is loaded or queried.
- `teacher.teacher_reward(row, action_index, mode='probability', invalid_reward=None, log_floor=1e-8)` uses the prepared row's teacher distribution. Probability is the default bounded reward. A legal action filtered out of the champion's deployment menu receives zero probability and remains distinct from malformed/illegal output. Log probability is optional; its invalid reward must be no larger than `log(log_floor)`, to avoid rewarding invalid output over valid low-probability choices. Default invalid reward is zero for probability or `log(log_floor)` for log probability.
- `dataset.prepare_rows(rows, teacher)` prepares messages and complete target distributions with a stable train/validation/test split over whole source worlds. Source file and public-record identities are checked before loading. Repeated decisions from one world remain in one split.
- `eval.evaluate_saved(rows, client, output=...)` reports legality, truncation, mean teacher probability, top-action agreement and filtered-menu actions. `eval.make_agent(...)` provides `choose_action` for the clean CLI. Invalid model output raises `InvalidModelAction`; it is not turned into a game loss. Forced one-option menus can be stepped without generation. Full response logs retain messages, parse results and raw API responses outside model inputs.

The HTTP client accepts explicit `extra_body` for the serving engine's model-specific reasoning/template options. It does not guess how an endpoint implements effort settings. Raw Hugging Face training completions should retain special tokens when decoding so reasoning boundaries remain visible.

## Prepare data and a portable teacher

Run with this package on `PYTHONPATH`; the teacher's Python needs PyTorch and NumPy. The clean parent runtime needs only the standard library for preparation/evaluation adapters.

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

Input rows can contain `observation` plus `world_identity`, or the immutable `source_record_identity` emitted by the throughput prompt preparer. Output JSONL contains `id`, `phase`, `legal_indices`, `messages`, `teacher`, `world_identity`, `split` and provenance. Submit only `messages` to the model. `teacher` uses `{legal_indices: [...], probabilities: [...], excluded_indices: [...]}`; its menu is a subset of the raw prompt menu because the champion hides OOC potion and key/recall actions.

The portable teacher export contains only the source modules actually imported for inference, a checkpoint if requested, and `teacher_runtime_manifest.json`. It is a dependency artifact for `data/`, not core environment code. Point `teacher_root` at this exported directory and `checkpoint` at the included file on another machine. Install compatible PyTorch/NumPy; no native simulator binary is needed. Source hashes record provenance and do not impose byte-for-byte checkpoint compatibility gates.

The prepared scratch example `data/teacher128.jsonl` has 128 distinct states from 93 worlds, 80 combat and 48 OOC; 92 training, 17 validation and 19 test rows. This tiny set is for smoke and timing checks, not a scientific training dataset. Its source observation files match `env-v0.1.5.1`; that speed-only release intentionally reports runtime `0.1.5` and `sts_env_public_v3`.

## Evaluate

```sh
PYTHONPATH=src python -m sts_llm.eval \
  --dataset data/llm_runs/pilot/teacher.jsonl --split test \
  --base-url http://127.0.0.1:8000/v1 --model MODEL_NAME \
  --max-tokens 4096 --temperature 0 \
  --extra-body '{}' \
  --output data/llm_runs/pilot/saved_eval.jsonl
```

The example template option must match the selected serving implementation. For full-game evaluation use the existing clean CLI's `--agent sts_llm.eval:make_agent`, with endpoint/model configuration and a fresh `response_log`. Its ordinary game trace remains the authority for game outcomes, simulator faults and truncations; saved-state agreement is not a win-rate estimate. Greedy and sampled decoding differ through the explicit temperature. Environment operation timeouts/process isolation stay in the clean CLI.

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
Validation/test rows are never sampled for updates. Benchmark modes may consume
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

Config records model revision, input and source hashes, teacher checkpoint,
actual framework/device details, and warm-start/resume semantics. Checkpoints
are staged in `.pending` directories and promoted only after state saving.
Failures write `failure.json` with the last completed checkpoint and retained
cohort paths. A pending checkpoint is never silently overwritten.

Tests: `PYTHONPATH=src python -m unittest tests.unit.test_llm_training -v`.
Numerical, gradient, token-alignment and cache-isolation tests require PyTorch.
Validate the full test set and a real CUDA update before a study.
