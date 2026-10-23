# Initial LLM training

This package trains generated reasoning and final actions with single-decision
non-LLM policy-probability rewards. It has no reasoning SFT targets. Merge its
`src/sts_llm/training.py` with the shared generation, observation and teacher
modules before running it.

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

Tests: `PYTHONPATH=src python -m unittest discover -s tests -p test_training.py -v`.
Five numerical/gradient/token-alignment tests require PyTorch; the other tests
run without it. Validate the full test set and a real CUDA update before a study.
